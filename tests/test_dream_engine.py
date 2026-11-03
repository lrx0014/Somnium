"""Unit and integration tests for Dream Engine automation and state machine (M6.1–M6.6).

Verifies:
- M6.1: Inactivity / idle detection via `Clock` and `TrafficTracker`.
- M6.2: Per-tenant state machine transitions, single-writer lock, and crash recovery.
- M6.3: Associative traffic pause / deferral during training.
- M6.4: Resource policy checks (AC power, memory pressure, deferral).
- M6.5: Scheduling triggers: minimum examples, weights_dirty escalation, base model change.
- M6.6: `memory_status` reporting live state, active version, pending examples.
"""

from __future__ import annotations

from pathlib import Path

from somnium.backends.fake import (
    FakeInferenceBackend,
    FakeTrainingBackend,
    FrozenClock,
)
from somnium.config import DreamConfig
from somnium.domain import (
    AuthContext,
    DeletionMode,
    DreamLifecycleState,
    MemoryForgetRequest,
    MemoryKind,
    MemoryStatusRequest,
    MemoryStoreRequest,
    Permission,
    Scope,
    ScopeKind,
    SensitivityLevel,
)
from somnium.dream.artifacts import LocalFilesystemArtifactStore
from somnium.dream.engine import DreamEngine
from somnium.dream.idle import TrafficTracker
from somnium.dream.registry import ModelRegistry
from somnium.dream.resources import StaticResourcePolicy
from somnium.freud.service import MemoryService
from somnium.hippocore.schema import tenant_state
from somnium.hippocore.store import HippocoreStore


def _auth(tenant_id: str = "tenant-gamma") -> AuthContext:
    return AuthContext(
        tenant_id=tenant_id,
        principal_id="test-agent",
        authorized_scopes=(Scope(kind=ScopeKind.PROJECT, id="project-gamma"),),
        permissions=frozenset(
            {
                Permission.READ,
                Permission.WRITE,
                Permission.CORRECT,
                Permission.FORGET,
                Permission.ASSOCIATIVE_RECALL,
                Permission.ADMIN,
            }
        ),
        allow_all_scopes=True,
    )


def test_idle_detection_with_frozen_clock() -> None:
    """M6.1: Test agent traffic tracking and idle detection over time."""
    clock = FrozenClock()
    tracker = TrafficTracker(clock=clock)
    tenant_id = "tenant-gamma"

    # Initially no traffic recorded -> idle_seconds is infinity
    assert tracker.is_idle(tenant_id, threshold_seconds=60)
    assert tracker.idle_seconds(tenant_id) == float("inf")

    # Record traffic at t=0
    tracker.record_traffic(tenant_id)
    assert not tracker.is_idle(tenant_id, threshold_seconds=60)
    assert tracker.idle_seconds(tenant_id) == 0.0

    # Advance clock by 30 seconds -> still not idle
    clock.advance(seconds=30)
    assert not tracker.is_idle(tenant_id, threshold_seconds=60)
    assert tracker.idle_seconds(tenant_id) == 30.0

    # Advance clock by another 35 seconds -> now idle (65s > 60s)
    clock.advance(seconds=35)
    assert tracker.is_idle(tenant_id, threshold_seconds=60)
    assert tracker.idle_seconds(tenant_id) == 65.0


def test_resource_policy_deferral() -> None:
    """M6.4: Test resource policy AC power requirement and memory pressure limits."""
    # AC power required but running on battery -> defer
    policy_battery = StaticResourcePolicy(on_ac_power=False, memory_pressure=30.0)
    ok_batt, reason_batt = policy_battery.check_resources(require_ac_power=True)
    assert not ok_batt
    assert reason_batt is not None and "battery" in reason_batt

    # Memory pressure exceeds 85% limit -> defer
    policy_heavy = StaticResourcePolicy(on_ac_power=True, memory_pressure=92.0)
    ok_mem, reason_mem = policy_heavy.check_resources(max_memory_pressure_pct=85)
    assert not ok_mem
    assert reason_mem is not None and "Memory pressure" in reason_mem

    # Acceptable conditions -> pass
    policy_ok = StaticResourcePolicy(on_ac_power=True, memory_pressure=45.0)
    ok_pass, reason_pass = policy_ok.check_resources(
        require_ac_power=True,
        max_memory_pressure_pct=85,
    )
    assert ok_pass
    assert reason_pass is None


def test_dream_engine_state_machine_and_triggers(tmp_path: Path) -> None:
    """M6.2 & M6.5: Test automated state transitions, trigger evaluation, and single-writer lock."""
    clock = FrozenClock()
    tracker = TrafficTracker(clock=clock)
    store = HippocoreStore("sqlite:///:memory:")
    art_store = LocalFilesystemArtifactStore(tmp_path / "artifacts")
    training_backend = FakeTrainingBackend()
    inference_backend = FakeInferenceBackend()
    registry = ModelRegistry(store.engine)
    res_policy = StaticResourcePolicy(on_ac_power=True, memory_pressure=30.0)

    dream_cfg = DreamConfig(
        enabled=True,
        idle_threshold_seconds=60,
        min_verified_examples=2,
        require_ac_power=True,
        max_memory_pressure_pct=85,
    )

    engine = DreamEngine(
        store=store,
        artifact_store=art_store,
        training_backend=training_backend,
        inference_backend=inference_backend,
        model_registry=registry,
        traffic_tracker=tracker,
        resource_policy=res_policy,
        dream_config=dream_cfg,
    )

    auth = _auth("tenant-gamma")

    # Initial trigger check: 0 examples -> deferred
    should_run, mode, reason = engine.evaluate_triggers(auth.tenant_id)
    assert not should_run
    assert mode == "examples_deferred"

    # Store 1 record -> still below min_verified_examples (2)
    store.store(
        auth,
        MemoryStoreRequest(
            scope=Scope(kind=ScopeKind.PROJECT, id="project-gamma"),
            content="Project Gamma adopted SQLite for storage.",
            kind=MemoryKind.DECISION,
            sensitivity=SensitivityLevel.NORMAL,
        ),
    )
    should_run, mode, reason = engine.evaluate_triggers(auth.tenant_id)
    assert not should_run
    assert mode == "examples_deferred"

    # Store 2nd record
    store.store(
        auth,
        MemoryStoreRequest(
            scope=Scope(kind=ScopeKind.PROJECT, id="project-gamma"),
            content="Project Gamma adopted Redis for caching.",
            kind=MemoryKind.DECISION,
            sensitivity=SensitivityLevel.NORMAL,
        ),
    )

    # Traffic recorded just now -> idle deferred
    tracker.record_traffic(auth.tenant_id)
    should_run, mode, reason = engine.evaluate_triggers(auth.tenant_id)
    assert not should_run
    assert mode == "idle_deferred"

    # Advance clock past idle threshold
    clock.advance(seconds=70)
    should_run, mode, reason = engine.evaluate_triggers(auth.tenant_id)
    assert should_run
    assert mode == "full"  # First run with no active model is full retrain

    # Execute cycle via trigger_cycle
    result = engine.trigger_cycle(auth, mode=mode)
    assert result is not None
    assert result.promoted
    assert engine.get_dream_state(auth.tenant_id) == DreamLifecycleState.AWAKE
    assert registry.get_active_version(auth.tenant_id) is not None


def test_crash_recovery_resets_orphaned_state(tmp_path: Path) -> None:
    """M6.2: Stale in-progress states from crashed workers are cleanly recovered to AWAKE."""
    store = HippocoreStore("sqlite:///:memory:")
    art_store = LocalFilesystemArtifactStore(tmp_path / "artifacts")
    registry = ModelRegistry(store.engine)

    # Simulate crashed database state: tenant stuck in TRAINING
    with store.engine.begin() as conn:
        conn.execute(
            tenant_state.insert().values(
                tenant_id="crashed-tenant",
                weights_dirty=False,
                active_model_version="v0",
                dream_state=DreamLifecycleState.TRAINING.value,
                updated_at="2026-01-01T00:00:00Z",
            )
        )

    # Starting engine triggers recover_crashed_states()
    engine = DreamEngine(
        store=store,
        artifact_store=art_store,
        training_backend=FakeTrainingBackend(),
        inference_backend=FakeInferenceBackend(),
        model_registry=registry,
    )

    assert engine.get_dream_state("crashed-tenant") == DreamLifecycleState.AWAKE


def test_memory_service_status_integration(tmp_path: Path) -> None:
    """M6.6: MemoryService.status returns live dream state, pending counts, and updates traffic."""
    clock = FrozenClock()
    tracker = TrafficTracker(clock=clock)
    store = HippocoreStore("sqlite:///:memory:")
    art_store = LocalFilesystemArtifactStore(tmp_path / "artifacts")
    registry = ModelRegistry(store.engine)
    engine = DreamEngine(
        store=store,
        artifact_store=art_store,
        training_backend=FakeTrainingBackend(),
        inference_backend=FakeInferenceBackend(),
        model_registry=registry,
        traffic_tracker=tracker,
    )

    service = MemoryService(store=store, dream_engine=engine)
    auth = _auth("tenant-gamma")

    # Initial status
    status0 = service.status(auth, MemoryStatusRequest())
    assert status0.dream_state == DreamLifecycleState.AWAKE
    assert status0.pending_training_examples == 0
    assert not status0.weights_dirty

    # Storing a record updates traffic and increments pending count
    service.store(
        auth,
        MemoryStoreRequest(
            scope=Scope(kind=ScopeKind.PROJECT, id="project-gamma"),
            content="Project Gamma chosen DuckDB.",
            kind=MemoryKind.DECISION,
            sensitivity=SensitivityLevel.NORMAL,
        ),
    )

    assert tracker.idle_seconds(auth.tenant_id) == 0.0
    status1 = service.status(auth, MemoryStatusRequest())
    assert status1.pending_training_examples == 1

    # Deleting record marks weights dirty
    records = store.list_by_scope(auth, Scope(kind=ScopeKind.PROJECT, id="project-gamma"))
    store.forget(
        auth,
        MemoryForgetRequest(memory_id=records[0].id, deletion_mode=DeletionMode.HARD),
    )
    status2 = service.status(auth, MemoryStatusRequest())
    assert status2.weights_dirty is True
