"""Dream Engine automation, state machine, and scheduling coordinator (M6.1–M6.5).

Per `design.md` §7:
- Per-tenant state machine:
  AWAKE -> IDLE_ELIGIBLE -> DREAM_PREPARING -> TRAINING -> EVALUATING -> PROMOTING / REJECTED ->
  AWAKE.
- Inactivity threshold detection backed by `TrafficTracker`.
- Evaluates triggers: minimum new verified examples, `weights_dirty` escalation,
  base model change, and environment resource policy.
- Single-writer per-tenant execution lock.
- Supports pause / deferral on associative recall traffic in dev profile.
- Crash recovery resets orphaned in-progress states back to AWAKE.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from sqlalchemy import and_, select

from somnium.backends.base import InferenceBackend, TrainingBackend
from somnium.config import DreamConfig
from somnium.domain import (
    AuthContext,
    DreamLifecycleState,
    MemoryStatus,
    SensitivityLevel,
    TenantId,
)
from somnium.dream.augmentation import AugmentationCache
from somnium.dream.gates import GateTolerances
from somnium.dream.idle import TrafficTracker
from somnium.dream.pipeline import DreamCycleResult, execute_dream_cycle
from somnium.dream.registry import ModelRegistry
from somnium.dream.resources import ResourcePolicy, SystemResourcePolicy
from somnium.hippocore.schema import (
    dataset_manifests,
    dream_runs,
    memory_facts,
    tenant_state,
)
from somnium.hippocore.store import HippocoreStore
from somnium.interfaces import ArtifactStore

logger = logging.getLogger("somnium.dream.engine")


class DreamEngine:
    """Coordinates automated background Dream cycles and per-tenant state machines (M6)."""

    def __init__(
        self,
        *,
        store: HippocoreStore,
        artifact_store: ArtifactStore,
        training_backend: TrainingBackend,
        inference_backend: InferenceBackend | None = None,
        model_registry: ModelRegistry | None = None,
        traffic_tracker: TrafficTracker | None = None,
        resource_policy: ResourcePolicy | None = None,
        augmentation_cache: AugmentationCache | None = None,
        dream_config: DreamConfig | None = None,
        base_model_id: str = "Qwen/Qwen3-1.7B",
        base_model_revision: str = "main",
        tolerances: GateTolerances | None = None,
        tick_interval_seconds: float = 1.0,
    ) -> None:
        self.store = store
        self.artifact_store = artifact_store
        self.training_backend = training_backend
        self.inference_backend = inference_backend
        self.registry = model_registry or ModelRegistry(store.engine)
        self.traffic_tracker = traffic_tracker or TrafficTracker()
        self.resource_policy = resource_policy or SystemResourcePolicy()
        self.augmentation_cache = augmentation_cache
        self.dream_config = dream_config or DreamConfig()
        self.base_model_id = base_model_id
        self.base_model_revision = base_model_revision
        self.tolerances = tolerances
        self.tick_interval_seconds = tick_interval_seconds

        self._tenant_locks: dict[TenantId, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._pause_events: dict[TenantId, threading.Event] = {}
        self._tracked_tenants: set[TenantId] = set()

        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Perform startup crash recovery
        self.recover_crashed_states()

    def _get_tenant_lock(self, tenant_id: TenantId) -> threading.Lock:
        with self._global_lock:
            if tenant_id not in self._tenant_locks:
                self._tenant_locks[tenant_id] = threading.Lock()
            return self._tenant_locks[tenant_id]

    def register_tenant(self, tenant_id: TenantId) -> None:
        """Register a tenant for automated background Dream monitoring."""
        with self._global_lock:
            self._tracked_tenants.add(tenant_id)

    def record_traffic(self, tenant_id: TenantId) -> None:
        """Record external agent traffic for `tenant_id`."""
        self.register_tenant(tenant_id)
        self.traffic_tracker.record_traffic(tenant_id)

    def notify_associative_traffic(self, tenant_id: TenantId) -> None:
        """Notify the engine that associative recall traffic has arrived for `tenant_id`."""
        self.record_traffic(tenant_id)
        if self.dream_config.pause_on_associative_traffic:
            with self._global_lock:
                ev = self._pause_events.get(tenant_id)
                if ev is not None:
                    ev.set()

    def get_dream_state(self, tenant_id: TenantId) -> DreamLifecycleState:
        """Fetch current persisted dream state from `tenant_state` table."""
        with self.store.engine.connect() as conn:
            row = conn.execute(
                select(tenant_state.c.dream_state).where(tenant_state.c.tenant_id == tenant_id)
            ).first()
            if row is not None and row[0]:
                try:
                    return DreamLifecycleState(str(row[0]))
                except ValueError:
                    return DreamLifecycleState.AWAKE
        return DreamLifecycleState.AWAKE

    def set_dream_state(self, tenant_id: TenantId, state: DreamLifecycleState) -> None:
        """Atomically persist tenant dream state in database."""
        now_iso = datetime.now(tz=UTC).isoformat()
        with self.store.engine.begin() as conn:
            row = conn.execute(
                select(tenant_state.c.tenant_id).where(tenant_state.c.tenant_id == tenant_id)
            ).first()
            if row is None:
                conn.execute(
                    tenant_state.insert().values(
                        tenant_id=tenant_id,
                        weights_dirty=False,
                        active_model_version=None,
                        dream_state=state.value,
                        updated_at=now_iso,
                    )
                )
            else:
                conn.execute(
                    tenant_state.update()
                    .where(tenant_state.c.tenant_id == tenant_id)
                    .values(dream_state=state.value, updated_at=now_iso)
                )

    def recover_crashed_states(self) -> None:
        """Scan for orphaned in-progress states on startup and reset them to AWAKE (M6.2)."""
        now_iso = datetime.now(tz=UTC).isoformat()
        with self.store.engine.begin() as conn:
            # Mark open dream_runs as interrupted
            conn.execute(
                dream_runs.update()
                .where(dream_runs.c.state.in_(["running", "training", "evaluating", "preparing"]))
                .values(
                    state="interrupted",
                    forced_full_retrain_reason="Interrupted by process restart or crash.",
                    completed_at=now_iso,
                )
            )

            # Reset any non-terminal tenant_state back to awake
            conn.execute(
                tenant_state.update()
                .where(
                    tenant_state.c.dream_state.in_(
                        [
                            DreamLifecycleState.DREAM_PREPARING.value,
                            DreamLifecycleState.TRAINING.value,
                            DreamLifecycleState.EVALUATING.value,
                            DreamLifecycleState.PROMOTING.value,
                        ]
                    )
                )
                .values(dream_state=DreamLifecycleState.AWAKE.value, updated_at=now_iso)
            )

    def count_pending_examples(self, tenant_id: TenantId) -> int:
        """Count training-eligible records for `tenant_id` not yet included in active adapter."""
        active_version = self.registry.get_active_version(tenant_id)
        parent_manifest_hash: str | None = None
        if active_version:
            v_rec = self.registry.get_version(active_version)
            if v_rec:
                parent_manifest_hash = v_rec.dataset_manifest_hash

        with self.store.engine.connect() as conn:
            parent_ids: set[str] = set()
            if parent_manifest_hash:
                stmt_m = select(dataset_manifests.c.source_record_ids_json).where(
                    dataset_manifests.c.dataset_hash == parent_manifest_hash
                )
                m_row = conn.execute(stmt_m).first()
                if m_row is not None and m_row[0]:
                    import json

                    parent_ids = set(json.loads(m_row[0]))

            stmt = select(memory_facts.c.id).where(
                and_(
                    memory_facts.c.tenant_id == tenant_id,
                    memory_facts.c.training_eligible.is_(True),
                    memory_facts.c.sensitivity == SensitivityLevel.NORMAL.value,
                    memory_facts.c.status.in_(
                        [MemoryStatus.ACTIVE.value, MemoryStatus.SUPERSEDED.value]
                    ),
                )
            )
            rows = conn.execute(stmt).fetchall()
            eligible_ids = [str(r[0]) for r in rows]

            if not parent_ids:
                return len(eligible_ids)
            return len([i for i in eligible_ids if i not in parent_ids])

    def evaluate_triggers(self, tenant_id: TenantId) -> tuple[bool, str, str]:
        """Evaluate scheduling triggers: (should_run, mode, reason) (M6.5)."""
        if not self.dream_config.enabled:
            return False, "disabled", "Dream Engine is disabled in configuration."

        # 1. Check if weights are dirty from a training-eligible deletion
        if self.registry.is_weights_dirty(tenant_id):
            return True, "full", "Tenant weights marked dirty from training-eligible deletion."

        # 2. Check base model change
        active_version = self.registry.get_active_version(tenant_id)
        if active_version:
            v_rec = self.registry.get_version(active_version)
            if v_rec and v_rec.base_model_id != self.base_model_id:
                return (
                    True,
                    "full",
                    f"Base model changed from {v_rec.base_model_id} to {self.base_model_id}.",
                )

        # 3. Check idle condition
        idle_sec = self.traffic_tracker.idle_seconds(tenant_id)
        if not self.traffic_tracker.is_idle(tenant_id, self.dream_config.idle_threshold_seconds):
            return (
                False,
                "idle_deferred",
                f"Tenant active within last {idle_sec:.1f}s "
                f"(threshold: {self.dream_config.idle_threshold_seconds}s).",
            )

        # 4. Check system resource policy
        res_ok, res_reason = self.resource_policy.check_resources(
            require_ac_power=self.dream_config.require_ac_power,
            max_memory_pressure_pct=self.dream_config.max_memory_pressure_pct,
        )
        if not res_ok:
            return False, "resource_deferred", res_reason or "System resources unavailable."

        # 5. Check pending training examples
        pending = self.count_pending_examples(tenant_id)
        if pending < self.dream_config.min_verified_examples:
            return (
                False,
                "examples_deferred",
                f"Pending examples ({pending}) below threshold "
                f"({self.dream_config.min_verified_examples}).",
            )

        # If no active model version exists, launch full retrain; otherwise incremental
        mode = "incremental" if active_version else "full"
        return True, mode, f"Eligible: idle for {idle_sec:.0f}s with {pending} pending examples."

    def trigger_cycle(
        self,
        auth: AuthContext,
        *,
        mode: str = "incremental",
        auto_promote: bool = True,
        force: bool = False,
    ) -> DreamCycleResult | None:
        """Execute a state-machine managed Dream cycle for `auth.tenant_id` (M6.2)."""
        tenant_id = auth.tenant_id
        lock = self._get_tenant_lock(tenant_id)

        if not lock.acquire(blocking=force):
            logger.info("Dream cycle skipped for %s: another cycle is already running.", tenant_id)
            return None

        pause_event = threading.Event()
        with self._global_lock:
            self._pause_events[tenant_id] = pause_event

        try:
            current_state = self.get_dream_state(tenant_id)
            if current_state != DreamLifecycleState.AWAKE and not force:
                logger.info(
                    "Tenant %s is in state %s; cannot start cycle.", tenant_id, current_state
                )
                return None

            # State transition: AWAKE -> IDLE_ELIGIBLE -> DREAM_PREPARING
            self.set_dream_state(tenant_id, DreamLifecycleState.IDLE_ELIGIBLE)
            self.set_dream_state(tenant_id, DreamLifecycleState.DREAM_PREPARING)

            if pause_event.is_set():
                logger.info("Dream cycle deferred for %s during preparation.", tenant_id)
                self.set_dream_state(tenant_id, DreamLifecycleState.AWAKE)
                return None

            # State transition: DREAM_PREPARING -> TRAINING
            self.set_dream_state(tenant_id, DreamLifecycleState.TRAINING)

            result = execute_dream_cycle(
                auth,
                store=self.store,
                artifact_store=self.artifact_store,
                training_backend=self.training_backend,
                inference_backend=self.inference_backend,
                model_registry=self.registry,
                augmentation_cache=self.augmentation_cache,
                requested_mode=mode,
                auto_promote=auto_promote,
                base_model_id=self.base_model_id,
                base_model_revision=self.base_model_revision,
                tolerances=self.tolerances,
            )

            # State transition: EVALUATING -> PROMOTING / REJECTED -> AWAKE
            if result.promoted:
                self.set_dream_state(tenant_id, DreamLifecycleState.PROMOTING)
            elif not result.gate_report.passed:
                self.set_dream_state(tenant_id, DreamLifecycleState.REJECTED)

            self.set_dream_state(tenant_id, DreamLifecycleState.AWAKE)
            return result

        except Exception as exc:
            logger.exception("Dream cycle failed for tenant %s: %s", tenant_id, exc)
            self.set_dream_state(tenant_id, DreamLifecycleState.AWAKE)
            raise
        finally:
            with self._global_lock:
                self._pause_events.pop(tenant_id, None)
            lock.release()

    def _worker_loop(self) -> None:
        """Background thread periodic polling loop."""
        while not self._stop_event.is_set():
            try:
                with self._global_lock:
                    tenants = list(self._tracked_tenants)

                for tid in tenants:
                    if self._stop_event.is_set():
                        break
                    should_run, mode, _reason = self.evaluate_triggers(tid)
                    if should_run:
                        from somnium.domain import Scope, ScopeKind

                        auth = AuthContext(
                            tenant_id=tid,
                            principal_id="dream-worker",
                            authorized_scopes=(Scope(kind=ScopeKind.PROJECT, id="all"),),
                            allow_all_scopes=True,
                        )
                        self.trigger_cycle(auth, mode=mode, force=False)
            except Exception as exc:
                logger.debug("Error in DreamEngine background tick: %s", exc)

            self._stop_event.wait(self.tick_interval_seconds)

    def start(self) -> None:
        """Start the background Dream Engine worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="somnium-dream-engine",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the background worker thread to stop and wait for it."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None


__all__ = ["DreamEngine"]
