"""Soak test: Continuous mixed queries during Dream training and atomic promotion (M6.7).

Verifies per `design.md` §7 and `Milestones.md` M6.7:
- Continuous mixed agent traffic (`store`, `ask` in evidence recall mode) runs concurrently.
- Zero failed requests (100% availability).
- Evidence recall is completely unaffected by Dream state (AWAKE, TRAINING, PROMOTING).
- Newly stored facts are immediately recallable via evidence recall even while training.
- Candidate adapter completes training, passes gates, and is atomically promoted.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from somnium.backends.fake import FakeInferenceBackend, FakeTrainingBackend
from somnium.config import DreamConfig
from somnium.domain import (
    AuthContext,
    EvidenceRecallResponse,
    MemoryAskRequest,
    MemoryKind,
    MemoryStoreRequest,
    Permission,
    RecallMode,
    Scope,
    ScopeKind,
    SensitivityLevel,
)
from somnium.dream.artifacts import LocalFilesystemArtifactStore
from somnium.dream.engine import DreamEngine
from somnium.dream.registry import ModelRegistry
from somnium.dream.resources import StaticResourcePolicy
from somnium.freud.service import MemoryService
from somnium.hippocore.store import HippocoreStore


def _auth(tenant_id: str = "soak-tenant") -> AuthContext:
    return AuthContext(
        tenant_id=tenant_id,
        principal_id="soak-agent",
        authorized_scopes=(Scope(kind=ScopeKind.PROJECT, id="project-soak"),),
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


def test_concurrent_soak_queries_during_training_and_promotion(tmp_path: Path) -> None:
    """M6.7: Continuous mixed traffic runs safely during Dream cycle with zero failures."""
    db_file = tmp_path / "soak.db"
    store = HippocoreStore(f"sqlite:///{db_file}")
    art_store = LocalFilesystemArtifactStore(tmp_path / "artifacts")
    registry = ModelRegistry(store.engine)
    inference_backend = FakeInferenceBackend()

    # Configure FakeTrainingBackend with a small deliberate delay to simulate GPU training time
    class SlowTrainingBackend(FakeTrainingBackend):
        def train_adapter(self, spec: Any) -> Any:
            time.sleep(0.3)
            return super().train_adapter(spec)

    training_backend = SlowTrainingBackend()
    res_policy = StaticResourcePolicy(on_ac_power=True, memory_pressure=30.0)

    dream_cfg = DreamConfig(
        enabled=True,
        idle_threshold_seconds=0,
        min_verified_examples=1,
        pause_on_associative_traffic=False,
    )

    engine = DreamEngine(
        store=store,
        artifact_store=art_store,
        training_backend=training_backend,
        inference_backend=inference_backend,
        model_registry=registry,
        resource_policy=res_policy,
        dream_config=dream_cfg,
    )

    service = MemoryService(
        store=store,
        engram=None,
        dream_engine=engine,
        default_scope=Scope(kind=ScopeKind.PROJECT, id="project-soak"),
    )

    auth = _auth()
    scope = Scope(kind=ScopeKind.PROJECT, id="project-soak")

    # Seed baseline fact
    seed_res = service.store(
        auth,
        MemoryStoreRequest(
            scope=scope,
            content="Project Soak selected DuckDB for fast local OLAP analytics.",
            kind=MemoryKind.DECISION,
            sensitivity=SensitivityLevel.NORMAL,
        ),
    )
    assert seed_res.memory_id is not None

    client_errors: list[Exception] = []
    queries_executed = 0
    stop_client = threading.Event()
    client_lock = threading.Lock()

    def client_worker() -> None:
        nonlocal queries_executed
        idx = 0
        while not stop_client.is_set():
            try:
                # 1. Evidence recall query for the baseline fact
                resp = service.ask(
                    auth,
                    MemoryAskRequest(
                        scope=scope,
                        question="DuckDB local OLAP analytics",
                        mode=RecallMode.EVIDENCE,
                    ),
                )
                assert isinstance(resp, EvidenceRecallResponse)
                assert resp.status == "ok"
                assert len(resp.records) > 0

                # 2. Store occasional new memories concurrently
                if idx % 5 == 0:
                    new_mem = service.store(
                        auth,
                        MemoryStoreRequest(
                            scope=scope,
                            content=f"Project Soak parameter checkpoint item {idx}.",
                            kind=MemoryKind.FACT,
                            sensitivity=SensitivityLevel.NORMAL,
                        ),
                    )
                    assert new_mem.memory_id is not None

                    # Immediately verify the newly stored memory via evidence recall
                    verify_resp = service.ask(
                        auth,
                        MemoryAskRequest(
                            scope=scope,
                            question=f"parameter checkpoint item {idx}",
                            mode=RecallMode.EVIDENCE,
                        ),
                    )
                    assert isinstance(verify_resp, EvidenceRecallResponse)
                    assert verify_resp.status == "ok"
                    assert len(verify_resp.records) > 0

                with client_lock:
                    queries_executed += 1
                idx += 1
                time.sleep(0.01)
            except Exception as exc:
                client_errors.append(exc)
                break

    client_thread = threading.Thread(target=client_worker, daemon=True)
    client_thread.start()

    try:
        # Run full dream cycle in main thread concurrently with client queries
        cycle_result = engine.trigger_cycle(auth, mode="full", force=True)
        assert cycle_result is not None
        assert cycle_result.promoted is True
        active_ver = registry.get_active_version(auth.tenant_id)
        assert active_ver == cycle_result.candidate_version_record.model_version
    finally:
        stop_client.set()
        client_thread.join(timeout=3.0)

    # Assert zero failed queries and evidence recall uninterrupted
    assert not client_errors, f"Client worker hit errors during soak: {client_errors}"
    assert queries_executed >= 20, f"Expected at least 20 queries, executed {queries_executed}"
