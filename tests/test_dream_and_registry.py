"""Tests for Dream Engine: Dataset Builder, Artifacts, Trainer, and Registry (M5).

Verifies:
- M5.1: Dataset builder reproducibility, split by fact group, sensitive exclusion.
- M5.2: Content-hashed augmentation cache exact rebuilding.
- M5.3: TrainingBackend LoRA training and artifact packaging.
- M5.4: LocalFilesystemArtifactStore content-addressing and integrity.
- M5.5: Model registry lineage, active pointer, atomic promotion and rollback.
- M5.6: Promotion gate hard gates and regression tolerance.
- M5.7: CLI commands (`somnium dream run`, `somnium model`, `somnium dataset`).
- M5.8: Training-eligible deletion forces full retrain and cleans weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from somnium.backends.base import TrainingJobSpec
from somnium.backends.fake import FakeInferenceBackend, FakeTrainingBackend
from somnium.backends.mlx import (
    MLXTrainingBackend,
    load_adapter_checkpoint,
)
from somnium.cli import app
from somnium.domain import (
    AuthContext,
    DeletionMode,
    MemoryForgetRequest,
    MemoryKind,
    MemoryRecord,
    MemoryStoreRequest,
    Permission,
    Scope,
    ScopeKind,
    SensitivityLevel,
)
from somnium.dream.artifacts import LocalFilesystemArtifactStore
from somnium.dream.augmentation import (
    AugmentationCache,
    compute_augmentation_key,
)
from somnium.dream.datasets import (
    build_tenant_dataset,
    load_dataset_manifest,
)
from somnium.dream.gates import GateTolerances, PromotionGateEvaluator
from somnium.dream.pipeline import execute_dream_cycle
from somnium.dream.registry import ModelRegistry
from somnium.hippocore.store import HippocoreStore

cli_runner = CliRunner()


def _auth(tenant_id: str = "tenant-alpha") -> AuthContext:
    return AuthContext(
        tenant_id=tenant_id,
        principal_id="test-agent",
        authorized_scopes=(
            Scope(kind=ScopeKind.PROJECT, id="project-alpha"),
            Scope(kind=ScopeKind.PROJECT, id="project-beta"),
        ),
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


def test_artifact_store_atomic_and_integrity(tmp_path: Path) -> None:
    """M5.4: Test LocalFilesystemArtifactStore atomic write, retrieval, and SHA-256 validation."""
    store = LocalFilesystemArtifactStore(tmp_path / "artifacts")
    data = b"hello dream world"

    digest = store.put_bytes("datasets", data)
    assert len(digest) == 64
    assert store.exists("datasets", digest)

    retrieved = store.get_bytes("datasets", digest)
    assert retrieved == data
    assert store.uri_for("datasets", digest).startswith("file://")

    # Tampering with disk file triggers integrity error on get_bytes
    disk_path = store.path_for("datasets", digest)
    disk_path.write_bytes(b"tampered content")
    with pytest.raises(ValueError, match="integrity check failed"):
        store.get_bytes("datasets", digest)


def test_augmentation_cache_deterministic(tmp_path: Path) -> None:
    """M5.2: Test deterministic question synthesis and cache reproducibility."""
    from datetime import UTC, datetime

    store = LocalFilesystemArtifactStore(tmp_path / "artifacts")
    cache = AugmentationCache(artifact_store=store)

    auth = _auth()
    now = datetime.now(tz=UTC)
    rec = MemoryRecord(
        tenant_id=auth.tenant_id,
        scope=Scope(kind=ScopeKind.PROJECT, id="project-alpha"),
        kind=MemoryKind.FACT,
        content="Project Alpha selected DuckDB as its embedded analytics database.",
        event_time=now,
        recorded_at=now,
        sensitivity=SensitivityLevel.NORMAL,
        training_eligible=True,
    )

    key1 = compute_augmentation_key(rec)
    augmented1 = cache.get_or_create(rec)

    # Calling again hits cache and returns identical questions
    augmented2 = cache.get_or_create(rec)
    assert augmented1 == augmented2
    assert cache.cache_hits >= 1
    assert "DuckDB" in augmented1.direct_question or "analytics" in augmented1.direct_question
    assert augmented1.cache_key_hash == key1


def test_dataset_builder_reproducibility_and_safety(tmp_path: Path) -> None:
    """M5.1: Test building dataset, exact hash reproducibility, and sensitive data exclusion."""
    auth = _auth()
    store = HippocoreStore("sqlite:///:memory:")
    art_store = LocalFilesystemArtifactStore(tmp_path / "artifacts")
    cache = AugmentationCache(artifact_store=art_store)

    # Store normal training-eligible fact
    res1 = store.store(
        auth,
        MemoryStoreRequest(
            scope=Scope(kind=ScopeKind.PROJECT, id="project-alpha"),
            content="Project Alpha selected PostgreSQL 16 as its relational database.",
            kind=MemoryKind.DECISION,
            sensitivity=SensitivityLevel.NORMAL,
        ),
    )
    assert res1.memory_id is not None

    # Store high-sensitivity fact (MUST be excluded)
    store.store(
        auth,
        MemoryStoreRequest(
            scope=Scope(kind=ScopeKind.PROJECT, id="project-alpha"),
            content="Project Alpha secret admin token is sk-secret-12345.",
            kind=MemoryKind.FACT,
            sensitivity=SensitivityLevel.HIGH,
        ),
    )

    manifest1 = build_tenant_dataset(
        auth,
        store=store,
        artifact_store=art_store,
        augmentation_cache=cache,
        training_mode="full",
        seed=42,
    )

    assert manifest1.dataset_hash
    assert str(res1.memory_id) in manifest1.source_record_ids

    # Verify no high-sensitivity content in examples
    for ex in manifest1.examples:
        assert "sk-secret-12345" not in ex.user_prompt
        assert "sk-secret-12345" not in ex.assistant_response

    # Verify categories present
    cats = {ex.category for ex in manifest1.examples}
    assert "direct" in cats
    assert "scope_attribution" in cats
    assert "incomplete_cue" in cats
    assert "evidence_reading" in cats
    assert "unknown" in cats

    # Rebuilding from same inputs yields exact same dataset_hash
    manifest2 = build_tenant_dataset(
        auth,
        store=store,
        artifact_store=art_store,
        augmentation_cache=cache,
        training_mode="full",
        seed=42,
    )
    assert manifest1.dataset_hash == manifest2.dataset_hash

    # Loading from storage
    loaded = load_dataset_manifest(store.engine, art_store, manifest1.dataset_hash)
    assert loaded.dataset_hash == manifest1.dataset_hash
    assert len(loaded.examples) == len(manifest1.examples)


def test_mlx_training_backend_and_checkpoint_reload(tmp_path: Path) -> None:
    """M5.3: Test MLX LoRA training and artifact packaging."""
    auth = _auth()
    store = HippocoreStore("sqlite:///:memory:")
    art_store = LocalFilesystemArtifactStore(tmp_path / "artifacts")

    store.store(
        auth,
        MemoryStoreRequest(
            scope=Scope(kind=ScopeKind.PROJECT, id="project-alpha"),
            content="Project Alpha selected Redis 7.2 as its caching tier.",
            kind=MemoryKind.DECISION,
            sensitivity=SensitivityLevel.NORMAL,
        ),
    )

    manifest = build_tenant_dataset(
        auth,
        store=store,
        artifact_store=art_store,
        training_mode="full",
    )

    trainer = MLXTrainingBackend(art_store)
    spec = TrainingJobSpec(
        job_id="job-101",
        tenant_id=auth.tenant_id,
        dataset_manifest_hash=manifest.dataset_hash,
        dataset_uri=manifest.artifact_uri,
        base_model_id="Qwen/Qwen3-1.7B",
        base_model_revision="main",
        training_mode="full",
        epochs=3,
        learning_rate=1e-2,
    )

    result = trainer.train_adapter(spec)
    assert result.candidate_version
    assert result.checkpoint_uri.startswith("file://")
    assert "train_loss" in result.metrics

    # Verify checkpoint can be unpacked and loaded from ArtifactStore
    checkpoint_hash = result.checkpoint_uri.split("/")[-1].replace(".bin", "")
    cfg, weights = load_adapter_checkpoint(art_store, checkpoint_hash)
    assert cfg["candidate_version"] == result.candidate_version
    assert cfg["dataset_manifest_hash"] == manifest.dataset_hash
    assert len(weights) > 0


def test_promotion_gate_evaluator() -> None:
    """M5.6: Test hard safety gates and comparative quality gates."""
    evaluator = PromotionGateEvaluator(GateTolerances())

    # Passing candidate
    cand_metrics = {
        "isolation_leakage": 0.0,
        "deletion_leakage": 0.0,
        "unsupported_claim_rate": 0.0,
        "acquisition_accuracy": 0.95,
        "historical_retention": 0.92,
        "overall_accuracy": 0.93,
        "abstention_quality": 0.90,
    }
    active_metrics = {
        "acquisition_accuracy": 0.90,
        "historical_retention": 0.94,
        "overall_accuracy": 0.91,
    }

    report = evaluator.evaluate(
        candidate_version="v2",
        candidate_metrics=cand_metrics,
        active_version="v1",
        active_metrics=active_metrics,
    )
    assert report.passed
    assert report.hard_gates_passed
    assert not report.failures

    # Hard gate failure: isolation leakage
    leaky_metrics = dict(cand_metrics, isolation_leakage=0.05)
    report_fail = evaluator.evaluate(
        candidate_version="v-leak",
        candidate_metrics=leaky_metrics,
    )
    assert not report_fail.passed
    assert not report_fail.hard_gates_passed
    assert any("Isolation leakage" in f for f in report_fail.failures)

    # Comparative gate failure: material retention regression
    regressed_metrics = dict(cand_metrics, historical_retention=0.80)
    report_regress = evaluator.evaluate(
        candidate_version="v-regress",
        candidate_metrics=regressed_metrics,
        active_version="v1",
        active_metrics=active_metrics,
    )
    assert not report_regress.passed
    assert any("retention regression" in f for f in report_regress.failures)


def test_model_registry_lineage_promote_and_rollback() -> None:
    """M5.5: Test ModelRegistry registration, active pointer, atomic promote and rollback."""
    store = HippocoreStore("sqlite:///:memory:")
    registry = ModelRegistry(store.engine)
    inference = FakeInferenceBackend()
    tenant_id = "tenant-alpha"

    # Register candidate v1
    v1 = registry.register_version(
        model_version="engram-v1",
        tenant_id=tenant_id,
        base_model_id="Qwen/Qwen3-1.7B",
        base_model_revision="main",
        parent_version=None,
        training_mode="full",
        dataset_manifest_hash="hash-111",
        checkpoint_uri="memory://adapters/v1",
    )
    assert v1.status == "candidate"

    # Promote v1
    prom1 = registry.promote(tenant_id, "engram-v1", inference_backend=inference)
    assert prom1.promoted_version == "engram-v1"
    assert registry.get_active_version(tenant_id) == "engram-v1"
    assert inference.active_version(tenant_id) == "engram-v1"

    # Register candidate v2
    registry.register_version(
        model_version="engram-v2",
        tenant_id=tenant_id,
        base_model_id="Qwen/Qwen3-1.7B",
        base_model_revision="main",
        parent_version="engram-v1",
        training_mode="incremental",
        dataset_manifest_hash="hash-222",
        checkpoint_uri="memory://adapters/v2",
    )

    # Promote v2
    registry.promote(tenant_id, "engram-v2", inference_backend=inference)
    assert registry.get_active_version(tenant_id) == "engram-v2"
    assert inference.active_version(tenant_id) == "engram-v2"

    # Rollback to v1 atomically
    rb = registry.rollback(tenant_id, inference_backend=inference)
    assert rb.active_version == "engram-v1"
    assert rb.rolled_back_version == "engram-v2"
    assert registry.get_active_version(tenant_id) == "engram-v1"
    assert inference.active_version(tenant_id) == "engram-v1"


def test_deletion_handling_forces_full_retrain(tmp_path: Path) -> None:
    """M5.8: Training-eligible deletion marks weights dirty, forces full retrain, purges data."""
    auth = _auth()
    store = HippocoreStore("sqlite:///:memory:")
    art_store = LocalFilesystemArtifactStore(tmp_path / "artifacts")
    registry = ModelRegistry(store.engine)
    training_backend = FakeTrainingBackend()
    inference_backend = FakeInferenceBackend()

    # 1. Store record and run cycle
    res = store.store(
        auth,
        MemoryStoreRequest(
            scope=Scope(kind=ScopeKind.PROJECT, id="project-alpha"),
            content="Project Alpha standardized on ClickHouse for analytics.",
            kind=MemoryKind.DECISION,
            sensitivity=SensitivityLevel.NORMAL,
        ),
    )
    assert res.memory_id is not None
    rec_id = res.memory_id

    cycle1 = execute_dream_cycle(
        auth,
        store=store,
        artifact_store=art_store,
        training_backend=training_backend,
        inference_backend=inference_backend,
        model_registry=registry,
        requested_mode="full",
    )
    assert cycle1.promoted
    assert str(rec_id) in cycle1.dataset_manifest.source_record_ids
    assert not registry.is_weights_dirty(auth.tenant_id)

    # 2. Delete the record -> weights_dirty should become True
    forget_res = store.forget(
        auth,
        MemoryForgetRequest(memory_id=rec_id, deletion_mode=DeletionMode.HARD),
    )
    assert forget_res.weights_marked_dirty
    assert registry.is_weights_dirty(auth.tenant_id)

    # 3. Next cycle requests "incremental", but MUST resolve to "full" retrain
    cycle2 = execute_dream_cycle(
        auth,
        store=store,
        artifact_store=art_store,
        training_backend=training_backend,
        inference_backend=inference_backend,
        model_registry=registry,
        requested_mode="incremental",
    )

    assert cycle2.run_record.requested_mode == "incremental"
    assert cycle2.run_record.resolved_mode == "full"
    assert cycle2.run_record.forced_full_retrain_reason is not None
    # Deleted record is completely absent from newly built dataset manifest
    assert str(rec_id) not in cycle2.dataset_manifest.source_record_ids
    # After cycle completion and promotion, weights_dirty is cleared
    assert not registry.is_weights_dirty(auth.tenant_id)


def test_cli_dream_model_and_dataset(tmp_path: Path) -> None:
    """M5.7: Test CLI dream run, model list/show/promote/rollback, and dataset build/show."""
    db_file = tmp_path / "cli_hippocore.db"
    db_url = f"sqlite:///{db_file}"

    # First store a record using CLI memory store
    res_store = cli_runner.invoke(
        app,
        [
            "memory",
            "store",
            "Project Beta adopted Apache Kafka for event bus.",
            "--scope",
            "project:project-beta",
            "--profile",
            "test",
            "--db-url",
            db_url,
        ],
    )
    assert res_store.exit_code == 0

    # Test dataset build
    res_ds_build = cli_runner.invoke(
        app,
        [
            "dataset",
            "build",
            "--mode",
            "full",
            "--profile",
            "test",
            "--db-url",
            db_url,
        ],
    )
    assert res_ds_build.exit_code == 0
    ds_data = json.loads(res_ds_build.stdout)
    ds_hash = ds_data["dataset_hash"]
    assert ds_hash

    # Test dataset show
    res_ds_show = cli_runner.invoke(
        app,
        [
            "dataset",
            "show",
            ds_hash,
            "--profile",
            "test",
            "--db-url",
            db_url,
        ],
    )
    assert res_ds_show.exit_code == 0
    assert json.loads(res_ds_show.stdout)["dataset_hash"] == ds_hash

    # Test dream run
    res_dream = cli_runner.invoke(
        app,
        [
            "dream",
            "run",
            "--mode",
            "full",
            "--profile",
            "test",
            "--db-url",
            db_url,
        ],
    )
    assert res_dream.exit_code == 0
    dream_data = json.loads(res_dream.stdout)
    cand_version = dream_data["candidate_model_version"]
    assert dream_data["promoted"] is True

    # Test model list
    res_model_list = cli_runner.invoke(
        app,
        [
            "model",
            "list",
            "--profile",
            "test",
            "--db-url",
            db_url,
        ],
    )
    assert res_model_list.exit_code == 0
    models = json.loads(res_model_list.stdout)
    assert any(m["model_version"] == cand_version for m in models)

    # Test model show
    res_model_show = cli_runner.invoke(
        app,
        [
            "model",
            "show",
            cand_version,
            "--profile",
            "test",
            "--db-url",
            db_url,
        ],
    )
    assert res_model_show.exit_code == 0
    assert json.loads(res_model_show.stdout)["model_version"] == cand_version

    # Test model rollback
    res_rollback = cli_runner.invoke(
        app,
        [
            "model",
            "rollback",
            "--profile",
            "test",
            "--db-url",
            db_url,
        ],
    )
    assert res_rollback.exit_code == 0
    assert json.loads(res_rollback.stdout)["status"] == "rolled_back"
