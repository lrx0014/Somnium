"""Unit tests for system Protocols (M0.6) and deterministic test doubles (M0.7)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from somnium.backends.base import (
    GenerationRequest,
    InferenceBackend,
    TrainingBackend,
    TrainingJobSpec,
)
from somnium.backends.fake import (
    FakeInferenceBackend,
    FakeTrainingBackend,
    FrozenClock,
    InMemoryArtifactStore,
    StaticAuthProvider,
    SystemClock,
)
from somnium.domain import Scope, ScopeKind
from somnium.interfaces import ArtifactStore, AuthProvider, Clock


def test_runtime_protocol_conformance() -> None:
    """Verify all test doubles satisfy their corresponding runtime_checkable Protocols."""
    assert isinstance(SystemClock(), Clock)
    assert isinstance(FrozenClock(), Clock)
    assert isinstance(StaticAuthProvider(), AuthProvider)
    assert isinstance(FakeInferenceBackend(), InferenceBackend)
    assert isinstance(FakeTrainingBackend(), TrainingBackend)
    assert isinstance(InMemoryArtifactStore(), ArtifactStore)


def test_frozen_clock_determinism() -> None:
    """Verify FrozenClock advances and sets timestamps deterministically."""
    t0 = datetime(2026, 11, 2, 1, 0, 0, tzinfo=UTC)
    clock = FrozenClock(t0)
    assert clock.now() == t0

    t1 = clock.advance(minutes=30, seconds=15)
    assert clock.now() == t1
    assert (t1 - t0).total_seconds() == 1815.0


def test_static_auth_provider_scope_and_token_validation() -> None:
    """Verify StaticAuthProvider rejects unauthorized scopes or invalid tokens."""
    orion = Scope(kind=ScopeKind.PROJECT, id="project-orion")
    lyra = Scope(kind=ScopeKind.PROJECT, id="project-lyra")
    provider = StaticAuthProvider(
        tenant_id="local-user",
        authorized_scopes=(orion,),
        expected_token="secret-token",
    )

    ctx = provider.authenticate(token="secret-token", requested_scopes=(orion,))
    assert ctx.tenant_id == "local-user"
    assert ctx.is_scope_authorized(orion) is True

    with pytest.raises(PermissionError, match="token"):
        provider.authenticate(token="wrong-token")

    with pytest.raises(PermissionError, match="Unauthorized scope"):
        provider.authenticate(token="secret-token", requested_scopes=(lyra,))


def test_fake_inference_backend_scripting_and_atomic_swap() -> None:
    """Verify FakeInferenceBackend scripts responses and swaps adapters per tenant."""
    backend = FakeInferenceBackend(default_model_version="engram-base-v0")
    backend.script_response("Orion database", "Project Orion selected PostgreSQL.")

    req = GenerationRequest(
        tenant_id="tenant-a",
        prompt="Question: What is the Orion database decision?",
    )
    res1 = backend.generate(req)
    assert res1.text == "Project Orion selected PostgreSQL."
    assert res1.model_version == "engram-base-v0"
    assert len(backend.call_history) == 1

    prev_version = backend.swap_adapter(
        tenant_id="tenant-a",
        new_model_version="engram-tenant-a-v1",
        adapter_uri="memory://adapters/tenant-a/v1",
    )
    assert prev_version == "engram-base-v0"
    assert backend.active_version("tenant-a") == "engram-tenant-a-v1"
    assert backend.active_version("tenant-b") == "engram-base-v0"

    res2 = backend.generate(req)
    assert res2.model_version == "engram-tenant-a-v1"


def test_fake_training_backend_and_artifact_store() -> None:
    """Verify FakeTrainingBackend produces deterministic hashes and supports failure injection."""
    trainer = FakeTrainingBackend()
    store = InMemoryArtifactStore()

    manifest_bytes = b'{"example_id": 1, "question": "What DB?"}\n'
    manifest_hash = store.put_bytes("datasets", manifest_bytes)
    assert store.exists("datasets", manifest_hash) is True
    assert store.get_bytes("datasets", manifest_hash) == manifest_bytes

    spec = TrainingJobSpec(
        job_id="dream-job-001",
        tenant_id="local-user",
        dataset_manifest_hash=manifest_hash,
        dataset_uri=store.uri_for("datasets", manifest_hash),
        base_model_id="Qwen/Qwen3-1.7B",
        base_model_revision="main",
    )
    result1 = trainer.train_adapter(spec)
    result2 = trainer.train_adapter(spec)
    assert result1.candidate_version == result2.candidate_version
    assert result1.checkpoint_uri == result2.checkpoint_uri
    assert "train_loss" in result1.metrics

    trainer.set_failure(RuntimeError("Simulated out-of-memory error"))
    with pytest.raises(RuntimeError, match="Simulated out-of-memory"):
        trainer.train_adapter(spec)
