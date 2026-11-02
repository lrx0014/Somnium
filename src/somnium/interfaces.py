"""Core system interfaces (Protocols) across Somnium modules (M0.6).

Every interface that accesses tenant or project data requires an explicit
`AuthContext` parameter so that isolation cannot be bypassed by callers.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from somnium.backends.base import (
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    TrainingBackend,
    TrainingJobResult,
    TrainingJobSpec,
)
from somnium.domain import (
    AuthContext,
    EvidenceRecordItem,
    MemoryAskRequest,
    MemoryCorrectRequest,
    MemoryCorrectResponse,
    MemoryForgetRequest,
    MemoryForgetResponse,
    MemoryRecord,
    MemoryStoreRequest,
    MemoryStoreResponse,
    RoutingInfo,
    Scope,
)


@runtime_checkable
class Clock(Protocol):
    """Injectable clock for deterministic timestamps, idle detection, and tests."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC datetime."""
        ...


@runtime_checkable
class AuthProvider(Protocol):
    """Pluggable authentication and authorization resolver."""

    def authenticate(
        self,
        token: str | None = None,
        requested_scopes: Sequence[Scope] | None = None,
    ) -> AuthContext:
        """Resolve and validate caller credentials into an AuthContext."""
        ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Content-addressed artifact storage for dataset snapshots and adapter checkpoints."""

    def put_bytes(self, namespace: str, content: bytes) -> str:
        """Store raw bytes under `namespace` and return their content hash (SHA-256 hex)."""
        ...

    def get_bytes(self, namespace: str, content_hash: str) -> bytes:
        """Retrieve raw bytes by namespace and content hash."""
        ...

    def exists(self, namespace: str, content_hash: str) -> bool:
        """Return True if the artifact exists in the store."""
        ...

    def uri_for(self, namespace: str, content_hash: str) -> str:
        """Return the canonical storage URI for the given artifact."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Authoritative episodic event and fact store interface (Hippocore)."""

    def store(self, auth: AuthContext, request: MemoryStoreRequest) -> MemoryStoreResponse:
        """Persist a new memory event within an authorized scope."""
        ...

    def get(self, auth: AuthContext, memory_id: UUID) -> MemoryRecord | None:
        """Fetch a single memory record if it belongs to an authorized scope."""
        ...

    def correct(self, auth: AuthContext, request: MemoryCorrectRequest) -> MemoryCorrectResponse:
        """Record an immutable revision superseding an existing memory record."""
        ...

    def forget(self, auth: AuthContext, request: MemoryForgetRequest) -> MemoryForgetResponse:
        """Delete a memory record and mark tenant weights dirty if training-eligible."""
        ...

    def list_by_scope(
        self,
        auth: AuthContext,
        scope: Scope,
        include_superseded: bool = False,
    ) -> Sequence[MemoryRecord]:
        """List memory records within a single authorized scope."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Lexical / vector search interface over Hippocore, strictly filtered by scope."""

    def retrieve(
        self,
        auth: AuthContext,
        query: str,
        scopes: Sequence[Scope],
        limit: int = 10,
    ) -> Sequence[EvidenceRecordItem]:
        """Search active records across `scopes` (must all be authorized in `auth`)."""
        ...


@runtime_checkable
class RecallRouter(Protocol):
    """Policy router deciding between `evidence` and `associative` recall for `mode=auto`."""

    def route(
        self,
        auth: AuthContext,
        request: MemoryAskRequest,
        evidence_items: Sequence[EvidenceRecordItem],
    ) -> RoutingInfo:
        """Determine the concrete recall mode and record the routing reason."""
        ...


__all__ = [
    "ArtifactStore",
    "AuthProvider",
    "Clock",
    "GenerationRequest",
    "GenerationResult",
    "InferenceBackend",
    "MemoryStore",
    "RecallRouter",
    "Retriever",
    "TrainingBackend",
    "TrainingJobResult",
    "TrainingJobSpec",
]
