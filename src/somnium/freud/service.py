"""Transport-independent MemoryService for the Freud gateway (M3.1, M3.5).

Implements `ask`, `store`, `correct`, `forget`, `status`, `get`, `list_by_scope`,
and `export_jsonl` over versioned domain models, enforcing `AuthContext` scope
isolation, untrusted-content delimiting, and structured `FreudError` codes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select

from somnium.domain import (
    AuthContext,
    DreamLifecycleState,
    EvidenceRecallResponse,
    MemoryAskRequest,
    MemoryAskResponse,
    MemoryCorrectRequest,
    MemoryCorrectResponse,
    MemoryForgetRequest,
    MemoryForgetResponse,
    MemoryRecord,
    MemoryStatus,
    MemoryStatusRequest,
    MemoryStatusResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
    Permission,
    RecallMode,
    ResolvedRecallMode,
    Scope,
    ScopeKind,
)
from somnium.engram.pipeline import EngramAssociativePipeline
from somnium.freud.auth import resolve_request_scopes
from somnium.freud.errors import ErrorCode, FreudError
from somnium.freud.formatting import attach_untrusted_delimiters
from somnium.freud.routing import RuleBasedRecallRouter
from somnium.hippocore import (
    HippocoreFtsRetriever,
    HippocoreStore,
    SecretDetectedError,
    memory_facts,
)
from somnium.interfaces import RecallRouter, Retriever


def _map_store_exception(exc: Exception) -> FreudError:
    """Translate Hippocore storage exceptions into structured `FreudError` instances."""
    if isinstance(exc, FreudError):
        return exc
    if isinstance(exc, SecretDetectedError):
        return FreudError(ErrorCode.SECRET_DETECTED, str(exc))
    if isinstance(exc, PermissionError):
        msg = str(exc)
        if "not authorized for scope" in msg or "no authorized scopes" in msg:
            return FreudError(ErrorCode.SCOPE_UNAUTHORIZED, msg)
        return FreudError(ErrorCode.PERMISSION_DENIED, msg)
    if isinstance(exc, KeyError):
        clean_msg = str(exc).strip("'\"")
        return FreudError(ErrorCode.MEMORY_NOT_FOUND, clean_msg)
    if isinstance(exc, ValueError):
        msg = str(exc)
        lowered = msg.lower()
        if "already superseded" in lowered:
            return FreudError(ErrorCode.CONFLICT_SUPERSEDED, msg)
        if "cannot correct deleted" in lowered or "deleted memory" in lowered:
            return FreudError(ErrorCode.CONFLICT_DELETED, msg)
        return FreudError(ErrorCode.INVALID_REQUEST, msg)
    return FreudError(ErrorCode.INTERNAL_ERROR, str(exc))


class MemoryService:
    """Transport-independent memory service backing MCP tools and CLI commands (M3.1, M4.7)."""

    def __init__(
        self,
        store: HippocoreStore,
        retriever: Retriever | None = None,
        *,
        router: RecallRouter | None = None,
        engram: EngramAssociativePipeline | None = None,
        dream_engine: Any | None = None,
        default_scope: Scope | None = None,
        min_evidence_relevance: float = 0.0,
        active_model_version: str = "engram-base-none",
    ) -> None:
        self._store = store
        self._retriever: Retriever = retriever or HippocoreFtsRetriever(store)
        self._engram = engram
        self._dream_engine = dream_engine
        self._router: RecallRouter = router or RuleBasedRecallRouter(
            store=store,
            associative_available=lambda: self._engram is not None,
        )
        self._default_scope = default_scope or Scope(kind=ScopeKind.PROJECT, id="default")
        self._min_evidence_relevance = min_evidence_relevance
        self._active_model_version = active_model_version

    @property
    def store_backend(self) -> HippocoreStore:
        """Underlying `HippocoreStore` instance."""
        return self._store

    @property
    def router(self) -> RecallRouter:
        """Active `RecallRouter` instance."""
        return self._router

    @property
    def engram_pipeline(self) -> EngramAssociativePipeline | None:
        """Configured `EngramAssociativePipeline`, if any."""
        return self._engram

    @property
    def dream_engine(self) -> Any | None:
        """Configured `DreamEngine` instance, if any."""
        return self._dream_engine

    @property
    def default_scope(self) -> Scope:
        """Default scope when a request omits an explicit scope."""
        return self._default_scope

    def store(self, auth: AuthContext, request: MemoryStoreRequest) -> MemoryStoreResponse:
        """Persist a new memory event within an authorized scope."""
        if self._dream_engine is not None:
            self._dream_engine.record_traffic(auth.tenant_id)
        try:
            return self._store.store(auth, request)
        except Exception as exc:
            raise _map_store_exception(exc) from exc

    def ask(self, auth: AuthContext, request: MemoryAskRequest) -> MemoryAskResponse:
        """Execute memory recall in `evidence`, `associative`, or `auto` mode (M3.1, M4.7)."""
        if self._dream_engine is not None:
            if request.mode == RecallMode.ASSOCIATIVE:
                self._dream_engine.notify_associative_traffic(auth.tenant_id)
            else:
                self._dream_engine.record_traffic(auth.tenant_id)

        if not auth.has_permission(Permission.READ):
            raise FreudError(
                ErrorCode.PERMISSION_DENIED,
                f"Principal '{auth.principal_id}' lacks required permission 'read'.",
            )

        # When cross_scope=True and associative or auto recall is requested,
        # validate requested_scope authorization if given, and allow evidence
        # retrieval across authorized scopes when cross_scope is explicitly True.
        if request.scope is not None and not auth.is_scope_authorized(request.scope):
            raise FreudError(
                ErrorCode.SCOPE_UNAUTHORIZED,
                f"Principal '{auth.principal_id}' is not authorized for scope '{request.scope}'.",
                details={"requested_scope": str(request.scope)},
            )

        if request.cross_scope is True and request.mode != RecallMode.EVIDENCE:
            target_scopes = resolve_request_scopes(
                auth,
                None,
                default_scope=self._default_scope,
            )
        else:
            target_scopes = resolve_request_scopes(
                auth,
                request.scope,
                default_scope=self._default_scope,
            )

        try:
            raw_items = self._retriever.retrieve(
                auth,
                request.question,
                scopes=target_scopes,
                limit=request.limit,
            )
        except Exception as exc:
            raise _map_store_exception(exc) from exc

        filtered_items = tuple(
            attach_untrusted_delimiters(item)
            for item in raw_items
            if item.relevance >= self._min_evidence_relevance
        )

        routing = self._router.route(auth, request, filtered_items)

        if routing.resolved == ResolvedRecallMode.ASSOCIATIVE and self._engram is not None:
            try:
                return self._engram.recall(
                    auth,
                    request,
                    routing=routing,
                    initial_evidence=filtered_items,
                )
            except Exception as exc:
                raise _map_store_exception(exc) from exc

        return EvidenceRecallResponse(
            status="ok" if filtered_items else "no_match",
            routing=routing,
            records=filtered_items,
        )

    def correct(self, auth: AuthContext, request: MemoryCorrectRequest) -> MemoryCorrectResponse:
        """Record an immutable revision superseding an existing memory record."""
        if self._dream_engine is not None:
            self._dream_engine.record_traffic(auth.tenant_id)
        try:
            return self._store.correct(auth, request)
        except Exception as exc:
            raise _map_store_exception(exc) from exc

    def forget(self, auth: AuthContext, request: MemoryForgetRequest) -> MemoryForgetResponse:
        """Delete a memory record and mark tenant weights dirty if training-eligible."""
        if self._dream_engine is not None:
            self._dream_engine.record_traffic(auth.tenant_id)
        try:
            return self._store.forget(auth, request)
        except Exception as exc:
            raise _map_store_exception(exc) from exc

    def status(
        self,
        auth: AuthContext,
        request: MemoryStatusRequest | None = None,
    ) -> MemoryStatusResponse:
        """Return current Dream lifecycle state, active model version, and pending examples."""
        _ = request
        try:
            dirty = self._store.is_weights_dirty(auth)
            if self._dream_engine is not None:
                pending_count = self._dream_engine.count_pending_examples(auth.tenant_id)
                dream_state = self._dream_engine.get_dream_state(auth.tenant_id)
                active_version = self._dream_engine.registry.get_active_version(auth.tenant_id)
            else:
                with self._store.engine.connect() as conn:
                    pending_count = int(
                        conn.execute(
                            select(func.count(memory_facts.c.id)).where(
                                and_(
                                    memory_facts.c.tenant_id == auth.tenant_id,
                                    memory_facts.c.status == MemoryStatus.ACTIVE.value,
                                    memory_facts.c.training_eligible.is_(True),
                                )
                            )
                        ).scalar_one()
                    )
                dream_state = DreamLifecycleState.AWAKE
                active_version = None

            if not active_version:
                active_version = (
                    self._engram.inference_backend.active_version(auth.tenant_id)
                    if self._engram is not None
                    else self._active_model_version
                )
        except Exception as exc:
            raise _map_store_exception(exc) from exc

        return MemoryStatusResponse(
            tenant_id=auth.tenant_id,
            dream_state=dream_state,
            active_model_version=active_version,
            pending_training_examples=pending_count,
            weights_dirty=dirty,
        )

    def get(self, auth: AuthContext, memory_id: UUID) -> MemoryRecord:
        """Fetch a single memory record by ID or raise `FreudError(MEMORY_NOT_FOUND)`."""
        try:
            rec = self._store.get(auth, memory_id)
        except Exception as exc:
            raise _map_store_exception(exc) from exc
        if rec is None:
            raise FreudError(
                ErrorCode.MEMORY_NOT_FOUND,
                f"Memory '{memory_id}' not found in authorized scopes.",
                details={"memory_id": str(memory_id)},
            )
        return rec

    def list_by_scope(
        self,
        auth: AuthContext,
        scope: Scope | None = None,
        *,
        include_superseded: bool = False,
    ) -> Sequence[MemoryRecord]:
        """List memory records within an authorized scope."""
        target_scope = scope or self._default_scope
        try:
            return self._store.list_by_scope(
                auth,
                target_scope,
                include_superseded=include_superseded,
            )
        except Exception as exc:
            raise _map_store_exception(exc) from exc

    def export_jsonl(self, auth: AuthContext) -> str:
        """Export all non-deleted records and revisions in `auth`'s authorized scopes as JSONL."""
        try:
            return self._store.export_jsonl(auth)
        except Exception as exc:
            raise _map_store_exception(exc) from exc


__all__ = ["MemoryService"]
