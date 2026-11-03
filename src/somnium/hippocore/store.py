"""Scoped repository and episodic store implementation for Hippocore (M2.2–M2.8).

Guarantees:
- Every read and write requires an explicit `AuthContext`.
- Tenant and scope filtering are applied inside the SQL query builder; no
  unscoped query API exists.
- Credentials and private keys are rejected at the write path.
- Corrections create immutable revisions and update materialized facts.
- Deletions purge content from tables and the FTS5 index, record a content-free
  audit log, and mark tenant `weights_dirty = True` if the deleted record was
  training-eligible.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from sqlalchemy import (
    ColumnElement,
    Connection,
    Engine,
    Table,
    and_,
    create_engine,
    delete,
    func,
    or_,
    select,
    text,
    tuple_,
    update,
)
from sqlalchemy.pool import StaticPool

from somnium.backends.fake import SystemClock
from somnium.domain import (
    AuthContext,
    DeletionMode,
    MemoryCorrectRequest,
    MemoryCorrectResponse,
    MemoryForgetRequest,
    MemoryForgetResponse,
    MemoryKind,
    MemoryRecord,
    MemorySource,
    MemoryStatus,
    MemoryStoreRequest,
    MemoryStoreResponse,
    Permission,
    Scope,
    ScopeKind,
    SensitivityLevel,
    VerificationLevel,
)
from somnium.hippocore.migrations import upgrade_database
from somnium.hippocore.schema import (
    deletion_requests,
    memory_access,
    memory_events,
    memory_facts,
    memory_revisions,
    memory_sources,
    tenant_state,
)
from somnium.hippocore.security import (
    assert_no_secrets,
    classify_sensitivity,
    compute_training_eligibility,
    normalize_content,
)
from somnium.interfaces import Clock


@dataclass(frozen=True, slots=True)
class AccessStats:
    """Aggregated access statistics distinguishing `retrieved` vs `used` (M2.7)."""

    memory_id: UUID
    retrieved_count: int
    used_count: int
    last_accessed_at: datetime | None


def build_scoped_where(
    table: Table,
    auth: AuthContext,
    target_scopes: Sequence[Scope] | None = None,
) -> ColumnElement[bool]:
    """Construct a mandatory SQL WHERE clause enforcing tenant and scope isolation.

    Raises `PermissionError` if any requested scope is not authorized in `auth`,
    or if the caller holds zero authorized scopes.
    """
    if target_scopes is not None:
        for s in target_scopes:
            if not auth.is_scope_authorized(s):
                raise PermissionError(
                    f"Principal '{auth.principal_id}' is not authorized for scope '{s}'."
                )
        effective_scopes = tuple(target_scopes)
    elif auth.allow_all_scopes or Permission.ADMIN in auth.permissions:
        return table.c.tenant_id == auth.tenant_id
    else:
        effective_scopes = auth.authorized_scopes

    if not effective_scopes:
        raise PermissionError(
            f"Principal '{auth.principal_id}' has no authorized scopes for query."
        )

    scope_pairs = [(s.kind.value, s.id) for s in effective_scopes]
    return and_(
        table.c.tenant_id == auth.tenant_id,
        tuple_(table.c.scope_kind, table.c.scope_id).in_(scope_pairs),
    )


def _require_permission(auth: AuthContext, permission: Permission) -> None:
    if not auth.has_permission(permission):
        raise PermissionError(
            f"Principal '{auth.principal_id}' lacks required permission '{permission.value}'."
        )


def _row_to_memory_record(row: Any, source: MemorySource | None = None) -> MemoryRecord:
    """Convert a SQLAlchemy Row from `memory_facts` into a `MemoryRecord`."""
    mapping = row._mapping
    supersedes_list = json.loads(mapping["supersedes_json"] or "[]")
    tags_list = json.loads(mapping["tags_json"] or "[]")
    return MemoryRecord(
        id=UUID(str(mapping["id"])),
        tenant_id=str(mapping["tenant_id"]),
        scope=Scope(
            kind=ScopeKind(str(mapping["scope_kind"])),
            id=str(mapping["scope_id"]),
        ),
        kind=MemoryKind(str(mapping["kind"])),
        content=str(mapping["content"]) if str(mapping["content"]) else "[TOMBSTONE]",
        event_time=datetime.fromisoformat(str(mapping["event_time"])),
        recorded_at=datetime.fromisoformat(str(mapping["recorded_at"])),
        source=source or MemorySource(),
        confidence=float(mapping["confidence"]),
        verification=VerificationLevel(str(mapping["verification"])),
        supersedes=tuple(UUID(x) for x in supersedes_list),
        sensitivity=SensitivityLevel(str(mapping["sensitivity"])),
        training_eligible=bool(mapping["training_eligible"]),
        retention_policy=str(mapping["retention_policy"]),
        status=MemoryStatus(str(mapping["status"])),
        tags=tuple(str(t) for t in tags_list),
        version=int(mapping["version"]),
    )


class HippocoreStore:
    """Authoritative episodic store with strict SQL-level scope isolation."""

    def __init__(
        self,
        engine: Engine | str = "sqlite:///:memory:",
        *,
        clock: Clock | None = None,
        auto_migrate: bool = True,
    ) -> None:
        if isinstance(engine, str):
            if engine == "sqlite:///:memory:":
                self._engine = create_engine(
                    engine,
                    future=True,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                )
            elif engine.startswith("sqlite"):
                self._engine = create_engine(
                    engine,
                    future=True,
                    connect_args={"check_same_thread": False},
                )
            else:
                self._engine = create_engine(engine, future=True)
        else:
            self._engine = engine
        self._clock = clock or SystemClock()
        if auto_migrate:
            upgrade_database(self._engine)

    @property
    def engine(self) -> Engine:
        """Underlying SQLAlchemy Engine."""
        return self._engine

    def _upsert_fts(
        self,
        conn: Connection,
        *,
        memory_id: str,
        tenant_id: str,
        scope_kind: str,
        scope_id: str,
        content: str,
        tags: Sequence[str],
    ) -> None:
        if conn.dialect.name != "sqlite":
            return
        conn.execute(
            text("DELETE FROM memory_fts WHERE memory_id = :mid AND tenant_id = :tid"),
            {"mid": memory_id, "tid": tenant_id},
        )
        conn.execute(
            text(
                "INSERT INTO memory_fts(memory_id, tenant_id, scope_kind, scope_id, content, tags) "
                "VALUES (:mid, :tid, :sk, :sid, :content, :tags)"
            ),
            {
                "mid": memory_id,
                "tid": tenant_id,
                "sk": scope_kind,
                "sid": scope_id,
                "content": content,
                "tags": " ".join(tags),
            },
        )

    def _delete_fts(self, conn: Connection, *, memory_id: str, tenant_id: str) -> None:
        if conn.dialect.name != "sqlite":
            return
        conn.execute(
            text("DELETE FROM memory_fts WHERE memory_id = :mid AND tenant_id = :tid"),
            {"mid": memory_id, "tid": tenant_id},
        )

    def _set_weights_dirty(self, conn: Connection, tenant_id: str, dirty: bool) -> None:
        now_iso = self._clock.now().isoformat()
        existing = conn.execute(
            select(tenant_state.c.tenant_id).where(tenant_state.c.tenant_id == tenant_id)
        ).first()
        if existing is None:
            conn.execute(
                tenant_state.insert().values(
                    tenant_id=tenant_id,
                    weights_dirty=dirty,
                    updated_at=now_iso,
                )
            )
        else:
            conn.execute(
                update(tenant_state)
                .where(tenant_state.c.tenant_id == tenant_id)
                .values(weights_dirty=dirty, updated_at=now_iso)
            )

    def is_weights_dirty(self, auth: AuthContext) -> bool:
        """Return True if a training-eligible deletion occurred for `auth.tenant_id`."""
        _require_permission(auth, Permission.READ)
        with self._engine.connect() as conn:
            row = conn.execute(
                select(tenant_state.c.weights_dirty).where(
                    tenant_state.c.tenant_id == auth.tenant_id
                )
            ).first()
            return bool(row[0]) if row is not None else False

    def clear_weights_dirty(self, auth: AuthContext) -> None:
        """Clear the `weights_dirty` flag after a clean full retrain completes."""
        _require_permission(auth, Permission.WRITE)
        with self._engine.begin() as conn:
            self._set_weights_dirty(conn, auth.tenant_id, False)

    def store(self, auth: AuthContext, request: MemoryStoreRequest) -> MemoryStoreResponse:
        """Normalize, validate secrets, deduplicate, and persist a new memory event (M2.3)."""
        _require_permission(auth, Permission.WRITE)
        scope_filter = build_scoped_where(memory_events, auth, target_scopes=(request.scope,))

        normalized = normalize_content(request.content)
        assert_no_secrets(normalized)
        sensitivity = classify_sensitivity(normalized, request.sensitivity)
        training_eligible = compute_training_eligibility(
            sensitivity=sensitivity,
            verification=request.verification,
            status=MemoryStatus.ACTIVE,
        )

        now = self._clock.now()
        event_time = request.event_time or now

        with self._engine.begin() as conn:
            # 1. Check deduplication by idempotency_key within authorized scope
            if request.idempotency_key:
                existing_idem = conn.execute(
                    select(
                        memory_events.c.id,
                        memory_events.c.recorded_at,
                        memory_events.c.training_eligible,
                    ).where(
                        and_(
                            scope_filter,
                            memory_events.c.idempotency_key == request.idempotency_key,
                        )
                    )
                ).first()
                if existing_idem is not None:
                    return MemoryStoreResponse(
                        memory_id=UUID(str(existing_idem[0])),
                        recorded_at=datetime.fromisoformat(str(existing_idem[1])),
                        training_eligible=bool(existing_idem[2]),
                        deduplicated=True,
                    )

            # 2. Check deduplication by source identity (agent + session_id + message_id)
            if request.source.session_id and request.source.message_id:
                src_scope_filter = build_scoped_where(
                    memory_sources, auth, target_scopes=(request.scope,)
                )
                existing_src = conn.execute(
                    select(memory_sources.c.memory_id).where(
                        and_(
                            src_scope_filter,
                            memory_sources.c.agent == request.source.agent,
                            memory_sources.c.session_id == request.source.session_id,
                            memory_sources.c.message_id == request.source.message_id,
                        )
                    )
                ).first()
                if existing_src is not None:
                    existing_ev = conn.execute(
                        select(
                            memory_events.c.id,
                            memory_events.c.recorded_at,
                            memory_events.c.training_eligible,
                        ).where(
                            and_(
                                scope_filter,
                                memory_events.c.id == str(existing_src[0]),
                            )
                        )
                    ).first()
                    if existing_ev is not None:
                        return MemoryStoreResponse(
                            memory_id=UUID(str(existing_ev[0])),
                            recorded_at=datetime.fromisoformat(str(existing_ev[1])),
                            training_eligible=bool(existing_ev[2]),
                            deduplicated=True,
                        )

            memory_id = uuid4()
            mid_str = str(memory_id)
            tags_json = json.dumps(list(request.tags))

            conn.execute(
                memory_events.insert().values(
                    id=mid_str,
                    tenant_id=auth.tenant_id,
                    scope_kind=request.scope.kind.value,
                    scope_id=request.scope.id,
                    kind=request.kind.value,
                    content=normalized,
                    event_time=event_time.isoformat(),
                    recorded_at=now.isoformat(),
                    confidence=1.0,
                    verification=request.verification.value,
                    supersedes_json="[]",
                    sensitivity=sensitivity.value,
                    training_eligible=training_eligible,
                    retention_policy="default",
                    status=MemoryStatus.ACTIVE.value,
                    tags_json=tags_json,
                    idempotency_key=request.idempotency_key,
                    version=1,
                )
            )
            conn.execute(
                memory_facts.insert().values(
                    id=mid_str,
                    tenant_id=auth.tenant_id,
                    scope_kind=request.scope.kind.value,
                    scope_id=request.scope.id,
                    kind=request.kind.value,
                    content=normalized,
                    event_time=event_time.isoformat(),
                    recorded_at=now.isoformat(),
                    confidence=1.0,
                    verification=request.verification.value,
                    supersedes_json="[]",
                    superseded_by=None,
                    sensitivity=sensitivity.value,
                    training_eligible=training_eligible,
                    retention_policy="default",
                    status=MemoryStatus.ACTIVE.value,
                    tags_json=tags_json,
                    version=1,
                )
            )
            conn.execute(
                memory_sources.insert().values(
                    memory_id=mid_str,
                    tenant_id=auth.tenant_id,
                    scope_kind=request.scope.kind.value,
                    scope_id=request.scope.id,
                    agent=request.source.agent,
                    session_id=request.source.session_id,
                    message_id=request.source.message_id,
                    recorded_at=now.isoformat(),
                )
            )
            self._upsert_fts(
                conn,
                memory_id=mid_str,
                tenant_id=auth.tenant_id,
                scope_kind=request.scope.kind.value,
                scope_id=request.scope.id,
                content=normalized,
                tags=request.tags,
            )

        return MemoryStoreResponse(
            memory_id=memory_id,
            recorded_at=now,
            training_eligible=training_eligible,
            deduplicated=False,
        )

    def get(
        self,
        auth: AuthContext,
        memory_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> MemoryRecord | None:
        """Fetch a single memory record if it belongs to an authorized scope."""
        _require_permission(auth, Permission.READ)
        scope_filter = build_scoped_where(memory_facts, auth)
        mid_str = str(memory_id)

        with self._engine.connect() as conn:
            stmt = select(memory_facts).where(and_(scope_filter, memory_facts.c.id == mid_str))
            if not include_deleted:
                stmt = stmt.where(memory_facts.c.status != MemoryStatus.DELETED.value)
            row = conn.execute(stmt).first()
            if row is None:
                return None

            src_row = conn.execute(
                select(
                    memory_sources.c.agent,
                    memory_sources.c.session_id,
                    memory_sources.c.message_id,
                ).where(
                    and_(
                        memory_sources.c.tenant_id == auth.tenant_id,
                        memory_sources.c.memory_id == mid_str,
                    )
                )
            ).first()
            source = (
                MemorySource(
                    agent=str(src_row[0]),
                    session_id=src_row[1],
                    message_id=src_row[2],
                )
                if src_row is not None
                else MemorySource()
            )
            return _row_to_memory_record(row, source=source)

    def correct(self, auth: AuthContext, request: MemoryCorrectRequest) -> MemoryCorrectResponse:
        """Record an immutable revision superseding an existing memory record (M2.4)."""
        _require_permission(auth, Permission.CORRECT)
        facts_filter = build_scoped_where(memory_facts, auth)
        revs_filter = build_scoped_where(memory_revisions, auth)

        normalized_replacement = normalize_content(request.replacement)
        assert_no_secrets(normalized_replacement)
        old_id_str = str(request.memory_id)
        now = self._clock.now()

        with self._engine.begin() as conn:
            if request.idempotency_key:
                existing_rev = conn.execute(
                    select(
                        memory_revisions.c.new_memory_id,
                        memory_revisions.c.superseded_memory_id,
                        memory_revisions.c.version,
                    ).where(
                        and_(
                            revs_filter,
                            memory_revisions.c.idempotency_key == request.idempotency_key,
                        )
                    )
                ).first()
                if existing_rev is not None:
                    return MemoryCorrectResponse(
                        new_memory_id=UUID(str(existing_rev[0])),
                        superseded_memory_id=UUID(str(existing_rev[1])),
                        version=int(existing_rev[2]),
                    )

            old_row = conn.execute(
                select(memory_facts).where(and_(facts_filter, memory_facts.c.id == old_id_str))
            ).first()
            if old_row is None:
                raise KeyError(f"Memory '{request.memory_id}' not found in authorized scopes.")

            old_map = old_row._mapping
            old_status = MemoryStatus(str(old_map["status"]))
            if old_status == MemoryStatus.DELETED:
                raise ValueError(f"Cannot correct deleted memory '{request.memory_id}'.")
            if old_status == MemoryStatus.SUPERSEDED:
                sup_by = old_map["superseded_by"]
                raise ValueError(
                    f"Memory '{request.memory_id}' is already superseded by '{sup_by}'."
                )

            new_id = uuid4()
            new_id_str = str(new_id)
            new_version = int(old_map["version"]) + 1
            scope_kind = str(old_map["scope_kind"])
            scope_id = str(old_map["scope_id"])
            old_tags = json.loads(old_map["tags_json"] or "[]")
            if "correction" not in old_tags:
                old_tags.append("correction")
            tags_json = json.dumps(old_tags)

            old_sens = SensitivityLevel(str(old_map["sensitivity"]))
            new_sens = classify_sensitivity(normalized_replacement, old_sens)
            old_verif = VerificationLevel(str(old_map["verification"]))
            new_eligible = compute_training_eligibility(
                sensitivity=new_sens,
                verification=old_verif,
                status=MemoryStatus.ACTIVE,
            )

            # Mark old fact and event as SUPERSEDED
            conn.execute(
                update(memory_facts)
                .where(and_(facts_filter, memory_facts.c.id == old_id_str))
                .values(
                    status=MemoryStatus.SUPERSEDED.value,
                    superseded_by=new_id_str,
                )
            )
            conn.execute(
                update(memory_events)
                .where(
                    and_(
                        memory_events.c.tenant_id == auth.tenant_id,
                        memory_events.c.id == old_id_str,
                    )
                )
                .values(status=MemoryStatus.SUPERSEDED.value)
            )

            supersedes_json = json.dumps([old_id_str])
            conn.execute(
                memory_events.insert().values(
                    id=new_id_str,
                    tenant_id=auth.tenant_id,
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    kind=MemoryKind.CORRECTION.value,
                    content=normalized_replacement,
                    event_time=now.isoformat(),
                    recorded_at=now.isoformat(),
                    confidence=float(old_map["confidence"]),
                    verification=old_verif.value,
                    supersedes_json=supersedes_json,
                    sensitivity=new_sens.value,
                    training_eligible=new_eligible,
                    retention_policy=str(old_map["retention_policy"]),
                    status=MemoryStatus.ACTIVE.value,
                    tags_json=tags_json,
                    idempotency_key=request.idempotency_key,
                    version=new_version,
                )
            )
            conn.execute(
                memory_facts.insert().values(
                    id=new_id_str,
                    tenant_id=auth.tenant_id,
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    kind=MemoryKind.CORRECTION.value,
                    content=normalized_replacement,
                    event_time=now.isoformat(),
                    recorded_at=now.isoformat(),
                    confidence=float(old_map["confidence"]),
                    verification=old_verif.value,
                    supersedes_json=supersedes_json,
                    superseded_by=None,
                    sensitivity=new_sens.value,
                    training_eligible=new_eligible,
                    retention_policy=str(old_map["retention_policy"]),
                    status=MemoryStatus.ACTIVE.value,
                    tags_json=tags_json,
                    version=new_version,
                )
            )
            conn.execute(
                memory_sources.insert().values(
                    memory_id=new_id_str,
                    tenant_id=auth.tenant_id,
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    agent=auth.principal_id,
                    session_id=None,
                    message_id=None,
                    recorded_at=now.isoformat(),
                )
            )
            conn.execute(
                memory_revisions.insert().values(
                    revision_id=str(uuid4()),
                    tenant_id=auth.tenant_id,
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    new_memory_id=new_id_str,
                    superseded_memory_id=old_id_str,
                    reason=request.reason,
                    recorded_at=now.isoformat(),
                    idempotency_key=request.idempotency_key,
                    version=new_version,
                )
            )
            self._upsert_fts(
                conn,
                memory_id=new_id_str,
                tenant_id=auth.tenant_id,
                scope_kind=scope_kind,
                scope_id=scope_id,
                content=normalized_replacement,
                tags=old_tags,
            )

        return MemoryCorrectResponse(
            new_memory_id=new_id,
            superseded_memory_id=request.memory_id,
            version=new_version,
        )

    def is_superseded(
        self,
        auth: AuthContext,
        memory_id: UUID,
    ) -> tuple[bool, UUID | None]:
        """Return `(is_superseded, active_replacement_id)` for `memory_id` (M2.4)."""
        _require_permission(auth, Permission.READ)
        facts_filter = build_scoped_where(memory_facts, auth)

        with self._engine.connect() as conn:
            row = conn.execute(
                select(memory_facts.c.status, memory_facts.c.superseded_by).where(
                    and_(facts_filter, memory_facts.c.id == str(memory_id))
                )
            ).first()
            if row is None:
                return False, None
            if str(row[0]) != MemoryStatus.SUPERSEDED.value or not row[1]:
                return False, None

            # Follow any chain of supersessions (A -> B -> C) to the latest record
            current_id_str = str(row[1])
            visited: set[str] = {str(memory_id)}
            while current_id_str and current_id_str not in visited:
                visited.add(current_id_str)
                next_row = conn.execute(
                    select(memory_facts.c.status, memory_facts.c.superseded_by).where(
                        and_(facts_filter, memory_facts.c.id == current_id_str)
                    )
                ).first()
                if next_row is None:
                    break
                if str(next_row[0]) == MemoryStatus.SUPERSEDED.value and next_row[1]:
                    current_id_str = str(next_row[1])
                else:
                    break
            return True, UUID(current_id_str)

    def get_active_replacement(
        self,
        auth: AuthContext,
        memory_id: UUID,
    ) -> MemoryRecord | None:
        """If `memory_id` was superseded, return the current active replacement record."""
        superseded, replacement_id = self.is_superseded(auth, memory_id)
        if not superseded or replacement_id is None:
            return None
        return self.get(auth, replacement_id)

    def forget(self, auth: AuthContext, request: MemoryForgetRequest) -> MemoryForgetResponse:
        """Delete a memory record, purge content/FTS, and mark weights dirty if needed (M2.5)."""
        _require_permission(auth, Permission.FORGET)
        facts_filter = build_scoped_where(memory_facts, auth)
        del_filter = build_scoped_where(deletion_requests, auth)
        mid_str = str(request.memory_id)
        now = self._clock.now()

        with self._engine.begin() as conn:
            if request.idempotency_key:
                existing_del = conn.execute(
                    select(
                        deletion_requests.c.memory_id,
                        deletion_requests.c.weights_marked_dirty,
                    ).where(
                        and_(
                            del_filter,
                            deletion_requests.c.idempotency_key == request.idempotency_key,
                        )
                    )
                ).first()
                if existing_del is not None:
                    return MemoryForgetResponse(
                        memory_id=UUID(str(existing_del[0])),
                        weights_marked_dirty=bool(existing_del[1]),
                    )

            row = conn.execute(
                select(
                    memory_facts.c.scope_kind,
                    memory_facts.c.scope_id,
                    memory_facts.c.training_eligible,
                    memory_facts.c.status,
                ).where(and_(facts_filter, memory_facts.c.id == mid_str))
            ).first()
            if row is None:
                raise KeyError(f"Memory '{request.memory_id}' not found in authorized scopes.")

            scope_kind = str(row[0])
            scope_id = str(row[1])
            was_eligible = bool(row[2])

            # Remove from FTS index immediately in all deletion modes
            self._delete_fts(conn, memory_id=mid_str, tenant_id=auth.tenant_id)

            if request.deletion_mode == DeletionMode.HARD:
                conn.execute(
                    delete(memory_facts).where(and_(facts_filter, memory_facts.c.id == mid_str))
                )
                conn.execute(
                    delete(memory_events).where(
                        and_(
                            memory_events.c.tenant_id == auth.tenant_id,
                            memory_events.c.id == mid_str,
                        )
                    )
                )
                conn.execute(
                    delete(memory_sources).where(
                        and_(
                            memory_sources.c.tenant_id == auth.tenant_id,
                            memory_sources.c.memory_id == mid_str,
                        )
                    )
                )
            else:
                # SOFT deletion: retain content-free tombstone row
                conn.execute(
                    update(memory_facts)
                    .where(and_(facts_filter, memory_facts.c.id == mid_str))
                    .values(
                        content="",
                        tags_json="[]",
                        training_eligible=False,
                        status=MemoryStatus.DELETED.value,
                    )
                )
                conn.execute(
                    update(memory_events)
                    .where(
                        and_(
                            memory_events.c.tenant_id == auth.tenant_id,
                            memory_events.c.id == mid_str,
                        )
                    )
                    .values(
                        content="",
                        tags_json="[]",
                        training_eligible=False,
                        status=MemoryStatus.DELETED.value,
                    )
                )

            if was_eligible:
                self._set_weights_dirty(conn, auth.tenant_id, True)

            conn.execute(
                deletion_requests.insert().values(
                    request_id=str(uuid4()),
                    memory_id=mid_str,
                    tenant_id=auth.tenant_id,
                    scope_kind=scope_kind,
                    scope_id=scope_id,
                    deletion_mode=request.deletion_mode.value,
                    was_training_eligible=was_eligible,
                    weights_marked_dirty=was_eligible,
                    reason=request.reason,
                    requested_by=auth.principal_id,
                    deleted_at=now.isoformat(),
                    idempotency_key=request.idempotency_key,
                )
            )

        return MemoryForgetResponse(
            memory_id=request.memory_id,
            weights_marked_dirty=was_eligible,
        )

    def list_by_scope(
        self,
        auth: AuthContext,
        scope: Scope,
        include_superseded: bool = False,
    ) -> Sequence[MemoryRecord]:
        """List memory records within a single authorized scope."""
        _require_permission(auth, Permission.READ)
        scope_filter = build_scoped_where(memory_facts, auth, target_scopes=(scope,))

        with self._engine.connect() as conn:
            stmt = select(memory_facts).where(scope_filter)
            if include_superseded:
                stmt = stmt.where(
                    or_(
                        memory_facts.c.status == MemoryStatus.ACTIVE.value,
                        memory_facts.c.status == MemoryStatus.SUPERSEDED.value,
                    )
                )
            else:
                stmt = stmt.where(memory_facts.c.status == MemoryStatus.ACTIVE.value)

            stmt = stmt.order_by(memory_facts.c.event_time.desc())
            rows = conn.execute(stmt).fetchall()
            return tuple(_row_to_memory_record(r) for r in rows)

    def list_training_eligible(
        self,
        auth: AuthContext,
        *,
        include_superseded: bool = True,
    ) -> Sequence[MemoryRecord]:
        """List all training-eligible records across `auth`'s authorized scopes (M5.1).

        Excludes `DELETED` records, `SensitivityLevel.HIGH` records, and any
        record with `training_eligible=False`. Orders deterministically by
        `(recorded_at, id)`.
        """
        _require_permission(auth, Permission.READ)
        scope_filter = build_scoped_where(memory_facts, auth)
        statuses = (
            [MemoryStatus.ACTIVE.value, MemoryStatus.SUPERSEDED.value]
            if include_superseded
            else [MemoryStatus.ACTIVE.value]
        )

        with self._engine.connect() as conn:
            stmt = (
                select(memory_facts)
                .where(
                    and_(
                        scope_filter,
                        memory_facts.c.training_eligible.is_(True),
                        memory_facts.c.sensitivity == SensitivityLevel.NORMAL.value,
                        memory_facts.c.status.in_(statuses),
                    )
                )
                .order_by(memory_facts.c.recorded_at.asc(), memory_facts.c.id.asc())
            )
            rows = conn.execute(stmt).fetchall()
            return tuple(_row_to_memory_record(r) for r in rows)

    def log_access(
        self,
        auth: AuthContext,
        memory_ids: Sequence[UUID],
        *,
        access_kind: Literal["retrieved", "used"],
    ) -> int:
        """Record `retrieved` or `used` access events for authorized memories (M2.7)."""
        _require_permission(auth, Permission.READ)
        if not memory_ids:
            return 0
        facts_filter = build_scoped_where(memory_facts, auth)
        now_iso = self._clock.now().isoformat()
        id_strs = [str(m) for m in memory_ids]

        with self._engine.begin() as conn:
            rows = conn.execute(
                select(
                    memory_facts.c.id,
                    memory_facts.c.scope_kind,
                    memory_facts.c.scope_id,
                ).where(and_(facts_filter, memory_facts.c.id.in_(id_strs)))
            ).fetchall()
            if not rows:
                return 0

            conn.execute(
                memory_access.insert(),
                [
                    {
                        "access_id": str(uuid4()),
                        "memory_id": str(r[0]),
                        "tenant_id": auth.tenant_id,
                        "scope_kind": str(r[1]),
                        "scope_id": str(r[2]),
                        "access_kind": access_kind,
                        "principal_id": auth.principal_id,
                        "accessed_at": now_iso,
                    }
                    for r in rows
                ],
            )
            return len(rows)

    def get_access_stats(self, auth: AuthContext, memory_id: UUID) -> AccessStats:
        """Return `retrieved` vs `used` counts and last access timestamp for `memory_id`."""
        _require_permission(auth, Permission.READ)
        acc_filter = build_scoped_where(memory_access, auth)
        mid_str = str(memory_id)

        with self._engine.connect() as conn:
            rows = conn.execute(
                select(
                    memory_access.c.access_kind,
                    func.count(memory_access.c.access_id),
                    func.max(memory_access.c.accessed_at),
                )
                .where(and_(acc_filter, memory_access.c.memory_id == mid_str))
                .group_by(memory_access.c.access_kind)
            ).fetchall()

        retrieved_count = 0
        used_count = 0
        latest_iso: str | None = None
        for kind_val, cnt, max_ts in rows:
            if str(kind_val) == "retrieved":
                retrieved_count = int(cnt)
            elif str(kind_val) == "used":
                used_count = int(cnt)
            if max_ts is not None and (latest_iso is None or str(max_ts) > latest_iso):
                latest_iso = str(max_ts)

        return AccessStats(
            memory_id=memory_id,
            retrieved_count=retrieved_count,
            used_count=used_count,
            last_accessed_at=datetime.fromisoformat(latest_iso) if latest_iso else None,
        )

    def export_jsonl(self, auth: AuthContext) -> str:
        """Export all non-deleted records and revisions in `auth`'s scopes as JSONL (M2.8)."""
        _require_permission(auth, Permission.READ)
        facts_filter = build_scoped_where(memory_facts, auth)
        revs_filter = build_scoped_where(memory_revisions, auth)

        lines: list[str] = []
        with self._engine.connect() as conn:
            fact_rows = conn.execute(
                select(memory_facts)
                .where(
                    and_(
                        facts_filter,
                        memory_facts.c.status != MemoryStatus.DELETED.value,
                    )
                )
                .order_by(memory_facts.c.recorded_at.asc(), memory_facts.c.id.asc())
            ).fetchall()

            for r in fact_rows:
                rec = _row_to_memory_record(r)
                payload = {
                    "record_type": "memory_fact",
                    "id": str(rec.id),
                    "tenant_id": rec.tenant_id,
                    "scope": {"kind": rec.scope.kind.value, "id": rec.scope.id},
                    "kind": rec.kind.value,
                    "content": rec.content,
                    "event_time": rec.event_time.isoformat(),
                    "recorded_at": rec.recorded_at.isoformat(),
                    "verification": rec.verification.value,
                    "supersedes": [str(x) for x in rec.supersedes],
                    "sensitivity": rec.sensitivity.value,
                    "training_eligible": rec.training_eligible,
                    "status": rec.status.value,
                    "tags": list(rec.tags),
                    "version": rec.version,
                }
                lines.append(json.dumps(payload, sort_keys=True))

            rev_rows = conn.execute(
                select(memory_revisions)
                .where(revs_filter)
                .order_by(
                    memory_revisions.c.recorded_at.asc(), memory_revisions.c.revision_id.asc()
                )
            ).fetchall()

            for rev in rev_rows:
                rmap = rev._mapping
                rev_payload = {
                    "record_type": "memory_revision",
                    "revision_id": str(rmap["revision_id"]),
                    "tenant_id": str(rmap["tenant_id"]),
                    "scope": {
                        "kind": str(rmap["scope_kind"]),
                        "id": str(rmap["scope_id"]),
                    },
                    "new_memory_id": str(rmap["new_memory_id"]),
                    "superseded_memory_id": str(rmap["superseded_memory_id"]),
                    "reason": rmap["reason"],
                    "recorded_at": str(rmap["recorded_at"]),
                    "version": int(rmap["version"]),
                }
                lines.append(json.dumps(rev_payload, sort_keys=True))

        return ("\n".join(lines) + "\n") if lines else ""

    def ingest_record(self, auth: AuthContext, record: MemoryRecord) -> None:
        """Persist a pre-constructed `MemoryRecord` (used for benchmark/fixture ingestion, M2.10).

        Still enforces `auth` tenant matching, scope authorization, and secret checks.
        """
        _require_permission(auth, Permission.WRITE)
        if record.tenant_id != auth.tenant_id:
            raise PermissionError(
                f"Cannot ingest record for tenant '{record.tenant_id}' "
                f"under auth tenant '{auth.tenant_id}'."
            )
        build_scoped_where(memory_facts, auth, target_scopes=(record.scope,))

        if record.status != MemoryStatus.DELETED:
            assert_no_secrets(record.content)

        mid_str = str(record.id)
        supersedes_strs = [str(u) for u in record.supersedes]
        supersedes_json = json.dumps(supersedes_strs)
        tags_json = json.dumps(list(record.tags))
        stored_content = "" if record.status == MemoryStatus.DELETED else record.content

        with self._engine.begin() as conn:
            conn.execute(
                memory_events.insert().values(
                    id=mid_str,
                    tenant_id=record.tenant_id,
                    scope_kind=record.scope.kind.value,
                    scope_id=record.scope.id,
                    kind=record.kind.value,
                    content=stored_content,
                    event_time=record.event_time.isoformat(),
                    recorded_at=record.recorded_at.isoformat(),
                    confidence=record.confidence,
                    verification=record.verification.value,
                    supersedes_json=supersedes_json,
                    sensitivity=record.sensitivity.value,
                    training_eligible=record.training_eligible,
                    retention_policy=record.retention_policy,
                    status=record.status.value,
                    tags_json=tags_json,
                    idempotency_key=None,
                    version=record.version,
                )
            )
            conn.execute(
                memory_facts.insert().values(
                    id=mid_str,
                    tenant_id=record.tenant_id,
                    scope_kind=record.scope.kind.value,
                    scope_id=record.scope.id,
                    kind=record.kind.value,
                    content=stored_content,
                    event_time=record.event_time.isoformat(),
                    recorded_at=record.recorded_at.isoformat(),
                    confidence=record.confidence,
                    verification=record.verification.value,
                    supersedes_json=supersedes_json,
                    superseded_by=None,
                    sensitivity=record.sensitivity.value,
                    training_eligible=record.training_eligible,
                    retention_policy=record.retention_policy,
                    status=record.status.value,
                    tags_json=tags_json,
                    version=record.version,
                )
            )
            conn.execute(
                memory_sources.insert().values(
                    memory_id=mid_str,
                    tenant_id=record.tenant_id,
                    scope_kind=record.scope.kind.value,
                    scope_id=record.scope.id,
                    agent=record.source.agent,
                    session_id=record.source.session_id,
                    message_id=record.source.message_id,
                    recorded_at=record.recorded_at.isoformat(),
                )
            )

            # Link superseding records to their predecessors
            for old_id_str in supersedes_strs:
                conn.execute(
                    update(memory_facts)
                    .where(
                        and_(
                            memory_facts.c.tenant_id == record.tenant_id,
                            memory_facts.c.id == old_id_str,
                        )
                    )
                    .values(
                        status=MemoryStatus.SUPERSEDED.value,
                        superseded_by=mid_str,
                    )
                )
                conn.execute(
                    memory_revisions.insert().values(
                        revision_id=str(uuid4()),
                        tenant_id=record.tenant_id,
                        scope_kind=record.scope.kind.value,
                        scope_id=record.scope.id,
                        new_memory_id=mid_str,
                        superseded_memory_id=old_id_str,
                        reason="ingested_supersession",
                        recorded_at=record.recorded_at.isoformat(),
                        idempotency_key=None,
                        version=record.version,
                    )
                )

            if record.status != MemoryStatus.DELETED:
                self._upsert_fts(
                    conn,
                    memory_id=mid_str,
                    tenant_id=record.tenant_id,
                    scope_kind=record.scope.kind.value,
                    scope_id=record.scope.id,
                    content=record.content,
                    tags=record.tags,
                )
