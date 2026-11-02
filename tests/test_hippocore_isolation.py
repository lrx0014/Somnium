"""Property-based isolation test suite for Hippocore using Hypothesis (M2.9).

Generates random tenants, principals, scope subsets, memory records, corrections,
and deletions, then verifies that every read, write, mutation, export, and FTS5
retrieval path enforces strict scope and tenant isolation with zero leakage.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from somnium.domain import (
    AuthContext,
    DeletionMode,
    MemoryCorrectRequest,
    MemoryForgetRequest,
    MemoryKind,
    MemoryStoreRequest,
    Scope,
    ScopeKind,
)
from somnium.hippocore import HippocoreFtsRetriever, HippocoreStore

TENANT_POOL = ("tenant-alpha", "tenant-bravo", "tenant-charlie")
SCOPE_POOL = (
    Scope(kind=ScopeKind.PROJECT, id="project-orion"),
    Scope(kind=ScopeKind.PROJECT, id="project-orion-core"),
    Scope(kind=ScopeKind.PROJECT, id="project-lyra"),
    Scope(kind=ScopeKind.PROJECT, id="project-vega"),
    Scope(kind=ScopeKind.SESSION, id="session-001"),
    Scope(kind=ScopeKind.GLOBAL, id="workspace-shared"),
)
KEYWORD_POOL = (
    "postgresql",
    "clickhouse",
    "kafka",
    "temporal",
    "valkey",
    "dragonfly",
    "opensearch",
    "scylladb",
)

_tenant_st = st.sampled_from(TENANT_POOL)
_scope_st = st.sampled_from(SCOPE_POOL)
_keyword_st = st.sampled_from(KEYWORD_POOL)

_write_op_st = st.tuples(
    _tenant_st,
    _scope_st,
    _keyword_st,
    st.booleans(),  # whether to also issue a correction
)


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    write_ops=st.lists(_write_op_st, min_size=4, max_size=18),
    caller_tenant=_tenant_st,
    caller_scopes=st.sets(_scope_st, min_size=1, max_size=3),
    search_keyword=_keyword_st,
)
def test_property_zero_cross_scope_and_cross_tenant_leakage(
    write_ops: list[tuple[str, Scope, str, bool]],
    caller_tenant: str,
    caller_scopes: set[Scope],
    search_keyword: str,
) -> None:
    """Assert zero cross-scope and zero cross-tenant leakage across all Hippocore APIs."""
    store = HippocoreStore("sqlite:///:memory:")
    retriever = HippocoreFtsRetriever(store, min_bm25_score=0.1)

    # Track ground-truth ownership: memory_id -> (tenant_id, scope)
    ownership: dict[UUID, tuple[str, Scope]] = {}

    for idx, (tenant_id, scope, keyword, do_correct) in enumerate(write_ops):
        writer_auth = AuthContext(
            tenant_id=tenant_id,
            principal_id=f"writer-{tenant_id}",
            authorized_scopes=(scope,),
        )
        secret_marker = f"MARKER_{tenant_id}_{scope.id}_{idx}"
        resp = store.store(
            writer_auth,
            MemoryStoreRequest(
                content=f"Architecture note {idx} using {keyword} [{secret_marker}].",
                scope=scope,
                kind=MemoryKind.DECISION,
                tags=(keyword, "architecture"),
            ),
        )
        ownership[resp.memory_id] = (tenant_id, scope)
        store.log_access(writer_auth, [resp.memory_id], access_kind="used")

        if do_correct:
            corr = store.correct(
                writer_auth,
                MemoryCorrectRequest(
                    memory_id=resp.memory_id,
                    replacement=(
                        f"Revised architecture note {idx} using {keyword} [{secret_marker}_V2]."
                    ),
                    reason="Property test revision",
                ),
            )
            ownership[corr.new_memory_id] = (tenant_id, scope)

    caller_scope_tuple = tuple(caller_scopes)
    caller_auth = AuthContext(
        tenant_id=caller_tenant,
        principal_id="caller-principal",
        authorized_scopes=caller_scope_tuple,
    )

    def is_authorized(mid: UUID) -> bool:
        owner_tenant, owner_scope = ownership[mid]
        return owner_tenant == caller_tenant and owner_scope in caller_scopes

    # 1. Verify `get`, `is_superseded`, `get_active_replacement`, `get_access_stats`,
    #    `correct`, and `forget` for EVERY stored memory ID
    for mid, (_owner_tenant, owner_scope) in ownership.items():
        rec = store.get(caller_auth, mid)
        sup_flag, sup_target = store.is_superseded(caller_auth, mid)
        active_rep = store.get_active_replacement(caller_auth, mid)
        stats = store.get_access_stats(caller_auth, mid)

        if is_authorized(mid):
            assert rec is not None
            assert rec.tenant_id == caller_tenant
            assert rec.scope == owner_scope
        else:
            # Must NEVER leak across scope or tenant boundaries
            assert rec is None
            assert sup_flag is False
            assert sup_target is None
            assert active_rep is None
            assert stats.retrieved_count == 0
            assert stats.used_count == 0
            assert stats.last_accessed_at is None

            # Unauthorized mutation attempts (`correct` / `forget`) must fail with KeyError
            with pytest.raises(KeyError):
                store.correct(
                    caller_auth,
                    MemoryCorrectRequest(
                        memory_id=mid,
                        replacement="Unauthorized modification attempt.",
                    ),
                )
            with pytest.raises(KeyError):
                store.forget(
                    caller_auth,
                    MemoryForgetRequest(
                        memory_id=mid,
                        deletion_mode=DeletionMode.HARD,
                    ),
                )

    # 2. Verify `list_by_scope` for every scope in SCOPE_POOL
    for candidate_scope in SCOPE_POOL:
        if candidate_scope in caller_scopes:
            listed = store.list_by_scope(caller_auth, candidate_scope, include_superseded=True)
            for item in listed:
                assert item.tenant_id == caller_tenant
                assert item.scope == candidate_scope
                assert is_authorized(item.id)
        else:
            with pytest.raises(PermissionError):
                store.list_by_scope(caller_auth, candidate_scope)

    # 3. Verify `HippocoreFtsRetriever.retrieve` across authorized scopes
    hits = retriever.retrieve(
        caller_auth,
        f"Architecture note using {search_keyword}",
        scopes=caller_scope_tuple,
        limit=50,
        include_superseded=True,
    )
    for hit in hits:
        assert is_authorized(hit.memory_id)
        assert hit.scope in caller_scopes
        # Ensure no unauthorized tenant/scope marker appears in returned content
        owner_tenant, owner_scope = ownership[hit.memory_id]
        assert owner_tenant == caller_tenant
        assert owner_scope in caller_scopes

    # Requesting an unauthorized scope in `retrieve` must raise PermissionError
    unauthorized_scopes = [s for s in SCOPE_POOL if s not in caller_scopes]
    if unauthorized_scopes:
        with pytest.raises(PermissionError):
            retriever.retrieve(
                caller_auth,
                search_keyword,
                scopes=(unauthorized_scopes[0],),
            )

    # 4. Verify `export_jsonl` never exports records or revisions from other tenants or scopes
    exported_text = store.export_jsonl(caller_auth)
    for line in exported_text.splitlines():
        row = json.loads(line)
        assert row["tenant_id"] == caller_tenant
        row_scope = Scope(
            kind=ScopeKind(row["scope"]["kind"]),
            id=row["scope"]["id"],
        )
        assert row_scope in caller_scopes
