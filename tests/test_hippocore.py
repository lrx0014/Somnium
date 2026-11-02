"""Unit and integration tests for Hippocore episodic store and FTS5 retriever (M2.1–M2.8, M2.10)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, inspect, select, text

from somnium.backends.fake import FrozenClock
from somnium.domain import (
    AuthContext,
    DeletionMode,
    MemoryCorrectRequest,
    MemoryForgetRequest,
    MemoryKind,
    MemorySource,
    MemoryStatus,
    MemoryStoreRequest,
    Permission,
    Scope,
    ScopeKind,
    SensitivityLevel,
    VerificationLevel,
)
from somnium.eval import (
    HippocoreBaseline,
    evaluate_system_on_world,
    generate_synthetic_world,
)
from somnium.hippocore import (
    HEAD_REVISION,
    HippocoreFtsRetriever,
    HippocoreStore,
    SecretDetectedError,
    current_revision,
    deletion_requests,
    downgrade_database,
    memory_events,
    memory_facts,
    upgrade_database,
)
from somnium.interfaces import MemoryStore, Retriever

SCOPE_ORION = Scope(kind=ScopeKind.PROJECT, id="project-orion")
SCOPE_LYRA = Scope(kind=ScopeKind.PROJECT, id="project-lyra")


def _make_auth(
    *,
    tenant_id: str = "local-user",
    principal_id: str = "codex-agent",
    scopes: tuple[Scope, ...] = (SCOPE_ORION,),
    permissions: frozenset[Permission] | None = None,
) -> AuthContext:
    if permissions is None:
        return AuthContext(
            tenant_id=tenant_id,
            principal_id=principal_id,
            authorized_scopes=scopes,
        )
    return AuthContext(
        tenant_id=tenant_id,
        principal_id=principal_id,
        authorized_scopes=scopes,
        permissions=permissions,
    )


def test_alembic_migrations_upgrade_and_downgrade() -> None:
    """Verify forward and reverse Alembic migrations create and drop all tables cleanly (M2.1)."""
    engine = create_engine("sqlite:///:memory:", future=True)
    assert current_revision(engine) is None

    rev = upgrade_database(engine)
    assert rev == HEAD_REVISION
    assert current_revision(engine) == HEAD_REVISION
    # Calling upgrade again is idempotent
    assert upgrade_database(engine) == HEAD_REVISION

    with engine.connect() as conn:
        tables = set(inspect(conn).get_table_names())
    expected_tables = {
        "alembic_version",
        "memory_events",
        "memory_facts",
        "memory_revisions",
        "memory_sources",
        "memory_access",
        "deletion_requests",
        "tenant_state",
        "memory_fts",
    }
    assert expected_tables.issubset(tables)

    downgrade_database(engine, "base")
    assert current_revision(engine) is None

    with engine.connect() as conn:
        remaining = set(inspect(conn).get_table_names())
    for core_table in expected_tables - {"alembic_version"}:
        assert core_table not in remaining


def test_protocols_satisfied() -> None:
    """Verify HippocoreStore and HippocoreFtsRetriever satisfy core Protocols (M2.2, M2.6)."""
    store = HippocoreStore("sqlite:///:memory:")
    retriever = HippocoreFtsRetriever(store)
    assert isinstance(store, MemoryStore)
    assert isinstance(retriever, Retriever)


def test_write_normalization_deduplication_and_sensitivity() -> None:
    """Verify write normalization, deduplication, and sensitivity rules (M2.3)."""
    clock = FrozenClock(datetime(2026, 3, 1, 12, 0, tzinfo=UTC))
    store = HippocoreStore("sqlite:///:memory:", clock=clock)
    auth = _make_auth(scopes=(SCOPE_ORION,))

    # 1. Normalization & idempotency_key deduplication
    req1 = MemoryStoreRequest(
        content="  Project Orion   uses   PostgreSQL 16.\r\n  ",
        scope=SCOPE_ORION,
        kind=MemoryKind.DECISION,
        source=MemorySource(agent="codex", session_id="s-1", message_id="m-1"),
        tags=("database", "architecture"),
        idempotency_key="idem-orion-db-1",
    )
    resp1 = store.store(auth, req1)
    assert not resp1.deduplicated
    assert resp1.training_eligible is True

    rec1 = store.get(auth, resp1.memory_id)
    assert rec1 is not None
    assert rec1.content == "Project Orion uses PostgreSQL 16."
    assert rec1.source.session_id == "s-1"

    # Duplicate by idempotency_key returns existing memory_id
    resp1_dup = store.store(auth, req1)
    assert resp1_dup.deduplicated is True
    assert resp1_dup.memory_id == resp1.memory_id

    # Duplicate by source identity (agent + session_id + message_id) even without idempotency_key
    req1_src_dup = MemoryStoreRequest(
        content="Project Orion uses PostgreSQL 16.",
        scope=SCOPE_ORION,
        kind=MemoryKind.DECISION,
        source=MemorySource(agent="codex", session_id="s-1", message_id="m-1"),
    )
    resp1_src_dup = store.store(auth, req1_src_dup)
    assert resp1_src_dup.deduplicated is True
    assert resp1_src_dup.memory_id == resp1.memory_id

    # 2. Rule-based high sensitivity classification (.corp.internal hostname / customer ID)
    req_sens = MemoryStoreRequest(
        content="Internal staging DB is hosted at db-01.corp.internal for CUST-994812.",
        scope=SCOPE_ORION,
        kind=MemoryKind.FACT,
    )
    resp_sens = store.store(auth, req_sens)
    assert resp_sens.training_eligible is False
    rec_sens = store.get(auth, resp_sens.memory_id)
    assert rec_sens is not None
    assert rec_sens.sensitivity == SensitivityLevel.HIGH
    assert rec_sens.training_eligible is False


@pytest.mark.parametrize(
    "secret_text",
    [
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmU=\n",
        "Use OpenAI key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890 for staging.",
        "GitHub token is ghp_1234567890abcdefghijklmnopqrstuvwxyzABCD.",
        "AWS key ID AKIAIOSFODNN7EXAMPLE was rotated.",
        "DATABASE_PASSWORD = 'super_secret_prod_password_123'",
    ],
)
def test_secret_detection_rejects_credentials(secret_text: str) -> None:
    """Verify credentials, private keys, and tokens are rejected on store and correct (M2.3)."""
    store = HippocoreStore("sqlite:///:memory:")
    auth = _make_auth(scopes=(SCOPE_ORION,))

    with pytest.raises(SecretDetectedError):
        store.store(
            auth,
            MemoryStoreRequest(
                content=secret_text,
                scope=SCOPE_ORION,
            ),
        )


def test_corrections_revisions_and_fts_ordering() -> None:
    """Verify corrections create revisions, chain properly, and update FTS (M2.4, M2.6)."""
    clock = FrozenClock(datetime(2026, 3, 1, 10, 0, tzinfo=UTC))
    store = HippocoreStore("sqlite:///:memory:", clock=clock)
    retriever = HippocoreFtsRetriever(store)
    auth = _make_auth(scopes=(SCOPE_ORION,))

    # Store v1
    v1_resp = store.store(
        auth,
        MemoryStoreRequest(
            content="Project Orion selected MongoDB as its primary database.",
            scope=SCOPE_ORION,
            kind=MemoryKind.DECISION,
            tags=("database",),
        ),
    )
    v1_id = v1_resp.memory_id

    # Correct v1 -> v2
    clock.advance(seconds=3600)
    corr1 = store.correct(
        auth,
        MemoryCorrectRequest(
            memory_id=v1_id,
            replacement=(
                "Project Orion replaced MongoDB with PostgreSQL 16 as its primary database."
            ),
            reason="Needed transactional DDL",
            idempotency_key="corr-1",
        ),
    )
    v2_id = corr1.new_memory_id
    assert corr1.superseded_memory_id == v1_id
    assert corr1.version == 2

    # Idempotent retry of the same correction
    corr1_retry = store.correct(
        auth,
        MemoryCorrectRequest(
            memory_id=v1_id,
            replacement=(
                "Project Orion replaced MongoDB with PostgreSQL 16 as its primary database."
            ),
            idempotency_key="corr-1",
        ),
    )
    assert corr1_retry.new_memory_id == v2_id

    # Attempting to correct an already-superseded memory without idempotency key raises ValueError
    with pytest.raises(ValueError, match="already superseded"):
        store.correct(
            auth,
            MemoryCorrectRequest(
                memory_id=v1_id,
                replacement="Another conflicting correction.",
            ),
        )

    # Correct v2 -> v3 to verify supersession chain resolution (v1 -> v2 -> v3)
    clock.advance(seconds=3600)
    corr2 = store.correct(
        auth,
        MemoryCorrectRequest(
            memory_id=v2_id,
            replacement="Project Orion upgraded its primary database to PostgreSQL 17.",
            reason="Major version upgrade",
        ),
    )
    v3_id = corr2.new_memory_id
    assert corr2.version == 3

    is_sup, active_id = store.is_superseded(auth, v1_id)
    assert is_sup is True
    assert active_id == v3_id

    active_rec = store.get_active_replacement(auth, v1_id)
    assert active_rec is not None
    assert active_rec.id == v3_id
    assert "PostgreSQL 17" in active_rec.content

    # Default FTS retrieval returns only ACTIVE records
    active_hits = retriever.retrieve(
        auth,
        "What is the primary database for project-orion?",
        scopes=(SCOPE_ORION,),
    )
    assert len(active_hits) == 1
    assert active_hits[0].memory_id == v3_id
    assert active_hits[0].status == MemoryStatus.ACTIVE
    assert active_hits[0].supersedes == (v2_id,)
    assert 0.0 < active_hits[0].relevance <= 1.0

    # When include_superseded=True, ACTIVE records are ordered first, followed by SUPERSEDED
    all_hits = retriever.retrieve(
        auth,
        "primary database",
        scopes=(SCOPE_ORION,),
        include_superseded=True,
    )
    assert len(all_hits) == 3
    assert all_hits[0].memory_id == v3_id
    assert all_hits[0].status == MemoryStatus.ACTIVE
    superseded_hits = [h for h in all_hits[1:] if h.status == MemoryStatus.SUPERSEDED]
    assert len(superseded_hits) == 2
    for sh in superseded_hits:
        assert sh.superseded_by in (v2_id, v3_id)


def test_deletion_modes_tombstones_and_weights_dirty() -> None:
    """Verify HARD and SOFT deletion purge content/FTS and set weights_dirty accurately (M2.5)."""
    store = HippocoreStore("sqlite:///:memory:")
    retriever = HippocoreFtsRetriever(store)
    auth = _make_auth(scopes=(SCOPE_ORION,))

    # 1. Deleting a HIGH sensitivity (non-training-eligible) record does NOT mark weights dirty
    sens_resp = store.store(
        auth,
        MemoryStoreRequest(
            content="Sensitive staging endpoint https://vault.corp.internal:8200.",
            scope=SCOPE_ORION,
            sensitivity=SensitivityLevel.HIGH,
        ),
    )
    assert sens_resp.training_eligible is False
    del_sens = store.forget(
        auth,
        MemoryForgetRequest(
            memory_id=sens_resp.memory_id,
            deletion_mode=DeletionMode.SOFT,
            reason="Decommissioned",
        ),
    )
    assert del_sens.weights_marked_dirty is False
    assert store.is_weights_dirty(auth) is False

    # Verify SOFT deletion leaves a content-free tombstone and removes from FTS
    assert store.get(auth, sens_resp.memory_id, include_deleted=False) is None
    tombstone = store.get(auth, sens_resp.memory_id, include_deleted=True)
    assert tombstone is not None
    assert tombstone.status == MemoryStatus.DELETED
    assert tombstone.content == "[TOMBSTONE]"

    with store.engine.connect() as conn:
        raw_fact_content = conn.execute(
            select(memory_facts.c.content).where(memory_facts.c.id == str(sens_resp.memory_id))
        ).scalar_one()
        raw_event_content = conn.execute(
            select(memory_events.c.content).where(memory_events.c.id == str(sens_resp.memory_id))
        ).scalar_one()
        fts_count = conn.execute(
            text("SELECT count(*) FROM memory_fts WHERE memory_id = :mid"),
            {"mid": str(sens_resp.memory_id)},
        ).scalar_one()
    assert raw_fact_content == ""
    assert raw_event_content == ""
    assert int(fts_count) == 0

    # 2. Deleting a training-eligible record with HARD mode removes rows and sets weights_dirty=True
    elig_resp = store.store(
        auth,
        MemoryStoreRequest(
            content="Project Orion uses Redis Streams for ephemeral event fanout.",
            scope=SCOPE_ORION,
            verification=VerificationLevel.USER_CONFIRMED,
        ),
    )
    assert elig_resp.training_eligible is True

    del_elig = store.forget(
        auth,
        MemoryForgetRequest(
            memory_id=elig_resp.memory_id,
            deletion_mode=DeletionMode.HARD,
            reason="Replaced architecture",
            idempotency_key="del-elig-1",
        ),
    )
    assert del_elig.weights_marked_dirty is True
    assert store.is_weights_dirty(auth) is True

    # Idempotent retry of forget
    del_retry = store.forget(
        auth,
        MemoryForgetRequest(
            memory_id=elig_resp.memory_id,
            deletion_mode=DeletionMode.HARD,
            idempotency_key="del-elig-1",
        ),
    )
    assert del_retry.weights_marked_dirty is True

    # Verify row is completely gone from memory_facts, memory_events, and memory_fts
    assert store.get(auth, elig_resp.memory_id, include_deleted=True) is None
    assert retriever.retrieve(auth, "Redis Streams ephemeral fanout", scopes=(SCOPE_ORION,)) == ()

    with store.engine.connect() as conn:
        audit_rows = conn.execute(select(deletion_requests)).fetchall()
    assert len(audit_rows) == 2

    # Clearing weights_dirty resets the flag
    store.clear_weights_dirty(auth)
    assert store.is_weights_dirty(auth) is False


def test_access_logging_distinguishes_retrieved_and_used() -> None:
    """Verify access logging tracks `retrieved` vs `used` counts separately (M2.7)."""
    clock = FrozenClock(datetime(2026, 3, 2, 9, 0, tzinfo=UTC))
    store = HippocoreStore("sqlite:///:memory:", clock=clock)
    retriever = HippocoreFtsRetriever(store)
    auth = _make_auth(scopes=(SCOPE_ORION,))

    stored = store.store(
        auth,
        MemoryStoreRequest(
            content="Project Orion uses Temporal for durable workflow orchestration.",
            scope=SCOPE_ORION,
            tags=("workflow", "temporal"),
        ),
    )
    mid = stored.memory_id

    # Retrieve twice via FTS
    hits1 = retriever.retrieve(auth, "Temporal workflow orchestration", scopes=(SCOPE_ORION,))
    assert len(hits1) == 1
    clock.advance(seconds=60)
    hits2 = retriever.retrieve(auth, "Temporal workflow", scopes=(SCOPE_ORION,))
    assert len(hits2) == 1

    # Log 1 `used` event when cited in an answer
    clock.advance(seconds=120)
    used_time = clock.now()
    logged = store.log_access(auth, [mid], access_kind="used")
    assert logged == 1

    stats = store.get_access_stats(auth, mid)
    assert stats.memory_id == mid
    assert stats.retrieved_count == 2
    assert stats.used_count == 1
    assert stats.last_accessed_at == used_time


def test_export_jsonl_includes_authorized_facts_and_revisions() -> None:
    """Verify JSONL export contains only authorized non-deleted facts and revisions (M2.8)."""
    store = HippocoreStore("sqlite:///:memory:")
    auth_orion = _make_auth(scopes=(SCOPE_ORION,))
    auth_lyra = _make_auth(scopes=(SCOPE_LYRA,))

    r_orion = store.store(
        auth_orion,
        MemoryStoreRequest(
            content="Orion initial queue: RabbitMQ.",
            scope=SCOPE_ORION,
        ),
    )
    store.correct(
        auth_orion,
        MemoryCorrectRequest(
            memory_id=r_orion.memory_id,
            replacement="Orion updated queue: NATS JetStream.",
            reason="Lower operational overhead",
        ),
    )
    store.store(
        auth_lyra,
        MemoryStoreRequest(
            content="Lyra confidential architecture note.",
            scope=SCOPE_LYRA,
        ),
    )

    exported = store.export_jsonl(auth_orion)
    records = [json.loads(line) for line in exported.strip().splitlines()]
    # Should have 2 facts (superseded + active) and 1 revision in project-orion, 0 from project-lyra
    fact_items = [r for r in records if r["record_type"] == "memory_fact"]
    rev_items = [r for r in records if r["record_type"] == "memory_revision"]
    assert len(fact_items) == 2
    assert len(rev_items) == 1
    for item in records:
        assert item["scope"]["id"] == "project-orion"
        assert "Lyra" not in json.dumps(item)


def test_hippocore_benchmark_integration() -> None:
    """Verify `hippocore-fts` ingests the M1 synthetic world and passes safety gates (M2.10)."""
    world = generate_synthetic_world(seed=42)
    sut = HippocoreBaseline()
    result = evaluate_system_on_world(
        sut,
        world,
        max_slice=2,
        compute_uplift_against_fts=False,
    )
    m = result.metrics
    assert m.isolation_leakage == 0.0
    assert m.deletion_leakage == 0.0
    assert m.unsupported_claim_rate == 0.0
    assert m.abstention_quality == 1.0
    assert m.category_accuracy["direct"] == 1.0
    assert m.category_accuracy["temporal"] == 1.0
    assert m.category_accuracy["correction"] == 1.0
