"""Portable SQLAlchemy Core table definitions for Hippocore (M2.1, design §10).

Defines the schema for:
- `memory_events`: Immutable raw event log (tombstoned/erased on deletion)
- `memory_facts`: Materialized current state of facts with supersession links
- `memory_revisions`: Immutable revision history tracking corrections
- `memory_sources`: Provenance metadata per memory record
- `memory_access`: Access log distinguishing `retrieved` vs `used`
- `deletion_requests`: Audit log of deletions (stores zero content)
- `tenant_state`: Per-tenant state including `weights_dirty` marker

Every table includes `tenant_id` and scope columns (`scope_kind`, `scope_id`)
where applicable so isolation is enforced at the SQL query level.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

metadata = MetaData()

memory_events = Table(
    "memory_events",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("scope_kind", String(32), nullable=False),
    Column("scope_id", String(256), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("content", Text, nullable=False),
    Column("event_time", String(64), nullable=False),
    Column("recorded_at", String(64), nullable=False),
    Column("confidence", Float, nullable=False, default=1.0),
    Column("verification", String(32), nullable=False),
    Column("supersedes_json", Text, nullable=False, default="[]"),
    Column("sensitivity", String(32), nullable=False),
    Column("training_eligible", Boolean, nullable=False),
    Column("retention_policy", String(64), nullable=False, default="default"),
    Column("status", String(32), nullable=False),
    Column("tags_json", Text, nullable=False, default="[]"),
    Column("idempotency_key", String(256), nullable=True),
    Column("version", Integer, nullable=False, default=1),
    Index("ix_memory_events_tenant_scope", "tenant_id", "scope_kind", "scope_id"),
    Index("ix_memory_events_idempotency", "tenant_id", "idempotency_key"),
)

memory_facts = Table(
    "memory_facts",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("scope_kind", String(32), nullable=False),
    Column("scope_id", String(256), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("content", Text, nullable=False),
    Column("event_time", String(64), nullable=False),
    Column("recorded_at", String(64), nullable=False),
    Column("confidence", Float, nullable=False, default=1.0),
    Column("verification", String(32), nullable=False),
    Column("supersedes_json", Text, nullable=False, default="[]"),
    Column("superseded_by", String(36), nullable=True),
    Column("sensitivity", String(32), nullable=False),
    Column("training_eligible", Boolean, nullable=False),
    Column("retention_policy", String(64), nullable=False, default="default"),
    Column("status", String(32), nullable=False),
    Column("tags_json", Text, nullable=False, default="[]"),
    Column("version", Integer, nullable=False, default=1),
    Index("ix_memory_facts_tenant_scope_status", "tenant_id", "scope_kind", "scope_id", "status"),
)

memory_revisions = Table(
    "memory_revisions",
    metadata,
    Column("revision_id", String(36), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("scope_kind", String(32), nullable=False),
    Column("scope_id", String(256), nullable=False),
    Column("new_memory_id", String(36), nullable=False),
    Column("superseded_memory_id", String(36), nullable=False),
    Column("reason", Text, nullable=True),
    Column("recorded_at", String(64), nullable=False),
    Column("idempotency_key", String(256), nullable=True),
    Column("version", Integer, nullable=False),
    Index("ix_memory_revisions_tenant_superseded", "tenant_id", "superseded_memory_id"),
    Index("ix_memory_revisions_tenant_new", "tenant_id", "new_memory_id"),
)

memory_sources = Table(
    "memory_sources",
    metadata,
    Column("memory_id", String(36), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("scope_kind", String(32), nullable=False),
    Column("scope_id", String(256), nullable=False),
    Column("agent", String(128), nullable=False),
    Column("session_id", String(256), nullable=True),
    Column("message_id", String(256), nullable=True),
    Column("recorded_at", String(64), nullable=False),
    Index(
        "ix_memory_sources_dedup",
        "tenant_id",
        "scope_kind",
        "scope_id",
        "agent",
        "session_id",
        "message_id",
    ),
)

memory_access = Table(
    "memory_access",
    metadata,
    Column("access_id", String(36), primary_key=True),
    Column("memory_id", String(36), nullable=False),
    Column("tenant_id", String(128), nullable=False),
    Column("scope_kind", String(32), nullable=False),
    Column("scope_id", String(256), nullable=False),
    Column("access_kind", String(32), nullable=False),  # "retrieved" or "used"
    Column("principal_id", String(256), nullable=False),
    Column("accessed_at", String(64), nullable=False),
    Index("ix_memory_access_tenant_memory", "tenant_id", "memory_id", "access_kind"),
)

deletion_requests = Table(
    "deletion_requests",
    metadata,
    Column("request_id", String(36), primary_key=True),
    Column("memory_id", String(36), nullable=False),
    Column("tenant_id", String(128), nullable=False),
    Column("scope_kind", String(32), nullable=False),
    Column("scope_id", String(256), nullable=False),
    Column("deletion_mode", String(32), nullable=False),
    Column("was_training_eligible", Boolean, nullable=False),
    Column("weights_marked_dirty", Boolean, nullable=False),
    Column("reason", Text, nullable=True),
    Column("requested_by", String(256), nullable=False),
    Column("deleted_at", String(64), nullable=False),
    Column("idempotency_key", String(256), nullable=True),
    Index("ix_deletion_requests_tenant_memory", "tenant_id", "memory_id"),
)

tenant_state = Table(
    "tenant_state",
    metadata,
    Column("tenant_id", String(128), primary_key=True),
    Column("weights_dirty", Boolean, nullable=False, default=False),
    Column("updated_at", String(64), nullable=False),
)

routing_decisions = Table(
    "routing_decisions",
    metadata,
    Column("decision_id", String(36), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("principal_id", String(256), nullable=False),
    Column("question", Text, nullable=False, default=""),
    Column("requested_scope", String(256), nullable=True),
    Column("requested_mode", String(32), nullable=False),
    Column("resolved_mode", String(32), nullable=False),
    Column("reason", String(128), nullable=False),
    Column("fallback_used", Boolean, nullable=False, default=False),
    Column("features_json", Text, nullable=False, default="{}"),
    Column("decided_at", String(64), nullable=False),
    Index("ix_routing_decisions_tenant_time", "tenant_id", "decided_at"),
)
