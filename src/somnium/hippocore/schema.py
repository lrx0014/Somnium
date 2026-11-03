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
    Column("active_model_version", String(256), nullable=True),
    Column("dream_state", String(64), nullable=False, default="awake"),
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

dataset_manifests = Table(
    "dataset_manifests",
    metadata,
    Column("dataset_hash", String(64), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("mode", String(32), nullable=False),
    Column("parent_manifest_hash", String(64), nullable=True),
    Column("generator_version", String(64), nullable=False),
    Column("artifact_uri", Text, nullable=False),
    Column("source_record_ids_json", Text, nullable=False, default="[]"),
    Column("example_count", Integer, nullable=False, default=0),
    Column("train_count", Integer, nullable=False, default=0),
    Column("held_out_count", Integer, nullable=False, default=0),
    Column("category_counts_json", Text, nullable=False, default="{}"),
    Column("created_at", String(64), nullable=False),
    Index("ix_dataset_manifests_tenant_time", "tenant_id", "created_at"),
)

training_examples = Table(
    "training_examples",
    metadata,
    Column("example_id", String(128), primary_key=True),
    Column("dataset_hash", String(64), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("category", String(64), nullable=False),
    Column("fact_group_id", String(128), nullable=False),
    Column("scope", String(256), nullable=False),
    Column("split", String(32), nullable=False, default="train"),
    Column("source_record_ids_json", Text, nullable=False, default="[]"),
    Column("system_prompt", Text, nullable=False),
    Column("user_prompt", Text, nullable=False),
    Column("assistant_response", Text, nullable=False),
    Column("created_at", String(64), nullable=False),
    Index("ix_training_examples_tenant_dataset", "tenant_id", "dataset_hash"),
)

model_versions = Table(
    "model_versions",
    metadata,
    Column("model_version", String(256), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("base_model_id", String(256), nullable=False),
    Column("base_model_revision", String(128), nullable=False),
    Column("parent_version", String(256), nullable=True),
    Column("training_mode", String(32), nullable=False),
    Column("dataset_manifest_hash", String(64), nullable=False),
    Column("checkpoint_uri", Text, nullable=False),
    Column("status", String(32), nullable=False),
    Column("metrics_json", Text, nullable=False, default="{}"),
    Column("gate_report_json", Text, nullable=False, default="{}"),
    Column("created_at", String(64), nullable=False),
    Column("promoted_at", String(64), nullable=True),
    Index("ix_model_versions_tenant_status", "tenant_id", "status"),
)

dream_runs = Table(
    "dream_runs",
    metadata,
    Column("run_id", String(64), primary_key=True),
    Column("tenant_id", String(128), nullable=False),
    Column("requested_mode", String(32), nullable=False),
    Column("resolved_mode", String(32), nullable=False),
    Column("forced_full_retrain_reason", String(256), nullable=True),
    Column("state", String(64), nullable=False),
    Column("parent_model_version", String(256), nullable=True),
    Column("candidate_model_version", String(256), nullable=True),
    Column("dataset_manifest_hash", String(64), nullable=True),
    Column("training_metrics_json", Text, nullable=False, default="{}"),
    Column("gate_passed", Boolean, nullable=False, default=False),
    Column("gate_report_json", Text, nullable=False, default="{}"),
    Column("started_at", String(64), nullable=False),
    Column("completed_at", String(64), nullable=True),
    Index("ix_dream_runs_tenant_started", "tenant_id", "started_at"),
)
