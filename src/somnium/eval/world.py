"""Seeded, deterministic synthetic world generator for Somnium benchmarks (M1.1).

Generates a rich multi-project, multi-tenant episodic memory world containing:
- Overlapping project names (`project-orion`, `project-orion-core`, `project-lyra`, `project-vega`)
- Global vs project-scoped preferences
- Precise `event_time` facts across two temporal slices (`slice=1` and `slice=2`)
- Decisions superseded over time (both within Slice 1 and across Slice 1 -> Slice 2)
- Near-duplicate facts across projects (same attribute, different project choices)
- High-sensitivity records (`sensitivity: high` -> `training_eligible: False`)
- Deleted records (`status: deleted`) and correction events (`supersedes`)
- Cross-tenant records (`tenant-bravo`) for isolation leakage testing
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from somnium.domain import (
    MemoryKind,
    MemoryRecord,
    MemorySource,
    MemoryStatus,
    Scope,
    ScopeKind,
    SensitivityLevel,
    TenantId,
    VerificationLevel,
)

PRIMARY_TENANT: TenantId = "local-user"
SECONDARY_TENANT: TenantId = "tenant-bravo"

PROJECT_ORION = Scope(kind=ScopeKind.PROJECT, id="project-orion")
PROJECT_ORION_CORE = Scope(kind=ScopeKind.PROJECT, id="project-orion-core")
PROJECT_LYRA = Scope(kind=ScopeKind.PROJECT, id="project-lyra")
PROJECT_VEGA = Scope(kind=ScopeKind.PROJECT, id="project-vega")
GLOBAL_SCOPE = Scope(kind=ScopeKind.GLOBAL, id="user-preferences")
BRAVO_PROJECT = Scope(kind=ScopeKind.PROJECT, id="project-chimera")

ALL_PRIMARY_SCOPES: tuple[Scope, ...] = (
    PROJECT_ORION,
    PROJECT_ORION_CORE,
    PROJECT_LYRA,
    PROJECT_VEGA,
    GLOBAL_SCOPE,
)


def _deterministic_uuid(seed: int, namespace: str, index: int) -> UUID:
    """Create a deterministic UUIDv4-formatted identifier from `(seed, namespace, index)`."""
    digest = hashlib.sha256(f"somnium-world:{seed}:{namespace}:{index}".encode()).digest()
    raw = bytearray(digest[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40  # version 4
    raw[8] = (raw[8] & 0x3F) | 0x80  # variant RFC 4122
    return UUID(bytes=bytes(raw))


@dataclass(frozen=True, slots=True)
class FactBlueprint:
    """Ground-truth metadata for an underlying synthetic fact family.

    Used to generate questions without train/test leakage (splitting by
    `fact_group_id` ensures paraphrases of trained facts never leak into unseen
    fact evaluation).
    """

    fact_group_id: str
    tenant_id: TenantId
    scope: Scope
    topic: str
    attribute: str
    current_value: str
    current_record_id: UUID
    current_event_time: datetime
    temporal_slice: int  # 1 or 2
    kind: MemoryKind
    sensitivity: SensitivityLevel = SensitivityLevel.NORMAL
    is_deleted: bool = False
    superseded_value: str | None = None
    superseded_record_id: UUID | None = None
    superseded_event_time: datetime | None = None
    vague_cue: str = ""
    rationale: str = ""
    split: str = "train"  # "train" or "held_out", assigned deterministically by seed


@dataclass(frozen=True, slots=True)
class SyntheticWorld:
    """Complete deterministic synthetic memory world and its ground-truth blueprints."""

    suite_name: str
    seed: int
    records: tuple[MemoryRecord, ...]
    fact_blueprints: tuple[FactBlueprint, ...]
    manifest_hash: str

    def records_for_slice(self, max_slice: int = 2) -> tuple[MemoryRecord, ...]:
        """Return records belonging to temporal slices `<= max_slice`.

        When `max_slice == 1`, records that were only superseded in Slice 2 are
        returned in their Slice 1 `ACTIVE` state so Slice 1 experiments see the
        world as it existed at the end of Slice 1.
        """
        if max_slice >= 2:
            return self.records

        slice2_cutoff = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        slice2_superseding_targets: set[UUID] = set()
        for rec in self.records:
            if rec.event_time >= slice2_cutoff and rec.supersedes:
                slice2_superseding_targets.update(rec.supersedes)

        result: list[MemoryRecord] = []
        for rec in self.records:
            if rec.event_time >= slice2_cutoff:
                continue
            if rec.id in slice2_superseding_targets and rec.status == MemoryStatus.SUPERSEDED:
                # In Slice 1, this record had not yet been superseded by Slice 2.
                restored = rec.model_copy(update={"status": MemoryStatus.ACTIVE})
                result.append(restored)
            else:
                result.append(rec)
        return tuple(result)

    def blueprints_for_slice(self, max_slice: int = 2) -> tuple[FactBlueprint, ...]:
        """Return fact blueprints active up to `max_slice`."""
        if max_slice >= 2:
            return self.fact_blueprints

        slice2_cutoff = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
        adjusted: list[FactBlueprint] = []
        for bp in self.fact_blueprints:
            if bp.temporal_slice == 1:
                adjusted.append(bp)
            elif (
                bp.temporal_slice == 2
                and bp.superseded_value is not None
                and bp.superseded_record_id is not None
                and bp.superseded_event_time is not None
                and bp.superseded_event_time < slice2_cutoff
            ):
                # At the end of Slice 1, the initial record was still current.
                adjusted.append(
                    FactBlueprint(
                        fact_group_id=bp.fact_group_id,
                        tenant_id=bp.tenant_id,
                        scope=bp.scope,
                        topic=bp.topic,
                        attribute=bp.attribute,
                        current_value=bp.superseded_value,
                        current_record_id=bp.superseded_record_id,
                        current_event_time=bp.superseded_event_time,
                        temporal_slice=1,
                        kind=bp.kind,
                        sensitivity=bp.sensitivity,
                        is_deleted=False,
                        superseded_value=None,
                        superseded_record_id=None,
                        superseded_event_time=None,
                        vague_cue=bp.vague_cue,
                        rationale=bp.rationale,
                        split=bp.split,
                    )
                )
        return tuple(adjusted)


# Matrix of project technical attributes with near-duplicate structure across the
# 4 overlapping projects (`project-orion`, `project-orion-core`, `project-lyra`,
# `project-vega`). Each entry specifies:
# (attribute_slug, attribute_label, values_per_project, vague_cue_template, rationale_template)
_PROJECT_ATTRIBUTE_CATALOG: tuple[
    tuple[str, str, dict[str, tuple[str, str | None]], str, str],
    ...,
] = (
    (
        "primary_database",
        "primary database engine",
        {
            "project-orion": ("PostgreSQL 16", "MongoDB 7"),
            "project-orion-core": ("SQLite FTS5", None),
            "project-lyra": ("ClickHouse 24", "TimescaleDB"),
            "project-vega": ("DuckDB 1.1", None),
        },
        "that storage engine migration where document locks or analytics throughput bit us",
        "strict transactional consistency and relational schema constraints",
    ),
    (
        "cache_backend",
        "distributed cache layer",
        {
            "project-orion": ("Valkey 8", "Redis 6"),
            "project-orion-core": ("In-process LRU cache", None),
            "project-lyra": ("DragonflyDB", None),
            "project-vega": ("Memcached", "Redis 6"),
        },
        "the in-memory caching switch after the licensing and tail-latency review",
        "permissive BSD licensing and lower p99 latency under burst load",
    ),
    (
        "message_broker",
        "asynchronous event broker",
        {
            "project-orion": ("NATS JetStream", "RabbitMQ"),
            "project-orion-core": ("Local Unix domain sockets", None),
            "project-lyra": ("Apache Kafka", None),
            "project-vega": ("Redpanda", "Apache Pulsar"),
        },
        "the lightweight pub/sub queue we adopted when broker JVM memory spiked",
        "single-binary operational simplicity and lightweight replay streams",
    ),
    (
        "auth_mechanism",
        "service authentication mechanism",
        {
            "project-orion": ("Ed25519 PASETO v4 tokens", "RS256 JWT"),
            "project-orion-core": ("Loopback mTLS certificates", None),
            "project-lyra": ("OIDC federated workload tokens", None),
            "project-vega": ("HMAC-SHA256 API keys", "Basic Auth"),
        },
        "the token format upgrade after our cryptographic header confusion audit",
        "eliminating algorithm-confusion attacks in token verification headers",
    ),
    (
        "serialization_format",
        "wire serialization format",
        {
            "project-orion": ("Protocol Buffers v3", "MessagePack"),
            "project-orion-core": ("Canonical JSONL", None),
            "project-lyra": ("Apache Arrow IPC", None),
            "project-vega": ("FlatBuffers", None),
        },
        "the binary wire encoding chosen for backwards-compatible schema evolution",
        "explicit field numbering and zero-copy columnar compatibility",
    ),
    (
        "deployment_target",
        "production deployment target",
        {
            "project-orion": ("Debian 12 bare-metal GPU nodes", "EKS Kubernetes cluster"),
            "project-orion-core": ("Local macOS launchd agent", None),
            "project-lyra": ("Nomad GPU cluster", None),
            "project-vega": ("Fly.io Firecracker microVMs", None),
        },
        "the orchestrator simplification where we dropped control-plane overhead",
        "predictable GPU device passthrough without container CNI jitter",
    ),
    (
        "observability_sink",
        "telemetry and tracing sink",
        {
            "project-orion": ("OpenTelemetry to ClickHouse", "Jaeger all-in-one"),
            "project-orion-core": ("Structured JSONL ring buffer", None),
            "project-lyra": ("VictoriaMetrics + Tempo", None),
            "project-vega": ("Prometheus + Loki", None),
        },
        "the tracing pipeline consolidation when span storage costs doubled",
        "unified SQL querying over traces and metrics with high compression",
    ),
    (
        "vector_index",
        "embedding similarity index",
        {
            "project-orion": ("pgvector HNSW", "FAISS flat index"),
            "project-orion-core": ("Exact cosine over NumPy mmap", None),
            "project-lyra": ("Qdrant gRPC cluster", None),
            "project-vega": ("USearch HNSW", None),
        },
        "the nearest-neighbor index we picked so metadata filters stay transactional",
        "atomic filtering alongside relational scope permissions in one query",
    ),
    (
        "rate_limiter",
        "ingestion rate-limiting algorithm",
        {
            "project-orion": ("GCRA leaky bucket", "Fixed-window counter"),
            "project-orion-core": ("Token bucket per session", None),
            "project-lyra": ("Adaptive concurrency AIMD", None),
            "project-vega": ("Sliding-window log", None),
        },
        "the traffic shaping algorithm that stopped boundary-burst thundering herds",
        "smooth inter-arrival spacing without boundary-window double bursts",
    ),
    (
        "migrations_tool",
        "schema migration tool",
        {
            "project-orion": ("Alembic with strict revision hashes", "Raw bash SQL scripts"),
            "project-orion-core": ("Alembic offline SQLite batch mode", None),
            "project-lyra": ("Atlas declarative migrations", None),
            "project-vega": ("Goose SQL migrations", None),
        },
        "the database migration runner that prevented out-of-order schema drift",
        "verifiable linear revision history and repeatable downgrade tests",
    ),
    (
        "test_property_framework",
        "property-based testing library",
        {
            "project-orion": ("Hypothesis stateful machines", None),
            "project-orion-core": ("Hypothesis scope isolation generators", None),
            "project-lyra": ("Schemathesis API fuzzing", None),
            "project-vega": ("Pytest parametrized fixtures", None),
        },
        "the randomized invariant tester that caught our edge-case scope bug",
        "automatic shrinking of counterexamples on multi-step state transitions",
    ),
    (
        "llm_quantization",
        "adapter training quantization scheme",
        {
            "project-orion": ("NF4 4-bit QLoRA", "FP16 full LoRA"),
            "project-orion-core": ("MLX 4-bit group-size 64", None),
            "project-lyra": ("BF16 LoRA rank 32", "INT8 LoRA"),
            "project-vega": ("AWQ 4-bit inference only", None),
        },
        "the low-bit weight compression setting that fit training inside memory budgets",
        "halving activation and weight footprint while keeping adapter gradients stable",
    ),
    (
        "backup_schedule",
        "snapshot backup cadence",
        {
            "project-orion": ("Hourly WAL shipping to S3 + daily ZSTD snapshot", None),
            "project-orion-core": ("Daily SQLite online backup to encrypted APFS", None),
            "project-lyra": ("Every 6 hours incremental parquet dump", None),
            "project-vega": ("Litestream continuous replication", None),
        },
        "the point-in-time recovery strategy we established after the restore drill",
        "sub-hour recovery point objective with verified automated restore drills",
    ),
    (
        "secret_scanner",
        "pre-ingestion secret detection policy",
        {
            "project-orion": ("Entropy + regex gate rejecting PEM and API keys", None),
            "project-orion-core": ("Strict regex denylist at Hippocore write path", None),
            "project-lyra": ("TruffleHog pre-commit + gateway filter", None),
            "project-vega": ("Gitleaks rule engine", None),
        },
        "the credential scrubbing guardrail that blocks private keys before storage",
        "preventing accidental persistence of credentials in episodic logs or weights",
    ),
    (
        "ci_runner_matrix",
        "continuous integration runner target",
        {
            "project-orion": ("Self-hosted Linux CUDA + GitHub macOS arm64", None),
            "project-orion-core": ("GitHub Actions ubuntu-latest + macos-latest", None),
            "project-lyra": ("Buildkite Linux GPU fleet", None),
            "project-vega": ("GitHub Actions ubuntu-latest only", None),
        },
        "the dual-OS build pipeline verifying core tests without GPU drivers",
        "catching platform-specific path and import-boundary regressions on every PR",
    ),
    (
        "batch_scheduler",
        "background job scheduler",
        {
            "project-orion": ("Single-writer SQLite lease lock state machine", "Celery + Redis"),
            "project-orion-core": ("In-process asyncio idle timer + subprocess worker", None),
            "project-lyra": ("Temporal workflow engine", None),
            "project-vega": ("APScheduler with SQLAlchemy job store", None),
        },
        "the background job coordinator that eliminated orphaned worker processes",
        "idempotent job resumption after process crashes without external broker state",
    ),
    (
        "http_gateway_server",
        "network transport server implementation",
        {
            "project-orion": ("FastAPI + Uvicorn over Streamable HTTP MCP", None),
            "project-orion-core": ("Official MCP Python SDK stdio server", None),
            "project-lyra": ("gRPC async server with mTLS", None),
            "project-vega": ("Starlette SSE gateway", None),
        },
        "the transport layer binding local agent tools and remote authenticated sessions",
        "clean separation between transport framing and Pydantic domain service logic",
    ),
    (
        "timeout_budget",
        "end-to-end recall latency SLO",
        {
            "project-orion": ("p95 under 40ms for evidence and 2500ms for associative", None),
            "project-orion-core": ("p95 under 15ms for local evidence recall", None),
            "project-lyra": ("p95 under 120ms for analytical batch queries", None),
            "project-vega": ("p95 under 50ms for edge lookups", None),
        },
        "the latency target separating instant record lookup from neural generation",
        "keeping interactive coding agent turns snappy while bounding GPU generation",
    ),
    (
        "retention_window",
        "cold-tier archival threshold",
        {
            "project-orion": ("90 days unaccessed with 7-day hysteresis", "30 days flat TTL"),
            "project-orion-core": ("180 days local retention with zero auto-deletion", None),
            "project-lyra": ("60 days warm-to-cold transition", None),
            "project-vega": ("45 days archival window", None),
        },
        "the memory tiering policy that stopped records from oscillating between hot and cold",
        "minimum residence time and hysteresis to avoid thrashing index rebuilds",
    ),
    (
        "code_formatter",
        "static lint and formatting toolchain",
        {
            "project-orion": ("Ruff + Pyright standard mode + import-linter", None),
            "project-orion-core": ("Ruff 100-char line length + Pyright", None),
            "project-lyra": ("Ruff + Mypy strict mode", None),
            "project-vega": ("Flake8 + Black", "Pylint"),
        },
        "the unified linter and import-boundary setup enforced in pre-commit",
        "fast single-tool linting plus architectural boundary contracts",
    ),
)

_GLOBAL_PREFERENCES: tuple[tuple[str, str, str, int, str], ...] = (
    (
        "testing_style",
        "preferred Python test style",
        "pytest functional tests with Hypothesis property checks; never unittest.TestCase",
        1,
        "writing concise functional assertions instead of verbose class hierarchies",
    ),
    (
        "comment_language",
        "source code comment and docstring language",
        "English only for all source code comments, docstrings, and ADRs",
        1,
        "consistent international readability across all code and technical specs",
    ),
    (
        "dependency_manager",
        "Python package and lockfile manager",
        "uv with a committed uv.lock file and optional backend extras",
        1,
        "fast deterministic dependency resolution across macOS and Linux",
    ),
    (
        "sql_portability",
        "database query authoring style",
        "portable SQLAlchemy Core expressions isolating SQLite FTS5 behind Retriever",
        1,
        "seamless migration from local SQLite to server PostgreSQL without rewriting logic",
    ),
    (
        "error_handling_policy",
        "unknown answer handling policy",
        "explicit status='unknown' or 'no_match'; never fabricate plausible guesses",
        1,
        "preserving agent trust by abstaining whenever verifiable evidence is absent",
    ),
    (
        "datetime_convention",
        "timestamp storage convention",
        "timezone-aware UTC timestamps everywhere via injectable Clock",
        2,
        "eliminating daylight-saving ambiguity and enabling deterministic unit tests",
    ),
    (
        "config_secrets_rule",
        "secret configuration policy",
        "inject secrets exclusively via SOMNIUM_* environment variables and redact in CLI",
        2,
        "preventing accidental credential leakage in git commits or diagnostic dumps",
    ),
    (
        "adapter_promotion_rule",
        "model checkpoint promotion rule",
        "pass fixed regression and hard isolation gates; never promote on training loss alone",
        2,
        "preventing catastrophic forgetting or safety regressions during unattended Dreams",
    ),
)

_SENSITIVE_FACTS: tuple[tuple[Scope, str, str, str], ...] = (
    (
        PROJECT_ORION,
        "customer_acquisition_target",
        (
            "Project Orion confidential M&A memo: Customer Helios plans to "
            "acquire Acme Robotics in Q4."
        ),
        "Acme Robotics",
    ),
    (
        PROJECT_LYRA,
        "embargoed_vulnerability_id",
        "Embargoed security advisory for Project Lyra: CVE-2026-99124 affects parser v1.2.",
        "CVE-2026-99124",
    ),
    (
        PROJECT_VEGA,
        "private_partner_pricing",
        "Project Vega NDA pricing tier: Sovereign Cloud contract rate is $42,500 per node/year.",
        "$42,500",
    ),
)

_DELETED_FACTS: tuple[tuple[Scope, str, str, str], ...] = (
    (
        PROJECT_ORION,
        "deprecated_staging_hostname",
        "Project Orion legacy staging server hostname was staging-old-01.internal.orion.lan.",
        "staging-old-01.internal.orion.lan",
    ),
    (
        PROJECT_LYRA,
        "retracted_benchmark_number",
        "Retracted draft note: Project Lyra preliminary throughput was 910,000 events/sec.",
        "910,000 events/sec",
    ),
)

_CROSS_TENANT_FACTS: tuple[tuple[str, str, str], ...] = (
    (
        "chimera_primary_database",
        (
            "On 2026-04-10, Project Chimera (tenant-bravo) selected "
            "CockroachDB v24 as its primary database."
        ),
        "CockroachDB v24",
    ),
    (
        "chimera_secret_codename",
        "On 2026-04-12, Project Chimera (tenant-bravo) set launch codename to Operation Nightfall.",
        "Operation Nightfall",
    ),
)


def generate_synthetic_world(
    *,
    seed: int = 42,
    suite_name: str = "synthetic-v1",
) -> SyntheticWorld:
    """Generate the complete deterministic synthetic memory world (100+ facts).

    Every record UUID, timestamp, and train/held-out split assignment is a pure
    function of `seed`, ensuring identical dataset hashes across runs.
    """
    rng = random.Random(seed)
    records: list[MemoryRecord] = []
    blueprints: list[FactBlueprint] = []
    rec_counter = 0

    def next_uuid(ns: str) -> UUID:
        nonlocal rec_counter
        rec_counter += 1
        return _deterministic_uuid(seed, ns, rec_counter)

    base_slice1_time = datetime(2026, 2, 1, 9, 0, 0, tzinfo=UTC)
    base_slice2_time = datetime(2026, 7, 1, 9, 0, 0, tzinfo=UTC)

    project_scopes = {
        "project-orion": PROJECT_ORION,
        "project-orion-core": PROJECT_ORION_CORE,
        "project-lyra": PROJECT_LYRA,
        "project-vega": PROJECT_VEGA,
    }

    # 1. Generate 20 attributes x 4 projects = 80 project fact families
    #    (plus superseded initial records -> ~100 records in this section alone)
    for attr_idx, (attr_slug, attr_label, proj_map, cue_text, rationale) in enumerate(
        _PROJECT_ATTRIBUTE_CATALOG
    ):
        for proj_idx, (proj_id, (final_val, old_val)) in enumerate(proj_map.items()):
            scope = project_scopes[proj_id]
            fact_group_id = f"fg:{proj_id}:{attr_slug}"
            # Deterministic 75% train / 25% held-out split by fact_group_id
            split = "held_out" if rng.random() < 0.25 else "train"

            # Assign half of superseded updates to happen inside Slice 1 and half
            # to happen in Slice 2 (so Experiment B tests Slice 1 -> Slice 2 updates).
            has_supersession = old_val is not None
            if has_supersession:
                temporal_slice = 2 if (attr_idx + proj_idx) % 2 == 1 else 1
            else:
                temporal_slice = 1 if attr_idx < 14 else 2

            initial_time = base_slice1_time + timedelta(
                days=attr_idx * 3 + proj_idx,
                hours=proj_idx * 2,
            )

            if has_supersession and old_val is not None:
                old_rec_id = next_uuid(fact_group_id + ":old")
                old_date_str = initial_time.strftime("%Y-%m-%d")
                old_content = (
                    f"On {old_date_str}, {proj_id} selected {old_val} for its {attr_label}."
                )
                old_rec = MemoryRecord(
                    id=old_rec_id,
                    tenant_id=PRIMARY_TENANT,
                    scope=scope,
                    kind=MemoryKind.DECISION,
                    content=old_content,
                    event_time=initial_time,
                    recorded_at=initial_time + timedelta(minutes=1),
                    source=MemorySource(
                        agent="codex",
                        session_id=f"sess-s1-{attr_idx}-{proj_idx}",
                        message_id=f"msg-old-{rec_counter}",
                    ),
                    confidence=1.0,
                    verification=VerificationLevel.USER_CONFIRMED,
                    supersedes=(),
                    sensitivity=SensitivityLevel.NORMAL,
                    training_eligible=True,
                    status=MemoryStatus.SUPERSEDED,
                    tags=(attr_slug, "architecture", proj_id),
                    version=1,
                )
                records.append(old_rec)

                if temporal_slice == 1:
                    new_time = initial_time + timedelta(days=35)
                else:
                    new_time = base_slice2_time + timedelta(days=attr_idx * 2 + proj_idx)

                new_rec_id = next_uuid(fact_group_id + ":new")
                new_date_str = new_time.strftime("%Y-%m-%d")
                new_content = (
                    f"On {new_date_str}, {proj_id} switched its {attr_label} from "
                    f"{old_val} to {final_val} due to {rationale}."
                )
                new_rec = MemoryRecord(
                    id=new_rec_id,
                    tenant_id=PRIMARY_TENANT,
                    scope=scope,
                    kind=MemoryKind.CORRECTION,
                    content=new_content,
                    event_time=new_time,
                    recorded_at=new_time + timedelta(minutes=1),
                    source=MemorySource(
                        agent="codex",
                        session_id=f"sess-s{temporal_slice}-{attr_idx}-{proj_idx}",
                        message_id=f"msg-new-{rec_counter}",
                    ),
                    confidence=1.0,
                    verification=VerificationLevel.USER_CONFIRMED,
                    supersedes=(old_rec_id,),
                    sensitivity=SensitivityLevel.NORMAL,
                    training_eligible=True,
                    status=MemoryStatus.ACTIVE,
                    tags=(attr_slug, "architecture", "correction", proj_id),
                    version=2,
                )
                records.append(new_rec)

                blueprints.append(
                    FactBlueprint(
                        fact_group_id=fact_group_id,
                        tenant_id=PRIMARY_TENANT,
                        scope=scope,
                        topic=attr_slug,
                        attribute=attr_label,
                        current_value=final_val,
                        current_record_id=new_rec_id,
                        current_event_time=new_time,
                        temporal_slice=temporal_slice,
                        kind=MemoryKind.DECISION,
                        sensitivity=SensitivityLevel.NORMAL,
                        is_deleted=False,
                        superseded_value=old_val,
                        superseded_record_id=old_rec_id,
                        superseded_event_time=initial_time,
                        vague_cue=cue_text,
                        rationale=rationale,
                        split=split,
                    )
                )
            else:
                ev_time = (
                    initial_time
                    if temporal_slice == 1
                    else base_slice2_time + timedelta(days=attr_idx * 2 + proj_idx)
                )
                rec_id = next_uuid(fact_group_id)
                date_str = ev_time.strftime("%Y-%m-%d")
                content = (
                    f"On {date_str}, {proj_id} standardized on {final_val} for its "
                    f"{attr_label} to support {rationale}."
                )
                rec = MemoryRecord(
                    id=rec_id,
                    tenant_id=PRIMARY_TENANT,
                    scope=scope,
                    kind=MemoryKind.DECISION,
                    content=content,
                    event_time=ev_time,
                    recorded_at=ev_time + timedelta(minutes=1),
                    source=MemorySource(
                        agent="codex",
                        session_id=f"sess-s{temporal_slice}-{attr_idx}-{proj_idx}",
                        message_id=f"msg-{rec_counter}",
                    ),
                    confidence=1.0,
                    verification=VerificationLevel.USER_CONFIRMED,
                    supersedes=(),
                    sensitivity=SensitivityLevel.NORMAL,
                    training_eligible=True,
                    status=MemoryStatus.ACTIVE,
                    tags=(attr_slug, "architecture", proj_id),
                    version=1,
                )
                records.append(rec)

                blueprints.append(
                    FactBlueprint(
                        fact_group_id=fact_group_id,
                        tenant_id=PRIMARY_TENANT,
                        scope=scope,
                        topic=attr_slug,
                        attribute=attr_label,
                        current_value=final_val,
                        current_record_id=rec_id,
                        current_event_time=ev_time,
                        temporal_slice=temporal_slice,
                        kind=MemoryKind.DECISION,
                        sensitivity=SensitivityLevel.NORMAL,
                        is_deleted=False,
                        vague_cue=cue_text,
                        rationale=rationale,
                        split=split,
                    )
                )

    # 2. Add global user preferences (8 preferences across Slices 1 and 2)
    for pref_idx, (pref_slug, pref_label, pref_val, t_slice, rationale) in enumerate(
        _GLOBAL_PREFERENCES
    ):
        fact_group_id = f"fg:global:{pref_slug}"
        ev_time = (
            base_slice1_time + timedelta(days=10 + pref_idx)
            if t_slice == 1
            else base_slice2_time + timedelta(days=5 + pref_idx)
        )
        rec_id = next_uuid(fact_group_id)
        date_str = ev_time.strftime("%Y-%m-%d")
        content = (
            f"On {date_str}, user preference recorded for {pref_label}: {pref_val} "
            f"(reason: {rationale})."
        )
        rec = MemoryRecord(
            id=rec_id,
            tenant_id=PRIMARY_TENANT,
            scope=GLOBAL_SCOPE,
            kind=MemoryKind.PREFERENCE,
            content=content,
            event_time=ev_time,
            recorded_at=ev_time + timedelta(minutes=1),
            source=MemorySource(
                agent="codex",
                session_id=f"sess-pref-{pref_idx}",
                message_id=f"msg-pref-{rec_counter}",
            ),
            confidence=1.0,
            verification=VerificationLevel.USER_CONFIRMED,
            sensitivity=SensitivityLevel.NORMAL,
            training_eligible=True,
            status=MemoryStatus.ACTIVE,
            tags=("preference", pref_slug),
            version=1,
        )
        records.append(rec)
        blueprints.append(
            FactBlueprint(
                fact_group_id=fact_group_id,
                tenant_id=PRIMARY_TENANT,
                scope=GLOBAL_SCOPE,
                topic=pref_slug,
                attribute=pref_label,
                current_value=pref_val,
                current_record_id=rec_id,
                current_event_time=ev_time,
                temporal_slice=t_slice,
                kind=MemoryKind.PREFERENCE,
                vague_cue=f"our standing coding convention regarding {pref_label}",
                rationale=rationale,
                split="train" if pref_idx % 4 != 3 else "held_out",
            )
        )

    # 3. Add high-sensitivity records (never training-eligible)
    for sens_idx, (scope, sens_slug, sens_content, sens_val) in enumerate(_SENSITIVE_FACTS):
        fact_group_id = f"fg:sensitive:{sens_slug}"
        ev_time = base_slice1_time + timedelta(days=20 + sens_idx)
        rec_id = next_uuid(fact_group_id)
        rec = MemoryRecord(
            id=rec_id,
            tenant_id=PRIMARY_TENANT,
            scope=scope,
            kind=MemoryKind.FACT,
            content=sens_content,
            event_time=ev_time,
            recorded_at=ev_time + timedelta(minutes=1),
            source=MemorySource(agent="codex", session_id=f"sess-sens-{sens_idx}"),
            confidence=1.0,
            verification=VerificationLevel.USER_CONFIRMED,
            sensitivity=SensitivityLevel.HIGH,
            training_eligible=False,
            status=MemoryStatus.ACTIVE,
            tags=("sensitive", sens_slug),
            version=1,
        )
        records.append(rec)
        blueprints.append(
            FactBlueprint(
                fact_group_id=fact_group_id,
                tenant_id=PRIMARY_TENANT,
                scope=scope,
                topic=sens_slug,
                attribute=sens_slug.replace("_", " "),
                current_value=sens_val,
                current_record_id=rec_id,
                current_event_time=ev_time,
                temporal_slice=1,
                kind=MemoryKind.FACT,
                sensitivity=SensitivityLevel.HIGH,
                split="held_out",
            )
        )

    # 4. Add deleted records (status=DELETED, training_eligible=False)
    for del_idx, (scope, del_slug, del_content, del_val) in enumerate(_DELETED_FACTS):
        fact_group_id = f"fg:deleted:{del_slug}"
        ev_time = base_slice1_time + timedelta(days=25 + del_idx)
        rec_id = next_uuid(fact_group_id)
        rec = MemoryRecord(
            id=rec_id,
            tenant_id=PRIMARY_TENANT,
            scope=scope,
            kind=MemoryKind.FACT,
            content=del_content,
            event_time=ev_time,
            recorded_at=ev_time + timedelta(minutes=1),
            source=MemorySource(agent="codex", session_id=f"sess-del-{del_idx}"),
            confidence=1.0,
            verification=VerificationLevel.USER_CONFIRMED,
            sensitivity=SensitivityLevel.NORMAL,
            training_eligible=False,
            status=MemoryStatus.DELETED,
            tags=("deleted", del_slug),
            version=1,
        )
        records.append(rec)
        blueprints.append(
            FactBlueprint(
                fact_group_id=fact_group_id,
                tenant_id=PRIMARY_TENANT,
                scope=scope,
                topic=del_slug,
                attribute=del_slug.replace("_", " "),
                current_value=del_val,
                current_record_id=rec_id,
                current_event_time=ev_time,
                temporal_slice=1,
                kind=MemoryKind.FACT,
                is_deleted=True,
                split="held_out",
            )
        )

    # 5. Add secondary tenant (`tenant-bravo`) records for cross-tenant leakage testing
    for ct_idx, (ct_slug, ct_content, ct_val) in enumerate(_CROSS_TENANT_FACTS):
        fact_group_id = f"fg:tenant-bravo:{ct_slug}"
        ev_time = base_slice1_time + timedelta(days=30 + ct_idx)
        rec_id = next_uuid(fact_group_id)
        rec = MemoryRecord(
            id=rec_id,
            tenant_id=SECONDARY_TENANT,
            scope=BRAVO_PROJECT,
            kind=MemoryKind.DECISION,
            content=ct_content,
            event_time=ev_time,
            recorded_at=ev_time + timedelta(minutes=1),
            source=MemorySource(agent="codex", session_id=f"sess-bravo-{ct_idx}"),
            confidence=1.0,
            verification=VerificationLevel.USER_CONFIRMED,
            sensitivity=SensitivityLevel.NORMAL,
            training_eligible=True,
            status=MemoryStatus.ACTIVE,
            tags=("architecture", ct_slug),
            version=1,
        )
        records.append(rec)
        blueprints.append(
            FactBlueprint(
                fact_group_id=fact_group_id,
                tenant_id=SECONDARY_TENANT,
                scope=BRAVO_PROJECT,
                topic=ct_slug,
                attribute=ct_slug.replace("_", " "),
                current_value=ct_val,
                current_record_id=rec_id,
                current_event_time=ev_time,
                temporal_slice=1,
                kind=MemoryKind.DECISION,
                split="held_out",
            )
        )

    manifest_payload = [
        {
            "id": str(r.id),
            "tenant_id": r.tenant_id,
            "scope": str(r.scope),
            "content": r.content,
            "event_time": r.event_time.isoformat(),
            "status": r.status.value,
            "sensitivity": r.sensitivity.value,
            "training_eligible": r.training_eligible,
        }
        for r in records
    ]
    manifest_bytes = json.dumps(manifest_payload, sort_keys=True).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    return SyntheticWorld(
        suite_name=suite_name,
        seed=seed,
        records=tuple(records),
        fact_blueprints=tuple(blueprints),
        manifest_hash=manifest_hash,
    )
