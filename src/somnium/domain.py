"""Core domain types and versioned API contracts for Somnium.

This module defines the fundamental models shared across Freud, Hippocore,
Engram, and the Dream Engine:
- Tenancy and authorization (`TenantId`, `Scope`, `Permission`, `AuthContext`)
- Episodic memory records (`MemoryRecord`, `MemoryKind`, `SensitivityLevel`, etc.)
- Recall modes and associative claims (`RecallMode`, `Claim`, `ClaimSupport`)
- Versioned API request and response payloads (`API_VERSION = "v1"`)
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from somnium import API_VERSION

# A validated non-empty identifier for a tenant (e.g., "local-user").
TenantId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9._-]+$",
    ),
]


class ScopeKind(StrEnum):
    """Granularity of a memory scope."""

    GLOBAL = "global"
    PROJECT = "project"
    SESSION = "session"
    AGENT = "agent"


class Scope(BaseModel):
    """Logical scope boundary for episodic memory records and queries."""

    model_config = ConfigDict(frozen=True)

    kind: ScopeKind = ScopeKind.PROJECT
    id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.id}"

    def __hash__(self) -> int:
        return hash((self.kind, self.id))


class Permission(StrEnum):
    """Fine-grained operation permissions carried by an AuthContext."""

    READ = "read"
    WRITE = "write"
    CORRECT = "correct"
    FORGET = "forget"
    ASSOCIATIVE_RECALL = "associative_recall"
    ADMIN = "admin"


class AuthContext(BaseModel):
    """Resolved authentication and authorization context for every request.

    Queries to Hippocore and calls to Engram require an explicit AuthContext so
    that tenant and scope isolation are enforced by construction.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: TenantId
    principal_id: str = Field(min_length=1)
    authorized_scopes: tuple[Scope, ...] = Field(default_factory=tuple)
    permissions: frozenset[Permission] = Field(
        default_factory=lambda: frozenset(
            {
                Permission.READ,
                Permission.WRITE,
                Permission.CORRECT,
                Permission.FORGET,
                Permission.ASSOCIATIVE_RECALL,
            }
        )
    )
    allow_all_scopes: bool = False

    def is_scope_authorized(self, scope: Scope) -> bool:
        """Return True if the caller is permitted to access the given scope."""
        if self.allow_all_scopes or Permission.ADMIN in self.permissions:
            return True
        return scope in self.authorized_scopes

    def has_permission(self, permission: Permission) -> bool:
        """Return True if the caller holds the given permission (or ADMIN)."""
        return Permission.ADMIN in self.permissions or permission in self.permissions

    @property
    def can_associative_recall(self) -> bool:
        """Return True if the caller may invoke Engram associative recall."""
        return self.has_permission(Permission.ASSOCIATIVE_RECALL)


class MemoryKind(StrEnum):
    """Category of an episodic memory record."""

    EVENT = "event"
    FACT = "fact"
    DECISION = "decision"
    PREFERENCE = "preference"
    CORRECTION = "correction"


class VerificationLevel(StrEnum):
    """Verification level of a stored memory record."""

    USER_CONFIRMED = "user_confirmed"
    AGENT_ASSERTED = "agent_asserted"
    UNVERIFIED = "unverified"


class SensitivityLevel(StrEnum):
    """Sensitivity classification of a memory record.

    Records marked HIGH must never enter model training datasets.
    """

    NORMAL = "normal"
    HIGH = "high"


class MemoryStatus(StrEnum):
    """Lifecycle status of a memory record."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class DeletionMode(StrEnum):
    """Mode for a memory_forget operation."""

    SOFT = "soft"
    HARD = "hard"


class MemorySource(BaseModel):
    """Provenance metadata identifying where a memory originated."""

    model_config = ConfigDict(frozen=True)

    agent: str = "codex"
    session_id: str | None = None
    message_id: str | None = None


class MemoryRecord(BaseModel):
    """Canonical episodic memory record stored in Hippocore (design §5.2).

    Mutable usage statistics (access_count, last_accessed_at) live in a separate
    access log table rather than on the immutable event record.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    tenant_id: TenantId = "local-user"
    scope: Scope
    kind: MemoryKind = MemoryKind.FACT
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    event_time: datetime
    recorded_at: datetime
    source: MemorySource = Field(default_factory=MemorySource)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    verification: VerificationLevel = VerificationLevel.AGENT_ASSERTED
    supersedes: tuple[UUID, ...] = Field(default_factory=tuple)
    sensitivity: SensitivityLevel = SensitivityLevel.NORMAL
    training_eligible: bool = True
    retention_policy: str = "default"
    status: MemoryStatus = MemoryStatus.ACTIVE
    tags: tuple[str, ...] = Field(default_factory=tuple)
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _enforce_training_eligibility(self) -> MemoryRecord:
        """Ensure high-sensitivity or deleted records are never training-eligible."""
        if (
            self.sensitivity == SensitivityLevel.HIGH
            or self.status == MemoryStatus.DELETED
            or self.verification == VerificationLevel.UNVERIFIED
        ) and self.training_eligible:
            object.__setattr__(self, "training_eligible", False)
        return self


class RecallMode(StrEnum):
    """Recall mode requested by the caller or selected by the router (design §6.1)."""

    AUTO = "auto"
    EVIDENCE = "evidence"
    ASSOCIATIVE = "associative"


class ResolvedRecallMode(StrEnum):
    """Concrete recall mode executed after routing."""

    EVIDENCE = "evidence"
    ASSOCIATIVE = "associative"


class ClaimSupport(StrEnum):
    """Verification status of an individual claim in an associative recall answer."""

    SUPPORTED = "supported"
    UNVERIFIED = "unverified"


class UncertaintyLevel(StrEnum):
    """Calibrated uncertainty level for an associative recall response."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Claim(BaseModel):
    """A single statement inside an associative recall answer with provenance."""

    model_config = ConfigDict(frozen=True)

    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    support: ClaimSupport
    evidence: tuple[UUID, ...] = Field(default_factory=tuple)
    scope: Scope | None = None

    @model_validator(mode="after")
    def _validate_support_evidence(self) -> Claim:
        """Require at least one evidence ID whenever a claim is marked SUPPORTED."""
        if self.support == ClaimSupport.SUPPORTED and not self.evidence:
            raise ValueError("A 'supported' claim must reference at least one evidence memory_id.")
        return self


class RoutingInfo(BaseModel):
    """Audit metadata describing how a recall request was routed."""

    model_config = ConfigDict(frozen=True)

    requested: RecallMode
    resolved: ResolvedRecallMode
    reason: str
    fallback_used: bool = False


class EvidenceRecordItem(BaseModel):
    """A single structured record returned by evidence recall."""

    model_config = ConfigDict(frozen=True)

    memory_id: UUID
    scope: Scope
    kind: MemoryKind
    content: str
    event_time: datetime
    status: MemoryStatus = MemoryStatus.ACTIVE
    relevance: float = Field(ge=0.0, le=1.0)
    supersedes: tuple[UUID, ...] = Field(default_factory=tuple)
    superseded_by: UUID | None = None


# ---------------------------------------------------------------------------
# Versioned API Request & Response Contracts (design §9, Milestones §3)
# ---------------------------------------------------------------------------


class MemoryAskRequest(BaseModel):
    """Request payload for the `memory_ask` operation."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    scope: Scope | None = None
    mode: RecallMode = RecallMode.AUTO
    cross_scope: bool | None = None
    require_synthesis: bool = False
    limit: int = Field(default=10, ge=1, le=100)


class EvidenceRecallResponse(BaseModel):
    """Response payload when `memory_ask` executes in `evidence` mode."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    status: Literal["ok", "no_match"]
    recall_mode: Literal[ResolvedRecallMode.EVIDENCE] = ResolvedRecallMode.EVIDENCE
    routing: RoutingInfo
    records: tuple[EvidenceRecordItem, ...] = Field(default_factory=tuple)


class AssociativeRecallResponse(BaseModel):
    """Response payload when `memory_ask` executes in `associative` mode."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    status: Literal["ok", "unknown"]
    recall_mode: Literal[ResolvedRecallMode.ASSOCIATIVE] = ResolvedRecallMode.ASSOCIATIVE
    routing: RoutingInfo
    answer: str | None = None
    claims: tuple[Claim, ...] = Field(default_factory=tuple)
    uncertainty: UncertaintyLevel = UncertaintyLevel.MEDIUM
    model_version: str


MemoryAskResponse = EvidenceRecallResponse | AssociativeRecallResponse


class MemoryStoreRequest(BaseModel):
    """Request payload for the `memory_store` operation."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    scope: Scope
    kind: MemoryKind = MemoryKind.FACT
    event_time: datetime | None = None
    source: MemorySource = Field(default_factory=MemorySource)
    sensitivity: SensitivityLevel = SensitivityLevel.NORMAL
    verification: VerificationLevel = VerificationLevel.AGENT_ASSERTED
    tags: tuple[str, ...] = Field(default_factory=tuple)
    idempotency_key: str | None = None


class MemoryStoreResponse(BaseModel):
    """Response payload for the `memory_store` operation."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    memory_id: UUID
    recorded_at: datetime
    training_eligible: bool
    deduplicated: bool = False


class MemoryCorrectRequest(BaseModel):
    """Request payload for the `memory_correct` operation."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    memory_id: UUID
    replacement: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    reason: str | None = None
    idempotency_key: str | None = None


class MemoryCorrectResponse(BaseModel):
    """Response payload for the `memory_correct` operation."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    new_memory_id: UUID
    superseded_memory_id: UUID
    version: int


class MemoryForgetRequest(BaseModel):
    """Request payload for the `memory_forget` operation."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    memory_id: UUID
    deletion_mode: DeletionMode = DeletionMode.HARD
    reason: str | None = None
    idempotency_key: str | None = None


class MemoryForgetResponse(BaseModel):
    """Response payload for the `memory_forget` operation."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    memory_id: UUID
    status: Literal["deleted"] = "deleted"
    weights_marked_dirty: bool = False


class DreamLifecycleState(StrEnum):
    """State of the Dream Engine lifecycle state machine (design §7)."""

    AWAKE = "awake"
    IDLE_ELIGIBLE = "idle_eligible"
    DREAM_PREPARING = "dream_preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    PROMOTING = "promoting"
    REJECTED = "rejected"


class MemoryStatusRequest(BaseModel):
    """Request payload for the `memory_status` operation."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION


class MemoryStatusResponse(BaseModel):
    """Response payload for the `memory_status` operation."""

    model_config = ConfigDict(frozen=True)

    api_version: str = API_VERSION
    tenant_id: TenantId
    dream_state: DreamLifecycleState = DreamLifecycleState.AWAKE
    active_model_version: str
    pending_training_examples: int = 0
    weights_dirty: bool = False
