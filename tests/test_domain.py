"""Unit tests for core domain models and versioned API contracts (M0.5)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from somnium import API_VERSION
from somnium.domain import (
    AssociativeRecallResponse,
    AuthContext,
    Claim,
    ClaimSupport,
    EvidenceRecallResponse,
    EvidenceRecordItem,
    MemoryAskRequest,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    Permission,
    RecallMode,
    ResolvedRecallMode,
    RoutingInfo,
    Scope,
    ScopeKind,
    SensitivityLevel,
    VerificationLevel,
)


def test_scope_and_auth_context_enforcement() -> None:
    """Verify AuthContext enforces scope grants and fine-grained permissions."""
    orion = Scope(kind=ScopeKind.PROJECT, id="project-orion")
    lyra = Scope(kind=ScopeKind.PROJECT, id="project-lyra")

    auth = AuthContext(
        tenant_id="user-alice",
        principal_id="codex-session-1",
        authorized_scopes=(orion,),
        permissions=frozenset({Permission.READ, Permission.WRITE}),
    )

    assert str(orion) == "project:project-orion"
    assert auth.is_scope_authorized(orion) is True
    assert auth.is_scope_authorized(lyra) is False
    assert auth.has_permission(Permission.READ) is True
    assert auth.has_permission(Permission.FORGET) is False
    assert auth.can_associative_recall is False


def test_memory_record_high_sensitivity_disables_training_eligibility() -> None:
    """Verify high-sensitivity records are never marked training_eligible (design §5.2, §5.4)."""
    now = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
    scope = Scope(kind=ScopeKind.PROJECT, id="project-orion")

    normal_rec = MemoryRecord(
        tenant_id="local-user",
        scope=scope,
        kind=MemoryKind.DECISION,
        content="Selected PostgreSQL over MongoDB.",
        event_time=now,
        recorded_at=now,
        verification=VerificationLevel.USER_CONFIRMED,
        sensitivity=SensitivityLevel.NORMAL,
        training_eligible=True,
    )
    assert normal_rec.training_eligible is True

    sensitive_rec = MemoryRecord(
        tenant_id="local-user",
        scope=scope,
        kind=MemoryKind.FACT,
        content="Confidential acquisition target.",
        event_time=now,
        recorded_at=now,
        verification=VerificationLevel.USER_CONFIRMED,
        sensitivity=SensitivityLevel.HIGH,
        training_eligible=True,
    )
    assert sensitive_rec.training_eligible is False

    unverified_rec = MemoryRecord(
        tenant_id="local-user",
        scope=scope,
        kind=MemoryKind.FACT,
        content="Unverified rumor.",
        event_time=now,
        recorded_at=now,
        verification=VerificationLevel.UNVERIFIED,
        sensitivity=SensitivityLevel.NORMAL,
        training_eligible=True,
    )
    assert unverified_rec.training_eligible is False


def test_claim_requires_evidence_when_supported() -> None:
    """Verify a Claim marked SUPPORTED cannot be constructed without evidence IDs."""
    evidence_id = uuid4()
    valid_supported = Claim(
        text="Project Orion uses PostgreSQL.",
        support=ClaimSupport.SUPPORTED,
        evidence=(evidence_id,),
        scope=Scope(kind=ScopeKind.PROJECT, id="project-orion"),
    )
    assert valid_supported.evidence == (evidence_id,)

    valid_unverified = Claim(
        text="We discussed a lock-timeout workaround.",
        support=ClaimSupport.UNVERIFIED,
        evidence=(),
    )
    assert valid_unverified.support == ClaimSupport.UNVERIFIED

    with pytest.raises(ValidationError, match="supported"):
        Claim(
            text="Invented unsupported claim labeled as supported.",
            support=ClaimSupport.SUPPORTED,
            evidence=(),
        )


def test_versioned_api_request_and_response_models() -> None:
    """Verify request and response models carry api_version and serialize cleanly."""
    scope = Scope(kind=ScopeKind.PROJECT, id="project-orion")
    ask_req = MemoryAskRequest(question="What database did we pick?", scope=scope)
    assert ask_req.api_version == API_VERSION
    assert ask_req.mode == RecallMode.AUTO

    now = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)
    mem_id = uuid4()
    ev_resp = EvidenceRecallResponse(
        status="ok",
        routing=RoutingInfo(
            requested=RecallMode.AUTO,
            resolved=ResolvedRecallMode.EVIDENCE,
            reason="exact_query_high_relevance",
        ),
        records=(
            EvidenceRecordItem(
                memory_id=mem_id,
                scope=scope,
                kind=MemoryKind.DECISION,
                content="Selected PostgreSQL.",
                event_time=now,
                status=MemoryStatus.ACTIVE,
                relevance=0.95,
            ),
        ),
    )
    assert ev_resp.api_version == API_VERSION
    assert ev_resp.recall_mode == ResolvedRecallMode.EVIDENCE

    assoc_resp = AssociativeRecallResponse(
        status="ok",
        routing=RoutingInfo(
            requested=RecallMode.ASSOCIATIVE,
            resolved=ResolvedRecallMode.ASSOCIATIVE,
            reason="explicit_mode",
        ),
        answer="Project Orion selected PostgreSQL.",
        claims=(
            Claim(
                text="Project Orion selected PostgreSQL",
                support=ClaimSupport.SUPPORTED,
                evidence=(mem_id,),
                scope=scope,
            ),
        ),
        model_version="engram-local-user-v0001",
    )
    assert assoc_resp.api_version == API_VERSION
    assert assoc_resp.recall_mode == ResolvedRecallMode.ASSOCIATIVE
