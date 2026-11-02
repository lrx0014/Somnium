"""Rule-based RecallRouter and routing decision audit logger (M4.6).

Implements `design.md` §6.4:
- Evaluates deterministic routing features (`has_exact_lexical_cue`,
  `has_associative_cue_pattern`, `require_synthesis`, `top_evidence_relevance`,
  `evidence_count`) against thresholds from `RoutingConfig`.
- Gracefully falls back to `ResolvedRecallMode.EVIDENCE` with `fallback_used=True`
  whenever associative recall is unavailable or the caller lacks
  `Permission.ASSOCIATIVE_RECALL`.
- Persists every routing decision and its feature vector to the
  `routing_decisions` table in `HippocoreStore` so learned routing policies can
  be trained and audited later.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import desc, insert, select

from somnium.backends.fake import SystemClock
from somnium.config import RoutingConfig
from somnium.domain import (
    AuthContext,
    EvidenceRecordItem,
    MemoryAskRequest,
    RecallMode,
    ResolvedRecallMode,
    RoutingInfo,
    TenantId,
)
from somnium.hippocore import HippocoreStore, routing_decisions
from somnium.interfaces import Clock

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_ISO_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
_FILE_PATH_RE = re.compile(
    r"(?:\b[a-zA-Z0-9_.-]+/[a-zA-Z0-9_./-]+\b|\b[a-zA-Z0-9_-]+\.(?:py|md|toml|yaml|yml|json|sql|sh|ts|js|rs|go)\b)"
)


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize a datetime loaded from SQLite/Postgres to timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class RoutingDecisionRecord:
    """Persisted audit record for a single `RecallRouter` decision (M4.6)."""

    id: UUID
    tenant_id: TenantId
    principal_id: str
    question: str
    requested_scope: str | None
    requested_mode: RecallMode
    resolved_mode: ResolvedRecallMode
    reason: str
    fallback_used: bool
    features: dict[str, Any]
    decided_at: datetime


class RuleBasedRecallRouter:
    """Deterministic rule-based `RecallRouter` with database audit logging (M4.6)."""

    def __init__(
        self,
        config: RoutingConfig | None = None,
        *,
        store: HippocoreStore | None = None,
        clock: Clock | None = None,
        associative_available: bool | Callable[[], bool] = True,
    ) -> None:
        self._config = config or RoutingConfig()
        self._store = store
        self._clock: Clock = clock or SystemClock()
        self._associative_available = associative_available

    @property
    def config(self) -> RoutingConfig:
        """Active `RoutingConfig` thresholds and cue patterns."""
        return self._config

    def is_associative_available(self) -> bool:
        """Return True if the Engram associative recall pipeline is currently available."""
        if callable(self._associative_available):
            return bool(self._associative_available())
        return bool(self._associative_available)

    def extract_features(
        self,
        request: MemoryAskRequest,
        evidence_items: Sequence[EvidenceRecordItem],
    ) -> dict[str, Any]:
        """Compute deterministic routing features for `request` and `evidence_items`."""
        q_lower = request.question.lower()
        has_uuid = bool(_UUID_RE.search(request.question))
        has_iso_date = bool(_ISO_DATE_RE.search(request.question))
        has_file_path = bool(_FILE_PATH_RE.search(request.question))
        has_exact_lexical_cue = has_uuid or has_iso_date or has_file_path

        matched_cues = [
            pattern
            for pattern in self._config.associative_cue_patterns
            if pattern.lower() in q_lower
        ]
        has_associative_cue_pattern = len(matched_cues) > 0
        top_relevance = max((item.relevance for item in evidence_items), default=0.0)

        return {
            "has_uuid": has_uuid,
            "has_iso_date": has_iso_date,
            "has_file_path": has_file_path,
            "has_exact_lexical_cue": has_exact_lexical_cue,
            "has_associative_cue_pattern": has_associative_cue_pattern,
            "matched_cue_patterns": matched_cues,
            "require_synthesis": request.require_synthesis,
            "cross_scope": request.cross_scope,
            "evidence_count": len(evidence_items),
            "top_evidence_relevance": round(float(top_relevance), 4),
            "min_evidence_relevance_threshold": self._config.min_evidence_relevance,
        }

    def route(
        self,
        auth: AuthContext,
        request: MemoryAskRequest,
        evidence_items: Sequence[EvidenceRecordItem],
    ) -> RoutingInfo:
        """Select `evidence` or `associative` recall mode and persist the decision."""
        features = self.extract_features(request, evidence_items)
        assoc_avail = self.is_associative_available()
        assoc_permitted = auth.can_associative_recall

        if request.mode == RecallMode.EVIDENCE:
            info = RoutingInfo(
                requested=RecallMode.EVIDENCE,
                resolved=ResolvedRecallMode.EVIDENCE,
                reason="explicit_evidence_mode",
                fallback_used=False,
            )
        elif request.mode == RecallMode.ASSOCIATIVE:
            if not assoc_permitted:
                info = RoutingInfo(
                    requested=RecallMode.ASSOCIATIVE,
                    resolved=ResolvedRecallMode.EVIDENCE,
                    reason="associative_unauthorized_fallback_to_evidence",
                    fallback_used=True,
                )
            elif not assoc_avail:
                info = RoutingInfo(
                    requested=RecallMode.ASSOCIATIVE,
                    resolved=ResolvedRecallMode.EVIDENCE,
                    reason="associative_engine_unavailable_fallback_to_evidence",
                    fallback_used=True,
                )
            else:
                info = RoutingInfo(
                    requested=RecallMode.ASSOCIATIVE,
                    resolved=ResolvedRecallMode.ASSOCIATIVE,
                    reason="explicit_associative_mode",
                    fallback_used=False,
                )
        else:
            # RecallMode.AUTO
            wants_associative = False
            desired_reason = "high_confidence_evidence_match"

            if (
                features["has_exact_lexical_cue"]
                and not features["require_synthesis"]
                and not features["has_associative_cue_pattern"]
            ):
                wants_associative = False
                desired_reason = "exact_lexical_cue_in_query"
            elif features["has_associative_cue_pattern"] or features["require_synthesis"]:
                wants_associative = True
                desired_reason = (
                    "associative_cue_pattern_matched"
                    if features["has_associative_cue_pattern"]
                    else "require_synthesis_requested"
                )
            elif (
                features["evidence_count"] > 0
                and features["top_evidence_relevance"] >= self._config.min_evidence_relevance
            ):
                wants_associative = False
                desired_reason = "high_confidence_evidence_match"
            else:
                wants_associative = True
                desired_reason = "low_evidence_relevance_escalate_to_associative"

            if wants_associative:
                if not assoc_permitted:
                    info = RoutingInfo(
                        requested=RecallMode.AUTO,
                        resolved=ResolvedRecallMode.EVIDENCE,
                        reason=f"{desired_reason}_unauthorized_fallback_to_evidence",
                        fallback_used=True,
                    )
                elif not assoc_avail:
                    info = RoutingInfo(
                        requested=RecallMode.AUTO,
                        resolved=ResolvedRecallMode.EVIDENCE,
                        reason=f"{desired_reason}_unavailable_fallback_to_evidence",
                        fallback_used=True,
                    )
                else:
                    info = RoutingInfo(
                        requested=RecallMode.AUTO,
                        resolved=ResolvedRecallMode.ASSOCIATIVE,
                        reason=desired_reason,
                        fallback_used=False,
                    )
            else:
                info = RoutingInfo(
                    requested=RecallMode.AUTO,
                    resolved=ResolvedRecallMode.EVIDENCE,
                    reason=desired_reason,
                    fallback_used=False,
                )

        self._persist_decision(auth, request, info, features)
        return info

    def _persist_decision(
        self,
        auth: AuthContext,
        request: MemoryAskRequest,
        info: RoutingInfo,
        features: dict[str, Any],
    ) -> None:
        """Write a row into `routing_decisions` if a `HippocoreStore` is attached."""
        if self._store is None:
            return

        now = self._clock.now()
        scope_str = str(request.scope) if request.scope is not None else None
        with self._store.engine.begin() as conn:
            conn.execute(
                insert(routing_decisions).values(
                    decision_id=str(uuid4()),
                    tenant_id=auth.tenant_id,
                    principal_id=auth.principal_id,
                    question=request.question,
                    requested_scope=scope_str,
                    requested_mode=info.requested.value,
                    resolved_mode=info.resolved.value,
                    reason=info.reason,
                    fallback_used=info.fallback_used,
                    features_json=json.dumps(features, sort_keys=True),
                    decided_at=now.isoformat(),
                )
            )

    def list_routing_decisions(
        self,
        auth: AuthContext,
        *,
        limit: int = 50,
    ) -> list[RoutingDecisionRecord]:
        """Return recent routing decisions for `auth.tenant_id` ordered newest-first."""
        if self._store is None:
            return []

        stmt = (
            select(routing_decisions)
            .where(routing_decisions.c.tenant_id == auth.tenant_id)
            .order_by(desc(routing_decisions.c.decided_at))
            .limit(max(1, limit))
        )
        with self._store.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()

        records: list[RoutingDecisionRecord] = []
        for row in rows:
            raw_decided = row["decided_at"]
            parsed_decided = (
                raw_decided
                if isinstance(raw_decided, datetime)
                else datetime.fromisoformat(str(raw_decided))
            )
            records.append(
                RoutingDecisionRecord(
                    id=UUID(str(row["decision_id"])),
                    tenant_id=str(row["tenant_id"]),
                    principal_id=str(row["principal_id"]),
                    question=str(row["question"]),
                    requested_scope=(
                        str(row["requested_scope"]) if row["requested_scope"] is not None else None
                    ),
                    requested_mode=RecallMode(str(row["requested_mode"])),
                    resolved_mode=ResolvedRecallMode(str(row["resolved_mode"])),
                    reason=str(row["reason"]),
                    fallback_used=bool(row["fallback_used"]),
                    features=json.loads(str(row["features_json"])),
                    decided_at=_ensure_utc(parsed_decided),
                )
            )
        return records


__all__ = [
    "RoutingDecisionRecord",
    "RuleBasedRecallRouter",
]
