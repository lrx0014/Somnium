"""Structured output validation, bounded repair, and post-generation safety guards (M4.4, M4.5).

Guarantees (enforced by deterministic code, never model trust):
1. Every `supported` claim must cite at least one existing `memory_id` in the
   caller's authorized active records; otherwise it is downgraded to `unverified`
   with `evidence=()`.
2. Every `unverified` claim has precise ISO dates, timestamps, UUIDs, and source
   citations deterministically stripped.
3. Conflict Rule (design §6.3): if a claim cites a superseded memory ID (or
   asserts a superseded record's content when its active replacement is known),
   the active Hippocore replacement wins and overrides the stale claim.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from somnium.backends.base import GenerationRequest, InferenceBackend
from somnium.domain import (
    AssociativeRecallResponse,
    Claim,
    ClaimSupport,
    MemoryRecord,
    RoutingInfo,
    Scope,
    TenantId,
    UncertaintyLevel,
)
from somnium.engram.prompts import render_repair_prompt

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_ISO_DATE_RE = re.compile(
    r"\b20\d{2}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?\b"
)
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_CITATION_TAG_RE = re.compile(
    r"\[(?:supported|source|evidence|id)\s*:[^\]]*\]|\((?:source|evidence|id)\s*:[^)]*\)",
    re.IGNORECASE,
)


class AssociateCandidateClaim(BaseModel):
    """Single candidate association produced by the `associate` pass (M4.3, M4.4)."""

    model_config = ConfigDict(extra="ignore")

    text: str
    cue_terms: tuple[str, ...] = Field(default_factory=tuple)
    cited_ids: tuple[str, ...] = Field(default_factory=tuple)
    scope: str | None = None


class AssociateStageOutput(BaseModel):
    """Validated structured output from the `associate` pass (M4.3, M4.4)."""

    model_config = ConfigDict(extra="ignore")

    template_version: str = "v1"
    cue_terms: tuple[str, ...] = Field(default_factory=tuple)
    candidate_claims: tuple[AssociateCandidateClaim, ...] = Field(default_factory=tuple)


class RawGeneratedClaim(BaseModel):
    """Raw claim emitted by the model before deterministic post-generation guards."""

    model_config = ConfigDict(extra="ignore")

    text: str
    support: ClaimSupport = ClaimSupport.UNVERIFIED
    evidence: tuple[str, ...] = Field(default_factory=tuple)
    scope: str | None = None


class AnswerStageOutput(BaseModel):
    """Validated structured output from the `answer` pass prior to safety guards (M4.4)."""

    model_config = ConfigDict(extra="ignore")

    status: Literal["ok", "unknown"] = "unknown"
    answer: str | None = None
    claims: tuple[RawGeneratedClaim, ...] = Field(default_factory=tuple)
    uncertainty: UncertaintyLevel = UncertaintyLevel.MEDIUM


@dataclass(slots=True)
class SchemaValidationStats:
    """Counters tracking structured output validation and repair rates (M4.4)."""

    total_calls: int = 0
    valid_first_try: int = 0
    repaired_on_retry: int = 0
    failed_after_retries: int = 0

    @property
    def first_try_valid_rate(self) -> float:
        """Fraction of calls that were schema-valid on the first generation attempt."""
        if self.total_calls == 0:
            return 1.0
        return self.valid_first_try / self.total_calls

    @property
    def overall_valid_rate(self) -> float:
        """Fraction of calls that produced schema-valid JSON (including bounded repair)."""
        if self.total_calls == 0:
            return 1.0
        return (self.valid_first_try + self.repaired_on_retry) / self.total_calls


def extract_json_object(raw_text: str) -> str:
    """Strip `<think>` blocks and markdown code fences to isolate a JSON object."""
    cleaned = _THINK_BLOCK_RE.sub("", raw_text).strip()
    fence_match = _CODE_FENCE_RE.match(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1]
    return cleaned


def parse_schema_payload[TModel: BaseModel](raw_text: str, schema_cls: type[TModel]) -> TModel:
    """Parse `raw_text` as JSON and validate against `schema_cls`."""
    json_str = extract_json_object(raw_text)
    parsed = json.loads(json_str)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected JSON object for {schema_cls.__name__}, got {type(parsed).__name__}."
        )
    return schema_cls.model_validate(parsed)


def generate_and_parse_with_repair[TModel: BaseModel](
    *,
    backend: InferenceBackend,
    tenant_id: TenantId,
    prompt: str,
    schema_cls: type[TModel],
    current_time: datetime,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_seconds: float = 30.0,
    max_retries: int = 2,
    stats: SchemaValidationStats | None = None,
) -> tuple[TModel | None, str]:
    """Generate structured output and apply bounded schema repair on validation failure (M4.4).

    Returns `(validated_model_or_none, model_version)`.
    """
    if stats is not None:
        stats.total_calls += 1

    gen = backend.generate(
        GenerationRequest(
            tenant_id=tenant_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
    )
    last_raw = gen.text
    model_version = gen.model_version

    try:
        validated = parse_schema_payload(last_raw, schema_cls)
        if stats is not None:
            stats.valid_first_try += 1
        return validated, model_version
    except (json.JSONDecodeError, ValidationError, ValueError) as first_err:
        last_error_str = str(first_err)

    for _attempt in range(max_retries):
        repair_prompt = render_repair_prompt(
            current_time=current_time,
            schema_name=schema_cls.__name__,
            validation_error=last_error_str,
            raw_output=last_raw,
        )
        repair_gen = backend.generate(
            GenerationRequest(
                tenant_id=tenant_id,
                prompt=repair_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                timeout_seconds=timeout_seconds,
            )
        )
        last_raw = repair_gen.text
        model_version = repair_gen.model_version
        try:
            validated = parse_schema_payload(last_raw, schema_cls)
            if stats is not None:
                stats.repaired_on_retry += 1
            return validated, model_version
        except (json.JSONDecodeError, ValidationError, ValueError) as retry_err:
            last_error_str = str(retry_err)

    if stats is not None:
        stats.failed_after_retries += 1
    return None, model_version


def strip_unverified_dates_and_sources(text: str) -> str:
    """Deterministically strip precise dates, UUIDs, and source tags from unverified text (M4.5)."""
    cleaned = _CITATION_TAG_RE.sub("", text)
    cleaned = _ISO_DATE_RE.sub("[date-omitted]", cleaned)
    cleaned = _UUID_RE.sub("[id-omitted]", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned or "[unverified association]"


def apply_post_generation_guards(
    raw_output: AnswerStageOutput,
    *,
    authorized_active_records: Mapping[UUID, MemoryRecord],
    superseded_to_active: Mapping[UUID, MemoryRecord] | None = None,
    superseded_records: Mapping[UUID, MemoryRecord] | None = None,
    requested_scope: Scope | None = None,
    cross_scope: bool = True,
    routing: RoutingInfo,
    model_version: str,
) -> AssociativeRecallResponse:
    """Enforce deterministic post-generation safety guards on model output (M4.5).

    Rules enforced in code:
    1. Filter `authorized_active_records` to `requested_scope` when `cross_scope=False`.
    2. Conflict Rule: any claim citing a superseded ID (or matching a superseded
       record whose active replacement is in `authorized_active_records`) is
       overridden by the active replacement record.
    3. Every `supported` claim must cite >= 1 UUID in `effective_active_records`;
       otherwise it is downgraded to `unverified` with `evidence=()`.
    4. Every `unverified` claim has precise ISO dates, UUIDs, and source citations
       stripped via `strip_unverified_dates_and_sources`.
    5. Authoritative `scope` on `supported` claims is taken from the Hippocore
       `MemoryRecord.scope`.
    """
    sup_map = dict(superseded_to_active or {})
    sup_recs = dict(superseded_records or {})

    if not cross_scope and requested_scope is not None:
        effective_active: dict[UUID, MemoryRecord] = {
            uid: rec
            for uid, rec in authorized_active_records.items()
            if rec.scope == requested_scope
        }
        sup_map = {
            old_uid: active_rec
            for old_uid, active_rec in sup_map.items()
            if active_rec.scope == requested_scope
        }
    else:
        effective_active = dict(authorized_active_records)

    if raw_output.status == "unknown" or (not raw_output.claims and not raw_output.answer):
        return AssociativeRecallResponse(
            status="unknown",
            routing=routing,
            answer=None,
            claims=(),
            uncertainty=UncertaintyLevel.HIGH,
            model_version=model_version,
        )

    guarded_claims: list[Claim] = []
    supported_dates: set[str] = set()
    conflict_overridden = False

    for raw_claim in raw_output.claims:
        claim_text = raw_claim.text.strip()
        if not claim_text:
            continue

        # Also extract any inline `[supported:<uuid>]` tags from `claim_text`
        candidate_id_strs = list(raw_claim.evidence) + _UUID_RE.findall(claim_text)
        valid_active_ids: list[UUID] = []
        replacement_rec: MemoryRecord | None = None

        for id_str in candidate_id_strs:
            try:
                parsed_uid = UUID(id_str.strip())
            except ValueError:
                continue

            if parsed_uid in effective_active:
                if parsed_uid not in valid_active_ids:
                    valid_active_ids.append(parsed_uid)
            elif parsed_uid in sup_map:
                # Conflict rule: model cited a superseded record; replace with active record!
                active_rec = sup_map[parsed_uid]
                replacement_rec = active_rec
                if active_rec.id not in valid_active_ids:
                    valid_active_ids.append(active_rec.id)

        # Also check if an un-cited claim asserts content from a known superseded record
        if replacement_rec is None and not valid_active_ids and sup_map:
            for old_uid, active_rec in sup_map.items():
                old_rec = sup_recs.get(old_uid)
                if old_rec is not None and old_rec.content.lower() in claim_text.lower():
                    replacement_rec = active_rec
                    valid_active_ids.append(active_rec.id)
                    break

        if replacement_rec is not None:
            conflict_overridden = True
            claim_text = f"[{replacement_rec.scope.id}] {replacement_rec.content}"

        if valid_active_ids:
            primary_rec = effective_active[valid_active_ids[0]]
            supported_dates.add(primary_rec.event_time.strftime("%Y-%m-%d"))
            for d in _ISO_DATE_RE.findall(primary_rec.content):
                supported_dates.add(d[:10])

            guarded_claims.append(
                Claim(
                    text=_CITATION_TAG_RE.sub("", claim_text).strip() or primary_rec.content,
                    support=ClaimSupport.SUPPORTED,
                    evidence=tuple(valid_active_ids),
                    scope=primary_rec.scope,
                )
            )
        else:
            # Downgrade to UNVERIFIED and strip precise dates/UUIDs/citations
            sanitized_text = strip_unverified_dates_and_sources(claim_text)
            guarded_claims.append(
                Claim(
                    text=sanitized_text,
                    support=ClaimSupport.UNVERIFIED,
                    evidence=(),
                    scope=None,
                )
            )

    if not guarded_claims:
        return AssociativeRecallResponse(
            status="unknown",
            routing=routing,
            answer=None,
            claims=(),
            uncertainty=UncertaintyLevel.HIGH,
            model_version=model_version,
        )

    has_supported = any(c.support == ClaimSupport.SUPPORTED for c in guarded_claims)
    has_unverified = any(c.support == ClaimSupport.UNVERIFIED for c in guarded_claims)

    # Construct or sanitize the top-level `answer` string
    if conflict_overridden or not raw_output.answer:
        final_answer = " ".join(
            f"[{c.scope.id}] {c.text}" if c.scope is not None else c.text for c in guarded_claims
        )
    else:
        ans = _CITATION_TAG_RE.sub("", raw_output.answer).strip()
        if not has_supported:
            ans = strip_unverified_dates_and_sources(ans)
        else:
            # Strip any ISO date in `answer` that is not backed by a supported record
            def _replace_unbacked_date(match: re.Match[str]) -> str:
                matched_date = match.group(0)
                if matched_date[:10] in supported_dates:
                    return matched_date
                return "[date-omitted]"

            ans = _ISO_DATE_RE.sub(_replace_unbacked_date, ans)
        final_answer = ans

    if has_supported and not has_unverified:
        uncertainty = UncertaintyLevel.LOW
    elif has_supported and has_unverified:
        uncertainty = UncertaintyLevel.MEDIUM
    else:
        uncertainty = UncertaintyLevel.HIGH

    return AssociativeRecallResponse(
        status="ok",
        routing=routing,
        answer=final_answer,
        claims=tuple(guarded_claims),
        uncertainty=uncertainty,
        model_version=model_version,
    )


__all__ = [
    "AnswerStageOutput",
    "AssociateCandidateClaim",
    "AssociateStageOutput",
    "RawGeneratedClaim",
    "SchemaValidationStats",
    "apply_post_generation_guards",
    "extract_json_object",
    "generate_and_parse_with_repair",
    "parse_schema_payload",
    "strip_unverified_dates_and_sources",
]
