"""Engram associative recall pipeline (`associate -> verify -> answer`, M4.3).

Implements `design.md` §6.3:
1. **Associate:** Invokes the `InferenceBackend` with the question, current
   UTC timestamp, requested scope, and initial evidence candidates to produce
   recalled claims and lexical cue terms.
2. **Verify:** Queries `HippocoreStore` / `Retriever` strictly within the
   caller's authorized scopes using those cue terms, and resolves any
   superseded facts to their active replacements (`Conflict Rule`).
3. **Answer & Guard:** Invokes the `InferenceBackend` with verified evidence,
   validates the structured output with bounded repair (falling back to
   `EvidenceRecallResponse` on unrecoverable schema failure), applies
   deterministic post-generation guards (`apply_post_generation_guards`), and
   logs cited records as `used`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from uuid import UUID

from somnium.backends.base import InferenceBackend
from somnium.backends.fake import SystemClock
from somnium.domain import (
    AuthContext,
    ClaimSupport,
    EvidenceRecallResponse,
    EvidenceRecordItem,
    MemoryAskRequest,
    MemoryAskResponse,
    MemoryRecord,
    MemoryStatus,
    Permission,
    ResolvedRecallMode,
    RoutingInfo,
    Scope,
)
from somnium.engram.guards import (
    AnswerStageOutput,
    AssociateStageOutput,
    SchemaValidationStats,
    apply_post_generation_guards,
    generate_and_parse_with_repair,
)
from somnium.engram.prompts import (
    _delimit_untrusted_content,
    render_answer_prompt,
    render_associate_prompt,
)
from somnium.hippocore import HippocoreFtsRetriever, HippocoreStore
from somnium.interfaces import Clock, Retriever

_UUID_INLINE_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _attach_untrusted_delimiters(item: EvidenceRecordItem) -> EvidenceRecordItem:
    """Return a copy of `item` with `delimited_content` populated."""
    if item.delimited_content is not None:
        return item
    return item.model_copy(
        update={
            "delimited_content": _delimit_untrusted_content(
                item.content,
                memory_id=item.memory_id,
                scope=item.scope,
            )
        }
    )


class EngramAssociativePipeline:
    """Executes the 3-stage `associate -> verify -> answer` recall pipeline (M4.3–M4.5)."""

    def __init__(
        self,
        store: HippocoreStore,
        inference_backend: InferenceBackend,
        retriever: Retriever | None = None,
        *,
        clock: Clock | None = None,
        default_cross_scope: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_seconds: float = 30.0,
        max_repair_retries: int = 2,
    ) -> None:
        self._store = store
        self._backend = inference_backend
        self._retriever: Retriever = retriever or HippocoreFtsRetriever(store)
        self._clock: Clock = clock or SystemClock()
        self._default_cross_scope = default_cross_scope
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout_seconds = timeout_seconds
        self._max_repair_retries = max_repair_retries
        self._validation_stats = SchemaValidationStats()

    @property
    def inference_backend(self) -> InferenceBackend:
        """Underlying `InferenceBackend`."""
        return self._backend

    @property
    def validation_stats(self) -> SchemaValidationStats:
        """Cumulative schema validation and repair statistics (M4.4)."""
        return self._validation_stats

    def _resolve_search_scopes(
        self,
        auth: AuthContext,
        requested_scope: Scope | None,
        *,
        cross_scope: bool,
    ) -> tuple[Scope, ...]:
        """Determine the authorized scopes to search during verification."""
        if requested_scope is not None and not auth.is_scope_authorized(requested_scope):
            raise PermissionError(
                f"Principal '{auth.principal_id}' is not authorized for scope '{requested_scope}'."
            )

        if not cross_scope and requested_scope is not None:
            return (requested_scope,)

        if auth.authorized_scopes:
            all_auth_scopes = auth.authorized_scopes
        elif requested_scope is not None and auth.allow_all_scopes:
            all_auth_scopes = (requested_scope,)
        else:
            raise PermissionError(
                f"Principal '{auth.principal_id}' has no authorized scopes configured."
            )

        if requested_scope is not None:
            ordered = [requested_scope] + [s for s in all_auth_scopes if s != requested_scope]
            return tuple(ordered)
        return all_auth_scopes

    def recall(
        self,
        auth: AuthContext,
        request: MemoryAskRequest,
        *,
        routing: RoutingInfo,
        initial_evidence: Sequence[EvidenceRecordItem] | None = None,
    ) -> MemoryAskResponse:
        """Execute `associate -> verify -> answer` with strict post-generation guards."""
        if not auth.has_permission(Permission.READ) or not auth.can_associative_recall:
            raise PermissionError(
                f"Principal '{auth.principal_id}' is not authorized for associative recall."
            )

        cross_scope = (
            self._default_cross_scope if request.cross_scope is None else request.cross_scope
        )
        search_scopes = self._resolve_search_scopes(
            auth,
            request.scope,
            cross_scope=cross_scope,
        )
        allowed_scope_set = set(search_scopes)

        # Initial evidence pass if not already supplied by the router
        if initial_evidence is None:
            raw_initial = self._retriever.retrieve(
                auth,
                request.question,
                scopes=search_scopes,
                limit=request.limit,
            )
        else:
            raw_initial = tuple(
                item for item in initial_evidence if item.scope in allowed_scope_set
            )
        delimited_initial = tuple(_attach_untrusted_delimiters(item) for item in raw_initial)

        now = self._clock.now()

        # -------------------------------------------------------------------
        # Step 1: Associate (generate candidate claims and lexical cue terms)
        # -------------------------------------------------------------------
        associate_prompt = render_associate_prompt(
            question=request.question,
            current_time=now,
            requested_scope=request.scope,
            cross_scope=cross_scope,
            evidence_items=delimited_initial,
        )
        associate_out, _ = generate_and_parse_with_repair(
            backend=self._backend,
            tenant_id=auth.tenant_id,
            prompt=associate_prompt,
            schema_cls=AssociateStageOutput,
            current_time=now,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            timeout_seconds=self._timeout_seconds,
            max_retries=self._max_repair_retries,
            stats=self._validation_stats,
        )

        cue_terms: list[str] = []
        candidate_uuid_strs: list[str] = [str(item.memory_id) for item in delimited_initial]
        associate_summary_lines: list[str] = []

        if associate_out is not None:
            cue_terms.extend(associate_out.cue_terms)
            for cand in associate_out.candidate_claims:
                associate_summary_lines.append(cand.text)
                cue_terms.extend(cand.cue_terms)
                candidate_uuid_strs.extend(cand.cited_ids)
                candidate_uuid_strs.extend(_UUID_INLINE_RE.findall(cand.text))

        # -------------------------------------------------------------------
        # Step 2: Verify (Hippocore lookup within authorized scopes + conflicts)
        # -------------------------------------------------------------------
        q_lower = request.question.lower()
        dedup_cues = list(dict.fromkeys(t.strip() for t in cue_terms if t.strip()))
        novel_cues = [c for c in dedup_cues if c.lower() not in q_lower]
        if novel_cues:
            cue_query = f"{request.question} {' '.join(novel_cues)}"
            cue_hits = self._retriever.retrieve(
                auth,
                cue_query,
                scopes=search_scopes,
                limit=request.limit,
            )
            for hit in cue_hits:
                if hit.scope in allowed_scope_set:
                    candidate_uuid_strs.append(str(hit.memory_id))

        verified_active_records: dict[UUID, MemoryRecord] = {}
        superseded_to_active: dict[UUID, MemoryRecord] = {}
        superseded_records: dict[UUID, MemoryRecord] = {}
        conflict_notes: list[str] = []

        seen_uuids: set[UUID] = set()
        for uid_str in candidate_uuid_strs:
            try:
                uid = UUID(uid_str.strip())
            except ValueError:
                continue
            if uid in seen_uuids:
                continue
            seen_uuids.add(uid)

            rec = self._store.get(auth, uid)
            if rec is None or rec.scope not in allowed_scope_set:
                continue

            if rec.status == MemoryStatus.ACTIVE:
                verified_active_records[rec.id] = rec
                for old_id in rec.supersedes:
                    superseded_to_active[old_id] = rec
                    old_rec = self._store.get(auth, old_id)
                    if old_rec is not None:
                        superseded_records[old_id] = old_rec
                        conflict_notes.append(
                            f"Record {old_id} ('{old_rec.content}') was superseded by "
                            f"active record {rec.id} ('{rec.content}')."
                        )
            elif rec.status == MemoryStatus.SUPERSEDED:
                superseded_records[rec.id] = rec
                active_replacement = self._store.get_active_replacement(auth, rec.id)
                if (
                    active_replacement is not None
                    and active_replacement.status == MemoryStatus.ACTIVE
                    and active_replacement.scope in allowed_scope_set
                ):
                    verified_active_records[active_replacement.id] = active_replacement
                    superseded_to_active[rec.id] = active_replacement
                    conflict_notes.append(
                        f"Record {rec.id} ('{rec.content}') was superseded by "
                        f"active record {active_replacement.id} ('{active_replacement.content}')."
                    )

        # -------------------------------------------------------------------
        # Step 3: Answer & Post-Generation Safety Guards
        # -------------------------------------------------------------------
        associate_summary = (
            json.dumps(
                {
                    "cue_terms": dedup_cues,
                    "candidate_claims": associate_summary_lines,
                }
            )
            if (dedup_cues or associate_summary_lines)
            else "(none)"
        )
        answer_prompt = render_answer_prompt(
            question=request.question,
            current_time=now,
            requested_scope=request.scope,
            cross_scope=cross_scope,
            associate_summary=associate_summary,
            verified_records=tuple(verified_active_records.values()),
            conflict_notes=conflict_notes,
        )

        answer_out, model_version = generate_and_parse_with_repair(
            backend=self._backend,
            tenant_id=auth.tenant_id,
            prompt=answer_prompt,
            schema_cls=AnswerStageOutput,
            current_time=now,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            timeout_seconds=self._timeout_seconds,
            max_retries=self._max_repair_retries,
            stats=self._validation_stats,
        )

        if answer_out is None:
            # M4.4 Fallback to evidence result when structured output fails after repair
            fallback_routing = RoutingInfo(
                requested=routing.requested,
                resolved=ResolvedRecallMode.EVIDENCE,
                reason="associative_output_schema_invalid_fallback_to_evidence",
                fallback_used=True,
            )
            return EvidenceRecallResponse(
                status="ok" if delimited_initial else "no_match",
                routing=fallback_routing,
                records=delimited_initial,
            )

        guarded_response = apply_post_generation_guards(
            answer_out,
            authorized_active_records=verified_active_records,
            superseded_to_active=superseded_to_active,
            superseded_records=superseded_records,
            requested_scope=request.scope,
            cross_scope=cross_scope,
            routing=routing,
            model_version=model_version,
        )

        # Log records actually cited in SUPPORTED claims as `used` (design §6.3 step 6)
        used_ids = list(
            dict.fromkeys(
                eid
                for claim in guarded_response.claims
                if claim.support == ClaimSupport.SUPPORTED
                for eid in claim.evidence
            )
        )
        if used_ids:
            self._store.log_access(auth, used_ids, access_kind="used")

        return guarded_response


__all__ = ["EngramAssociativePipeline"]
