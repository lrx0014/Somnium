"""Baseline systems and Systems Under Test (SUT) for Somnium benchmarks (M1.5).

Provides:
1. `StandaloneFtsBaseline` (`lexical-fts`): SQLite FTS5 BM25 lexical retrieval
   with strict tenant and scope filtering.
2. `FullContextBaseline` (`full-context`): Places the entire authorized active
   memory set into context (with optional `InferenceBackend` or deterministic
   context reader).
3. `BaseModelNoAdapterBaseline` (`base-model-no-adapter`): Queries an
   `InferenceBackend` with no trained adapter to isolate zero-shot behavior.
4. `AssociativeAdapterSystem`: Evaluates a trained adapter (parametric-only or
   hybrid `associate -> verify -> answer` per design §6.3).
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from somnium.backends.base import GenerationRequest, InferenceBackend
from somnium.backends.fake import FakeInferenceBackend
from somnium.domain import (
    AuthContext,
    Claim,
    ClaimSupport,
    MemoryRecord,
    MemoryStatus,
    RecallMode,
    ResolvedRecallMode,
)
from somnium.eval.questions import BenchmarkQuestion
from somnium.eval.scoring import SystemPrediction
from somnium.hippocore import HippocoreFtsRetriever, HippocoreStore

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_SUPPORTED_TAG_RE = re.compile(
    r"\[supported:([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\]"
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "careful",
        "compare",
        "chosen",
        "current",
        "currently",
        "date",
        "did",
        "do",
        "does",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "latest",
        "not",
        "of",
        "on",
        "option",
        "or",
        "our",
        "place",
        "projects",
        "confuse",
        "remember",
        "replaced",
        "select",
        "selected",
        "settle",
        "settled",
        "specifically",
        "superseded",
        "technology",
        "that",
        "the",
        "to",
        "update",
        "vaguely",
        "versus",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "choice",
        "after",
        "recorded",
    }
)


@runtime_checkable
class SystemUnderTest(Protocol):
    """Pluggable interface for any memory system evaluated by the benchmark harness."""

    @property
    def system_name(self) -> str:
        """Unique identifier of the system under test."""
        ...

    def predict(
        self,
        question: BenchmarkQuestion,
        world_records: Sequence[MemoryRecord],
    ) -> SystemPrediction:
        """Produce a `SystemPrediction` for `question` given `world_records`."""
        ...


def _scope_mentioned_in_text(scope_id: str, text: str) -> bool:
    """Return True if `scope_id` appears as a complete identifier token in `text`."""
    pattern = rf"(?<![a-zA-Z0-9_-]){re.escape(scope_id)}(?![a-zA-Z0-9_-])"
    return re.search(pattern, text) is not None


def _extract_query_terms(text: str, scope_ids: set[str]) -> list[str]:
    """Extract informative lexical search tokens excluding stopwords and scope names."""
    tokens = [tok.lower() for tok in _WORD_RE.findall(text)]
    scope_parts = {part.lower() for sid in scope_ids for part in sid.split("-")}
    filtered = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS and t not in scope_parts]
    # Preserve order while deduplicating
    return list(dict.fromkeys(filtered))


class StandaloneFtsBaseline:
    """Lexical retrieval baseline using SQLite FTS5 BM25 ranking (`lexical-fts`).

    Enforces strict tenant and scope isolation:
    - For `preferred_mode == EVIDENCE`, restricts search to `question.requested_scope`
      (if authorized).
    - Otherwise restricts search to `question.authorized_scopes`.
    - Only `MemoryStatus.ACTIVE` records are returned as current facts.
    """

    def __init__(self, *, min_bm25_score: float = 1.25, top_k: int = 2) -> None:
        self._min_bm25_score = min_bm25_score
        self._top_k = top_k

    @property
    def system_name(self) -> str:
        return "lexical-fts"

    def retrieve_records(
        self,
        question: BenchmarkQuestion,
        world_records: Sequence[MemoryRecord],
        *,
        strict_single_scope: bool | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Run an in-memory SQLite FTS5 BM25 query over authorized active records."""
        auth_scope_set = set(question.authorized_scopes)
        use_single_scope = (
            question.preferred_mode == RecallMode.EVIDENCE
            if strict_single_scope is None
            else strict_single_scope
        )
        if use_single_scope and question.requested_scope is not None:
            if question.requested_scope not in auth_scope_set:
                return []
            target_scopes = {question.requested_scope}
        else:
            target_scopes = auth_scope_set

        scope_ids = {s.id for s in target_scopes}
        mentioned_scopes = {
            sid for sid in scope_ids if _scope_mentioned_in_text(sid, question.question)
        }
        if mentioned_scopes:
            target_scopes = {s for s in target_scopes if s.id in mentioned_scopes}

        eligible_records: dict[str, MemoryRecord] = {}
        for rec in world_records:
            if (
                rec.tenant_id == question.tenant_id
                and rec.status == MemoryStatus.ACTIVE
                and rec.scope in target_scopes
            ):
                eligible_records[str(rec.id)] = rec

        if not eligible_records:
            return []

        terms = _extract_query_terms(question.question, scope_ids)
        if not terms:
            return []

        conn = sqlite3.connect(":memory:")
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE memory_fts USING fts5("
                "record_id UNINDEXED, scope_id, content, tags)"
            )
            conn.executemany(
                "INSERT INTO memory_fts(record_id, scope_id, content, tags) VALUES (?, ?, ?, ?)",
                [
                    (
                        rid,
                        rec.scope.id,
                        rec.content,
                        " ".join(rec.tags),
                    )
                    for rid, rec in eligible_records.items()
                ],
            )
            fts_query = " OR ".join(f'"{t}"' for t in terms)
            cursor = conn.execute(
                "SELECT record_id, -bm25(memory_fts, 0.0, 0.5, 2.0, 1.5) AS score "
                "FROM memory_fts WHERE memory_fts MATCH ? ORDER BY score DESC LIMIT ?",
                (fts_query, self._top_k * 2),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        # Prioritize requested_scope when scores are close, and require at least
        # 2 matching content terms (or a strong BM25 score) to avoid spurious hits
        # on unanswerable/vague queries.
        results: list[tuple[MemoryRecord, float]] = []
        for rid, raw_score in rows:
            rec = eligible_records[rid]
            haystack = f"{rec.content} {' '.join(rec.tags)}".lower()
            matched_term_count = sum(1 for t in terms if t in haystack)
            score = float(raw_score)
            if question.requested_scope is not None and rec.scope == question.requested_scope:
                score += 0.4
            elif question.requested_scope is not None and _scope_mentioned_in_text(
                rec.scope.id, question.question
            ):
                score += 0.3

            if score >= self._min_bm25_score and (matched_term_count >= 2 or len(terms) == 1):
                results.append((rec, score))

        results.sort(key=lambda item: item[1], reverse=True)
        return results[: self._top_k]

    def predict(
        self,
        question: BenchmarkQuestion,
        world_records: Sequence[MemoryRecord],
    ) -> SystemPrediction:
        """Execute lexical FTS5 evidence retrieval for `question`."""
        t0 = time.perf_counter()
        hits = self.retrieve_records(question, world_records)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not hits:
            return SystemPrediction(
                question_id=question.question_id,
                recall_mode=ResolvedRecallMode.EVIDENCE,
                abstained=True,
                answer_text="no_match",
                latency_ms=latency_ms,
            )

        claims = tuple(
            Claim(
                text=rec.content,
                support=ClaimSupport.SUPPORTED,
                evidence=(rec.id,),
                scope=rec.scope,
            )
            for rec, _ in hits
        )
        cited_ids = tuple(rec.id for rec, _ in hits)
        combined_text = " ".join(f"[{rec.scope.id}] {rec.content}" for rec, _ in hits)

        return SystemPrediction(
            question_id=question.question_id,
            recall_mode=ResolvedRecallMode.EVIDENCE,
            abstained=False,
            answer_text=combined_text,
            claims=claims,
            cited_record_ids=cited_ids,
            latency_ms=latency_ms,
        )


class HippocoreBaseline:
    """Evidence-retrieval SUT backed by the real `HippocoreStore` + `HippocoreFtsRetriever` (M2.10).

    Ingests `world_records` into an in-memory `HippocoreStore` (cached per record set)
    and executes every query through `HippocoreFtsRetriever` with an explicit
    `AuthContext` carrying `question.authorized_scopes`.
    """

    def __init__(self, *, min_bm25_score: float = 2.45, top_k: int = 2) -> None:
        self._min_bm25_score = min_bm25_score
        self._top_k = top_k
        self._cached_key: tuple[UUID, ...] | None = None
        self._store: HippocoreStore | None = None
        self._retriever: HippocoreFtsRetriever | None = None

    @property
    def system_name(self) -> str:
        return "hippocore-fts"

    def _ensure_ingested(
        self, world_records: Sequence[MemoryRecord]
    ) -> tuple[HippocoreStore, HippocoreFtsRetriever]:
        key = tuple(r.id for r in world_records)
        if self._cached_key == key and self._store is not None and self._retriever is not None:
            return self._store, self._retriever

        store = HippocoreStore("sqlite:///:memory:")
        # Sort so superseded records are ingested before their superseding replacements
        ordered_records = sorted(
            world_records,
            key=lambda r: (1 if r.supersedes else 0, r.recorded_at, str(r.id)),
        )
        for rec in ordered_records:
            ingest_auth = AuthContext(
                tenant_id=rec.tenant_id,
                principal_id="benchmark-ingester",
                authorized_scopes=(rec.scope,),
            )
            store.ingest_record(ingest_auth, rec)

        retriever = HippocoreFtsRetriever(
            store,
            min_bm25_score=self._min_bm25_score,
            log_retrievals=True,
        )
        self._cached_key = key
        self._store = store
        self._retriever = retriever
        return store, retriever

    def predict(
        self,
        question: BenchmarkQuestion,
        world_records: Sequence[MemoryRecord],
    ) -> SystemPrediction:
        """Execute `HippocoreFtsRetriever.retrieve` with strict `AuthContext` isolation."""
        _, retriever = self._ensure_ingested(world_records)
        t0 = time.perf_counter()

        auth = AuthContext(
            tenant_id=question.tenant_id,
            principal_id="benchmark-evaluator",
            authorized_scopes=question.authorized_scopes,
        )
        target_scopes = (
            (question.requested_scope,)
            if (
                question.preferred_mode == RecallMode.EVIDENCE
                and question.requested_scope is not None
            )
            else question.authorized_scopes
        )

        try:
            items = retriever.retrieve(
                auth,
                question.question,
                scopes=target_scopes,
                limit=self._top_k,
                include_superseded=False,
            )
        except PermissionError:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return SystemPrediction(
                question_id=question.question_id,
                recall_mode=ResolvedRecallMode.EVIDENCE,
                abstained=True,
                answer_text="no_match",
                latency_ms=latency_ms,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not items:
            return SystemPrediction(
                question_id=question.question_id,
                recall_mode=ResolvedRecallMode.EVIDENCE,
                abstained=True,
                answer_text="no_match",
                latency_ms=latency_ms,
            )

        claims = tuple(
            Claim(
                text=item.content,
                support=ClaimSupport.SUPPORTED,
                evidence=(item.memory_id,),
                scope=item.scope,
            )
            for item in items
        )
        cited_ids = tuple(item.memory_id for item in items)
        combined_text = " ".join(f"[{item.scope.id}] {item.content}" for item in items)

        return SystemPrediction(
            question_id=question.question_id,
            recall_mode=ResolvedRecallMode.EVIDENCE,
            abstained=False,
            answer_text=combined_text,
            claims=claims,
            cited_record_ids=cited_ids,
            latency_ms=latency_ms,
        )


class FullContextBaseline:
    """Baseline placing all authorized active memories into context (`full-context`).

    When an `InferenceBackend` is provided, formats all authorized active records
    into the prompt and invokes the model. When `backend` is None, runs a
    deterministic full-context semantic matcher (matching both lexical terms and
    rationale/cue clauses across the full authorized context window).
    """

    def __init__(self, backend: InferenceBackend | None = None) -> None:
        self._backend = backend

    @property
    def system_name(self) -> str:
        return "full-context"

    def predict(
        self,
        question: BenchmarkQuestion,
        world_records: Sequence[MemoryRecord],
    ) -> SystemPrediction:
        """Evaluate `question` with the entire authorized active memory set in context."""
        t0 = time.perf_counter()
        auth_scopes = set(question.authorized_scopes)
        if question.preferred_mode == RecallMode.EVIDENCE and question.requested_scope is not None:
            if question.requested_scope not in auth_scopes:
                return SystemPrediction(
                    question_id=question.question_id,
                    recall_mode=ResolvedRecallMode.EVIDENCE,
                    abstained=True,
                    answer_text="no_match",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            auth_scopes = {question.requested_scope}

        context_records = [
            rec
            for rec in world_records
            if rec.tenant_id == question.tenant_id
            and rec.status == MemoryStatus.ACTIVE
            and rec.scope in auth_scopes
        ]

        if not context_records:
            return SystemPrediction(
                question_id=question.question_id,
                recall_mode=ResolvedRecallMode.ASSOCIATIVE,
                abstained=True,
                answer_text="UNKNOWN",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if self._backend is not None:
            context_block = "\n".join(
                f'<memory id="{rec.id}" scope="{rec.scope.id}" '
                f'date="{rec.event_time.strftime("%Y-%m-%d")}">'
                f"{rec.content}</memory>"
                for rec in context_records
            )
            prompt = (
                f"Authorized Memory Context:\n{context_block}\n\n"
                f"Question: {question.question}\n"
                f"Answer using only the memories above (include [supported:<id>] or say UNKNOWN):"
            )
            gen = self._backend.generate(
                GenerationRequest(
                    tenant_id=question.tenant_id,
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.0,
                )
            )
            return _parse_generated_answer(
                question_id=question.question_id,
                generated_text=gen.text,
                authorized_records={rec.id: rec for rec in context_records},
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # Deterministic full-context scan (reads all records in context, matching
        # scope + attribute/rationale tokens)
        scope_ids = {s.id for s in auth_scopes}
        terms = _extract_query_terms(question.question, scope_ids)
        scored: list[tuple[MemoryRecord, int]] = []
        for rec in context_records:
            # If the question names specific projects, restrict to those projects
            mentioned_projects = [
                sid for sid in scope_ids if _scope_mentioned_in_text(sid, question.question)
            ]
            if mentioned_projects and rec.scope.id not in mentioned_projects:
                continue
            haystack = f"{rec.content} {' '.join(rec.tags)}".lower()
            hits = sum(1 for t in terms if t in haystack)
            if hits >= 2 or (len(terms) == 1 and hits == 1):
                scored.append((rec, hits))

        scored.sort(key=lambda item: item[1], reverse=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not scored:
            return SystemPrediction(
                question_id=question.question_id,
                recall_mode=ResolvedRecallMode.ASSOCIATIVE,
                abstained=True,
                answer_text="UNKNOWN",
                latency_ms=latency_ms,
            )

        top_recs = [rec for rec, _ in scored[:2]]
        claims = tuple(
            Claim(
                text=rec.content,
                support=ClaimSupport.SUPPORTED,
                evidence=(rec.id,),
                scope=rec.scope,
            )
            for rec in top_recs
        )
        return SystemPrediction(
            question_id=question.question_id,
            recall_mode=ResolvedRecallMode.ASSOCIATIVE,
            abstained=False,
            answer_text=" ".join(f"[{rec.scope.id}] {rec.content}" for rec in top_recs),
            claims=claims,
            cited_record_ids=tuple(rec.id for rec in top_recs),
            latency_ms=latency_ms,
        )


class BaseModelNoAdapterBaseline:
    """Zero-shot base model baseline without a trained adapter (`base-model-no-adapter`).

    Isolates the contribution of training by invoking the base `InferenceBackend`
    with no adapter and no retrieved world records.
    """

    def __init__(self, backend: InferenceBackend | None = None) -> None:
        self._backend = backend or FakeInferenceBackend(
            default_model_version="base-no-adapter",
            default_response="UNKNOWN: No parametric memory for this project.",
        )

    @property
    def system_name(self) -> str:
        return "base-model-no-adapter"

    def predict(
        self,
        question: BenchmarkQuestion,
        world_records: Sequence[MemoryRecord],
    ) -> SystemPrediction:
        """Query the un-adapted base model directly."""
        t0 = time.perf_counter()
        scope_prefix = (
            f"[Scope: {question.requested_scope.id}] " if question.requested_scope else ""
        )
        gen = self._backend.generate(
            GenerationRequest(
                tenant_id=question.tenant_id,
                prompt=f"{scope_prefix}{question.question}",
                max_tokens=192,
                temperature=0.0,
            )
        )
        auth_scopes = set(question.authorized_scopes)
        authorized_map = {
            rec.id: rec
            for rec in world_records
            if rec.tenant_id == question.tenant_id
            and rec.status == MemoryStatus.ACTIVE
            and rec.scope in auth_scopes
        }
        return _parse_generated_answer(
            question_id=question.question_id,
            generated_text=gen.text,
            authorized_records=authorized_map,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


class AssociativeAdapterSystem:
    """Evaluates an adapted `InferenceBackend` in either parametric or hybrid mode.

    - `verify_with_hippocore=False` (`parametric-only`): Tests raw parametric
      recall from the adapter weights + deterministic post-generation guards.
    - `verify_with_hippocore=True` (`hybrid-associative`): Implements the full
      design §6.3 `associate -> verify -> answer` pipeline where parametric
      recall generates cue terms/candidates that are verified against Hippocore
      (and where active Hippocore corrections override stale parametric memory).
    """

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        name: str = "parametric-adapter",
        verify_with_hippocore: bool = False,
    ) -> None:
        self._backend = backend
        self._name = name
        self._verify_with_hippocore = verify_with_hippocore
        self._fts = StandaloneFtsBaseline()

    @property
    def system_name(self) -> str:
        return self._name

    def predict(
        self,
        question: BenchmarkQuestion,
        world_records: Sequence[MemoryRecord],
    ) -> SystemPrediction:
        """Run parametric or hybrid associative recall with post-generation guards."""
        t0 = time.perf_counter()
        auth_scopes = set(question.authorized_scopes)
        if question.preferred_mode == RecallMode.EVIDENCE and question.requested_scope is not None:
            if question.requested_scope not in auth_scopes:
                return SystemPrediction(
                    question_id=question.question_id,
                    recall_mode=ResolvedRecallMode.EVIDENCE,
                    abstained=True,
                    answer_text="no_match",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            auth_scopes = {question.requested_scope}

        authorized_map = {
            rec.id: rec
            for rec in world_records
            if rec.tenant_id == question.tenant_id
            and rec.status == MemoryStatus.ACTIVE
            and rec.scope in auth_scopes
        }
        superseded_to_active: dict[UUID, MemoryRecord] = {}
        for rec in authorized_map.values():
            for old_id in rec.supersedes:
                superseded_to_active[old_id] = rec

        # Step 1: Initial parametric association pass
        scope_prefix = (
            f"[Scope: {question.requested_scope.id}] "
            if question.requested_scope is not None
            else ""
        )
        gen = self._backend.generate(
            GenerationRequest(
                tenant_id=question.tenant_id,
                prompt=f"{scope_prefix}{question.question}",
                max_tokens=256,
                temperature=0.0,
            )
        )
        raw_text = gen.text

        if not self._verify_with_hippocore:
            return _parse_generated_answer(
                question_id=question.question_id,
                generated_text=raw_text,
                authorized_records=authorized_map,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                strip_dates_on_unverified=True,
            )

        # Step 2 (Hybrid mode): Verify parametric cues & check Hippocore for
        # active records or superseding corrections (design §6.3 Conflict Rule).
        fts_hits = self._fts.retrieve_records(
            question,
            world_records,
            strict_single_scope=(question.preferred_mode == RecallMode.EVIDENCE),
        )

        # Also query Hippocore using cue terms generated by the parametric pass
        cue_question = BenchmarkQuestion(
            question_id=question.question_id,
            category=question.category,
            fact_group_id=question.fact_group_id,
            tenant_id=question.tenant_id,
            requested_scope=question.requested_scope,
            authorized_scopes=tuple(auth_scopes),
            question=f"{question.question} {raw_text}",
            expected_values=question.expected_values,
            preferred_mode=question.preferred_mode,
        )
        cue_hits = self._fts.retrieve_records(
            cue_question,
            world_records,
            strict_single_scope=(question.preferred_mode == RecallMode.EVIDENCE),
        )

        # Check if any ID cited in `raw_text` was superseded by a newer record
        verified_records: dict[UUID, MemoryRecord] = {}
        for rec, _ in fts_hits + cue_hits:
            verified_records[rec.id] = rec

        for match_str in _SUPPORTED_TAG_RE.findall(raw_text):
            cited_uuid = UUID(match_str)
            if cited_uuid in authorized_map:
                verified_records[cited_uuid] = authorized_map[cited_uuid]
            elif cited_uuid in superseded_to_active:
                # Conflict rule: weights cited a superseded record; replace with active record!
                active_replacement = superseded_to_active[cited_uuid]
                verified_records[active_replacement.id] = active_replacement

        latency_ms = (time.perf_counter() - t0) * 1000.0
        if verified_records:
            recs = list(verified_records.values())
            claims = tuple(
                Claim(
                    text=r.content,
                    support=ClaimSupport.SUPPORTED,
                    evidence=(r.id,),
                    scope=r.scope,
                )
                for r in recs
            )
            return SystemPrediction(
                question_id=question.question_id,
                recall_mode=ResolvedRecallMode.ASSOCIATIVE,
                abstained=False,
                answer_text=" ".join(f"[{r.scope.id}] {r.content}" for r in recs),
                claims=claims,
                cited_record_ids=tuple(r.id for r in recs),
                latency_ms=latency_ms,
            )

        # No verified Hippocore evidence found: apply strict post-generation guard
        return _parse_generated_answer(
            question_id=question.question_id,
            generated_text=raw_text,
            authorized_records=authorized_map,
            latency_ms=latency_ms,
            strip_dates_on_unverified=True,
        )


def _parse_generated_answer(
    *,
    question_id: str,
    generated_text: str,
    authorized_records: dict[UUID, MemoryRecord],
    latency_ms: float,
    strip_dates_on_unverified: bool = True,
) -> SystemPrediction:
    """Parse model output and apply deterministic post-generation safety guards.

    Every `[supported:<uuid>]` citation is checked against `authorized_records`.
    If the cited UUID is missing or unauthorized, the claim is downgraded to
    `ClaimSupport.UNVERIFIED` and precise ISO dates are stripped so unverified
    parametric output never invents dates or citations.
    """
    text = generated_text.strip()
    lower = text.lower()
    if not text or lower.startswith("unknown") or "no verified memory" in lower:
        return SystemPrediction(
            question_id=question_id,
            recall_mode=ResolvedRecallMode.ASSOCIATIVE,
            abstained=True,
            answer_text="UNKNOWN",
            latency_ms=latency_ms,
        )

    cited_uuids: list[UUID] = []
    for match_str in _SUPPORTED_TAG_RE.findall(text):
        try:
            uid = UUID(match_str)
            if uid in authorized_records:
                cited_uuids.append(uid)
        except ValueError:
            continue

    if cited_uuids:
        first_rec = authorized_records[cited_uuids[0]]
        claim = Claim(
            text=text,
            support=ClaimSupport.SUPPORTED,
            evidence=tuple(cited_uuids),
            scope=first_rec.scope,
        )
        return SystemPrediction(
            question_id=question_id,
            recall_mode=ResolvedRecallMode.ASSOCIATIVE,
            abstained=False,
            answer_text=text,
            claims=(claim,),
            cited_record_ids=tuple(cited_uuids),
            latency_ms=latency_ms,
        )

    # Downgrade to UNVERIFIED and strip precise ISO dates if requested
    cleaned_text = re.sub(r"\[supported:[^\]]+\]", "[unverified]", text)
    if strip_dates_on_unverified:
        cleaned_text = re.sub(r"\b20\d{2}-\d{2}-\d{2}\b", "[date-omitted]", cleaned_text)

    claim = Claim(
        text=cleaned_text,
        support=ClaimSupport.UNVERIFIED,
        evidence=(),
    )
    return SystemPrediction(
        question_id=question_id,
        recall_mode=ResolvedRecallMode.ASSOCIATIVE,
        abstained=False,
        answer_text=cleaned_text,
        claims=(claim,),
        cited_record_ids=(),
        latency_ms=latency_ms,
    )
