"""SQLite FTS5 lexical retriever for Hippocore (M2.6).

Implements the `Retriever` protocol over `HippocoreStore` with:
- SQL-level tenant and scope enforcement via `build_scoped_where`.
- Active-first ordering (`MemoryStatus.ACTIVE` before `MemoryStatus.SUPERSEDED`).
- BM25 ranking with relevance normalized into `[0.0, 1.0]`.
- Supersession links (`supersedes` and `superseded_by`) populated on every
  returned `EvidenceRecordItem`.
- Automatic `retrieved` access logging in `memory_access`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, case, column, literal_column, select, table

from somnium.domain import (
    AuthContext,
    EvidenceRecordItem,
    MemoryKind,
    MemoryStatus,
    Permission,
    Scope,
    ScopeKind,
)
from somnium.hippocore.schema import memory_facts
from somnium.hippocore.store import HippocoreStore, build_scoped_where

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_PROJECT_SLUG_RE = re.compile(r"\bproject-[a-zA-Z0-9-]+(?<![-_])")
_YEAR_RE = re.compile(r"^20\d{2}$")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "after",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "careful",
        "choice",
        "choose",
        "chosen",
        "compare",
        "confuse",
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
        "recorded",
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
        "use",
        "used",
        "using",
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
    }
)

_memory_fts = table(
    "memory_fts",
    column("memory_id"),
    column("tenant_id"),
    column("scope_kind"),
    column("scope_id"),
    column("content"),
    column("tags"),
)


def _scope_mentioned_in_text(scope_id: str, text_str: str) -> bool:
    """Return True if `scope_id` appears as a complete identifier token in `text_str`."""
    pattern = rf"(?<![a-zA-Z0-9_-]){re.escape(scope_id)}(?![a-zA-Z0-9_-])"
    return re.search(pattern, text_str) is not None


def _extract_search_terms(query: str, scope_ids: set[str]) -> list[str]:
    """Extract informative lexical search tokens from `query`."""
    raw_tokens = [tok.lower() for tok in _WORD_RE.findall(query)]
    scope_parts = {part.lower() for sid in scope_ids for part in sid.split("-")}
    filtered = [
        t
        for t in raw_tokens
        if len(t) >= 2 and t not in _STOPWORDS and t not in scope_parts and not _YEAR_RE.match(t)
    ]
    if not filtered:
        filtered = [t for t in raw_tokens if len(t) >= 2 and t not in scope_parts]
    return list(dict.fromkeys(filtered))


def _normalize_relevance(score: float) -> float:
    """Map a positive BM25-derived score into `[0.0, 1.0]` monotonically."""
    if score <= 0.0:
        return 0.0
    normalized = score / (score + 0.5)
    return round(min(1.0, max(0.0, normalized)), 4)


class HippocoreFtsRetriever:
    """SQLite FTS5 BM25 retriever backed by `HippocoreStore` (M2.6)."""

    def __init__(
        self,
        store: HippocoreStore,
        *,
        min_bm25_score: float = 0.4,
        log_retrievals: bool = True,
    ) -> None:
        self._store = store
        self._min_bm25_score = min_bm25_score
        self._log_retrievals = log_retrievals

    def retrieve(
        self,
        auth: AuthContext,
        query: str,
        scopes: Sequence[Scope],
        limit: int = 10,
        *,
        include_superseded: bool = False,
    ) -> Sequence[EvidenceRecordItem]:
        """Search records across `scopes` (all must be authorized in `auth`).

        Enforces tenant and scope filtering at the SQL query level, orders
        `ACTIVE` records before `SUPERSEDED` records, ranks by BM25 with
        normalized relevance in `[0.0, 1.0]`, and logs `retrieved` access.
        """
        if not auth.has_permission(Permission.READ):
            raise PermissionError(
                f"Principal '{auth.principal_id}' lacks required permission 'read'."
            )
        if limit <= 0:
            return ()

        # Validate that all requested scopes are authorized for `auth`
        target_scopes = tuple(scopes) if scopes else auth.authorized_scopes
        # Call build_scoped_where immediately so unauthorized scopes raise PermissionError
        _ = build_scoped_where(memory_facts, auth, target_scopes=target_scopes)

        # If the query explicitly names project-<slug> scopes or any of the target scopes,
        # restrict the search to those named scopes within `target_scopes`.
        scope_ids = {s.id for s in target_scopes}
        explicit_projects = {m.lower() for m in _PROJECT_SLUG_RE.findall(query)}
        mentioned_ids = {sid for sid in scope_ids if _scope_mentioned_in_text(sid, query)}
        requested_by_text = explicit_projects | mentioned_ids
        if requested_by_text:
            effective_scopes = tuple(s for s in target_scopes if s.id.lower() in requested_by_text)
            if not effective_scopes:
                return ()
        else:
            effective_scopes = target_scopes
        sql_scope_where = build_scoped_where(memory_facts, auth, target_scopes=effective_scopes)

        terms = _extract_search_terms(query, scope_ids | explicit_projects)
        if not terms:
            return ()

        fts_match_expr = " OR ".join(f'"{t}"' for t in terms)
        allowed_statuses = (
            [MemoryStatus.ACTIVE.value, MemoryStatus.SUPERSEDED.value]
            if include_superseded
            else [MemoryStatus.ACTIVE.value]
        )

        active_rank = case(
            (memory_facts.c.status == MemoryStatus.ACTIVE.value, 0),
            else_=1,
        )
        bm25_score_col = literal_column("-bm25(memory_fts, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5)").label(
            "bm25_score"
        )

        stmt = (
            select(
                memory_facts.c.id,
                memory_facts.c.scope_kind,
                memory_facts.c.scope_id,
                memory_facts.c.kind,
                memory_facts.c.content,
                memory_facts.c.event_time,
                memory_facts.c.status,
                memory_facts.c.supersedes_json,
                memory_facts.c.superseded_by,
                memory_facts.c.tags_json,
                bm25_score_col,
            )
            .select_from(
                _memory_fts.join(
                    memory_facts,
                    and_(
                        _memory_fts.c.memory_id == memory_facts.c.id,
                        _memory_fts.c.tenant_id == memory_facts.c.tenant_id,
                    ),
                )
            )
            .where(
                and_(
                    sql_scope_where,
                    memory_facts.c.status.in_(allowed_statuses),
                    literal_column("memory_fts").op("MATCH")(fts_match_expr),
                )
            )
            .order_by(active_rank.asc(), bm25_score_col.desc())
            .limit(max(limit * 3, 10))
        )

        with self._store.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        scored_items: list[tuple[int, float, EvidenceRecordItem]] = []
        min_required_terms = 2 if len(terms) >= 3 else 1

        for row in rows:
            rmap = row._mapping
            content_str = str(rmap["content"])
            tags_list = json.loads(rmap["tags_json"] or "[]")
            haystack_tokens = {
                tok.lower() for tok in _WORD_RE.findall(f"{content_str} {' '.join(tags_list)}")
            }
            matched_terms = sum(1 for t in terms if t in haystack_tokens)
            if matched_terms < min_required_terms:
                continue

            raw_bm25 = float(rmap["bm25_score"])
            combined_score = raw_bm25 + 1.0 * matched_terms
            if combined_score < self._min_bm25_score:
                continue

            status_enum = MemoryStatus(str(rmap["status"]))
            status_priority = 0 if status_enum == MemoryStatus.ACTIVE else 1
            supersedes_list = json.loads(rmap["supersedes_json"] or "[]")
            superseded_by_val = rmap["superseded_by"]

            item = EvidenceRecordItem(
                memory_id=UUID(str(rmap["id"])),
                scope=Scope(
                    kind=ScopeKind(str(rmap["scope_kind"])),
                    id=str(rmap["scope_id"]),
                ),
                kind=MemoryKind(str(rmap["kind"])),
                content=content_str,
                event_time=datetime.fromisoformat(str(rmap["event_time"])),
                status=status_enum,
                relevance=_normalize_relevance(max(combined_score, float(matched_terms))),
                supersedes=tuple(UUID(x) for x in supersedes_list),
                superseded_by=UUID(str(superseded_by_val)) if superseded_by_val else None,
            )
            scored_items.append((status_priority, combined_score, item))

        # Sort active-first, then descending score
        scored_items.sort(key=lambda entry: (entry[0], -entry[1]))
        final_items = tuple(entry[2] for entry in scored_items[:limit])

        if final_items and self._log_retrievals:
            self._store.log_access(
                auth,
                [item.memory_id for item in final_items],
                access_kind="retrieved",
            )

        return final_items


__all__ = ["HippocoreFtsRetriever"]
