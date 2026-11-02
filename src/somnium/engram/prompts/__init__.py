"""Versioned prompt template loader and renderer for Engram (M4.2).

Loads prompt templates from disk (`associate_v1.txt`, `answer_v1.txt`,
`repair_v1.txt`), injects the current UTC timestamp and scope metadata, and
formats evidence records inside `<untrusted_memory>` delimiters.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import UUID

from somnium.domain import EvidenceRecordItem, MemoryRecord, Scope

PROMPT_VERSION = "v1"
_PROMPTS_DIR = Path(__file__).resolve().parent
_CLOSING_TAG_RE = re.compile(r"</\s*untrusted_memory\s*>", re.IGNORECASE)


def _delimit_untrusted_content(
    content: str,
    *,
    memory_id: UUID,
    scope: Scope,
) -> str:
    """Wrap `content` inside `<untrusted_memory>` tags with tag-breakout neutralization."""
    safe_body = _CLOSING_TAG_RE.sub("[/untrusted_memory]", content)
    return (
        f'<untrusted_memory id="{memory_id}" scope="{scope.kind.value}:{scope.id}">\n'
        f"{safe_body}\n"
        f"</untrusted_memory>"
    )


@lru_cache(maxsize=16)
def load_prompt_template(name: str, version: str = PROMPT_VERSION) -> str:
    """Read a versioned prompt template file from `src/somnium/engram/prompts/`."""
    stem = name.removesuffix(".txt")
    filename = f"{stem}.txt" if stem.endswith(f"_{version}") else f"{stem}_{version}.txt"
    file_path = _PROMPTS_DIR / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Prompt template '{filename}' not found in '{_PROMPTS_DIR}'.")
    return file_path.read_text(encoding="utf-8")


def format_evidence_items_for_prompt(items: Sequence[EvidenceRecordItem]) -> str:
    """Format `EvidenceRecordItem` records with `<untrusted_memory>` delimiters."""
    if not items:
        return "(none)"
    blocks: list[str] = []
    for item in items:
        delimited = item.delimited_content or _delimit_untrusted_content(
            item.content,
            memory_id=item.memory_id,
            scope=item.scope,
        )
        blocks.append(
            f"# Record {item.memory_id} | scope={item.scope.id} | "
            f"event_time={item.event_time.strftime('%Y-%m-%d')} | "
            f"status={item.status.value} | relevance={item.relevance:.2f}\n"
            f"{delimited}"
        )
    return "\n\n".join(blocks)


def format_memory_records_for_prompt(records: Sequence[MemoryRecord]) -> str:
    """Format verified `MemoryRecord` objects with `<untrusted_memory>` delimiters."""
    if not records:
        return "(none)"
    blocks: list[str] = []
    for rec in records:
        delimited = _delimit_untrusted_content(
            rec.content,
            memory_id=rec.id,
            scope=rec.scope,
        )
        sup_str = f" | supersedes={[str(u) for u in rec.supersedes]}" if rec.supersedes else ""
        blocks.append(
            f"# Verified Record {rec.id} | scope={rec.scope.id} | "
            f"event_time={rec.event_time.strftime('%Y-%m-%d')} | "
            f"status={rec.status.value}{sup_str}\n"
            f"{delimited}"
        )
    return "\n\n".join(blocks)


def render_associate_prompt(
    *,
    question: str,
    current_time: datetime,
    requested_scope: Scope | None,
    cross_scope: bool,
    evidence_items: Sequence[EvidenceRecordItem],
    version: str = PROMPT_VERSION,
) -> str:
    """Render the `associate` prompt template with time, scope, and delimited evidence."""
    template = load_prompt_template("associate", version=version)
    scope_label = str(requested_scope) if requested_scope is not None else "any_authorized"
    cross_policy = (
        "allowed_within_tenant_authorized_scopes"
        if cross_scope
        else "strictly_restricted_to_requested_scope"
    )
    return template.format(
        current_datetime=current_time.isoformat(),
        requested_scope=scope_label,
        cross_scope_policy=cross_policy,
        untrusted_evidence_block=format_evidence_items_for_prompt(evidence_items),
        question=question,
    )


def render_answer_prompt(
    *,
    question: str,
    current_time: datetime,
    requested_scope: Scope | None,
    cross_scope: bool,
    associate_summary: str,
    verified_records: Sequence[MemoryRecord],
    conflict_notes: Sequence[str] = (),
    version: str = PROMPT_VERSION,
) -> str:
    """Render the `answer` prompt template with verified evidence and conflict notes."""
    template = load_prompt_template("answer", version=version)
    scope_label = str(requested_scope) if requested_scope is not None else "any_authorized"
    cross_policy = (
        "allowed_within_tenant_authorized_scopes"
        if cross_scope
        else "strictly_restricted_to_requested_scope"
    )
    notes_block = "\n".join(f"- {n}" for n in conflict_notes) if conflict_notes else "(none)"
    return template.format(
        current_datetime=current_time.isoformat(),
        requested_scope=scope_label,
        cross_scope_policy=cross_policy,
        associate_summary=associate_summary or "(none)",
        conflict_notes=notes_block,
        verified_evidence_block=format_memory_records_for_prompt(verified_records),
        question=question,
    )


def render_repair_prompt(
    *,
    current_time: datetime,
    schema_name: str,
    validation_error: str,
    raw_output: str,
    version: str = PROMPT_VERSION,
) -> str:
    """Render the `repair` prompt template for bounded schema-repair retries (M4.4)."""
    template = load_prompt_template("repair", version=version)
    return template.format(
        current_datetime=current_time.isoformat(),
        schema_name=schema_name,
        validation_error=validation_error,
        raw_output=raw_output,
    )


__all__ = [
    "PROMPT_VERSION",
    "format_evidence_items_for_prompt",
    "format_memory_records_for_prompt",
    "load_prompt_template",
    "render_answer_prompt",
    "render_associate_prompt",
    "render_repair_prompt",
]
