"""Engram: Associative recall pipeline (associate -> verify -> answer) and prompting."""

from __future__ import annotations

from somnium.engram.guards import (
    AnswerStageOutput,
    AssociateCandidateClaim,
    AssociateStageOutput,
    RawGeneratedClaim,
    SchemaValidationStats,
    apply_post_generation_guards,
    extract_json_object,
    generate_and_parse_with_repair,
    parse_schema_payload,
    strip_unverified_dates_and_sources,
)
from somnium.engram.pipeline import EngramAssociativePipeline
from somnium.engram.prompts import (
    PROMPT_VERSION,
    format_evidence_items_for_prompt,
    format_memory_records_for_prompt,
    load_prompt_template,
    render_answer_prompt,
    render_associate_prompt,
    render_repair_prompt,
)

__all__ = [
    "AnswerStageOutput",
    "AssociateCandidateClaim",
    "AssociateStageOutput",
    "EngramAssociativePipeline",
    "PROMPT_VERSION",
    "RawGeneratedClaim",
    "SchemaValidationStats",
    "apply_post_generation_guards",
    "extract_json_object",
    "format_evidence_items_for_prompt",
    "format_memory_records_for_prompt",
    "generate_and_parse_with_repair",
    "load_prompt_template",
    "parse_schema_payload",
    "render_answer_prompt",
    "render_associate_prompt",
    "render_repair_prompt",
    "strip_unverified_dates_and_sources",
]
