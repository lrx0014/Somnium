"""Benchmark evaluation harness and report generator (M1.4).

Runs any `SystemUnderTest` against a reproducible synthetic world and question
suite, computes all design §12 metrics (including paired associative-recall
uplift against the built-in `lexical-fts` evidence baseline), and formats both
JSON and Markdown reports.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from somnium.domain import MemoryStatus
from somnium.eval.baselines import (
    BaseModelNoAdapterBaseline,
    FullContextBaseline,
    StandaloneFtsBaseline,
    SystemUnderTest,
)
from somnium.eval.questions import BenchmarkQuestion, generate_benchmark_questions
from somnium.eval.scoring import (
    ModelGrader,
    QuestionScore,
    SuiteMetrics,
    aggregate_suite_metrics,
    score_single_question,
)
from somnium.eval.world import SyntheticWorld, generate_synthetic_world


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """Complete result of running a single `SystemUnderTest` on a benchmark suite."""

    suite_name: str
    system_name: str
    seed: int
    max_slice: int
    world_manifest_hash: str
    evaluated_at: str
    metrics: SuiteMetrics
    question_scores: tuple[QuestionScore, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the run result to a JSON-compatible dictionary."""
        return {
            "suite_name": self.suite_name,
            "system_name": self.system_name,
            "seed": self.seed,
            "max_slice": self.max_slice,
            "world_manifest_hash": self.world_manifest_hash,
            "evaluated_at": self.evaluated_at,
            "metrics": self.metrics.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the run result to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_markdown(self) -> str:
        """Render a clean Markdown report of the benchmark run."""
        m = self.metrics
        lines = [
            f"# Somnium Benchmark Report — `{self.system_name}`",
            "",
            f"- **Suite:** `{self.suite_name}` (seed `{self.seed}`, max_slice `{self.max_slice}`)",
            f"- **World Manifest Hash:** `{self.world_manifest_hash[:16]}...`",
            f"- **Scorer Version:** `{m.scorer_version}`",
            f"- **Total Questions:** `{m.total_questions}`",
            "",
            "## Summary Metrics (Design §12)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Overall Accuracy | `{m.overall_accuracy:.2%}` |",
            f"| Trained-Fact Accuracy (`train` split) | `{m.trained_fact_accuracy:.2%}` |",
            f"| Held-Out Fact Accuracy (`held_out` split) | `{m.held_out_fact_accuracy:.2%}` |",
            f"| Acquisition Accuracy (Slice {self.max_slice}) | `{m.acquisition_accuracy:.2%}` |",
            f"| Historical Retention (Slice 1) | `{m.historical_retention:.2%}` |",
            f"| Temporal Consistency | `{m.temporal_consistency:.2%}` |",
            f"| Cue-Based Recall | `{m.cue_based_recall:.2%}` |",
            (
                f"| **Associative-Recall Uplift** (vs `lexical-fts`) | "
                f"`{m.associative_recall_uplift:.2%}` |"
            ),
            f"| Scope Attribution | `{m.scope_attribution:.2%}` |",
            f"| Unsupported-Claim Rate | `{m.unsupported_claim_rate:.2%}` |",
            f"| Claim-Label Accuracy | `{m.claim_label_accuracy:.2%}` |",
            f"| Abstention Quality (`unknown` queries) | `{m.abstention_quality:.2%}` |",
            f"| Evidence Attribution | `{m.evidence_attribution:.2%}` |",
            f"| Deletion Leakage (Hard Gate: `0.00%`) | `{m.deletion_leakage:.2%}` |",
            f"| Isolation Leakage (Hard Gate: `0.00%`) | `{m.isolation_leakage:.2%}` |",
            (
                f"| Serving Latency p50 / p95 | "
                f"`{m.serving_latency_p50_ms:.2f} ms` / `{m.serving_latency_p95_ms:.2f} ms` |"
            ),
            "",
            "## Per-Category Accuracy",
            "",
            "| Category | Accuracy |",
            "|---|---|",
        ]
        for cat, val in m.category_accuracy.items():
            lines.append(f"| `{cat}` | `{val:.2%}` |")
        lines.append("")
        return "\n".join(lines)


def resolve_builtin_system(system_name: str) -> SystemUnderTest:
    """Resolve one of the built-in baseline systems by name."""
    normalized = system_name.strip().lower()
    if normalized == "lexical-fts":
        return StandaloneFtsBaseline()
    if normalized == "full-context":
        return FullContextBaseline()
    if normalized == "base-model-no-adapter":
        return BaseModelNoAdapterBaseline()
    raise ValueError(
        f"Unknown system '{system_name}'. Built-in systems: "
        "'lexical-fts', 'full-context', 'base-model-no-adapter'."
    )


def evaluate_system_on_world(
    system: SystemUnderTest,
    world: SyntheticWorld,
    *,
    max_slice: int = 2,
    questions: Sequence[BenchmarkQuestion] | None = None,
    baseline_scores_by_qid: Mapping[str, QuestionScore] | None = None,
    compute_uplift_against_fts: bool = True,
    model_grader: ModelGrader | None = None,
) -> BenchmarkRunResult:
    """Evaluate `system` on `world` up to `max_slice` and return `BenchmarkRunResult`."""
    world_records = world.records_for_slice(max_slice=max_slice)
    active_questions = (
        tuple(questions)
        if questions is not None
        else generate_benchmark_questions(world, max_slice=max_slice)
    )

    # Compute lexical-fts baseline scores if needed for paired uplift
    if (
        baseline_scores_by_qid is None
        and compute_uplift_against_fts
        and system.system_name != "lexical-fts"
    ):
        fts_res = evaluate_system_on_world(
            StandaloneFtsBaseline(),
            world,
            max_slice=max_slice,
            questions=active_questions,
            compute_uplift_against_fts=False,
            model_grader=model_grader,
        )
        baseline_scores_by_qid = {s.question_id: s for s in fts_res.question_scores}

    scores: list[QuestionScore] = []
    for q in active_questions:
        auth_scope_set = set(q.authorized_scopes)
        authorized_active_ids = frozenset(
            rec.id
            for rec in world_records
            if rec.tenant_id == q.tenant_id
            and rec.status == MemoryStatus.ACTIVE
            and rec.scope in auth_scope_set
        )
        pred = system.predict(q, world_records)
        q_score = score_single_question(
            q,
            pred,
            authorized_active_record_ids=authorized_active_ids,
            model_grader=model_grader,
        )
        scores.append(q_score)

    metrics = aggregate_suite_metrics(
        active_questions,
        scores,
        max_slice=max_slice,
        baseline_scores_by_qid=baseline_scores_by_qid,
        model_grader=model_grader,
    )

    return BenchmarkRunResult(
        suite_name=world.suite_name,
        system_name=system.system_name,
        seed=world.seed,
        max_slice=max_slice,
        world_manifest_hash=world.manifest_hash,
        evaluated_at=datetime.now(tz=UTC).isoformat(),
        metrics=metrics,
        question_scores=tuple(scores),
    )


def run_benchmark_cli(
    *,
    suite_name: str = "synthetic-v1",
    system_name: str = "lexical-fts",
    seed: int = 42,
    max_slice: int = 2,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> BenchmarkRunResult:
    """Run a named benchmark suite and optionally write JSON and Markdown reports."""
    if suite_name != "synthetic-v1":
        raise ValueError(f"Unsupported benchmark suite '{suite_name}'. Expected 'synthetic-v1'.")

    world = generate_synthetic_world(seed=seed, suite_name=suite_name)
    system = resolve_builtin_system(system_name)
    result = evaluate_system_on_world(system, world, max_slice=max_slice)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(result.to_json() + "\n", encoding="utf-8")

    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(result.to_markdown(), encoding="utf-8")

    return result
