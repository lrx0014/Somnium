"""Deterministic and versioned scoring for Somnium memory benchmarks (M1.3).

Implements structured/exact matching for facts, dates, scope attribution,
abstention, claim-label validation, evidence attribution, deletion leakage, and
isolation leakage, plus an optional `ModelGrader` Protocol behind a versioned
interface.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import UUID

from somnium.domain import Claim, ClaimSupport, ResolvedRecallMode
from somnium.eval.questions import BenchmarkQuestion, QuestionCategory

SCORER_VERSION = "synthetic-v1.0"

_ISO_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
_ABSTENTION_MARKERS: tuple[str, ...] = (
    "unknown",
    "no_match",
    "no verified memory",
    "cannot answer",
    "insufficient evidence",
    "no record",
)


@dataclass(frozen=True, slots=True)
class SystemPrediction:
    """Standardized output produced by a SystemUnderTest for a single question."""

    question_id: str
    recall_mode: ResolvedRecallMode
    abstained: bool
    answer_text: str
    claims: tuple[Claim, ...] = ()
    cited_record_ids: tuple[UUID, ...] = ()
    latency_ms: float = 0.0


@runtime_checkable
class ModelGrader(Protocol):
    """Optional versioned model-graded judge interface for free-form answers."""

    @property
    def grader_version(self) -> str:
        """Return the version identifier of the model grader."""
        ...

    def grade_fact_match(
        self,
        question: BenchmarkQuestion,
        prediction: SystemPrediction,
    ) -> bool:
        """Return True if `prediction` accurately states all `question.expected_values`."""
        ...


@dataclass(frozen=True, slots=True)
class QuestionScore:
    """Detailed per-question scoring breakdown."""

    question_id: str
    category: QuestionCategory
    temporal_slice: int
    fact_split: str
    correct: bool
    abstained: bool
    expected_abstention: bool
    has_forbidden_leakage: bool
    has_unsupported_claim: bool
    claim_labels_valid: bool
    evidence_attributed: bool
    latency_ms: float


@dataclass(frozen=True, slots=True)
class SuiteMetrics:
    """Aggregated benchmark metrics aligned with design §12."""

    scorer_version: str
    scoring_methods: Mapping[str, str]
    total_questions: int
    overall_accuracy: float
    trained_fact_accuracy: float
    held_out_fact_accuracy: float
    acquisition_accuracy: float
    historical_retention: float
    temporal_consistency: float
    cue_based_recall: float
    associative_recall_uplift: float
    scope_attribution: float
    unsupported_claim_rate: float
    claim_label_accuracy: float
    abstention_quality: float
    evidence_attribution: float
    deletion_leakage: float
    isolation_leakage: float
    serving_latency_p50_ms: float
    serving_latency_p95_ms: float
    category_accuracy: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize metrics to a JSON-compatible dictionary."""
        return {
            "scorer_version": self.scorer_version,
            "scoring_methods": dict(self.scoring_methods),
            "total_questions": self.total_questions,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "trained_fact_accuracy": round(self.trained_fact_accuracy, 4),
            "held_out_fact_accuracy": round(self.held_out_fact_accuracy, 4),
            "acquisition_accuracy": round(self.acquisition_accuracy, 4),
            "historical_retention": round(self.historical_retention, 4),
            "temporal_consistency": round(self.temporal_consistency, 4),
            "cue_based_recall": round(self.cue_based_recall, 4),
            "associative_recall_uplift": round(self.associative_recall_uplift, 4),
            "scope_attribution": round(self.scope_attribution, 4),
            "unsupported_claim_rate": round(self.unsupported_claim_rate, 4),
            "claim_label_accuracy": round(self.claim_label_accuracy, 4),
            "abstention_quality": round(self.abstention_quality, 4),
            "evidence_attribution": round(self.evidence_attribution, 4),
            "deletion_leakage": round(self.deletion_leakage, 4),
            "isolation_leakage": round(self.isolation_leakage, 4),
            "serving_latency_p50_ms": round(self.serving_latency_p50_ms, 3),
            "serving_latency_p95_ms": round(self.serving_latency_p95_ms, 3),
            "category_accuracy": {k: round(v, 4) for k, v in self.category_accuracy.items()},
        }


SCORING_METHODS_MANIFEST: dict[str, str] = {
    "acquisition_accuracy": "case_insensitive_substring_all_expected_no_forbidden",
    "historical_retention": "case_insensitive_substring_all_expected_no_forbidden",
    "temporal_consistency": "current_value_present_and_superseded_value_not_asserted_current",
    "cue_based_recall": "case_insensitive_substring_all_expected_no_forbidden",
    "associative_recall_uplift": "paired_win_rate_over_evidence_baseline_on_associative_queries",
    "scope_attribution": "target_value_and_scope_present_and_zero_distractor_values",
    "unsupported_claim_rate": "invented_dates_or_unbacked_supported_claims_or_forbidden_values",
    "claim_label_accuracy": (
        "supported_claims_cite_authorized_active_records_unverified_have_no_dates"
    ),
    "abstention_quality": "exact_abstention_flag_or_marker_on_unanswerable_queries",
    "evidence_attribution": "cited_record_ids_intersect_expected_record_ids",
    "deletion_leakage": "forbidden_deleted_value_present_in_answer",
    "isolation_leakage": "forbidden_cross_scope_or_sensitive_or_cross_tenant_value_present",
}


def _contains_ci(haystack: str, needle: str) -> bool:
    """Case-insensitive substring check."""
    return needle.lower() in haystack.lower()


def _detect_abstention(prediction: SystemPrediction) -> bool:
    """Return True if `prediction` explicitly abstains or has status unknown/no_match."""
    if prediction.abstained:
        return True
    lowered = prediction.answer_text.strip().lower()
    if not lowered:
        return True
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


def score_single_question(
    question: BenchmarkQuestion,
    prediction: SystemPrediction,
    *,
    authorized_active_record_ids: frozenset[UUID],
    model_grader: ModelGrader | None = None,
) -> QuestionScore:
    """Score a single `SystemPrediction` against a `BenchmarkQuestion`."""
    abstained = _detect_abstention(prediction)
    answer_text = prediction.answer_text

    # Check if any forbidden value (superseded decision, distractor project value,
    # deleted value, sensitive value, or cross-tenant value) is asserted as current.
    # For CORRECTION / TEMPORAL questions where the user prompt itself mentions the
    # old value, a response that explicitly frames the old value as "superseded" or
    # "replaced" is allowed provided the current expected value is present.
    has_forbidden = False
    strict_forbidden_categories = {
        QuestionCategory.SCOPE_ATTRIBUTION,
        QuestionCategory.DELETION_PROBE,
        QuestionCategory.ISOLATION_PROBE,
    }
    for forbidden in question.forbidden_values:
        if forbidden and _contains_ci(answer_text, forbidden):
            if question.category not in strict_forbidden_categories:
                # Allow mentioning a superseded value if all expected current values are
                # present and the answer explicitly frames the old value as superseded/replaced.
                all_expected_present = bool(question.expected_values) and all(
                    _contains_ci(answer_text, exp) for exp in question.expected_values
                )
                supersession_framed = any(
                    kw in answer_text.lower()
                    for kw in ("superseded", "replaced", "switched", "instead of", "previously")
                )
                if not (all_expected_present and supersession_framed):
                    has_forbidden = True
            else:
                has_forbidden = True

    # Validate claim labels and check for invented dates/citations
    claim_labels_valid = True
    has_unsupported_claim = False
    for claim in prediction.claims:
        if claim.support == ClaimSupport.SUPPORTED:
            if not claim.evidence or not all(
                eid in authorized_active_record_ids for eid in claim.evidence
            ):
                claim_labels_valid = False
                has_unsupported_claim = True
        elif claim.support == ClaimSupport.UNVERIFIED and _ISO_DATE_RE.search(claim.text):
            # Unverified parametric claims must never invent a precise ISO date (design §6.3)
            claim_labels_valid = False
            has_unsupported_claim = True

    # If the model answered a factual question without abstaining and without any cited
    # evidence, check if it invented an incorrect ISO date.
    if not abstained and question.expected_event_date is not None:
        dates_in_answer = _ISO_DATE_RE.findall(answer_text)
        for d in dates_in_answer:
            if d != question.expected_event_date:
                has_unsupported_claim = True

    if has_forbidden:
        has_unsupported_claim = True

    # Determine correctness
    if question.expect_abstention:
        correct = abstained and not has_forbidden
        evidence_attributed = abstained
    else:
        if abstained or has_forbidden:
            correct = False
        elif model_grader is not None:
            correct = model_grader.grade_fact_match(question, prediction)
        else:
            correct = all(_contains_ci(answer_text, exp) for exp in question.expected_values)

        if question.expected_record_ids:
            cited_set = set(prediction.cited_record_ids) | {
                eid for c in prediction.claims for eid in c.evidence
            }
            evidence_attributed = bool(cited_set.intersection(question.expected_record_ids))
        else:
            evidence_attributed = True

    return QuestionScore(
        question_id=question.question_id,
        category=question.category,
        temporal_slice=question.temporal_slice,
        fact_split=question.fact_split,
        correct=correct,
        abstained=abstained,
        expected_abstention=question.expect_abstention,
        has_forbidden_leakage=has_forbidden,
        has_unsupported_claim=has_unsupported_claim,
        claim_labels_valid=claim_labels_valid,
        evidence_attributed=evidence_attributed,
        latency_ms=prediction.latency_ms,
    )


def _mean_bool(items: Sequence[bool], default: float = 0.0) -> float:
    if not items:
        return default
    return sum(1 for x in items if x) / len(items)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def aggregate_suite_metrics(
    questions: Sequence[BenchmarkQuestion],
    scores: Sequence[QuestionScore],
    *,
    max_slice: int = 2,
    baseline_scores_by_qid: Mapping[str, QuestionScore] | None = None,
    model_grader: ModelGrader | None = None,
) -> SuiteMetrics:
    """Compute all §12 benchmark metrics from per-question scores."""
    score_by_id = {s.question_id: s for s in scores}

    factual_categories = {
        QuestionCategory.DIRECT,
        QuestionCategory.PARAPHRASE,
        QuestionCategory.INCOMPLETE_CUE,
        QuestionCategory.TEMPORAL,
        QuestionCategory.CORRECTION,
        QuestionCategory.SCOPE_ATTRIBUTION,
        QuestionCategory.CROSS_SCOPE,
    }
    factual_scores = [s for s in scores if s.category in factual_categories]

    # Acquisition accuracy: accuracy on the newest slice in the run (`temporal_slice == max_slice`)
    new_slice_scores = [s for s in factual_scores if s.temporal_slice == max_slice]
    acquisition_accuracy = _mean_bool([s.correct for s in new_slice_scores])

    # Historical retention: accuracy on Slice 1 facts
    slice1_scores = [s for s in factual_scores if s.temporal_slice == 1]
    historical_retention = _mean_bool([s.correct for s in slice1_scores])

    # Trained vs Held-out fact split accuracy
    trained_scores = [s for s in factual_scores if s.fact_split == "train"]
    held_out_scores = [s for s in factual_scores if s.fact_split == "held_out"]

    # Temporal consistency
    temporal_scores = [
        s for s in scores if s.category in (QuestionCategory.TEMPORAL, QuestionCategory.CORRECTION)
    ]
    temporal_consistency = _mean_bool([s.correct for s in temporal_scores])

    # Cue-based recall (paraphrase + incomplete_cue)
    cue_scores = [
        s
        for s in scores
        if s.category in (QuestionCategory.PARAPHRASE, QuestionCategory.INCOMPLETE_CUE)
    ]
    cue_based_recall = _mean_bool([s.correct for s in cue_scores])

    # Scope attribution (scope_attribution + cross_scope)
    scope_scores = [
        s
        for s in scores
        if s.category in (QuestionCategory.SCOPE_ATTRIBUTION, QuestionCategory.CROSS_SCOPE)
    ]
    scope_attribution = _mean_bool([s.correct for s in scope_scores])

    # Abstention quality on UNKNOWN questions
    unknown_scores = [s for s in scores if s.category == QuestionCategory.UNKNOWN]
    abstention_quality = _mean_bool([s.correct for s in unknown_scores])

    # Deletion leakage on DELETION_PROBE questions (target: 0.0)
    del_scores = [s for s in scores if s.category == QuestionCategory.DELETION_PROBE]
    deletion_leakage = _mean_bool([s.has_forbidden_leakage for s in del_scores], default=0.0)

    # Isolation leakage on ISOLATION_PROBE questions (target: 0.0)
    iso_scores = [s for s in scores if s.category == QuestionCategory.ISOLATION_PROBE]
    isolation_leakage = _mean_bool([s.has_forbidden_leakage for s in iso_scores], default=0.0)

    # Unsupported claim rate across all questions
    unsupported_claim_rate = _mean_bool([s.has_unsupported_claim for s in scores])

    # Claim-label accuracy
    claim_label_accuracy = _mean_bool([s.claim_labels_valid for s in scores], default=1.0)

    # Evidence attribution on factual questions
    evidence_attribution = _mean_bool([s.evidence_attributed for s in factual_scores])

    # Associative-recall uplift over lexical evidence baseline on associative/cue queries
    associative_qids = [
        q.question_id
        for q in questions
        if q.category
        in (
            QuestionCategory.INCOMPLETE_CUE,
            QuestionCategory.PARAPHRASE,
            QuestionCategory.CROSS_SCOPE,
        )
    ]
    if baseline_scores_by_qid and associative_qids:
        uplift_flags: list[bool] = []
        for qid in associative_qids:
            sys_sc = score_by_id.get(qid)
            base_sc = baseline_scores_by_qid.get(qid)
            if sys_sc is not None and base_sc is not None:
                uplift_flags.append(sys_sc.correct and not base_sc.correct)
        associative_recall_uplift = _mean_bool(uplift_flags, default=0.0)
    else:
        associative_recall_uplift = 0.0

    latencies = sorted(s.latency_ms for s in scores)
    p50 = _percentile(latencies, 0.50)
    p95 = _percentile(latencies, 0.95)

    cat_acc: dict[str, float] = {}
    for cat in QuestionCategory:
        cat_items = [s.correct for s in scores if s.category == cat]
        if cat_items:
            cat_acc[cat.value] = _mean_bool(cat_items)

    methods = dict(SCORING_METHODS_MANIFEST)
    if model_grader is not None:
        methods["model_grader"] = model_grader.grader_version

    return SuiteMetrics(
        scorer_version=SCORER_VERSION,
        scoring_methods=methods,
        total_questions=len(scores),
        overall_accuracy=_mean_bool([s.correct for s in scores]),
        trained_fact_accuracy=_mean_bool([s.correct for s in trained_scores]),
        held_out_fact_accuracy=_mean_bool([s.correct for s in held_out_scores]),
        acquisition_accuracy=acquisition_accuracy,
        historical_retention=historical_retention,
        temporal_consistency=temporal_consistency,
        cue_based_recall=cue_based_recall,
        associative_recall_uplift=associative_recall_uplift,
        scope_attribution=scope_attribution,
        unsupported_claim_rate=unsupported_claim_rate,
        claim_label_accuracy=claim_label_accuracy,
        abstention_quality=abstention_quality,
        evidence_attribution=evidence_attribution,
        deletion_leakage=deletion_leakage,
        isolation_leakage=isolation_leakage,
        serving_latency_p50_ms=p50,
        serving_latency_p95_ms=p95,
        category_accuracy=cat_acc,
    )
