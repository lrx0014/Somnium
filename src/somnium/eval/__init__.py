"""Synthetic benchmark generation, evaluation harness, metrics, and baselines (M1)."""

from __future__ import annotations

from somnium.eval.baselines import (
    AssociativeAdapterSystem,
    BaseModelNoAdapterBaseline,
    FullContextBaseline,
    StandaloneFtsBaseline,
    SystemUnderTest,
)
from somnium.eval.harness import (
    BenchmarkRunResult,
    evaluate_system_on_world,
    resolve_builtin_system,
    run_benchmark_cli,
)
from somnium.eval.questions import (
    BenchmarkQuestion,
    DatasetManifest,
    QuestionCategory,
    TrainingExample,
    build_training_dataset,
    generate_benchmark_questions,
)
from somnium.eval.scoring import (
    SCORER_VERSION,
    ModelGrader,
    QuestionScore,
    SuiteMetrics,
    SystemPrediction,
    aggregate_suite_metrics,
    score_single_question,
)
from somnium.eval.world import (
    ALL_PRIMARY_SCOPES,
    PRIMARY_TENANT,
    SECONDARY_TENANT,
    FactBlueprint,
    SyntheticWorld,
    generate_synthetic_world,
)

__all__ = [
    "ALL_PRIMARY_SCOPES",
    "AssociativeAdapterSystem",
    "BaseModelNoAdapterBaseline",
    "BenchmarkQuestion",
    "BenchmarkRunResult",
    "DatasetManifest",
    "FactBlueprint",
    "FullContextBaseline",
    "ModelGrader",
    "PRIMARY_TENANT",
    "QuestionCategory",
    "QuestionScore",
    "SCORER_VERSION",
    "SECONDARY_TENANT",
    "StandaloneFtsBaseline",
    "SuiteMetrics",
    "SyntheticWorld",
    "SystemPrediction",
    "SystemUnderTest",
    "TrainingExample",
    "aggregate_suite_metrics",
    "build_training_dataset",
    "evaluate_system_on_world",
    "generate_benchmark_questions",
    "generate_synthetic_world",
    "resolve_builtin_system",
    "run_benchmark_cli",
    "score_single_question",
]
