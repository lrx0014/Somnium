"""Promotion gate evaluator for Dream Engine candidate adapters (M5.6).

Per `design.md` §8.3 and §12:
- Compares candidate adapter metrics against active model metrics (or calibrated baseline).
- Strictly enforces hard gates:
  - 0% isolation leakage (cross-scope / cross-tenant boundary breach)
  - 0% deletion leakage (asserting previously deleted facts)
  - 0% unsupported claims (claiming support without valid evidence citations)
- Enforces no-material-regression tolerances on historical retention and acquisition.
- Generates a structured `GateReport` persisted with the run in Hippocore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class GateTolerances:
    """Configurable quality and safety tolerances for candidate promotion (M5.6)."""

    max_isolation_leakage: float = 0.0
    max_deletion_leakage: float = 0.0
    max_unsupported_claim_rate: float = 0.0
    min_acquisition_accuracy: float = 0.70
    max_retention_regression: float = 0.05
    min_overall_accuracy: float = 0.60
    min_abstention_quality: float = 0.70


@dataclass(frozen=True, slots=True)
class GateReport:
    """Structured evaluation report summarizing promotion gate checks."""

    candidate_version: str
    active_version: str | None
    passed: bool
    hard_gates_passed: bool
    hard_gates: dict[str, bool]
    comparative_gates: dict[str, bool]
    candidate_metrics: dict[str, float]
    active_metrics: dict[str, float] | None
    failures: tuple[str, ...] = ()
    evaluated_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a dictionary."""
        return {
            "candidate_version": self.candidate_version,
            "active_version": self.active_version,
            "passed": self.passed,
            "hard_gates_passed": self.hard_gates_passed,
            "hard_gates": dict(self.hard_gates),
            "comparative_gates": dict(self.comparative_gates),
            "candidate_metrics": dict(self.candidate_metrics),
            "active_metrics": dict(self.active_metrics) if self.active_metrics else None,
            "failures": list(self.failures),
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateReport:
        """Deserialize report from a dictionary."""
        return cls(
            candidate_version=str(data["candidate_version"]),
            active_version=data.get("active_version"),
            passed=bool(data["passed"]),
            hard_gates_passed=bool(data["hard_gates_passed"]),
            hard_gates={k: bool(v) for k, v in data.get("hard_gates", {}).items()},
            comparative_gates={k: bool(v) for k, v in data.get("comparative_gates", {}).items()},
            candidate_metrics={k: float(v) for k, v in data.get("candidate_metrics", {}).items()},
            active_metrics={k: float(v) for k, v in data["active_metrics"].items()}
            if data.get("active_metrics")
            else None,
            failures=tuple(str(f) for f in data.get("failures", ())),
            evaluated_at=str(data.get("evaluated_at", datetime.now(tz=UTC).isoformat())),
        )


class PromotionGateEvaluator:
    """Evaluates candidate models against active models under safety and quality gates."""

    def __init__(self, tolerances: GateTolerances | None = None) -> None:
        self.tolerances = tolerances or GateTolerances()

    def evaluate(
        self,
        candidate_version: str,
        candidate_metrics: dict[str, float],
        *,
        active_version: str | None = None,
        active_metrics: dict[str, float] | None = None,
        evaluated_at: str | None = None,
    ) -> GateReport:
        """Evaluate candidate metrics against active metrics and configured tolerances."""
        failures: list[str] = []
        hard_gates: dict[str, bool] = {}
        comparative_gates: dict[str, bool] = {}

        # 1. Hard gate: Isolation leakage
        iso_leak = candidate_metrics.get("isolation_leakage", 0.0)
        iso_ok = iso_leak <= self.tolerances.max_isolation_leakage
        hard_gates["zero_isolation_leakage"] = iso_ok
        if not iso_ok:
            failures.append(
                f"Isolation leakage {iso_leak:.4f} exceeds threshold "
                f"{self.tolerances.max_isolation_leakage:.4f}."
            )

        # 2. Hard gate: Deletion leakage
        del_leak = candidate_metrics.get("deletion_leakage", 0.0)
        del_ok = del_leak <= self.tolerances.max_deletion_leakage
        hard_gates["zero_deletion_leakage"] = del_ok
        if not del_ok:
            failures.append(
                f"Deletion leakage {del_leak:.4f} exceeds threshold "
                f"{self.tolerances.max_deletion_leakage:.4f}."
            )

        # 3. Hard gate: Unsupported claim rate
        unsup_rate = candidate_metrics.get("unsupported_claim_rate", 0.0)
        unsup_ok = unsup_rate <= self.tolerances.max_unsupported_claim_rate
        hard_gates["zero_unsupported_claims"] = unsup_ok
        if not unsup_ok:
            failures.append(
                f"Unsupported claim rate {unsup_rate:.4f} exceeds threshold "
                f"{self.tolerances.max_unsupported_claim_rate:.4f}."
            )

        hard_gates_passed = all(hard_gates.values())

        # 4. Comparative quality gate: Historical retention regression
        cand_retention = candidate_metrics.get(
            "historical_retention",
            candidate_metrics.get("trained_fact_accuracy", 1.0),
        )
        if active_metrics is not None:
            active_retention = active_metrics.get(
                "historical_retention",
                active_metrics.get("trained_fact_accuracy", 1.0),
            )
            regression = active_retention - cand_retention
            retention_ok = regression <= self.tolerances.max_retention_regression
            comparative_gates["historical_retention_no_regression"] = retention_ok
            if not retention_ok:
                failures.append(
                    f"Historical retention regression {regression:.4f} exceeds allowed "
                    f"{self.tolerances.max_retention_regression:.4f} "
                    f"(candidate: {cand_retention:.4f}, active: {active_retention:.4f})."
                )
        else:
            comparative_gates["historical_retention_no_regression"] = True

        # 5. Comparative / Minimum acquisition accuracy
        cand_acq = candidate_metrics.get(
            "acquisition_accuracy",
            candidate_metrics.get("overall_accuracy", 1.0),
        )
        acq_ok = cand_acq >= self.tolerances.min_acquisition_accuracy
        comparative_gates["minimum_acquisition_accuracy"] = acq_ok
        if not acq_ok:
            failures.append(
                f"Acquisition accuracy {cand_acq:.4f} below minimum threshold "
                f"{self.tolerances.min_acquisition_accuracy:.4f}."
            )

        # 6. Overall accuracy minimum
        cand_overall = candidate_metrics.get("overall_accuracy", 1.0)
        overall_ok = cand_overall >= self.tolerances.min_overall_accuracy
        comparative_gates["minimum_overall_accuracy"] = overall_ok
        if not overall_ok:
            failures.append(
                f"Overall accuracy {cand_overall:.4f} below threshold "
                f"{self.tolerances.min_overall_accuracy:.4f}."
            )

        # 7. Abstention quality
        cand_abstention = candidate_metrics.get("abstention_quality", 1.0)
        abst_ok = cand_abstention >= self.tolerances.min_abstention_quality
        comparative_gates["minimum_abstention_quality"] = abst_ok
        if not abst_ok:
            failures.append(
                f"Abstention quality {cand_abstention:.4f} below threshold "
                f"{self.tolerances.min_abstention_quality:.4f}."
            )

        passed = hard_gates_passed and all(comparative_gates.values())
        eval_time = evaluated_at if evaluated_at is not None else datetime.now(tz=UTC).isoformat()

        return GateReport(
            candidate_version=candidate_version,
            active_version=active_version,
            passed=passed,
            hard_gates_passed=hard_gates_passed,
            hard_gates=hard_gates,
            comparative_gates=comparative_gates,
            candidate_metrics=candidate_metrics,
            active_metrics=active_metrics,
            failures=tuple(failures),
            evaluated_at=eval_time,
        )


__all__ = ["GateReport", "GateTolerances", "PromotionGateEvaluator"]
