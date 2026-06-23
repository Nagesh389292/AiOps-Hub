from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MigrationVerdict(str, Enum):
    RECOMMENDED = "Recommended"
    NOT_RECOMMENDED = "Not Recommended"
    CONDITIONAL = "Conditionally Recommended"


@dataclass
class ModelProfile:
    """Snapshot of a model's performance metrics for migration comparison."""
    model_name: str
    accuracy: float            # 0-100
    validation_success_rate: float  # 0-100
    average_latency_ms: float
    average_cost_per_request: float
    monthly_cost_usd: float
    reliability_score: float   # 0-1


class MigrationAdvisor:
    """
    Compares a current production model against a candidate model and
    produces a structured business recommendation.

    Decision logic:
    - Accuracy gain  >= 5%  → strong positive signal
    - Cost increase  >= 30% → negative unless accuracy gain is large
    - Latency increase >= 50% → negative for latency-sensitive workloads
    - Reliability drop  → always negative
    """

    # Thresholds
    MIN_ACCURACY_GAIN_FOR_UPGRADE: float = 3.0        # percentage points
    MAX_TOLERABLE_COST_INCREASE_PCT: float = 30.0     # percent
    MAX_TOLERABLE_LATENCY_INCREASE_PCT: float = 50.0  # percent
    MIN_RELIABILITY_DELTA: float = -0.05              # reliability must not drop more than 5%

    def compare(
        self,
        current: ModelProfile,
        candidate: ModelProfile,
        workload_description: str = "",
        latency_sensitive: bool = False,
        budget_constrained: bool = False,
    ) -> dict[str, Any]:
        """
        Produce a full migration report comparing current vs candidate model.

        Returns a dict suitable for DB storage and dashboard display.
        """
        deltas = self._compute_deltas(current, candidate)
        verdict, reasoning_points = self._evaluate_verdict(
            deltas, latency_sensitive, budget_constrained
        )
        risk_level = self._compute_risk_level(deltas, verdict)
        migration_score = self._compute_migration_score(deltas)

        return {
            "current_model": current.model_name,
            "candidate_model": candidate.model_name,
            "verdict": verdict.value,
            "migration_score": round(migration_score, 2),
            "risk_level": risk_level,
            "workload_description": workload_description,
            "deltas": {
                "accuracy_delta_pct": round(deltas["accuracy_delta"], 2),
                "validation_rate_delta_pct": round(deltas["validation_delta"], 2),
                "latency_delta_ms": round(deltas["latency_delta"], 2),
                "latency_change_pct": round(deltas["latency_change_pct"], 2),
                "cost_delta_usd": round(deltas["cost_delta"], 8),
                "cost_change_pct": round(deltas["cost_change_pct"], 2),
                "monthly_cost_delta_usd": round(deltas["monthly_cost_delta"], 2),
                "reliability_delta": round(deltas["reliability_delta"], 4),
            },
            "current_metrics": {
                "model": current.model_name,
                "accuracy": round(current.accuracy, 2),
                "validation_success_rate": round(current.validation_success_rate, 2),
                "average_latency_ms": round(current.average_latency_ms, 2),
                "cost_per_request": round(current.average_cost_per_request, 8),
                "monthly_cost_usd": round(current.monthly_cost_usd, 2),
                "reliability_score": round(current.reliability_score, 4),
            },
            "candidate_metrics": {
                "model": candidate.model_name,
                "accuracy": round(candidate.accuracy, 2),
                "validation_success_rate": round(candidate.validation_success_rate, 2),
                "average_latency_ms": round(candidate.average_latency_ms, 2),
                "cost_per_request": round(candidate.average_cost_per_request, 8),
                "monthly_cost_usd": round(candidate.monthly_cost_usd, 2),
                "reliability_score": round(candidate.reliability_score, 4),
            },
            "reasoning": reasoning_points,
            "action_items": self._generate_action_items(verdict, deltas, current, candidate),
        }

    def _compute_deltas(self, current: ModelProfile, candidate: ModelProfile) -> dict[str, float]:
        accuracy_delta = candidate.accuracy - current.accuracy
        validation_delta = candidate.validation_success_rate - current.validation_success_rate
        latency_delta = candidate.average_latency_ms - current.average_latency_ms
        latency_change_pct = (
            (latency_delta / max(0.001, current.average_latency_ms)) * 100
        )
        cost_delta = candidate.average_cost_per_request - current.average_cost_per_request
        cost_change_pct = (
            (cost_delta / max(0.0000001, current.average_cost_per_request)) * 100
        )
        monthly_cost_delta = candidate.monthly_cost_usd - current.monthly_cost_usd
        reliability_delta = candidate.reliability_score - current.reliability_score

        return {
            "accuracy_delta": accuracy_delta,
            "validation_delta": validation_delta,
            "latency_delta": latency_delta,
            "latency_change_pct": latency_change_pct,
            "cost_delta": cost_delta,
            "cost_change_pct": cost_change_pct,
            "monthly_cost_delta": monthly_cost_delta,
            "reliability_delta": reliability_delta,
        }

    def _evaluate_verdict(
        self,
        deltas: dict[str, float],
        latency_sensitive: bool,
        budget_constrained: bool,
    ) -> tuple[MigrationVerdict, list[str]]:
        positives: list[str] = []
        negatives: list[str] = []
        conditions: list[str] = []

        acc = deltas["accuracy_delta"]
        cost_pct = deltas["cost_change_pct"]
        lat_pct = deltas["latency_change_pct"]
        rel = deltas["reliability_delta"]
        monthly_delta = deltas["monthly_cost_delta"]

        # Accuracy signals
        if acc >= self.MIN_ACCURACY_GAIN_FOR_UPGRADE:
            positives.append(
                f"Accuracy improves by {acc:.1f}pp — candidate is meaningfully better."
            )
        elif acc > 0:
            positives.append(f"Minor accuracy gain of {acc:.1f}pp.")
        elif acc < -2:
            negatives.append(
                f"Accuracy drops {abs(acc):.1f}pp — candidate performs worse."
            )

        # Reliability
        if rel >= 0.05:
            positives.append(f"Reliability improves by {rel:.2%}.")
        elif rel < self.MIN_RELIABILITY_DELTA:
            negatives.append(
                f"Reliability degrades by {abs(rel):.2%} — production risk."
            )

        # Cost signals
        if cost_pct > self.MAX_TOLERABLE_COST_INCREASE_PCT:
            msg = (
                f"Cost increases {cost_pct:.1f}% (${monthly_delta:+.2f}/month). "
                f"Exceeds the {self.MAX_TOLERABLE_COST_INCREASE_PCT}% tolerance threshold."
            )
            if budget_constrained or cost_pct > 100.0:
                # Large cost spike always treated as a hard negative
                negatives.append(msg)
            else:
                conditions.append(msg + " Acceptable only with significant accuracy gains.")
        elif cost_pct < -10:
            positives.append(
                f"Cost reduces by {abs(cost_pct):.1f}% (saves ${abs(monthly_delta):.2f}/month)."
            )
        elif cost_pct < 0:
            positives.append(f"Slight cost reduction of {abs(cost_pct):.1f}%.")

        # Latency signals
        if lat_pct > self.MAX_TOLERABLE_LATENCY_INCREASE_PCT:
            msg = f"Latency increases {lat_pct:.1f}% — may degrade user experience."
            if latency_sensitive:
                negatives.append(msg)
            else:
                conditions.append(msg + " Acceptable for async/batch workloads.")
        elif lat_pct < -15:
            positives.append(f"Latency reduces by {abs(lat_pct):.1f}%.")

        # Compute verdict
        if len(negatives) >= 2:
            verdict = MigrationVerdict.NOT_RECOMMENDED
        elif len(negatives) == 1 and len(positives) == 0:
            verdict = MigrationVerdict.NOT_RECOMMENDED
        elif conditions and len(positives) < 2:
            verdict = MigrationVerdict.CONDITIONAL
        elif len(positives) >= 2 and len(negatives) == 0:
            verdict = MigrationVerdict.RECOMMENDED
        elif len(positives) >= 1 and len(conditions) == 0 and len(negatives) == 0:
            verdict = MigrationVerdict.CONDITIONAL
        else:
            verdict = MigrationVerdict.CONDITIONAL

        all_reasons = (
            [f"[+] {p}" for p in positives]
            + [f"[-] {n}" for n in negatives]
            + [f"[~] {c}" for c in conditions]
        )
        return verdict, all_reasons

    @staticmethod
    def _compute_risk_level(deltas: dict[str, float], verdict: MigrationVerdict) -> str:
        if verdict == MigrationVerdict.NOT_RECOMMENDED:
            return "High"
        if verdict == MigrationVerdict.CONDITIONAL:
            return "Medium"
        if deltas["cost_change_pct"] > 20 or deltas["latency_change_pct"] > 30:
            return "Medium"
        return "Low"

    def _compute_migration_score(self, deltas: dict[str, float]) -> float:
        """
        Score from -100 (very bad migration) to +100 (very good migration).
        """
        score = 0.0
        # Accuracy (weight 40)
        score += min(40.0, deltas["accuracy_delta"] * 4)
        # Reliability (weight 20)
        score += min(20.0, deltas["reliability_delta"] * 200)
        # Cost (weight 25): negative change is good
        cost_score = -deltas["cost_change_pct"] * 0.25
        score += max(-25.0, min(25.0, cost_score))
        # Latency (weight 15): negative change is good
        lat_score = -deltas["latency_change_pct"] * 0.15
        score += max(-15.0, min(15.0, lat_score))
        return round(max(-100.0, min(100.0, score)), 2)

    @staticmethod
    def _generate_action_items(
        verdict: MigrationVerdict,
        deltas: dict[str, float],
        current: ModelProfile,
        candidate: ModelProfile,
    ) -> list[str]:
        actions: list[str] = []

        if verdict == MigrationVerdict.RECOMMENDED:
            actions.append(f"Plan migration from {current.model_name} to {candidate.model_name}.")
            actions.append("Run a 2-week shadow-mode evaluation before full cutover.")
            actions.append("Update API keys and model names in configuration.")
            if deltas["cost_change_pct"] > 0:
                actions.append(
                    f"Budget an additional ${abs(deltas['monthly_cost_delta']):.2f}/month for the upgrade."
                )

        elif verdict == MigrationVerdict.CONDITIONAL:
            actions.append(
                f"Pilot {candidate.model_name} on a subset of traffic (10-20%) for 2 weeks."
            )
            actions.append("Measure live accuracy and latency before deciding on full migration.")
            if deltas["cost_change_pct"] > MigrationAdvisor.MAX_TOLERABLE_COST_INCREASE_PCT:
                actions.append("Negotiate volume pricing with provider before committing.")

        else:  # NOT_RECOMMENDED
            actions.append(f"Remain on {current.model_name} for now.")
            actions.append(
                f"Re-evaluate {candidate.model_name} when a new version or price drop is available."
            )
            if deltas["accuracy_delta"] < 0:
                actions.append(
                    "Investigate fine-tuning options for the current model to improve accuracy."
                )

        return actions

    def compare_from_leaderboard(
        self,
        current_model_key: str,
        candidate_model_key: str,
        leaderboard: list[dict[str, Any]],
        daily_requests: int = 10000,
        latency_sensitive: bool = False,
        budget_constrained: bool = False,
    ) -> dict[str, Any]:
        """
        Convenience method that builds ModelProfiles from leaderboard data
        (as returned by BenchmarkEngine) and runs a full comparison.
        """
        def find_row(key: str) -> dict[str, Any] | None:
            for row in leaderboard:
                if key.lower() in row.get("model", "").lower():
                    return row
            return None

        cur_row = find_row(current_model_key)
        cand_row = find_row(candidate_model_key)

        if not cur_row or not cand_row:
            missing = []
            if not cur_row:
                missing.append(current_model_key)
            if not cand_row:
                missing.append(candidate_model_key)
            return {
                "error": f"Model(s) not found in leaderboard: {', '.join(missing)}",
                "verdict": MigrationVerdict.NOT_RECOMMENDED.value,
            }

        def make_profile(row: dict[str, Any]) -> ModelProfile:
            avg_cost = row.get("average_cost", 0.0)
            return ModelProfile(
                model_name=row["model"],
                accuracy=row.get("accuracy", 0.0),
                validation_success_rate=row.get("validation_success_rate", 0.0),
                average_latency_ms=row.get("average_latency_ms", 0.0),
                average_cost_per_request=avg_cost,
                monthly_cost_usd=avg_cost * daily_requests * 30,
                reliability_score=row.get("validation_success_rate", 0.0) / 100,
            )

        current_profile = make_profile(cur_row)
        candidate_profile = make_profile(cand_row)

        return self.compare(
            current=current_profile,
            candidate=candidate_profile,
            workload_description=f"Benchmark-derived comparison ({daily_requests:,} daily requests)",
            latency_sensitive=latency_sensitive,
            budget_constrained=budget_constrained,
        )
