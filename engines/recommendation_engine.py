from __future__ import annotations

from typing import Any

from config import settings


# Business workload profiles: which properties matter most per use-case
_WORKLOAD_PROFILES: dict[str, dict[str, float]] = {
    "coding": {
        "accuracy": 0.50, "validation": 0.30, "latency": 0.10, "cost": 0.10
    },
    "sql": {
        "accuracy": 0.50, "validation": 0.30, "latency": 0.10, "cost": 0.10
    },
    "support": {
        "accuracy": 0.35, "validation": 0.25, "latency": 0.20, "cost": 0.20
    },
    "summarization": {
        "accuracy": 0.35, "validation": 0.20, "latency": 0.15, "cost": 0.30
    },
    "reasoning": {
        "accuracy": 0.50, "validation": 0.25, "latency": 0.15, "cost": 0.10
    },
    "general": {
        "accuracy": 0.40, "validation": 0.30, "latency": 0.15, "cost": 0.15
    },
}

# Known strengths per model — used for narrative business recommendations
_MODEL_STRENGTHS: dict[str, str] = {
    "gpt": "GPT excels at complex reasoning, coding, and nuanced instruction-following.",
    "gpt-4o": "GPT-4o provides frontier performance with multimodal capabilities.",
    "gpt-4o-mini": "GPT-4o-mini delivers strong coding quality at low cost.",
    "claude": "Claude is optimised for safety, customer-facing text, and document analysis.",
    "claude-3-5-haiku-latest": "Claude Haiku is ideal for high-volume, cost-sensitive customer support.",
    "claude-3-5-sonnet-latest": "Claude Sonnet balances intelligence and throughput for enterprise workflows.",
    "gemini": "Gemini Flash is Google's fastest and most cost-efficient model.",
    "gemini-1.5-flash": "Gemini Flash offers the best cost-per-token for batch summarization tasks.",
    "deepseek": "DeepSeek offers exceptional cost efficiency for bulk inference at massive scale.",
    "deepseek-chat": "DeepSeek-Chat is the most affordable option for simple classification and extraction.",
    "llama": "Llama provides open-weight inference with no vendor lock-in.",
    "llama-3.1-8b-instant": "Llama 3.1 8B is extremely fast via Groq for latency-critical applications.",
    "mistral": "Mistral delivers strong European-model performance with competitive pricing.",
    "mistral-small-latest": "Mistral Small is suitable for medium-complexity tasks at mid-range cost.",
}


class RecommendationEngine:
    def __init__(
        self,
        wa: float = settings.recommendation_weight_accuracy,
        wv: float = settings.recommendation_weight_validation,
        wl: float = settings.recommendation_weight_latency,
        wc: float = settings.recommendation_weight_cost,
    ) -> None:
        self.wa = wa
        self.wv = wv
        self.wl = wl
        self.wc = wc

    @staticmethod
    def _minmax(value: float, min_v: float, max_v: float) -> float:
        if max_v == min_v:
            return 0.0
        return (value - min_v) / (max_v - min_v)

    def recommend(
        self,
        leaderboard: list[dict[str, Any]],
        workload: str = "general",
    ) -> dict[str, Any]:
        if not leaderboard:
            return {
                "best_coding_model": "N/A",
                "best_cost_efficient_model": "N/A",
                "best_overall_model": "N/A",
                "reason": "No benchmark data available",
                "scored_rows": [],
                "workload": workload,
                "workload_recommendations": self._static_workload_recommendations(),
            }

        # Pick workload weights
        profile = _WORKLOAD_PROFILES.get(workload, _WORKLOAD_PROFILES["general"])
        wa = profile["accuracy"]
        wv = profile["validation"]
        wl = profile["latency"]
        wc = profile["cost"]

        latencies = [row["average_latency_ms"] for row in leaderboard]
        costs = [row["average_cost"] for row in leaderboard]

        min_lat, max_lat = min(latencies), max(latencies)
        min_cost, max_cost = min(costs), max(costs)

        scored = []
        for row in leaderboard:
            latency_norm = self._minmax(row["average_latency_ms"], min_lat, max_lat)
            cost_norm = self._minmax(row["average_cost"], min_cost, max_cost)
            accuracy_norm = row["accuracy"] / 100
            validation_norm = row.get("validation_success_rate", row.get("accuracy", 0)) / 100

            score = (
                wa * accuracy_norm
                + wv * validation_norm
                - wl * latency_norm
                - wc * cost_norm
            )

            eligible = (
                row["accuracy"] >= settings.min_required_accuracy
                and row.get("validation_success_rate", row["accuracy"])
                >= settings.min_required_validation_success
                and row["average_latency_ms"] <= settings.max_allowed_average_latency_ms
            )

            scored_row = dict(row)
            scored_row["overall_score"] = round(score, 4)
            scored_row["eligible_for_production"] = eligible
            scored.append(scored_row)

        best_quality = max(scored, key=lambda x: x["accuracy"])
        best_cost = min(scored, key=lambda x: x["average_cost"])
        eligible_rows = [r for r in scored if r["eligible_for_production"]]
        best_overall = (
            max(eligible_rows, key=lambda x: x["overall_score"])
            if eligible_rows
            else max(scored, key=lambda x: x["overall_score"])
        )

        strength = _MODEL_STRENGTHS.get(
            best_overall["model"],
            _MODEL_STRENGTHS.get(
                next(
                    (k for k in _MODEL_STRENGTHS if k in best_overall["model"].lower()),
                    "",
                ),
                f"{best_overall['model']} is the top-ranked model for this benchmark.",
            ),
        )

        reason = (
            f"{best_overall['model']} is recommended for {workload} workloads. "
            f"Score: {best_overall['accuracy']:.1f}% accuracy, "
            f"${best_overall['average_cost']:.6f} avg cost/request. {strength}"
        )

        if not eligible_rows:
            reason += " No model passed all production thresholds — top scorer selected as fallback."

        return {
            "best_coding_model": best_quality["model"],
            "best_cost_efficient_model": best_cost["model"],
            "best_overall_model": best_overall["model"],
            "reason": reason,
            "workload": workload,
            "scored_rows": sorted(scored, key=lambda x: x["overall_score"], reverse=True),
            "workload_recommendations": self._static_workload_recommendations(),
        }

    @staticmethod
    def _static_workload_recommendations() -> dict[str, dict[str, str]]:
        """
        Business-oriented workload routing guide.
        Returns recommendations even before any benchmark data is available.
        """
        return {
            "coding": {
                "recommended_model": "GPT-4o-mini / Claude Sonnet",
                "rationale": "Complex multi-step code generation requires high reasoning accuracy.",
                "cost_tier": "mid",
            },
            "sql": {
                "recommended_model": "GPT-4o-mini",
                "rationale": "SQL generation benefits from precise instruction-following.",
                "cost_tier": "mid",
            },
            "customer_support": {
                "recommended_model": "Claude Haiku",
                "rationale": "Claude is optimised for safe, empathetic customer-facing responses.",
                "cost_tier": "budget-mid",
            },
            "bulk_summarization": {
                "recommended_model": "Gemini Flash / DeepSeek",
                "rationale": "High-volume document processing demands the lowest cost per token.",
                "cost_tier": "budget",
            },
            "reasoning": {
                "recommended_model": "GPT-4o / Claude Sonnet",
                "rationale": "Multi-step logical reasoning requires frontier model capability.",
                "cost_tier": "premium",
            },
            "budget_workloads": {
                "recommended_model": "DeepSeek-Chat / Llama 3.1 8B",
                "rationale": "For cost-sensitive batch inference where accuracy can be relaxed.",
                "cost_tier": "budget",
            },
        }

