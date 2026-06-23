from __future__ import annotations

from typing import Any


class CostEngine:
    """
    Calculates and compares LLM costs across all supported providers.

    Prices are per 1K tokens (USD) and reflect published pricing as of mid-2025.
    """

    MODEL_PRICING_USD_PER_1K: dict[str, dict[str, float]] = {
        # OpenAI
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        # Anthropic
        "claude-3-5-haiku-latest": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet-latest": {"input": 0.003, "output": 0.015},
        "claude-3-opus-latest": {"input": 0.015, "output": 0.075},
        # Google
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
        # DeepSeek
        "deepseek-chat": {"input": 0.000014, "output": 0.000028},
        "deepseek-reasoner": {"input": 0.00055, "output": 0.00219},
        # Llama (via Groq)
        "llama-3.1-8b-instant": {"input": 0.000059, "output": 0.000079},
        "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
        # Mistral
        "mistral-small-latest": {"input": 0.0002, "output": 0.0006},
        "mistral-medium-latest": {"input": 0.0027, "output": 0.0081},
        "mistral-large-latest": {"input": 0.002, "output": 0.006},
    }

    # Fallback pricing for unknown models
    _DEFAULT_PRICING: dict[str, float] = {"input": 0.0002, "output": 0.0008}

    def calculate_request_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> float:
        pricing = self.MODEL_PRICING_USD_PER_1K.get(model_name, self._DEFAULT_PRICING)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 8)

    def estimate_period_cost(
        self, request_cost: float, daily_requests: int
    ) -> dict[str, float]:
        daily_cost = request_cost * daily_requests
        monthly_cost = daily_cost * 30
        return {
            "cost_per_request": round(request_cost, 8),
            "daily_cost": round(daily_cost, 4),
            "monthly_cost": round(monthly_cost, 4),
        }

    def compare_model_costs(
        self, model_metrics: list[dict[str, Any]], daily_requests: int
    ) -> list[dict[str, Any]]:
        if not model_metrics:
            return []

        baseline = max(model_metrics, key=lambda x: x["average_cost"])
        comparison = []
        for row in model_metrics:
            period = self.estimate_period_cost(
                row["average_cost"], daily_requests=daily_requests
            )
            savings = max(0.0, baseline["average_cost"] - row["average_cost"])
            savings_percent = (
                savings / baseline["average_cost"] * 100
                if baseline["average_cost"]
                else 0.0
            )
            comparison.append(
                {
                    "model": row["model"],
                    "cost_per_request": period["cost_per_request"],
                    "daily_cost": period["daily_cost"],
                    "monthly_cost": period["monthly_cost"],
                    "projected_savings_percent": round(savings_percent, 2),
                }
            )
        return sorted(comparison, key=lambda x: x["monthly_cost"])

    def get_budget_tier(self, model_name: str) -> str:
        """Classify model into budget, mid, or premium tier."""
        pricing = self.MODEL_PRICING_USD_PER_1K.get(model_name, self._DEFAULT_PRICING)
        avg = (pricing["input"] + pricing["output"]) / 2
        if avg < 0.0005:
            return "budget"
        if avg < 0.002:
            return "mid"
        return "premium"

