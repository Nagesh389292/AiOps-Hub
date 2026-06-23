from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ComplexityTier(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class RoutingDecision:
    complexity: ComplexityTier
    recommended_model: str
    routing_reason: str
    estimated_input_tokens: int
    confidence: float


# Routing table: maps complexity → preferred model key (must match BenchmarkEngine runners)
_ROUTING_TABLE: dict[ComplexityTier, list[str]] = {
    ComplexityTier.SIMPLE: ["deepseek", "gemini"],
    ComplexityTier.MEDIUM: ["claude", "mistral"],
    ComplexityTier.COMPLEX: ["gpt", "claude"],
}

# Signals that indicate complex tasks
_COMPLEX_SIGNALS = re.compile(
    r"\b(multi[\s-]?step|chain[\s-]?of[\s-]?thought|compare\s+\w+\s+and\s+\w+|"
    r"optimize|trade[\s-]?off|architecture|design\s+pattern|refactor|"
    r"distributed|concurrency|algorithm|complexity\s+analysis|"
    r"implement\s+\w+\s+from\s+scratch|explain\s+in\s+detail|"
    r"pros\s+and\s+cons|cost[\s-]?benefit|regulatory|compliance|"
    r"sql.*join.*subquery|recursive|dynamic\s+programming)\b",
    re.IGNORECASE,
)

_MEDIUM_SIGNALS = re.compile(
    r"\b(summarize|classify|categorize|translate|convert|format|"
    r"fix\s+the\s+bug|debug|review|customer\s+support|ticket|"
    r"extract|parse|list\s+the|enumerate|write\s+a\s+function|"
    r"explain\s+briefly|simple\s+sql|basic\s+query)\b",
    re.IGNORECASE,
)

_SIMPLE_SIGNALS = re.compile(
    r"\b(what\s+is|define|yes\s+or\s+no|true\s+or\s+false|"
    r"single[\s-]?word|one[\s-]?sentence|hello|ping|status|"
    r"count|list\s+names|simple\s+math|multiply|add|subtract|divide)\b",
    re.IGNORECASE,
)


class QueryComplexityAnalyzer:
    """
    Classifies an incoming query into Simple, Medium, or Complex
    using token-count heuristics and keyword signal detection.

    Simple  → short prompts, factual lookups, basic math
    Medium  → moderate tasks, classification, debugging, short summaries
    Complex → multi-step reasoning, architecture, algorithm design, long context
    """

    SIMPLE_TOKEN_THRESHOLD: int = 40
    MEDIUM_TOKEN_THRESHOLD: int = 200

    def classify(self, prompt: str) -> tuple[ComplexityTier, float]:
        """Returns (complexity_tier, confidence_0_to_1)."""
        token_count = max(1, len(prompt.split()))

        complex_hits = len(_COMPLEX_SIGNALS.findall(prompt))
        medium_hits = len(_MEDIUM_SIGNALS.findall(prompt))
        simple_hits = len(_SIMPLE_SIGNALS.findall(prompt))

        # Hard rules first
        if token_count > self.MEDIUM_TOKEN_THRESHOLD or complex_hits >= 2:
            return ComplexityTier.COMPLEX, min(1.0, 0.6 + complex_hits * 0.1)
        if token_count > self.SIMPLE_TOKEN_THRESHOLD or medium_hits >= 1:
            return ComplexityTier.MEDIUM, min(1.0, 0.55 + medium_hits * 0.1)
        if simple_hits >= 1 or token_count <= self.SIMPLE_TOKEN_THRESHOLD:
            return ComplexityTier.SIMPLE, min(1.0, 0.65 + simple_hits * 0.1)

        # Fallback based on token count alone
        if token_count <= self.SIMPLE_TOKEN_THRESHOLD:
            return ComplexityTier.SIMPLE, 0.5
        if token_count <= self.MEDIUM_TOKEN_THRESHOLD:
            return ComplexityTier.MEDIUM, 0.5
        return ComplexityTier.COMPLEX, 0.5


class RoutingEngine:
    """
    Routes incoming requests to the most cost-effective model for their complexity.

    Routing strategy:
        Simple  → DeepSeek (cheapest) or Gemini Flash
        Medium  → Claude Haiku or Mistral Small
        Complex → GPT-4o-mini or Claude Sonnet

    The engine also calculates projected cost savings versus always routing to GPT.
    """

    # Cost per 1K tokens (USD) — kept in sync with CostEngine
    MODEL_PRICING: dict[str, dict[str, float]] = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "claude-3-5-haiku-latest": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet-latest": {"input": 0.003, "output": 0.015},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "deepseek-chat": {"input": 0.000014, "output": 0.000028},
        "llama-3.1-8b-instant": {"input": 0.000059, "output": 0.000079},
        "mistral-small-latest": {"input": 0.0002, "output": 0.0006},
    }

    # Default model name per runner key
    RUNNER_TO_MODEL: dict[str, str] = {
        "gpt": "gpt-4o-mini",
        "claude": "claude-3-5-haiku-latest",
        "gemini": "gemini-1.5-flash",
        "deepseek": "deepseek-chat",
        "llama": "llama-3.1-8b-instant",
        "mistral": "mistral-small-latest",
    }

    def __init__(self, available_runners: list[str] | None = None) -> None:
        self.analyzer = QueryComplexityAnalyzer()
        # Runners that are currently initialised and available
        self.available_runners: set[str] = set(available_runners or list(self.RUNNER_TO_MODEL.keys()))

    def route(self, prompt: str) -> RoutingDecision:
        """Classify prompt and return the best model routing decision."""
        complexity, confidence = self.analyzer.classify(prompt)
        candidates = _ROUTING_TABLE[complexity]

        # Pick first available candidate
        chosen_runner = next(
            (r for r in candidates if r in self.available_runners),
            "gpt",  # safe fallback
        )
        model_name = self.RUNNER_TO_MODEL.get(chosen_runner, "gpt-4o-mini")
        token_estimate = max(1, len(prompt.split()))

        reasons = {
            ComplexityTier.SIMPLE: f"Simple query ({token_estimate} tokens) — routing to cost-efficient {model_name}.",
            ComplexityTier.MEDIUM: f"Medium complexity query — routing to balanced {model_name}.",
            ComplexityTier.COMPLEX: f"Complex query requiring high reasoning — routing to capable {model_name}.",
        }

        return RoutingDecision(
            complexity=complexity,
            recommended_model=chosen_runner,
            routing_reason=reasons[complexity],
            estimated_input_tokens=token_estimate,
            confidence=round(confidence, 4),
        )

    def simulate_routing_savings(
        self,
        request_distribution: dict[str, int],  # {"simple": N, "medium": N, "complex": N}
        baseline_model: str = "gpt",
    ) -> dict[str, Any]:
        """
        Compare costs of always-GPT vs smart routing across a request distribution.

        Args:
            request_distribution: {"simple": 5000, "medium": 3000, "complex": 2000}
            baseline_model: runner key of the always-on expensive model

        Returns:
            dict with baseline_monthly_cost, optimised_monthly_cost, savings_usd, savings_pct,
            and per-tier breakdown.
        """
        baseline_model_name = self.RUNNER_TO_MODEL.get(baseline_model, "gpt-4o-mini")
        baseline_pricing = self.MODEL_PRICING.get(baseline_model_name, {"input": 0.00015, "output": 0.0006})

        # Assume ~200 input tokens + ~500 output tokens per request (typical)
        avg_input = 200
        avg_output = 500

        def cost_per_request(model_name: str) -> float:
            p = self.MODEL_PRICING.get(model_name, baseline_pricing)
            return (avg_input / 1000 * p["input"]) + (avg_output / 1000 * p["output"])

        baseline_cpr = cost_per_request(baseline_model_name)
        total_requests = sum(request_distribution.values())
        baseline_daily = baseline_cpr * total_requests
        baseline_monthly = baseline_daily * 30

        optimised_daily = 0.0
        tier_breakdown: list[dict[str, Any]] = []

        for tier_name, count in request_distribution.items():
            try:
                tier = ComplexityTier(tier_name)
            except ValueError:
                continue
            candidates = _ROUTING_TABLE[tier]
            chosen_runner = next(
                (r for r in candidates if r in self.available_runners), baseline_model
            )
            chosen_model = self.RUNNER_TO_MODEL.get(chosen_runner, baseline_model_name)
            cpr = cost_per_request(chosen_model)
            tier_daily = cpr * count

            tier_breakdown.append({
                "tier": tier_name,
                "request_count": count,
                "routed_to": chosen_model,
                "cost_per_request_usd": round(cpr, 8),
                "daily_cost_usd": round(tier_daily, 4),
                "monthly_cost_usd": round(tier_daily * 30, 2),
            })
            optimised_daily += tier_daily

        optimised_monthly = optimised_daily * 30
        savings_usd = max(0.0, baseline_monthly - optimised_monthly)
        savings_pct = (savings_usd / baseline_monthly * 100) if baseline_monthly > 0 else 0.0

        return {
            "total_daily_requests": total_requests,
            "baseline_model": baseline_model_name,
            "baseline_monthly_cost_usd": round(baseline_monthly, 2),
            "optimised_monthly_cost_usd": round(optimised_monthly, 2),
            "monthly_savings_usd": round(savings_usd, 2),
            "savings_percentage": round(savings_pct, 2),
            "tier_breakdown": tier_breakdown,
            "business_recommendation": self._generate_routing_recommendation(savings_pct, savings_usd),
        }

    @staticmethod
    def _generate_routing_recommendation(savings_pct: float, savings_usd: float) -> str:
        if savings_pct >= 40:
            return (
                f"Strong recommendation to adopt smart routing. "
                f"Projected monthly savings of ${savings_usd:,.2f} ({savings_pct:.1f}%) "
                f"by routing simple and medium tasks to cost-efficient models."
            )
        if savings_pct >= 20:
            return (
                f"Moderate cost reduction possible through routing. "
                f"Estimated ${savings_usd:,.2f}/month ({savings_pct:.1f}%) in savings."
            )
        return (
            f"Marginal routing savings ({savings_pct:.1f}%). "
            f"Consider workload optimisation before changing routing strategy."
        )
