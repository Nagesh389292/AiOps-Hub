"""
Unit tests for routing engine and query complexity analyzer.
"""
from __future__ import annotations

import pytest

from engines.routing_engine import (
    ComplexityTier,
    QueryComplexityAnalyzer,
    RoutingEngine,
)


@pytest.fixture
def analyzer() -> QueryComplexityAnalyzer:
    return QueryComplexityAnalyzer()


@pytest.fixture
def engine() -> RoutingEngine:
    return RoutingEngine()


class TestQueryComplexityAnalyzer:
    def test_simple_question_classified_simple(self, analyzer: QueryComplexityAnalyzer):
        tier, conf = analyzer.classify("What is the capital of France?")
        assert tier == ComplexityTier.SIMPLE
        assert conf > 0

    def test_long_prompt_classified_complex(self, analyzer: QueryComplexityAnalyzer):
        long_prompt = " ".join(["word"] * 250)
        tier, _ = analyzer.classify(long_prompt)
        assert tier == ComplexityTier.COMPLEX

    def test_medium_prompt_summarize(self, analyzer: QueryComplexityAnalyzer):
        tier, _ = analyzer.classify("Summarize the following document in 3 bullet points.")
        assert tier in [ComplexityTier.MEDIUM, ComplexityTier.SIMPLE]

    def test_complex_architecture_prompt(self, analyzer: QueryComplexityAnalyzer):
        tier, _ = analyzer.classify(
            "Design a distributed microservices architecture for a multi-tenant SaaS platform. "
            "Include trade-offs and compare different approaches."
        )
        assert tier == ComplexityTier.COMPLEX

    def test_confidence_between_0_and_1(self, analyzer: QueryComplexityAnalyzer):
        for prompt in ["What is 1+1?", "Summarize this.", "Design a system" + " x" * 200]:
            _, conf = analyzer.classify(prompt)
            assert 0.0 <= conf <= 1.0


class TestRoutingEngine:
    def test_route_simple_to_budget_model(self, engine: RoutingEngine):
        decision = engine.route("What is the capital of France?")
        assert decision.complexity == ComplexityTier.SIMPLE
        assert decision.recommended_model in ["deepseek", "gemini"]

    def test_route_complex_to_capable_model(self, engine: RoutingEngine):
        decision = engine.route(
            "Implement a distributed rate limiter in Python using Redis "
            "with multi-step fallback and dynamic configuration from scratch."
        )
        assert decision.complexity == ComplexityTier.COMPLEX
        assert decision.recommended_model in ["gpt", "claude"]

    def test_decision_has_routing_reason(self, engine: RoutingEngine):
        decision = engine.route("Hello")
        assert len(decision.routing_reason) > 0

    def test_decision_token_estimate_positive(self, engine: RoutingEngine):
        decision = engine.route("Short prompt")
        assert decision.estimated_input_tokens > 0

    def test_simulate_routing_savings_structure(self, engine: RoutingEngine):
        result = engine.simulate_routing_savings(
            {"simple": 5000, "medium": 3000, "complex": 2000},
            baseline_model="gpt",
        )
        assert "baseline_monthly_cost_usd" in result
        assert "optimised_monthly_cost_usd" in result
        assert "monthly_savings_usd" in result
        assert result["monthly_savings_usd"] >= 0
        assert "tier_breakdown" in result
        assert len(result["tier_breakdown"]) == 3

    def test_savings_lower_than_baseline(self, engine: RoutingEngine):
        result = engine.simulate_routing_savings(
            {"simple": 5000, "medium": 3000, "complex": 2000},
            baseline_model="gpt",
        )
        assert result["optimised_monthly_cost_usd"] <= result["baseline_monthly_cost_usd"]

    def test_available_runners_restricts_routing(self):
        limited_engine = RoutingEngine(available_runners=["gpt", "claude"])
        decision = limited_engine.route("What is the capital of France?")
        # deepseek/gemini unavailable, should fall back within available runners
        assert decision.recommended_model in ["gpt", "claude"]
