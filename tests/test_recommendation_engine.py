"""
Unit tests for the RecommendationEngine.
"""
from __future__ import annotations

import pytest

from engines.recommendation_engine import RecommendationEngine


@pytest.fixture
def engine() -> RecommendationEngine:
    return RecommendationEngine()


SAMPLE_LEADERBOARD = [
    {
        "model": "gpt-4o-mini",
        "accuracy": 92.5,
        "average_latency_ms": 850.0,
        "average_cost": 0.00012,
        "validation_success_rate": 95.0,
    },
    {
        "model": "claude-3-5-haiku-latest",
        "accuracy": 88.0,
        "average_latency_ms": 620.0,
        "average_cost": 0.00009,
        "validation_success_rate": 90.0,
    },
    {
        "model": "gemini-1.5-flash",
        "accuracy": 81.0,
        "average_latency_ms": 450.0,
        "average_cost": 0.00003,
        "validation_success_rate": 85.0,
    },
]


class TestRecommendationEngine:
    def test_returns_best_overall(self, engine: RecommendationEngine):
        result = engine.recommend(SAMPLE_LEADERBOARD)
        assert result["best_overall_model"] in ["gpt-4o-mini", "claude-3-5-haiku-latest"]

    def test_best_quality_model_is_highest_accuracy(self, engine: RecommendationEngine):
        result = engine.recommend(SAMPLE_LEADERBOARD)
        assert result["best_coding_model"] == "gpt-4o-mini"

    def test_best_cost_model_is_cheapest(self, engine: RecommendationEngine):
        result = engine.recommend(SAMPLE_LEADERBOARD)
        assert result["best_cost_efficient_model"] == "gemini-1.5-flash"

    def test_empty_leaderboard_returns_na(self, engine: RecommendationEngine):
        result = engine.recommend([])
        assert result["best_overall_model"] == "N/A"
        assert result["best_coding_model"] == "N/A"

    def test_reason_is_non_empty_string(self, engine: RecommendationEngine):
        result = engine.recommend(SAMPLE_LEADERBOARD)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 10

    def test_scored_rows_have_overall_score(self, engine: RecommendationEngine):
        result = engine.recommend(SAMPLE_LEADERBOARD)
        for row in result["scored_rows"]:
            assert "overall_score" in row
            assert isinstance(row["overall_score"], float)

    def test_workload_coding_weights_accuracy(self, engine: RecommendationEngine):
        result = engine.recommend(SAMPLE_LEADERBOARD, workload="coding")
        # highest accuracy model should be preferred for coding
        assert result["best_coding_model"] == "gpt-4o-mini"

    def test_workload_included_in_result(self, engine: RecommendationEngine):
        result = engine.recommend(SAMPLE_LEADERBOARD, workload="sql")
        assert result["workload"] == "sql"

    def test_workload_recommendations_present(self, engine: RecommendationEngine):
        result = engine.recommend(SAMPLE_LEADERBOARD)
        assert "workload_recommendations" in result
        assert "coding" in result["workload_recommendations"]
        assert "customer_support" in result["workload_recommendations"]

    def test_single_model_still_returns_result(self, engine: RecommendationEngine):
        single = [SAMPLE_LEADERBOARD[0]]
        result = engine.recommend(single)
        assert result["best_overall_model"] == "gpt-4o-mini"

    def test_eligible_for_production_flag(self, engine: RecommendationEngine):
        result = engine.recommend(SAMPLE_LEADERBOARD)
        for row in result["scored_rows"]:
            assert "eligible_for_production" in row
            assert isinstance(row["eligible_for_production"], bool)
