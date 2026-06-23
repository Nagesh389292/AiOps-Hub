"""
Integration tests for the BenchmarkEngine in mock mode.
Tests full pipeline: load dataset → run models → validate → cost → recommend.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("AIOPS_RUNTIME_MODE", "mock")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from database.db import init_db
from engines.benchmark_engine import BenchmarkEngine


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def engine() -> BenchmarkEngine:
    return BenchmarkEngine()


class TestBenchmarkEngineMock:
    def test_coding_run_returns_leaderboard(self, engine: BenchmarkEngine):
        result = engine.run("coding", ["gpt", "claude"], daily_requests=1000)
        assert "leaderboard" in result
        assert len(result["leaderboard"]) == 2

    def test_sql_run_works(self, engine: BenchmarkEngine):
        result = engine.run("sql", ["gpt"], daily_requests=1000)
        assert len(result["leaderboard"]) == 1

    def test_support_run_works(self, engine: BenchmarkEngine):
        result = engine.run("support", ["gemini"], daily_requests=1000)
        assert len(result["leaderboard"]) == 1

    def test_leaderboard_fields_present(self, engine: BenchmarkEngine):
        result = engine.run("coding", ["gpt"], daily_requests=1000)
        lb = result["leaderboard"][0]
        for field in ["model", "accuracy", "average_latency_ms", "average_cost", "validation_success_rate"]:
            assert field in lb, f"Missing field: {field}"

    def test_cost_comparison_present(self, engine: BenchmarkEngine):
        result = engine.run("coding", ["gpt", "gemini"], daily_requests=5000)
        assert len(result["cost_comparison"]) == 2
        for row in result["cost_comparison"]:
            assert "monthly_cost" in row
            assert "projected_savings_percent" in row

    def test_recommendation_present(self, engine: BenchmarkEngine):
        result = engine.run("coding", ["gpt", "claude"], daily_requests=1000)
        rec = result["recommendation"]
        assert rec["best_overall_model"] != ""
        assert rec["best_cost_efficient_model"] != ""
        assert len(rec["reason"]) > 0

    def test_run_id_is_positive_integer(self, engine: BenchmarkEngine):
        result = engine.run("support", ["gpt"], daily_requests=1000)
        assert isinstance(result["run_id"], int)
        assert result["run_id"] > 0

    def test_raw_results_contain_reliability(self, engine: BenchmarkEngine):
        result = engine.run("coding", ["gpt"], daily_requests=1000)
        for row in result["raw_results"]:
            assert "reliability_score" in row
            assert "confidence_score" in row

    def test_summarization_run_works(self, engine: BenchmarkEngine):
        result = engine.run("summarization", ["gpt"], daily_requests=1000)
        assert len(result["leaderboard"]) == 1

    def test_reasoning_run_works(self, engine: BenchmarkEngine):
        result = engine.run("reasoning", ["claude"], daily_requests=1000)
        assert len(result["leaderboard"]) == 1

    def test_invalid_category_raises(self, engine: BenchmarkEngine):
        with pytest.raises(FileNotFoundError):
            engine.run("nonexistent_category", ["gpt"])

    def test_all_six_models_run(self, engine: BenchmarkEngine):
        result = engine.run(
            "support",
            ["gpt", "claude", "gemini", "deepseek", "llama", "mistral"],
            daily_requests=1000,
        )
        assert len(result["leaderboard"]) == 6
