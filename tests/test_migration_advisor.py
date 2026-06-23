"""
Unit tests for the MigrationAdvisor.
"""
from __future__ import annotations

import pytest

from engines.migration_advisor import MigrationAdvisor, MigrationVerdict, ModelProfile


@pytest.fixture
def advisor() -> MigrationAdvisor:
    return MigrationAdvisor()


def _profile(name: str, accuracy: float, cost: float, latency: float, reliability: float = 0.9) -> ModelProfile:
    return ModelProfile(
        model_name=name,
        accuracy=accuracy,
        validation_success_rate=reliability * 100,
        average_latency_ms=latency,
        average_cost_per_request=cost,
        monthly_cost_usd=cost * 10_000 * 30,
        reliability_score=reliability,
    )


class TestMigrationAdvisor:
    def test_clear_upgrade_recommended(self, advisor: MigrationAdvisor):
        current = _profile("gpt-4o-mini", accuracy=82.0, cost=0.00015, latency=900.0)
        candidate = _profile("gpt-4o", accuracy=92.0, cost=0.00013, latency=850.0)
        report = advisor.compare(current, candidate)
        assert report["verdict"] == MigrationVerdict.RECOMMENDED.value
        assert report["migration_score"] > 0

    def test_cost_spike_no_gain_not_recommended(self, advisor: MigrationAdvisor):
        # Same accuracy + massive cost spike = Not Recommended for budget-constrained org
        current = _profile("claude-haiku", accuracy=88.0, cost=0.00009, latency=600.0)
        candidate = _profile("gpt-4o", accuracy=88.0, cost=0.0025, latency=900.0)
        report = advisor.compare(current, candidate, budget_constrained=True)
        assert report["verdict"] == MigrationVerdict.NOT_RECOMMENDED.value

    def test_conditional_on_trade_offs(self, advisor: MigrationAdvisor):
        current = _profile("gemini-flash", accuracy=80.0, cost=0.00003, latency=450.0)
        candidate = _profile("claude-haiku", accuracy=88.0, cost=0.00009, latency=650.0)
        report = advisor.compare(current, candidate)
        assert report["verdict"] in [MigrationVerdict.RECOMMENDED.value, MigrationVerdict.CONDITIONAL.value]

    def test_deltas_computed_correctly(self, advisor: MigrationAdvisor):
        current = _profile("A", accuracy=80.0, cost=0.0001, latency=500.0)
        candidate = _profile("B", accuracy=90.0, cost=0.00015, latency=600.0)
        report = advisor.compare(current, candidate)
        deltas = report["deltas"]
        assert abs(deltas["accuracy_delta_pct"] - 10.0) < 0.01
        assert abs(deltas["cost_change_pct"] - 50.0) < 0.1
        assert abs(deltas["latency_change_pct"] - 20.0) < 0.1

    def test_reasoning_points_not_empty(self, advisor: MigrationAdvisor):
        current = _profile("A", accuracy=85.0, cost=0.0001, latency=500.0)
        candidate = _profile("B", accuracy=90.0, cost=0.00008, latency=450.0)
        report = advisor.compare(current, candidate)
        assert len(report["reasoning"]) > 0

    def test_action_items_present(self, advisor: MigrationAdvisor):
        current = _profile("A", accuracy=85.0, cost=0.0001, latency=500.0)
        candidate = _profile("B", accuracy=91.0, cost=0.00008, latency=450.0)
        report = advisor.compare(current, candidate)
        assert len(report["action_items"]) > 0

    def test_risk_level_values(self, advisor: MigrationAdvisor):
        current = _profile("A", accuracy=85.0, cost=0.0001, latency=500.0)
        candidate = _profile("B", accuracy=91.0, cost=0.00008, latency=450.0)
        report = advisor.compare(current, candidate)
        assert report["risk_level"] in ["Low", "Medium", "High"]

    def test_compare_from_leaderboard_missing_model(self, advisor: MigrationAdvisor):
        leaderboard = [
            {"model": "gpt-4o-mini", "accuracy": 90.0, "average_latency_ms": 800.0,
             "average_cost": 0.00012, "validation_success_rate": 95.0}
        ]
        result = advisor.compare_from_leaderboard(
            current_model_key="gpt-4o-mini",
            candidate_model_key="missing-model",
            leaderboard=leaderboard,
        )
        assert "error" in result
        assert result["verdict"] == MigrationVerdict.NOT_RECOMMENDED.value

    def test_same_model_still_produces_report(self, advisor: MigrationAdvisor):
        current = _profile("same-model", accuracy=85.0, cost=0.0001, latency=500.0)
        report = advisor.compare(current, current)
        # Should produce a report even if models are identical
        assert "verdict" in report
        assert "deltas" in report
