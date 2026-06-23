"""
Unit tests for the SQL and Support evaluators.
"""
from __future__ import annotations

import pytest

from evaluators.sql_evaluator import SQLEvaluator
from evaluators.support_evaluator import SupportEvaluator


# ---------------------------------------------------------------------------
# SQL Evaluator
# ---------------------------------------------------------------------------

class TestSQLEvaluator:
    @pytest.fixture
    def evaluator(self) -> SQLEvaluator:
        return SQLEvaluator()

    def test_correct_query_passes(self, evaluator: SQLEvaluator):
        result = evaluator.evaluate(
            "SELECT * FROM users WHERE status = 'active'",
            "SELECT * FROM users WHERE status = 'active'",
        )
        assert result["validation_status"] == "Pass"
        assert result["accuracy"] == 100.0

    def test_wrong_query_fails(self, evaluator: SQLEvaluator):
        result = evaluator.evaluate(
            "SELECT * FROM users WHERE status = 'inactive'",
            "SELECT * FROM users WHERE status = 'active'",
        )
        assert result["validation_status"] == "Fail"
        assert result["accuracy"] == 0.0

    def test_execution_based_comparison(self, evaluator: SQLEvaluator):
        # Both queries select same data — test execution-based comparison
        result = evaluator.evaluate(
            "SELECT id, name FROM users WHERE status = 'active' ORDER BY id",
            "SELECT id, name FROM users WHERE status = 'active' ORDER BY id",
        )
        assert result["accuracy"] == 100.0

    def test_malformed_query_fails_gracefully(self, evaluator: SQLEvaluator):
        result = evaluator.evaluate(
            "NOT VALID SQL !!!",
            "SELECT * FROM users",
        )
        assert result["validation_status"] == "Fail"
        assert result["accuracy"] == 0.0

    def test_empty_query_fails(self, evaluator: SQLEvaluator):
        result = evaluator.evaluate("", "SELECT * FROM users")
        assert result["validation_status"] == "Fail"

    def test_reliability_score_present(self, evaluator: SQLEvaluator):
        result = evaluator.evaluate(
            "SELECT * FROM users WHERE status = 'active'",
            "SELECT * FROM users WHERE status = 'active'",
        )
        assert "reliability_score" in result
        assert result["reliability_score"] > 0

    def test_validation_method_recorded(self, evaluator: SQLEvaluator):
        result = evaluator.evaluate(
            "SELECT * FROM users WHERE status = 'active'",
            "SELECT * FROM users WHERE status = 'active'",
        )
        assert "validation_method" in result


# ---------------------------------------------------------------------------
# Support Evaluator
# ---------------------------------------------------------------------------

class TestSupportEvaluator:
    @pytest.fixture
    def evaluator(self) -> SupportEvaluator:
        return SupportEvaluator()

    def test_exact_match_passes(self, evaluator: SupportEvaluator):
        result = evaluator.evaluate("refund", "refund")
        assert result["validation_status"] == "Pass"
        assert result["accuracy"] == 100.0
        assert result["reliability_score"] == 1.0

    def test_wrong_label_fails(self, evaluator: SupportEvaluator):
        result = evaluator.evaluate("cancellation", "refund")
        assert result["validation_status"] == "Fail"
        assert result["accuracy"] == 0.0

    def test_alias_normalisation(self, evaluator: SupportEvaluator):
        # "cancel" should normalise to "cancellation"
        result = evaluator.evaluate("cancel", "cancellation")
        assert result["validation_status"] == "Pass"

    def test_payment_failure_alias(self, evaluator: SupportEvaluator):
        result = evaluator.evaluate("payment failure", "payment_failure")
        assert result["validation_status"] == "Pass"

    def test_technical_issue_alias(self, evaluator: SupportEvaluator):
        result = evaluator.evaluate("technical issue", "technical_issue")
        assert result["validation_status"] == "Pass"

    def test_failure_reason_populated_on_mismatch(self, evaluator: SupportEvaluator):
        result = evaluator.evaluate("shipping", "refund")
        assert result["failure_reason"] != ""

    def test_confidence_score_present(self, evaluator: SupportEvaluator):
        result = evaluator.evaluate("refund", "refund")
        assert "confidence_score" in result
        assert result["confidence_score"] > 0
