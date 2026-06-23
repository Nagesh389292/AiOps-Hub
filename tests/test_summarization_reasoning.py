"""
Unit tests for SummarizationEvaluator and ReasoningEvaluator.
"""
from __future__ import annotations

import pytest

from evaluators.reasoning_evaluator import ReasoningEvaluator
from evaluators.summarization_evaluator import SummarizationEvaluator


# ---------------------------------------------------------------------------
# Summarization Evaluator
# ---------------------------------------------------------------------------

class TestSummarizationEvaluator:
    @pytest.fixture
    def evaluator(self) -> SummarizationEvaluator:
        return SummarizationEvaluator()

    SOURCE = (
        "OpenAI released GPT-5, a powerful language model with improved reasoning and coding. "
        "The model achieves new benchmarks in MMLU and HumanEval. It is available via the API "
        "at $15 per million input tokens. Safety features reduce harmful outputs by 78%."
    )

    def test_good_summary_passes(self, evaluator: SummarizationEvaluator):
        candidates = {
            "gpt": "GPT-5 by OpenAI improves reasoning and coding, sets new benchmarks, "
                   "costs $15/M tokens, and reduces harmful outputs by 78%."
        }
        result = evaluator.evaluate(candidates, self.SOURCE)
        assert result["validation_status"] == "Pass"
        assert result["accuracy"] > 0

    def test_empty_summary_fails(self, evaluator: SummarizationEvaluator):
        result = evaluator.evaluate({"gpt": ""}, self.SOURCE)
        assert result["validation_status"] == "Fail"

    def test_no_candidates_fails(self, evaluator: SummarizationEvaluator):
        result = evaluator.evaluate({}, self.SOURCE)
        assert result["validation_status"] == "Fail"

    def test_multi_candidate_ranking(self, evaluator: SummarizationEvaluator):
        candidates = {
            "gpt": "GPT-5 improves reasoning, coding, and safety. Available at $15/M tokens.",
            "gemini": "New model.",
        }
        result = evaluator.evaluate(candidates, self.SOURCE)
        assert len(result["ranking"]) == 2
        # Better summary should rank first
        assert result["ranking"][0]["quality_score"] >= result["ranking"][1]["quality_score"]

    def test_pairwise_result_present(self, evaluator: SummarizationEvaluator):
        candidates = {"gpt": "GPT-5 improves reasoning.", "claude": "New OpenAI model released."}
        result = evaluator.evaluate(candidates, self.SOURCE)
        assert len(result["pairwise_results"]) == 1
        assert "winner" in result["pairwise_results"][0]

    def test_reliability_score_between_0_and_1(self, evaluator: SummarizationEvaluator):
        candidates = {"gpt": "GPT-5 is a powerful new AI model from OpenAI."}
        result = evaluator.evaluate(candidates, self.SOURCE)
        assert 0.0 <= result["reliability_score"] <= 1.0


# ---------------------------------------------------------------------------
# Reasoning Evaluator
# ---------------------------------------------------------------------------

class TestReasoningEvaluator:
    @pytest.fixture
    def evaluator(self) -> ReasoningEvaluator:
        return ReasoningEvaluator()

    def test_correct_answer_with_reasoning_passes(self, evaluator: ReasoningEvaluator):
        answer = (
            "First, I identify the pattern. The differences are 4, 6, 8, 10. "
            "Therefore, the next difference is 12, so the answer is 42."
        )
        result = evaluator.evaluate_single(answer, "42")
        assert result["validation_status"] == "Pass"
        assert result["answer_correct"] is True
        assert result["has_reasoning_chain"] is True

    def test_wrong_answer_fails(self, evaluator: ReasoningEvaluator):
        result = evaluator.evaluate_single("The answer is 100.", "42")
        assert result["validation_status"] == "Fail"
        assert result["answer_correct"] is False

    def test_empty_answer_fails(self, evaluator: ReasoningEvaluator):
        result = evaluator.evaluate_single("", "42")
        assert result["validation_status"] == "Fail"
        assert result["accuracy"] == 0.0

    def test_reliability_score_between_0_and_1(self, evaluator: ReasoningEvaluator):
        result = evaluator.evaluate_single("The answer is 42. Therefore, 42.", "42")
        assert 0.0 <= result["reliability_score"] <= 1.0

    def test_multi_candidate_ranking(self, evaluator: ReasoningEvaluator):
        candidates = {
            "gpt": "First, I see differences of 4,6,8,10 which increase by 2. Therefore the answer is 42.",
            "gemini": "42",
        }
        result = evaluator.evaluate_multi(candidates, "42")
        assert len(result["ranking"]) == 2
        # gpt should rank higher (has reasoning chain)
        assert result["ranking"][0]["model"] == "gpt"

    def test_pairwise_in_multi(self, evaluator: ReasoningEvaluator):
        candidates = {"gpt": "Step 1: observe differences. Step 2: 42.", "claude": "The answer is 42."}
        result = evaluator.evaluate_multi(candidates, "42")
        assert len(result["pairwise_results"]) == 1
        assert result["pairwise_results"][0]["winner"] in ["gpt", "claude", "tie"]

    def test_consensus_best_not_empty(self, evaluator: ReasoningEvaluator):
        candidates = {"gpt": "First step: therefore the answer is 42.", "claude": "42 is the answer."}
        result = evaluator.evaluate_multi(candidates, "42")
        assert result["consensus_best"] != ""
        assert result["consensus_best"] in ["gpt", "claude"]
