from __future__ import annotations

from evaluators.coding_evaluator import CodingEvaluator
from evaluators.reasoning_evaluator import ReasoningEvaluator
from evaluators.sql_evaluator import SQLEvaluator
from evaluators.summarization_evaluator import SummarizationEvaluator
from evaluators.support_evaluator import SupportEvaluator


class ValidationEngine:
    def __init__(self) -> None:
        self.coding_evaluator = CodingEvaluator()
        self.sql_evaluator = SQLEvaluator()
        self.support_evaluator = SupportEvaluator()
        self.summarization_evaluator = SummarizationEvaluator()
        self.reasoning_evaluator = ReasoningEvaluator()

    def validate(self, category: str, output: str, benchmark_item: dict) -> dict:
        if category == "coding":
            return self.coding_evaluator.evaluate(
                output,
                benchmark_item["unit_tests"],
                benchmark_item.get("hidden_tests", []),
            )

        if category == "sql":
            return self.sql_evaluator.evaluate(
                output, benchmark_item["expected_sql"]
            )

        if category == "support":
            return self.support_evaluator.evaluate(
                output, benchmark_item["expected_label"]
            )

        if category == "summarization":
            # For single-model evaluation wrap in dict for the multi-candidate evaluator
            source_text = benchmark_item.get("source_text", benchmark_item.get("prompt", ""))
            reference = benchmark_item.get("reference_summary")
            # Run single-candidate scoring via the multi-evaluator
            candidates = {"model": output}
            result = self.summarization_evaluator.evaluate(
                candidates=candidates,
                source_text=source_text,
                reference_summary=reference,
            )
            # Flatten to standard result shape
            ranking = result.get("ranking", [])
            top = ranking[0] if ranking else {}
            return {
                "accuracy": result.get("accuracy", 0.0),
                "pass_percentage": result.get("pass_percentage", 0.0),
                "validation_status": result.get("validation_status", "Fail"),
                "failure_reason": result.get("failure_reason", ""),
                "reliability_score": top.get("reliability_score", result.get("reliability_score", 0.0)),
                "confidence_score": top.get("confidence_score", result.get("confidence_score", 0.0)),
            }

        if category == "reasoning":
            expected = benchmark_item.get("expected_answer", "")
            question = benchmark_item.get("prompt", "")
            return self.reasoning_evaluator.evaluate_single(
                generated_answer=output,
                expected_answer=expected,
                reasoning_question=question,
            )

        return {
            "accuracy": 0.0,
            "pass_percentage": 0.0,
            "validation_status": "Fail",
            "failure_reason": f"Unsupported category: {category}",
            "reliability_score": 0.0,
            "confidence_score": 0.0,
        }

