from __future__ import annotations

import re
from typing import Any


class ReasoningEvaluator:
    """
    Multi-judge reasoning evaluator using ranking and confidence scoring.

    Evaluates whether a model's reasoning chain is logically sound,
    reaches the correct conclusion, and explains its steps.

    Returns per-candidate scores, a consensus ranking, and a pairwise
    comparison matrix — not a simple pass/fail accuracy metric.
    """

    MIN_PASSING_SCORE: float = 60.0

    def evaluate_single(
        self,
        generated_answer: str,
        expected_answer: str,
        reasoning_question: str = "",
    ) -> dict[str, Any]:
        """Evaluate a single model response for a reasoning problem."""
        if not generated_answer.strip():
            return self._fail_result("Empty response")

        score, details = self._score_reasoning(
            generated_answer, expected_answer, reasoning_question
        )
        passed = score >= self.MIN_PASSING_SCORE

        return {
            "accuracy": round(score, 2),
            "pass_percentage": round(score, 2),
            "validation_status": "Pass" if passed else "Fail",
            "failure_reason": details.get("failure_reason", ""),
            "answer_correct": details.get("answer_correct", False),
            "has_reasoning_chain": details.get("has_reasoning_chain", False),
            "reasoning_depth": details.get("reasoning_depth", 0),
            "reliability_score": round(score / 100, 4),
            "confidence_score": round(min(1.0, score / 100 * 0.95), 4),
        }

    def evaluate_multi(
        self,
        candidates: dict[str, str],  # {model_name: answer_text}
        expected_answer: str,
        question: str = "",
    ) -> dict[str, Any]:
        """Rank multiple model responses for the same reasoning problem."""
        if not candidates:
            return self._empty_result("No candidates provided")

        scores: dict[str, float] = {}
        details: dict[str, dict] = {}

        for model, answer in candidates.items():
            score, det = self._score_reasoning(answer, expected_answer, question)
            scores[model] = score
            details[model] = det

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Pairwise matrix
        models = list(candidates.keys())
        pairwise: list[dict[str, Any]] = []
        for i, m_a in enumerate(models):
            for m_b in models[i + 1:]:
                winner, reasoning = self._pairwise_judge(
                    m_a, scores[m_a], details[m_a],
                    m_b, scores[m_b], details[m_b],
                )
                pairwise.append({
                    "model_a": m_a, "model_b": m_b,
                    "winner": winner, "reasoning": reasoning,
                    "score_a": round(scores[m_a], 2),
                    "score_b": round(scores[m_b], 2),
                })

        best_model = ranked[0][0] if ranked else "N/A"
        best_score = ranked[0][1] if ranked else 0.0
        passed = best_score >= self.MIN_PASSING_SCORE

        ranking_list = [
            {
                "model": m,
                "reasoning_score": round(s, 2),
                "answer_correct": details[m].get("answer_correct", False),
                "reasoning_depth": details[m].get("reasoning_depth", 0),
                "reliability_score": round(s / 100, 4),
                "confidence_score": round(min(1.0, s / 100 * 0.95), 4),
            }
            for m, s in ranked
        ]

        return {
            "validation_status": "Pass" if passed else "Fail",
            "accuracy": round(best_score, 2),
            "pass_percentage": round(best_score, 2),
            "consensus_best": best_model,
            "ranking": ranking_list,
            "pairwise_results": pairwise,
            "failure_reason": details.get(best_model, {}).get("failure_reason", ""),
            "reliability_score": round(best_score / 100, 4),
            "confidence_score": round(min(1.0, best_score / 100 * 0.95), 4),
        }

    def _score_reasoning(
        self, answer: str, expected: str, question: str
    ) -> tuple[float, dict[str, Any]]:
        score = 0.0
        details: dict[str, Any] = {}

        # 1. Answer correctness (40 points)
        answer_correct = self._check_answer_correct(answer, expected)
        details["answer_correct"] = answer_correct
        if answer_correct:
            score += 40.0

        # 2. Reasoning chain presence (30 points)
        chain_markers = [
            r"\bfirst\b", r"\bsecond\b", r"\btherefore\b", r"\bbecause\b",
            r"\bsince\b", r"\bthus\b", r"\bso\b", r"\bhence\b",
            r"\bstep\s*\d", r"\d+\.", r"→", r"=>",
        ]
        chain_count = sum(1 for pat in chain_markers if re.search(pat, answer.lower()))
        has_chain = chain_count >= 2
        details["has_reasoning_chain"] = has_chain
        details["reasoning_depth"] = chain_count
        if has_chain:
            score += min(30.0, chain_count * 5.0)

        # 3. Answer completeness (20 points)
        word_count = len(answer.split())
        if word_count >= 20:
            score += 20.0
        elif word_count >= 10:
            score += 10.0

        # 4. Penalise hedging/uncertainty without conclusion (−10 points)
        uncertainty_markers = [r"\bi don't know\b", r"\bi'm not sure\b", r"\bcannot determine\b"]
        if any(re.search(p, answer.lower()) for p in uncertainty_markers) and not answer_correct:
            score = max(0.0, score - 10.0)
            details["failure_reason"] = "Uncertain answer without correct conclusion"

        if not details.get("failure_reason"):
            details["failure_reason"] = "" if score >= self.MIN_PASSING_SCORE else "Insufficient reasoning depth or incorrect answer"

        return round(min(100.0, score), 2), details

    @staticmethod
    def _check_answer_correct(answer: str, expected: str) -> bool:
        ans = answer.strip().lower()
        exp = expected.strip().lower()
        # Exact substring match or first sentence contains expected
        first_sentence = re.split(r"[.!?\n]", ans)[0].strip()
        return exp in ans or exp in first_sentence

    @staticmethod
    def _pairwise_judge(
        m_a: str, score_a: float, det_a: dict,
        m_b: str, score_b: float, det_b: dict,
    ) -> tuple[str, str]:
        diff = score_a - score_b
        if abs(diff) < 5:
            return "tie", f"{m_a} and {m_b} produced comparable reasoning quality."
        winner = m_a if diff > 0 else m_b
        loser = m_b if diff > 0 else m_a
        reasoning = (
            f"{winner} scored higher: correct answer = {det_a['answer_correct'] if diff>0 else det_b['answer_correct']}, "
            f"reasoning depth = {det_a['reasoning_depth'] if diff>0 else det_b['reasoning_depth']} markers."
        )
        return winner, reasoning

    @staticmethod
    def _fail_result(reason: str) -> dict[str, Any]:
        return {
            "accuracy": 0.0,
            "pass_percentage": 0.0,
            "validation_status": "Fail",
            "failure_reason": reason,
            "answer_correct": False,
            "has_reasoning_chain": False,
            "reasoning_depth": 0,
            "reliability_score": 0.0,
            "confidence_score": 0.0,
        }

    @staticmethod
    def _empty_result(reason: str) -> dict[str, Any]:
        return {
            "validation_status": "Fail",
            "accuracy": 0.0,
            "pass_percentage": 0.0,
            "consensus_best": "N/A",
            "ranking": [],
            "pairwise_results": [],
            "failure_reason": reason,
            "reliability_score": 0.0,
            "confidence_score": 0.0,
        }
