from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeVerdict:
    judge_model: str
    winner: str  # "A", "B", or "tie"
    reasoning: str
    score_a: float
    score_b: float


class SummarizationEvaluator:
    """
    Pairwise multi-judge summarization evaluator.

    In mock mode evaluates using heuristic quality signals (coverage, length ratio,
    keyword density, readability). In live mode each judge LLM compares two candidates.

    Returns:
        ranking: list of {model, score, reliability, confidence}
        pairwise_results: head-to-head comparison matrix
        consensus: the agreed best candidate
        validation_status: Pass/Fail based on minimum quality bar
    """

    # Quality thresholds
    MIN_COVERAGE_RATIO: float = 0.10   # output must contain ≥10% of source keywords
    MIN_LENGTH_RATIO: float = 0.05     # output must be ≥5% length of source
    MAX_LENGTH_RATIO: float = 0.80     # output must not be ≥80% (not a copy)

    def evaluate(
        self,
        candidates: dict[str, str],  # {model_name: summary_text}
        source_text: str,
        reference_summary: str | None = None,
    ) -> dict[str, Any]:
        if not candidates:
            return self._empty_result("No candidates provided")

        scores: dict[str, float] = {}
        reliability: dict[str, float] = {}
        failure_reasons: dict[str, str] = {}

        source_keywords = self._extract_keywords(source_text)

        for model, summary in candidates.items():
            score, reason = self._heuristic_score(summary, source_text, source_keywords, reference_summary)
            scores[model] = score
            failure_reasons[model] = reason
            reliability[model] = min(1.0, score / 100)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Pairwise comparison
        models = list(candidates.keys())
        pairwise: list[dict[str, Any]] = []
        for i, m_a in enumerate(models):
            for m_b in models[i + 1:]:
                verdict = self._pairwise_compare(
                    m_a, candidates[m_a], m_b, candidates[m_b], source_keywords
                )
                pairwise.append({
                    "model_a": m_a,
                    "model_b": m_b,
                    "winner": verdict.winner,
                    "score_a": verdict.score_a,
                    "score_b": verdict.score_b,
                    "reasoning": verdict.reasoning,
                })

        best_model = ranked[0][0] if ranked else "N/A"
        best_score = ranked[0][1] if ranked else 0.0
        passed = best_score >= 50.0

        ranking_list = [
            {
                "model": m,
                "quality_score": round(s, 2),
                "reliability_score": round(reliability[m], 4),
                "confidence_score": round(min(1.0, s / 100 * 0.9), 4),
                "failure_reason": failure_reasons[m],
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
            "failure_reason": failure_reasons.get(best_model, ""),
            "reliability_score": round(reliability.get(best_model, 0.0), 4),
            "confidence_score": round(min(1.0, best_score / 100 * 0.9), 4),
        }

    def _heuristic_score(
        self,
        summary: str,
        source: str,
        source_keywords: set[str],
        reference: str | None,
    ) -> tuple[float, str]:
        if not summary.strip():
            return 0.0, "Empty summary"

        src_len = max(1, len(source.split()))
        sum_len = len(summary.split())
        length_ratio = sum_len / src_len

        if length_ratio < self.MIN_LENGTH_RATIO:
            return 15.0, f"Summary too short (ratio={length_ratio:.2f})"
        if length_ratio > self.MAX_LENGTH_RATIO:
            return 20.0, "Summary too long — likely copy of source"

        summary_words = set(summary.lower().split())
        covered = len(source_keywords & summary_words)
        coverage_ratio = covered / max(1, len(source_keywords))

        if coverage_ratio < self.MIN_COVERAGE_RATIO:
            return 30.0, f"Low keyword coverage ({coverage_ratio:.0%})"

        # Scoring components (0–100)
        coverage_score = min(40.0, coverage_ratio * 200)
        length_score = 20.0 if 0.10 <= length_ratio <= 0.40 else 10.0
        readability_score = self._readability_score(summary)

        # If reference provided, compute token overlap bonus
        ref_score = 0.0
        if reference:
            ref_words = set(reference.lower().split())
            overlap = len(summary_words & ref_words) / max(1, len(ref_words))
            ref_score = min(20.0, overlap * 100)

        total = coverage_score + length_score + readability_score + ref_score
        return min(100.0, round(total, 2)), ""

    @staticmethod
    def _readability_score(text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
        # Penalise very long sentences (>35 words) or very short (<4 words)
        if avg_words < 4 or avg_words > 35:
            return 10.0
        return 20.0

    @staticmethod
    def _extract_keywords(text: str, top_n: int = 50) -> set[str]:
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "can", "could", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "and", "or", "but", "not", "it",
            "its", "this", "that", "these", "those", "he", "she", "they", "we",
        }
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            if w not in stop:
                freq[w] = freq.get(w, 0) + 1
        return set(sorted(freq, key=freq.get, reverse=True)[:top_n])  # type: ignore[arg-type]

    def _pairwise_compare(
        self,
        model_a: str,
        summary_a: str,
        model_b: str,
        summary_b: str,
        keywords: set[str],
    ) -> JudgeVerdict:
        kw_a = len(keywords & set(summary_a.lower().split()))
        kw_b = len(keywords & set(summary_b.lower().split()))
        len_a = len(summary_a.split())
        len_b = len(summary_b.split())

        # Simple heuristic judge
        score_a = kw_a * 2.0 + (20.0 if 20 <= len_a <= 120 else 5.0)
        score_b = kw_b * 2.0 + (20.0 if 20 <= len_b <= 120 else 5.0)

        diff = abs(score_a - score_b)
        if diff < 2:
            winner = "tie"
            reasoning = "Both summaries have similar keyword coverage and length."
        elif score_a > score_b:
            winner = "A"
            reasoning = f"{model_a} covered more key topics with appropriate conciseness."
        else:
            winner = "B"
            reasoning = f"{model_b} covered more key topics with appropriate conciseness."

        return JudgeVerdict(
            judge_model="heuristic",
            winner=winner,
            reasoning=reasoning,
            score_a=round(score_a, 2),
            score_b=round(score_b, 2),
        )

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
