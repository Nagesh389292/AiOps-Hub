from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from config import settings
from database.db import (
    add_benchmark_result,
    add_cost_report,
    add_recommendation,
    add_audit_log,
    add_validation_result,
    create_benchmark_run,
    get_custom_models,
)
from engines.cost_engine import CostEngine
from engines.recommendation_engine import RecommendationEngine
from engines.routing_engine import RoutingEngine
from engines.validation_engine import ValidationEngine
from models.claude_runner import ClaudeRunner
from models.deepseek_runner import DeepSeekRunner
from models.gemini_runner import GeminiRunner
from models.llama_runner import LlamaRunner
from models.mistral_runner import MistralRunner
from models.openai_runner import OpenAIRunner
from models.dynamic_runner import DynamicRunner

# Canonical dataset filename per category
_DATASET_FILE: dict[str, str] = {
    "coding": "coding_benchmarks.json",
    "sql": "sql_benchmarks.json",
    "support": "support_benchmarks.json",
    "summarization": "summarization_benchmarks.json",
    "reasoning": "reasoning_benchmarks.json",
}


class BenchmarkEngine:
    def __init__(self) -> None:
        self.validation_engine = ValidationEngine()
        self.cost_engine = CostEngine()
        self.recommendation_engine = RecommendationEngine()
        self.routing_engine = RoutingEngine()

        mode = settings.runtime_mode
        self.runners = {
            "gpt": OpenAIRunner(runtime_mode=mode),
            "claude": ClaudeRunner(runtime_mode=mode),
            "gemini": GeminiRunner(runtime_mode=mode),
            "deepseek": DeepSeekRunner(runtime_mode=mode),
            "llama": LlamaRunner(runtime_mode=mode),
            "mistral": MistralRunner(runtime_mode=mode),
        }

        # Auto-load user-added custom models from the database
        try:
            for cm in get_custom_models():
                self.runners[cm["key"]] = DynamicRunner(
                    model_key=cm["key"],
                    model_id=cm["model_id"],
                    base_url=cm["base_url"],
                    api_key=cm["api_key"],
                    runtime_mode=mode,
                )
        except Exception:
            pass  # DB not yet initialised on first run

    @staticmethod
    def _dataset_path(category: str) -> Path:
        filename = _DATASET_FILE.get(category, f"{category}_benchmarks.json")
        return settings.base_dir / "datasets" / category / filename

    def load_dataset(self, category: str) -> list[dict]:
        path = self._dataset_path(category)
        if not path.exists():
            raise FileNotFoundError(f"Dataset missing at {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def run(
        self,
        category: str,
        selected_models: list[str],
        daily_requests: int = 10000,
    ) -> dict:
        dataset = self.load_dataset(category)
        run_id = create_benchmark_run(
            category=category,
            models=selected_models,
            total_benchmarks=len(dataset),
            metadata={
                "dataset_version": settings.dataset_version,
                "prompt_template_version": settings.prompt_template_version,
                "evaluation_rubric_version": settings.evaluation_rubric_version,
                "runtime_mode": settings.runtime_mode,
            },
        )

        raw_results: list[dict] = []

        for model_key in selected_models:
            runner = self.runners.get(model_key)
            if runner is None:
                continue

            for item in dataset:
                prompt = item.get("prompt") or item.get("text", "")
                output = None
                attempt = 0
                for attempt in range(settings.provider_max_retries + 1):
                    output = runner.generate(prompt=prompt, category=category)
                    if not output.error:
                        break
                    if attempt < settings.provider_max_retries:
                        time.sleep(settings.provider_retry_backoff_seconds * (attempt + 1))

                if output is None:
                    raise RuntimeError("Model runner did not return output")

                if output.error:
                    validation: dict = {
                        "accuracy": 0.0,
                        "pass_percentage": 0.0,
                        "validation_status": "Fail",
                        "failure_reason": output.error,
                        "reliability_score": 0.0,
                        "confidence_score": 0.0,
                    }
                else:
                    validation = self.validation_engine.validate(
                        category, output.response_text, item
                    )

                estimated_cost = self.cost_engine.calculate_request_cost(
                    model_name=output.model_name,
                    input_tokens=output.input_tokens,
                    output_tokens=output.output_tokens,
                )

                row = {
                    "model": output.model_name,
                    "model_key": model_key,
                    "problem_title": item.get("title", "N/A"),
                    "prompt": prompt,
                    "response": output.response_text,
                    "accuracy": validation["accuracy"],
                    "latency_ms": round(output.latency_ms, 2),
                    "input_tokens": output.input_tokens,
                    "output_tokens": output.output_tokens,
                    "estimated_cost": estimated_cost,
                    "validation_status": validation["validation_status"],
                    "failure_reason": validation.get("failure_reason", ""),
                    "reliability_score": validation.get("reliability_score", 0.0),
                    "confidence_score": validation.get("confidence_score", 0.0),
                    "provider_attempts": attempt + 1,
                }
                raw_results.append(row)
                add_benchmark_result(run_id, row)
                add_validation_result(run_id, row)

        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in raw_results:
            grouped[row["model"]].append(row)

        leaderboard = []
        for model, rows in grouped.items():
            count = len(rows)
            passes = sum(1 for r in rows if r["validation_status"] == "Pass")
            leaderboard.append(
                {
                    "model": model,
                    "accuracy": round(sum(r["accuracy"] for r in rows) / count, 2),
                    "average_latency_ms": round(
                        sum(r["latency_ms"] for r in rows) / count, 2
                    ),
                    "average_cost": round(
                        sum(r["estimated_cost"] for r in rows) / count, 8
                    ),
                    "validation_success_rate": round((passes / count) * 100, 2),
                    "average_reliability": round(
                        sum(r["reliability_score"] for r in rows) / count, 4
                    ),
                    "average_confidence": round(
                        sum(r["confidence_score"] for r in rows) / count, 4
                    ),
                }
            )

        leaderboard = sorted(leaderboard, key=lambda x: x["accuracy"], reverse=True)

        cost_comparison = self.cost_engine.compare_model_costs(
            leaderboard, daily_requests=daily_requests
        )
        add_cost_report(run_id, cost_comparison)

        recommendation = self.recommendation_engine.recommend(
            leaderboard, workload=category
        )
        recommendation["workload"] = category
        add_recommendation(run_id, recommendation)

        add_audit_log(
            run_id,
            "recommendation_generated",
            {
                "best_overall_model": recommendation["best_overall_model"],
                "best_cost_efficient_model": recommendation["best_cost_efficient_model"],
            },
        )
        add_audit_log(
            run_id,
            "benchmark_completed",
            {
                "category": category,
                "selected_models": selected_models,
                "result_rows": len(raw_results),
            },
        )

        return {
            "run_id": run_id,
            "leaderboard": leaderboard,
            "raw_results": raw_results,
            "cost_comparison": cost_comparison,
            "recommendation": recommendation,
        }

