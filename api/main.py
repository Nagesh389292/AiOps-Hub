from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from config import settings
from database.db import (
    add_migration_report,
    get_accuracy_trend,
    get_cost_trend,
    get_home_metrics,
    get_leaderboard_by_category,
    get_migration_history,
    get_validation_failures,
    init_db,
)
from engines.benchmark_engine import BenchmarkEngine
from engines.migration_advisor import MigrationAdvisor
from engines.recommendation_engine import RecommendationEngine
from engines.routing_engine import RoutingEngine
from .schemas import (
    AIInsightResponse,
    BenchmarkRunRequest,
    CostSimulationRequest,
    HealthResponse,
    MigrationRequest,
    RouteRequest,
)

app = FastAPI(
    title="AIOps Hub API",
    version="1.0.0",
    description="Enterprise AI model evaluation, optimization, validation, and migration API",
)

benchmark_engine = BenchmarkEngine()
routing_engine = RoutingEngine()
migration_advisor = MigrationAdvisor()
recommendation_engine = RecommendationEngine()


@app.on_event("startup")
def startup_event() -> None:
    init_db()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        runtime_mode=settings.runtime_mode,
        database_url=settings.database_url,
    )


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return get_home_metrics()


@app.post("/benchmarks/run")
def run_benchmark(payload: BenchmarkRunRequest) -> dict[str, Any]:
    try:
        return benchmark_engine.run(
            category=payload.category,
            selected_models=payload.selected_models,
            daily_requests=payload.daily_requests,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {exc}") from exc


@app.post("/routing/classify")
def classify_route(payload: RouteRequest) -> dict[str, Any]:
    engine = RoutingEngine(available_runners=payload.available_runners)
    decision = engine.route(payload.prompt)
    return {
        "complexity": decision.complexity.value,
        "recommended_runner": decision.recommended_model,
        "routing_reason": decision.routing_reason,
        "estimated_input_tokens": decision.estimated_input_tokens,
        "confidence": decision.confidence,
    }


@app.post("/routing/simulate")
def simulate_routing(payload: CostSimulationRequest) -> dict[str, Any]:
    engine = RoutingEngine(available_runners=payload.available_runners)
    return engine.simulate_routing_savings(
        request_distribution={
            "simple": payload.simple_requests,
            "medium": payload.medium_requests,
            "complex": payload.complex_requests,
        },
        baseline_model=payload.baseline_model,
    )


@app.post("/migration/analyze")
def analyze_migration(payload: MigrationRequest) -> dict[str, Any]:
    report = migration_advisor.compare_from_leaderboard(
        current_model_key=payload.current_model_key,
        candidate_model_key=payload.candidate_model_key,
        leaderboard=payload.leaderboard,
        daily_requests=payload.daily_requests,
        latency_sensitive=payload.latency_sensitive,
        budget_constrained=payload.budget_constrained,
    )
    if "error" not in report:
        add_migration_report(report)
    return report


@app.get("/history/accuracy")
def accuracy_history(model: str | None = None, limit: int = 60) -> list[dict[str, Any]]:
    return get_accuracy_trend(model_name=model, limit=limit)


@app.get("/history/cost")
def cost_history(limit: int = 60) -> list[dict[str, Any]]:
    return get_cost_trend(limit=limit)


@app.get("/history/migration")
def migration_history(limit: int = 20) -> list[dict[str, Any]]:
    return get_migration_history(limit=limit)


@app.get("/history/validation-failures")
def validation_failures(run_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return get_validation_failures(run_id=run_id, limit=limit)


@app.get("/leaderboard/{category}")
def leaderboard(category: str) -> list[dict[str, Any]]:
    return get_leaderboard_by_category(category)


@app.get("/ai/insight/{category}", response_model=AIInsightResponse)
def ai_insight(category: str) -> AIInsightResponse:
    leaderboard_rows = get_leaderboard_by_category(category)
    if not leaderboard_rows:
        raise HTTPException(status_code=404, detail="No benchmark history available for this category")

    recommendation = recommendation_engine.recommend(leaderboard_rows, workload=category)
    best = recommendation["best_overall_model"]
    top = recommendation["scored_rows"][0]

    risk_level = "Low"
    if top["validation_success_rate"] < 85 or top["accuracy"] < 80:
        risk_level = "High"
    elif top["validation_success_rate"] < 92 or top["accuracy"] < 88:
        risk_level = "Medium"

    return AIInsightResponse(
        workload=category,
        best_model=best,
        risk_level=risk_level,
        executive_summary=(
            f"For {category} workloads, {best} currently provides the strongest overall enterprise profile "
            f"across quality, reliability, latency, and cost."
        ),
        kpis={
            "accuracy": top["accuracy"],
            "validation_success_rate": top["validation_success_rate"],
            "average_latency_ms": top["average_latency_ms"],
            "average_cost": top["average_cost"],
            "average_reliability": top.get("average_reliability", 0),
        },
        actions=[
            "Run migration shadow test against current production model",
            "Configure workload routing policy with complexity-based dispatch",
            "Set alert thresholds for reliability and validation failures",
        ],
    )
