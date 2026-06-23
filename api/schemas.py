from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BenchmarkRunRequest(BaseModel):
    category: Literal["coding", "sql", "support", "summarization", "reasoning"]
    selected_models: list[str] = Field(min_length=1)
    daily_requests: int = Field(default=10000, ge=1)


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    available_runners: list[str] | None = None


class CostSimulationRequest(BaseModel):
    simple_requests: int = Field(ge=0)
    medium_requests: int = Field(ge=0)
    complex_requests: int = Field(ge=0)
    baseline_model: str = Field(default="gpt")
    available_runners: list[str] | None = None


class MigrationRequest(BaseModel):
    current_model_key: str
    candidate_model_key: str
    leaderboard: list[dict[str, Any]]
    daily_requests: int = Field(default=10000, ge=1)
    latency_sensitive: bool = False
    budget_constrained: bool = False


class HealthResponse(BaseModel):
    status: str
    app: str
    runtime_mode: str
    database_url: str


class AIInsightResponse(BaseModel):
    workload: str
    best_model: str
    risk_level: str
    executive_summary: str
    kpis: dict[str, Any]
    actions: list[str]
