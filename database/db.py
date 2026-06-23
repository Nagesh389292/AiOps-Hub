from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from config import settings


class Base(DeclarativeBase):
    pass


class ModelRegistry(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    # Custom / user-added model fields
    display_name: Mapped[str] = mapped_column(String(128), default="")
    model_id: Mapped[str] = mapped_column(String(256), default="")
    base_url: Mapped[str] = mapped_column(String(512), default="")
    api_key: Mapped[str] = mapped_column(String(512), default="")
    is_custom: Mapped[int] = mapped_column(Integer, default=0)   # 1 = user-added
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"
    __table_args__ = (
        Index("ix_benchmark_runs_category", "category"),
        Index("ix_benchmark_runs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    models_csv: Mapped[str] = mapped_column(String(300), nullable=False)
    total_benchmarks: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    __table_args__ = (
        Index("ix_benchmark_results_run_id", "run_id"),
        Index("ix_benchmark_results_model_name", "model_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    problem_title: Mapped[str] = mapped_column(String(200), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    validation_status: Mapped[str] = mapped_column(String(20), default="Fail")
    failure_reason: Mapped[str] = mapped_column(Text, default="")
    reliability_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ValidationResult(Base):
    __tablename__ = "validation_results"
    __table_args__ = (
        Index("ix_validation_results_run_id", "run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    problem_title: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="")
    reliability_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CostReport(Base):
    __tablename__ = "cost_reports"
    __table_args__ = (
        Index("ix_cost_reports_run_id", "run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    cost_per_request: Mapped[float] = mapped_column(Float, default=0.0)
    daily_cost: Mapped[float] = mapped_column(Float, default=0.0)
    monthly_cost: Mapped[float] = mapped_column(Float, default=0.0)
    projected_savings_percent: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Recommendation(Base):
    __tablename__ = "recommendations"
    __table_args__ = (
        Index("ix_recommendations_run_id", "run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    best_model: Mapped[str] = mapped_column(String(128), nullable=False)
    cost_efficient_model: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    workload: Mapped[str] = mapped_column(String(100), default="general")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class MigrationReport(Base):
    __tablename__ = "migration_reports"
    __table_args__ = (
        Index("ix_migration_reports_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    current_model: Mapped[str] = mapped_column(String(128), nullable=False)
    candidate_model: Mapped[str] = mapped_column(String(128), nullable=False)
    verdict: Mapped[str] = mapped_column(String(50), nullable=False)
    migration_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(20), default="Medium")
    accuracy_delta: Mapped[float] = mapped_column(Float, default=0.0)
    cost_change_pct: Mapped[float] = mapped_column(Float, default=0.0)
    latency_change_pct: Mapped[float] = mapped_column(Float, default=0.0)
    monthly_cost_delta: Mapped[float] = mapped_column(Float, default=0.0)
    full_report_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    event_payload: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


_extra_connect: dict[str, Any] = (
    {"check_same_thread": False} if "sqlite" in settings.database_url else {}
)
_engine = create_engine(
    settings.database_url,
    future=True,
    connect_args=_extra_connect,
)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=_engine)


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, default=str)


json_dumps = _json_dumps


def create_benchmark_run(
    category: str,
    models: list[str],
    total_benchmarks: int,
    metadata: dict[str, Any] | None = None,
) -> int:
    with SessionLocal() as session:
        run = BenchmarkRun(
            category=category,
            models_csv=",".join(models),
            total_benchmarks=total_benchmarks,
            metadata_json=_json_dumps(metadata or {}),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id


def add_benchmark_result(run_id: int, row: dict[str, Any]) -> None:
    with SessionLocal() as session:
        result = BenchmarkResult(
            run_id=run_id,
            model_name=row["model"],
            problem_title=row["problem_title"],
            prompt=row["prompt"],
            response=row["response"],
            accuracy=row["accuracy"],
            latency_ms=row["latency_ms"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            estimated_cost=row["estimated_cost"],
            validation_status=row["validation_status"],
            failure_reason=row.get("failure_reason", ""),
            reliability_score=row.get("reliability_score", 0.0),
            confidence_score=row.get("confidence_score", 0.0),
        )
        session.add(result)
        session.commit()


def add_validation_result(run_id: int, row: dict[str, Any]) -> None:
    with SessionLocal() as session:
        item = ValidationResult(
            run_id=run_id,
            model_name=row["model"],
            problem_title=row["problem_title"],
            status=row["validation_status"],
            reason=row.get("failure_reason", ""),
            reliability_score=row.get("reliability_score", 0.0),
            confidence_score=row.get("confidence_score", 0.0),
        )
        session.add(item)
        session.commit()


def add_cost_report(run_id: int, cost_rows: list[dict[str, Any]]) -> None:
    with SessionLocal() as session:
        for row in cost_rows:
            item = CostReport(
                run_id=run_id,
                model_name=row["model"],
                cost_per_request=row["cost_per_request"],
                daily_cost=row["daily_cost"],
                monthly_cost=row["monthly_cost"],
                projected_savings_percent=row["projected_savings_percent"],
            )
            session.add(item)
        session.commit()


def add_recommendation(run_id: int, recommendation: dict[str, Any]) -> None:
    with SessionLocal() as session:
        row = Recommendation(
            run_id=run_id,
            best_model=recommendation["best_overall_model"],
            cost_efficient_model=recommendation["best_cost_efficient_model"],
            reason=recommendation["reason"],
            workload=recommendation.get("workload", "general"),
        )
        session.add(row)
        session.commit()


def add_migration_report(report: dict[str, Any]) -> int:
    with SessionLocal() as session:
        deltas = report.get("deltas", {})
        row = MigrationReport(
            current_model=report["current_model"],
            candidate_model=report["candidate_model"],
            verdict=report["verdict"],
            migration_score=report.get("migration_score", 0.0),
            risk_level=report.get("risk_level", "Medium"),
            accuracy_delta=deltas.get("accuracy_delta_pct", 0.0),
            cost_change_pct=deltas.get("cost_change_pct", 0.0),
            latency_change_pct=deltas.get("latency_change_pct", 0.0),
            monthly_cost_delta=deltas.get("monthly_cost_delta_usd", 0.0),
            full_report_json=_json_dumps(report),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def add_audit_log(run_id: int, event_type: str, payload: dict[str, Any]) -> None:
    with SessionLocal() as session:
        item = AuditLog(
            run_id=run_id,
            event_type=event_type,
            event_payload=_json_dumps(payload),
        )
        session.add(item)
        session.commit()


# ---------------------------------------------------------------------------
# Custom model registry helpers
# ---------------------------------------------------------------------------

def register_custom_model(
    display_name: str,
    model_id: str,
    base_url: str,
    api_key: str,
    provider: str = "custom",
) -> str:
    """Save a user-added model to the registry. Returns the internal key."""
    key = display_name.lower().replace(" ", "_")
    with SessionLocal() as session:
        existing = session.query(ModelRegistry).filter_by(model_name=key).first()
        if existing:
            existing.model_id = model_id
            existing.base_url = base_url
            existing.api_key = api_key
            existing.display_name = display_name
            existing.provider = provider
        else:
            row = ModelRegistry(
                model_name=key,
                provider=provider,
                display_name=display_name,
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                is_custom=1,
            )
            session.add(row)
        session.commit()
    return key


def get_custom_models() -> list[dict[str, Any]]:
    """Return all user-added models from the registry."""
    with SessionLocal() as session:
        rows = session.query(ModelRegistry).filter_by(is_custom=1).all()
        return [
            {
                "key": r.model_name,
                "display_name": r.display_name,
                "model_id": r.model_id,
                "base_url": r.base_url,
                "api_key": r.api_key,
                "provider": r.provider,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in rows
        ]


def delete_custom_model(key: str) -> None:
    """Remove a user-added model from the registry."""
    with SessionLocal() as session:
        session.query(ModelRegistry).filter_by(model_name=key, is_custom=1).delete()
        session.commit()


def get_home_metrics() -> dict[str, Any]:
    with SessionLocal() as session:
        total_benchmarks = session.query(BenchmarkResult).count()
        models_evaluated = session.query(BenchmarkResult.model_name).distinct().count()
        total_runs = session.query(BenchmarkRun).count()
        migration_count = session.query(MigrationReport).count()
        avg_accuracy_rows = session.query(BenchmarkResult.accuracy).all()
        avg_cost_rows = session.query(BenchmarkResult.estimated_cost).all()
        avg_reliability_rows = session.query(BenchmarkResult.reliability_score).all()

    accuracy_value = (
        round(sum(x[0] for x in avg_accuracy_rows) / len(avg_accuracy_rows), 2)
        if avg_accuracy_rows else 0.0
    )
    cost_value = (
        round(sum(x[0] for x in avg_cost_rows) / len(avg_cost_rows), 6)
        if avg_cost_rows else 0.0
    )
    reliability_value = (
        round(sum(x[0] for x in avg_reliability_rows) / len(avg_reliability_rows), 4)
        if avg_reliability_rows else 0.0
    )

    return {
        "total_benchmarks": total_benchmarks,
        "total_runs": total_runs,
        "models_evaluated": models_evaluated,
        "migration_reports": migration_count,
        "average_accuracy": accuracy_value,
        "average_cost": cost_value,
        "average_reliability": reliability_value,
    }


def get_accuracy_trend(model_name: str | None = None, limit: int = 60) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        q = (
            session.query(
                BenchmarkRun.created_at,
                BenchmarkRun.category,
                BenchmarkResult.model_name,
                BenchmarkResult.accuracy,
            )
            .join(BenchmarkResult, BenchmarkResult.run_id == BenchmarkRun.id)
            .order_by(BenchmarkRun.created_at.asc())
        )
        if model_name:
            q = q.filter(BenchmarkResult.model_name.ilike(f"%{model_name}%"))
        rows = q.limit(limit).all()

    return [
        {
            "timestamp": r[0].isoformat() if r[0] else "",
            "category": r[1],
            "model": r[2],
            "accuracy": round(r[3], 2),
        }
        for r in rows
    ]


def get_cost_trend(limit: int = 60) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        rows = (
            session.query(
                CostReport.created_at,
                CostReport.model_name,
                CostReport.monthly_cost,
                CostReport.projected_savings_percent,
            )
            .order_by(CostReport.created_at.asc())
            .limit(limit)
            .all()
        )
    return [
        {
            "timestamp": r[0].isoformat() if r[0] else "",
            "model": r[1],
            "monthly_cost": round(r[2], 4),
            "savings_pct": round(r[3], 2),
        }
        for r in rows
    ]


def get_migration_history(limit: int = 20) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        rows = (
            session.query(MigrationReport)
            .order_by(MigrationReport.created_at.desc())
            .limit(limit)
            .all()
        )
    return [
        {
            "id": r.id,
            "current_model": r.current_model,
            "candidate_model": r.candidate_model,
            "verdict": r.verdict,
            "migration_score": r.migration_score,
            "risk_level": r.risk_level,
            "accuracy_delta": r.accuracy_delta,
            "cost_change_pct": r.cost_change_pct,
            "monthly_cost_delta": r.monthly_cost_delta,
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "full_report": json.loads(r.full_report_json) if r.full_report_json else {},
        }
        for r in rows
    ]


def get_validation_failures(run_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        q = (
            session.query(ValidationResult)
            .filter(ValidationResult.status == "Fail")
            .order_by(ValidationResult.created_at.desc())
        )
        if run_id is not None:
            q = q.filter(ValidationResult.run_id == run_id)
        rows = q.limit(limit).all()
    return [
        {
            "run_id": r.run_id,
            "model": r.model_name,
            "problem": r.problem_title,
            "reason": r.reason,
            "reliability": r.reliability_score,
            "confidence": r.confidence_score,
        }
        for r in rows
    ]


def get_leaderboard_by_category(category: str) -> list[dict[str, Any]]:
    with SessionLocal() as session:
        run_ids = [
            r[0]
            for r in session.query(BenchmarkRun.id)
            .filter(BenchmarkRun.category == category)
            .all()
        ]
        if not run_ids:
            return []
        rows = (
            session.query(BenchmarkResult)
            .filter(BenchmarkResult.run_id.in_(run_ids))
            .all()
        )

    grouped: dict[str, list] = {}
    for r in rows:
        grouped.setdefault(r.model_name, []).append(r)

    result = []
    for model, items in grouped.items():
        count = len(items)
        passes = sum(1 for i in items if i.validation_status == "Pass")
        result.append(
            {
                "model": model,
                "accuracy": round(sum(i.accuracy for i in items) / count, 2),
                "average_latency_ms": round(sum(i.latency_ms for i in items) / count, 2),
                "average_cost": round(sum(i.estimated_cost for i in items) / count, 8),
                "validation_success_rate": round((passes / count) * 100, 2),
                "average_reliability": round(
                    sum(i.reliability_score for i in items) / count, 4
                ),
            }
        )
    return sorted(result, key=lambda x: x["accuracy"], reverse=True)
