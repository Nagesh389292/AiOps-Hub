# AIOps Hub

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aiops-app-hcslsomk6ftmawnujknsht.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Nagesh389292%2FAiOps--Hub-blue?logo=github)](https://github.com/Nagesh389292/AiOps-Hub)

> 🚀 **Live Dashboard:** https://aiops-app-hcslsomk6ftmawnujknsht.streamlit.app/

AIOps Hub is a production-ready Python platform for enterprise AI model evaluation, cost optimization, output validation, and recommendation.

## Features

- Benchmark GPT, Claude, Gemini across coding, SQL, and support datasets.
- Calculate accuracy, latency, token usage, and estimated cost.
- Validate outputs (coding execution + tests, SQL matching, intent classification).
- Generate model recommendations using weighted scoring.
- Enforce recommendation eligibility with configurable production thresholds.
- Capture reproducibility metadata (dataset/prompt/rubric versions) per run.
- Persist audit trail events for recommendation and benchmark completion.
- Visualize results via Streamlit dashboard.
- Persist runs/results/reports in SQLite (PostgreSQL ready).

## Project Structure
```
aiops-hub/
  datasets/
    coding/
    sql/
    support/
  models/
    openai_runner.py
    claude_runner.py
    gemini_runner.py
  evaluators/
    coding_evaluator.py
    sql_evaluator.py
    support_evaluator.py
  engines/
    benchmark_engine.py
    cost_engine.py
    validation_engine.py
    recommendation_engine.py
  database/
    db.py
  dashboard/
    app.py
  reports/
  tests/
  IMPLEMENTATION_PLAN.md
  requirements.txt
  README.md
```

## Setup
1. Create environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment:

```bash
copy .env.example .env
```

3. Run dashboard:

```bash
streamlit run dashboard/app.py
```

## Runtime Modes
- `mock` (default): uses deterministic local responses for development and interviews.
- `live`: uses provider APIs (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).

Set in `.env`:

```env
AIOPS_RUNTIME_MODE=mock
DATABASE_URL=sqlite:///aiops_hub.db
```

## Testing
```bash
pytest -q
```

## Docker
Build and run locally with Docker:

```bash
docker build -t aiops-hub .
docker run --rm -p 8501:8501 --env-file .env aiops-hub
```

Or use Docker Compose:

```bash
docker compose up --build
```

## Deployment Notes
- Docker and platform-agnostic deployment steps are in `IMPLEMENTATION_PLAN.md`.
- To switch to PostgreSQL, set `DATABASE_URL` with a SQLAlchemy-compatible PostgreSQL URI.

## Interview Value
This project demonstrates practical GenAI + MLOps capabilities:
- LLM benchmarking and evaluation rigor.
- Quality-cost-latency trade-off analysis.
- Validation-first architecture for production reliability.
- Extensible modular system design.
