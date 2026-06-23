from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import settings
from database.db import (
    get_accuracy_trend,
    get_cost_trend,
    get_home_metrics,
    get_leaderboard_by_category,
    get_migration_history,
    get_validation_failures,
    init_db,
    add_migration_report,
    register_custom_model,
    get_custom_models,
    delete_custom_model,
)
from engines.benchmark_engine import BenchmarkEngine
from engines.migration_advisor import MigrationAdvisor
from engines.recommendation_engine import RecommendationEngine
from engines.routing_engine import RoutingEngine
from logging_config import configure_logging

configure_logging(settings.log_level)
init_db()

st.set_page_config(
    page_title="AIOps Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #f8fafc;
        --surface: #ffffff;
        --line: rgba(30,58,138,0.10);
        --primary: #1e40af;
        --secondary: #3b82f6;
        --accent: #f59e0b;
        --ink: #0f172a;
        --muted: #475569;
        --success: #0f766e;
        --danger: #b91c1c;
        --warning: #b45309;
        --shadow: 0 20px 50px rgba(15,23,42,0.08);
    }

    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #f8fafc 100%);
        color: var(--ink);
        font-family: 'Fira Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] * { color: #e2e8f0; font-family: 'Fira Sans', sans-serif; }

    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    h1, h2, h3 {
        font-family: 'Fira Code', monospace;
        letter-spacing: -0.02em;
        color: #1e3a8a;
    }

    .hero-shell {
        background: linear-gradient(135deg, rgba(30,64,175,0.98), rgba(15,23,42,0.95));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 1.8rem;
        box-shadow: var(--shadow);
        color: white;
        margin-bottom: 1rem;
    }

    .panel {
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: var(--shadow);
        padding: 1rem 1.1rem 1.2rem 1.1rem;
        margin-top: 1rem;
    }

    .panel-title {
        font-family: 'Fira Code', monospace;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--primary);
        margin-bottom: 0.35rem;
    }

    .panel-copy { color: var(--muted); margin-bottom: 0.8rem; }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: var(--shadow);
        padding: 1rem;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 5px;
        background: linear-gradient(180deg, var(--secondary), var(--accent));
    }

    .metric-card.green::before { background: linear-gradient(180deg, #10b981, #059669); }
    .metric-card.red::before   { background: linear-gradient(180deg, #ef4444, #b91c1c); }
    .metric-card.amber::before { background: linear-gradient(180deg, #f59e0b, #d97706); }

    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
    }

    .metric-value {
        margin-top: 0.35rem;
        font-family: 'Fira Code', monospace;
        font-size: 1.65rem;
        color: var(--ink);
    }

    .metric-foot { margin-top: 0.45rem; color: var(--muted); font-size: 0.88rem; }

    .band-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.2rem;
        margin-bottom: 1rem;
    }

    .kpi-band {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: var(--shadow);
        padding: 0.95rem 1rem;
    }

    .band-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
    }

    .band-value { margin-top: 0.4rem; color: var(--primary); font-family: 'Fira Code', monospace; font-size: 1.05rem; }

    .verdict-recommended  { color: #059669; font-weight: 700; font-size: 1.2rem; }
    .verdict-not          { color: #b91c1c; font-weight: 700; font-size: 1.2rem; }
    .verdict-conditional  { color: #d97706; font-weight: 700; font-size: 1.2rem; }

    @media (max-width: 1100px) {
        .kpi-grid, .band-grid { grid-template-columns: 1fr 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def _mc(label: str, value: str, foot: str, color: str = "") -> str:
    cls = f"metric-card {color}" if color else "metric-card"
    return (
        f'<div class="{cls}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-foot">{foot}</div>'
        "</div>"
    )


def _band(label: str, value: str) -> str:
    return (
        '<div class="kpi-band">'
        f'<div class="band-label">{label}</div>'
        f'<div class="band-value">{value}</div>'
        "</div>"
    )


def _hero(title: str, copy: str, badge: str = "") -> None:
    st.markdown(
        f'<div class="hero-shell">'
        f'<h1 style="color:white;font-family:\'Fira Code\',monospace;margin:0 0 0.5rem 0;">{title}</h1>'
        f'<div style="color:rgba(226,232,240,0.9);font-size:1rem;line-height:1.6;">{copy}</div>'
        + (f'<div style="margin-top:0.8rem;"><span style="background:rgba(245,158,11,0.25);color:#fde68a;padding:0.3rem 0.8rem;border-radius:999px;font-size:0.8rem;font-family:Fira Code,monospace;">{badge}</span></div>' if badge else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def _verdict_html(verdict: str) -> str:
    cls = {
        "Recommended": "verdict-recommended",
        "Not Recommended": "verdict-not",
        "Conditionally Recommended": "verdict-conditional",
    }.get(verdict, "verdict-conditional")
    icon = {
        "Recommended": "✅",
        "Not Recommended": "❌",
        "Conditionally Recommended": "⚠️",
    }.get(verdict, "⚠️")
    return f'<span class="{cls}">{icon} {verdict}</span>'


def _style_status(df: pd.DataFrame, col: str = "validation_status") -> "pd.io.formats.style.Styler":
    def _s(v: str) -> str:
        if v == "Pass":
            return "color:#059669;font-weight:700;"
        return "color:#b91c1c;font-weight:700;"
    return df.style.map(_s, subset=[col])


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

benchmark_engine = BenchmarkEngine()
migration_advisor = MigrationAdvisor()
routing_engine = RoutingEngine()
recommendation_engine = RecommendationEngine()

if "last_run" not in st.session_state:
    st.session_state["last_run"] = None
if "migration_report" not in st.session_state:
    st.session_state["migration_report"] = None

last_run: dict | None = st.session_state["last_run"]
migration_report: dict | None = st.session_state["migration_report"]
metrics = get_home_metrics()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🚀 AIOps Hub")
    st.caption("Enterprise AI Model Operations Platform")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "AI Command Center",
            "Benchmark",
            "Cost Optimizer",
            "Validation",
            "Migration Advisor",
            "Recommendations",
            "Historical Trends",
            "Model Comparison",
            "Model Registry",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**Runtime:** `{settings.runtime_mode.upper()}`")
    custom_count = len(get_custom_models())
    st.markdown(f"**Providers:** GPT · Claude · Gemini · DeepSeek · Llama · Mistral" + (f" + {custom_count} custom" if custom_count else ""))


def _generate_ai_brief_for_workload(workload: str) -> dict:
    leaderboard_rows = get_leaderboard_by_category(workload)
    if not leaderboard_rows:
        return {
            "workload": workload,
            "summary": "No historical benchmark data yet. Run a benchmark to generate an AI executive brief.",
            "best_model": "N/A",
            "risk_level": "Unknown",
            "actions": [
                "Run benchmark on at least three models for this workload",
                "Enable daily trend tracking for accuracy and validation",
                "Set budget baseline for cost optimizer simulation",
            ],
            "kpis": {},
        }

    recommendation = recommendation_engine.recommend(leaderboard_rows, workload=workload)
    top = recommendation["scored_rows"][0]
    risk_level = "Low"
    if top["validation_success_rate"] < 85 or top["accuracy"] < 80:
        risk_level = "High"
    elif top["validation_success_rate"] < 92 or top["accuracy"] < 88:
        risk_level = "Medium"

    summary = (
        f"For {workload}, {recommendation['best_overall_model']} is currently the best enterprise choice "
        f"based on {top['accuracy']:.1f}% accuracy, {top['validation_success_rate']:.1f}% validation success, "
        f"{top['average_latency_ms']:.0f}ms latency, and ${top['average_cost']:.6f} average cost per request."
    )
    return {
        "workload": workload,
        "summary": summary,
        "best_model": recommendation["best_overall_model"],
        "risk_level": risk_level,
        "actions": [
            "Run migration advisor against current production model",
            "Apply complexity-based routing to reduce spend",
            "Monitor validation failures and reliability drift daily",
        ],
        "kpis": {
            "accuracy": top["accuracy"],
            "validation_success_rate": top["validation_success_rate"],
            "average_latency_ms": top["average_latency_ms"],
            "average_cost": top["average_cost"],
        },
    }


# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    _hero(
        "AIOps Hub Control Center",
        "Evaluate frontier AI models, compare performance, optimise costs, validate outputs, "
        "and make data-driven migration decisions — all in one platform.",
        f"Runtime: {settings.runtime_mode.upper()}",
    )

    st.markdown(
        "<div class='kpi-grid'>"
        + _mc("Total Benchmarks", str(metrics["total_benchmarks"]), "Stored result rows", "blue")
        + _mc("Benchmark Runs", str(metrics["total_runs"]), "Evaluation sessions")
        + _mc("Models Evaluated", str(metrics["models_evaluated"]), "Distinct providers")
        + _mc("Migration Reports", str(metrics["migration_reports"]), "Migration analyses", "amber")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='kpi-grid'>"
        + _mc("Avg Accuracy", f"{metrics['average_accuracy']}%", "Cross-run accuracy", "green")
        + _mc("Avg Cost/Request", f"${metrics['average_cost']:.6f}", "Mean estimated cost")
        + _mc("Avg Reliability", f"{metrics['average_reliability']:.2%}", "Output reliability score")
        + _mc("Domains", "5", "Coding · SQL · Support · Summ · Reasoning")
        + "</div>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        st.markdown("### Platform Modules")
        modules_df = pd.DataFrame([
            {"Module": "Model Evaluation", "Status": "Active", "Scope": "6 models · 5 domains"},
            {"Module": "Cost Optimizer", "Status": "Active", "Scope": "Routing · Savings analysis"},
            {"Module": "Validation Engine", "Status": "Active", "Scope": "Sandbox · Multi-judge"},
            {"Module": "Migration Advisor", "Status": "Active", "Scope": "Compare · Recommend"},
            {"Module": "Recommendation Engine", "Status": "Active", "Scope": "Workload-aware scoring"},
            {"Module": "Historical Tracking", "Status": "Active", "Scope": "Trends · Audit log"},
        ])
        st.dataframe(modules_df, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("### Supported Models")
        models_df = pd.DataFrame([
            {"Model": "GPT-4o-mini", "Provider": "OpenAI", "Tier": "mid"},
            {"Model": "Claude Haiku", "Provider": "Anthropic", "Tier": "mid"},
            {"Model": "Gemini Flash", "Provider": "Google", "Tier": "budget"},
            {"Model": "DeepSeek-Chat", "Provider": "DeepSeek", "Tier": "budget"},
            {"Model": "Llama 3.1 8B", "Provider": "Meta/Groq", "Tier": "budget"},
            {"Model": "Mistral Small", "Provider": "Mistral AI", "Tier": "mid"},
        ])
        st.dataframe(models_df, use_container_width=True, hide_index=True)

    st.markdown("### Workload Routing Guide")
    wl_recs = benchmark_engine.recommendation_engine._static_workload_recommendations()
    wl_rows = [
        {"Workload": k, "Recommended Model": v["recommended_model"], "Rationale": v["rationale"], "Cost Tier": v["cost_tier"]}
        for k, v in wl_recs.items()
    ]
    st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)

    st.markdown("### AI Executive Brief")
    brief_cols = st.columns(2)
    with brief_cols[0]:
        brief_workload = st.selectbox(
            "Select workload for AI brief",
            ["coding", "sql", "support", "summarization", "reasoning"],
            key="overview_brief_workload",
        )
    with brief_cols[1]:
        if st.button("Generate AI Brief", use_container_width=True):
            st.session_state["overview_ai_brief"] = _generate_ai_brief_for_workload(brief_workload)

    if "overview_ai_brief" in st.session_state:
        brief = st.session_state["overview_ai_brief"]
        st.markdown(
            f'<div class="panel"><div class="panel-title">AI Summary ({brief["workload"].title()})</div>'
            f'<div class="panel-copy">{brief["summary"]}</div></div>',
            unsafe_allow_html=True,
        )
        if brief.get("kpis"):
            k = brief["kpis"]
            st.markdown(
                "<div class='band-grid'>"
                + _band("Best Model", brief["best_model"])
                + _band("Risk Level", brief["risk_level"])
                + _band("Accuracy", f"{k['accuracy']:.1f}%")
                + "</div>",
                unsafe_allow_html=True,
            )
        for action in brief["actions"]:
            st.markdown(f"- {action}")


# ---------------------------------------------------------------------------
# PAGE: AI Command Center
# ---------------------------------------------------------------------------

elif page == "AI Command Center":
    _hero(
        "AI Command Center",
        "Enterprise AI copilot for executive decisions, risk signals, and intelligent routing recommendations.",
        "AI Decision Intelligence",
    )

    top_row = st.columns([1.1, 0.9])
    with top_row[0]:
        workload = st.selectbox(
            "Workload",
            ["coding", "sql", "support", "summarization", "reasoning"],
            key="ai_cc_workload",
        )
        if st.button("Generate Executive AI Recommendation", type="primary", use_container_width=True):
            st.session_state["ai_cc_brief"] = _generate_ai_brief_for_workload(workload)

        if "ai_cc_brief" in st.session_state:
            brief = st.session_state["ai_cc_brief"]
            st.markdown(
                f'<div class="panel"><div class="panel-title">Executive AI Narrative</div>'
                f'<div class="panel-copy">{brief["summary"]}</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='kpi-grid'>"
                + _mc("Recommended Model", brief["best_model"], "AI-selected best fit", "green")
                + _mc("Risk Level", brief["risk_level"], "Production suitability")
                + _mc("Actions", str(len(brief["actions"])), "Suggested next steps")
                + _mc("Workload", brief["workload"].title(), "Decision context")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("#### Suggested Actions")
            for action in brief["actions"]:
                st.markdown(f"- {action}")

    with top_row[1]:
        st.markdown("#### Intelligent Request Router")
        prompt = st.text_area(
            "Paste business prompt",
            value="Analyze customer ticket themes and summarize root causes for leadership.",
            height=120,
            key="ai_cc_prompt",
        )
        if st.button("AI Route Decision", use_container_width=True):
            decision = routing_engine.route(prompt)
            st.markdown(
                "<div class='panel'>"
                f"<div class='panel-title'>Routing Decision: {decision.complexity.value.upper()}</div>"
                f"<div class='panel-copy'>Route to <b>{decision.recommended_model}</b> with "
                f"confidence {decision.confidence:.0%}. {decision.routing_reason}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### AI Risk Signals")
        failures = get_validation_failures(limit=15)
        if failures:
            fail_df = pd.DataFrame(failures)
            st.dataframe(fail_df[["model", "problem", "reason", "reliability", "confidence"]], use_container_width=True, hide_index=True)
        else:
            st.info("No recent validation failures detected.")


# ---------------------------------------------------------------------------
# PAGE: Benchmark
# ---------------------------------------------------------------------------

elif page == "Benchmark":
    _hero(
        "Benchmark Command Deck",
        "Run multi-model evaluation suites across Coding, SQL, Support, Summarization, and Reasoning domains.",
    )

    ctrl, info = st.columns([1.1, 0.9])

    with ctrl:
        category = st.selectbox(
            "Benchmark Domain",
            ["coding", "sql", "support", "summarization", "reasoning"],
        )
        _builtin_models = ["gpt", "claude", "gemini", "deepseek", "llama", "mistral"]
        _custom_model_keys = [cm["key"] for cm in get_custom_models()]
        _all_models = _builtin_models + _custom_model_keys
        models = st.multiselect(
            "Select Models",
            _all_models,
            default=["gpt", "claude", "gemini"],
        )
        daily_requests = st.number_input(
            "Expected Daily Requests", min_value=100, value=10_000, step=500
        )

        if st.button("▶ Run Evaluation", type="primary", use_container_width=True):
            if not models:
                st.error("Select at least one model.")
            else:
                with st.spinner("Executing benchmark suite…"):
                    st.session_state["last_run"] = benchmark_engine.run(
                        category=category,
                        selected_models=models,
                        daily_requests=int(daily_requests),
                    )
                last_run = st.session_state["last_run"]
                st.success(f"Benchmark complete — Run #{last_run['run_id']}")

    with info:
        domain_info = {
            "coding": "Python code executed in isolated sandbox. Unit tests validated with timeout & security checks.",
            "sql": "SQL executed against in-memory SQLite. Result sets compared row-by-row.",
            "support": "Predicted support label compared to ground-truth with alias normalisation.",
            "summarization": "Heuristic multi-judge evaluation: keyword coverage, length ratio, readability.",
            "reasoning": "Multi-criteria scoring: answer correctness, reasoning chain depth, completeness.",
        }
        st.info(f"**{category.title()} evaluation:** {domain_info.get(category, '')}")

    if last_run:
        st.markdown("### 🏆 Leaderboard")
        lb_df = pd.DataFrame(last_run["leaderboard"])
        st.dataframe(lb_df, use_container_width=True, hide_index=True)

        with st.expander("View Detailed Results"):
            detail_df = pd.DataFrame(last_run["raw_results"])
            st.dataframe(
                _style_status(detail_df),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# PAGE: Cost Optimizer
# ---------------------------------------------------------------------------

elif page == "Cost Optimizer":
    _hero(
        "Cost Optimizer",
        "Analyse LLM spend, simulate smart routing, and project monthly savings.",
    )

    tab_bench, tab_routing = st.tabs(["Benchmark-Based Analysis", "Smart Routing Simulator"])

    with tab_bench:
        if not last_run:
            st.info("Run a benchmark first to see cost analysis.")
        else:
            cost_df = pd.DataFrame(last_run["cost_comparison"])
            cheapest = cost_df.iloc[0] if not cost_df.empty else None

            if cheapest is not None:
                most_exp = cost_df.iloc[-1] if len(cost_df) > 1 else cheapest
                savings = most_exp["monthly_cost"] - cheapest["monthly_cost"]
                savings_pct = cheapest.get("projected_savings_percent", 0.0)

                st.markdown(
                    "<div class='kpi-grid'>"
                    + _mc("Cheapest Model", cheapest["model"], f"${cheapest['monthly_cost']:.2f}/month", "green")
                    + _mc("Monthly Savings vs Baseline", f"${savings:.2f}", f"{savings_pct:.1f}% reduction")
                    + _mc("Daily Cost (cheapest)", f"${cheapest['daily_cost']:.4f}", "Projected at current volume")
                    + _mc("Cost/Request", f"${cheapest['cost_per_request']:.8f}", "Per API call estimate")
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.dataframe(cost_df, use_container_width=True, hide_index=True)
            if len(cost_df) > 1:
                chart_data = cost_df.set_index("model")[["monthly_cost", "daily_cost"]]
                st.bar_chart(chart_data)

    with tab_routing:
        st.markdown("#### Query Routing Simulator")
        st.caption("Model smart routing savings by distributing requests across Simple / Medium / Complex tiers.")

        col_a, col_b, col_c = st.columns(3)
        simple_n = col_a.number_input("Simple requests/day", min_value=0, value=5000, step=500)
        medium_n = col_b.number_input("Medium requests/day", min_value=0, value=3000, step=500)
        complex_n = col_c.number_input("Complex requests/day", min_value=0, value=2000, step=500)

        available_runners = st.multiselect(
            "Available model runners",
            ["gpt", "claude", "gemini", "deepseek", "llama", "mistral"],
            default=["gpt", "claude", "gemini", "deepseek"],
        )
        baseline_model = st.selectbox("Baseline model (always-on)", ["gpt", "claude", "gemini"])

        if st.button("Simulate Routing Savings", use_container_width=True):
            re = RoutingEngine(available_runners=available_runners)
            result = re.simulate_routing_savings(
                request_distribution={
                    "simple": int(simple_n),
                    "medium": int(medium_n),
                    "complex": int(complex_n),
                },
                baseline_model=baseline_model,
            )
            st.markdown(
                "<div class='kpi-grid'>"
                + _mc("Baseline Monthly Cost", f"${result['baseline_monthly_cost_usd']:,.2f}", f"All requests → {result['baseline_model']}")
                + _mc("Optimised Monthly Cost", f"${result['optimised_monthly_cost_usd']:,.2f}", "Smart routing", "green")
                + _mc("Monthly Savings", f"${result['monthly_savings_usd']:,.2f}", f"{result['savings_percentage']:.1f}% reduction", "amber")
                + _mc("Total Daily Requests", f"{result['total_daily_requests']:,}", "Across all tiers")
                + "</div>",
                unsafe_allow_html=True,
            )

            st.info(result["business_recommendation"])

            tier_df = pd.DataFrame(result["tier_breakdown"])
            st.dataframe(tier_df, use_container_width=True, hide_index=True)

        st.markdown("#### Route a Single Query")
        sample_prompt = st.text_area("Enter a prompt to classify", value="What is the capital of France?", height=80)
        if st.button("Classify & Route", use_container_width=True):
            re2 = RoutingEngine(available_runners=available_runners)
            decision = re2.route(sample_prompt)
            st.markdown(
                f"**Complexity:** `{decision.complexity.value.upper()}`  \n"
                f"**Route to:** `{decision.recommended_model}` (confidence: {decision.confidence:.0%})  \n"
                f"**Reason:** {decision.routing_reason}"
            )


# ---------------------------------------------------------------------------
# PAGE: Validation
# ---------------------------------------------------------------------------

elif page == "Validation":
    _hero(
        "Validation Dashboard",
        "Inspect output quality, reliability scores, confidence levels, and failure analysis.",
    )

    tab_live, tab_history = st.tabs(["Last Benchmark Run", "All Historical Failures"])

    with tab_live:
        if not last_run:
            st.info("Run a benchmark first.")
        else:
            details = pd.DataFrame(last_run["raw_results"])
            passed = int((details["validation_status"] == "Pass").sum())
            failed = int((details["validation_status"] == "Fail").sum())
            total = passed + failed
            pass_rate = (passed / max(1, total)) * 100

            avg_rel = details.get("reliability_score", pd.Series([0.0])).mean()
            avg_conf = details.get("confidence_score", pd.Series([0.0])).mean()

            st.markdown(
                "<div class='kpi-grid'>"
                + _mc("Passed", str(passed), f"{pass_rate:.1f}% pass rate", "green")
                + _mc("Failed", str(failed), "Failures captured", "red")
                + _mc("Avg Reliability", f"{avg_rel:.2%}", "Mean reliability score")
                + _mc("Avg Confidence", f"{avg_conf:.2%}", "Mean confidence score")
                + "</div>",
                unsafe_allow_html=True,
            )

            display_cols = [
                c for c in ["model", "problem_title", "validation_status", "failure_reason",
                             "reliability_score", "confidence_score"]
                if c in details.columns
            ]
            st.dataframe(
                _style_status(details[display_cols]),
                use_container_width=True,
                hide_index=True,
            )

    with tab_history:
        failures = get_validation_failures(limit=100)
        if not failures:
            st.info("No failures recorded yet.")
        else:
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE: Migration Advisor
# ---------------------------------------------------------------------------

elif page == "Migration Advisor":
    _hero(
        "Migration Advisor",
        "Compare your current model against a candidate and get a business-grade migration recommendation.",
    )

    if not last_run:
        st.info("Run a benchmark first so the advisor has performance data to compare.")
    else:
        lb = last_run["leaderboard"]
        model_options = [r["model"] for r in lb]
        display_names = [r["model"] for r in lb]

        col1, col2 = st.columns(2)
        current_idx = col1.selectbox("Current (production) model", range(len(display_names)), format_func=lambda i: display_names[i])
        candidate_idx = col2.selectbox("Candidate model", range(len(display_names)), format_func=lambda i: display_names[i], index=min(1, len(display_names)-1))

        adv_col1, adv_col2 = st.columns(2)
        latency_sensitive = adv_col1.checkbox("Latency-sensitive workload")
        budget_constrained = adv_col2.checkbox("Budget-constrained organisation")
        daily_req = st.number_input("Daily Requests (for cost projection)", min_value=100, value=10_000, step=1000)

        if st.button("🔬 Run Migration Analysis", type="primary", use_container_width=True):
            current_key = display_names[current_idx]
            candidate_key = display_names[candidate_idx]

            if current_key == candidate_key:
                st.warning("Select different models for comparison.")
            else:
                report = migration_advisor.compare_from_leaderboard(
                    current_model_key=current_key,
                    candidate_model_key=candidate_key,
                    leaderboard=lb,
                    daily_requests=int(daily_req),
                    latency_sensitive=latency_sensitive,
                    budget_constrained=budget_constrained,
                )
                st.session_state["migration_report"] = report
                migration_report = report

                if "error" not in report:
                    add_migration_report(report)

        if migration_report and "error" not in migration_report:
            verdict = migration_report["verdict"]
            st.markdown(
                f"### Verdict: {_verdict_html(verdict)}",
                unsafe_allow_html=True,
            )

            deltas = migration_report.get("deltas", {})
            st.markdown(
                "<div class='kpi-grid'>"
                + _mc("Migration Score", f"{migration_report.get('migration_score', 0):+.1f}", "Range -100 to +100")
                + _mc("Accuracy Δ", f"{deltas.get('accuracy_delta_pct', 0):+.1f}pp", "Percentage points")
                + _mc("Cost Change", f"{deltas.get('cost_change_pct', 0):+.1f}%", f"${deltas.get('monthly_cost_delta_usd', 0):+.2f}/month")
                + _mc("Latency Change", f"{deltas.get('latency_change_pct', 0):+.1f}%", "Response time impact")
                + "</div>",
                unsafe_allow_html=True,
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Current Model Metrics")
                cur = migration_report.get("current_metrics", {})
                st.dataframe(pd.DataFrame([cur]).T.rename(columns={0: "Value"}), use_container_width=True)

            with col_b:
                st.markdown("#### Candidate Model Metrics")
                cand = migration_report.get("candidate_metrics", {})
                st.dataframe(pd.DataFrame([cand]).T.rename(columns={0: "Value"}), use_container_width=True)

            st.markdown("#### Reasoning")
            for point in migration_report.get("reasoning", []):
                st.markdown(f"- {point}")

            st.markdown("#### Recommended Action Items")
            for action in migration_report.get("action_items", []):
                st.markdown(f"✔ {action}")

        elif migration_report and "error" in migration_report:
            st.error(migration_report["error"])


# ---------------------------------------------------------------------------
# PAGE: Recommendations
# ---------------------------------------------------------------------------

elif page == "Recommendations":
    _hero(
        "Recommendation Engine",
        "Workload-aware model recommendations powered by accuracy, cost, latency, and reliability scoring.",
    )

    if last_run:
        rec = last_run["recommendation"]

        st.markdown(
            "<div class='band-grid'>"
            + _band("Best for Quality", rec["best_coding_model"])
            + _band("Most Cost-Efficient", rec["best_cost_efficient_model"])
            + _band("Best Overall", rec["best_overall_model"])
            + "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="panel"><div class="panel-title">Decision Rationale</div>'
            f'<div class="panel-copy">{rec["reason"]}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Scored Models")
        scored_df = pd.DataFrame(rec["scored_rows"])
        st.dataframe(scored_df, use_container_width=True, hide_index=True)

    st.markdown("### Workload Routing Playbook")
    wl = benchmark_engine.recommendation_engine._static_workload_recommendations()
    playbook_rows = [
        {
            "Workload": k,
            "Recommended Model": v["recommended_model"],
            "Rationale": v["rationale"],
            "Cost Tier": v["cost_tier"],
        }
        for k, v in wl.items()
    ]
    st.dataframe(pd.DataFrame(playbook_rows), use_container_width=True, hide_index=True)

    if not last_run:
        st.info("Run a benchmark to see scored recommendations.")


# ---------------------------------------------------------------------------
# PAGE: Historical Trends
# ---------------------------------------------------------------------------

elif page == "Historical Trends":
    _hero(
        "Historical Trends",
        "Track accuracy, cost, validation, and migration decisions over time.",
    )

    tab_acc, tab_cost, tab_mig, tab_val = st.tabs(
        ["Accuracy Trend", "Cost Trend", "Migration History", "Validation History"]
    )

    with tab_acc:
        model_filter = st.text_input("Filter by model name (optional)", "")
        acc_data = get_accuracy_trend(model_name=model_filter or None)
        if acc_data:
            df = pd.DataFrame(acc_data)
            st.line_chart(df.pivot_table(index="timestamp", columns="model", values="accuracy", aggfunc="mean"))
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No accuracy trend data yet. Run some benchmarks first.")

    with tab_cost:
        cost_data = get_cost_trend()
        if cost_data:
            df = pd.DataFrame(cost_data)
            st.line_chart(df.pivot_table(index="timestamp", columns="model", values="monthly_cost", aggfunc="mean"))
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No cost trend data yet.")

    with tab_mig:
        mig_hist = get_migration_history()
        if mig_hist:
            mig_df = pd.DataFrame([
                {
                    "ID": r["id"],
                    "Current": r["current_model"],
                    "Candidate": r["candidate_model"],
                    "Verdict": r["verdict"],
                    "Score": r["migration_score"],
                    "Risk": r["risk_level"],
                    "Accuracy Δ": r["accuracy_delta"],
                    "Cost Δ %": r["cost_change_pct"],
                    "Month Cost Δ $": r["monthly_cost_delta"],
                    "Date": r["created_at"][:10] if r["created_at"] else "",
                }
                for r in mig_hist
            ])
            st.dataframe(mig_df, use_container_width=True, hide_index=True)
        else:
            st.info("No migration reports yet. Use the Migration Advisor page.")

    with tab_val:
        failures = get_validation_failures(limit=200)
        if failures:
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)
        else:
            st.info("No validation failures recorded.")


# ---------------------------------------------------------------------------
# PAGE: Model Comparison
# ---------------------------------------------------------------------------

elif page == "Model Registry":
    _hero(
        "Model Registry",
        "Add any OpenAI-compatible LLM API. Once registered, the model is immediately available in Benchmark runs — no code changes needed.",
        "Custom LLM Integration",
    )

    st.markdown("### ➕ Register a New LLM")
    st.info("Works with any provider that has an OpenAI-compatible `/v1/chat/completions` endpoint (Groq, Together AI, OpenRouter, Ollama, Fireworks, etc.)")

    with st.form("add_model_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            display_name = st.text_input("Model Display Name", placeholder="e.g. Qwen 2.5 72B")
            model_id = st.text_input("Model ID", placeholder="e.g. qwen/qwen2.5-72b-instruct")
        with col2:
            base_url = st.text_input("API Base URL", placeholder="e.g. https://openrouter.ai/api/v1")
            api_key = st.text_input("API Key", type="password", placeholder="sk-...")
        provider = st.selectbox("Provider / Source", ["openrouter", "together", "fireworks", "ollama", "groq", "huggingface", "custom"])
        submitted = st.form_submit_button("Register Model", type="primary")

    if submitted:
        if not all([display_name, model_id, base_url, api_key]):
            st.error("All fields are required.")
        else:
            key = register_custom_model(
                display_name=display_name,
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                provider=provider,
            )
            st.success(f"✅ Model **{display_name}** registered with key `{key}`. It will appear in Benchmark model selection immediately.")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Registered Custom Models")
    custom_models = get_custom_models()
    if not custom_models:
        st.info("No custom models registered yet. Use the form above to add one.")
    else:
        for cm in custom_models:
            with st.expander(f"🤖 {cm['display_name']}  —  key: `{cm['key']}`"):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Model ID:** `{cm['model_id']}`")
                c2.markdown(f"**Provider:** `{cm['provider']}`")
                c3.markdown(f"**Added:** {cm['created_at'][:10]}")
                st.markdown(f"**Base URL:** `{cm['base_url']}`")
                st.markdown(f"**API Key:** `{'*' * 8}{cm['api_key'][-4:] if len(cm['api_key']) > 4 else '****'}`")
                st.markdown(f"**Usage in benchmark:** select `{cm['key']}` from model list in Benchmark page")
                if st.button(f"🗑️ Remove {cm['display_name']}", key=f"del_{cm['key']}"):
                    delete_custom_model(cm['key'])
                    st.success(f"Removed {cm['display_name']}")
                    st.rerun()

    st.markdown("---")
    st.markdown("### 🌐 Popular Free / Open-Source Endpoints")
    popular = [
        {"name": "OpenRouter (Llama 3.3 70B free)", "base_url": "https://openrouter.ai/api/v1", "model_id": "meta-llama/llama-3.3-70b-instruct:free", "key_url": "https://openrouter.ai/keys"},
        {"name": "Together AI (Llama 3.1 8B)", "base_url": "https://api.together.xyz/v1", "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "key_url": "https://api.together.ai/settings/api-keys"},
        {"name": "Fireworks (Llama 3.1 8B)", "base_url": "https://api.fireworks.ai/inference/v1", "model_id": "accounts/fireworks/models/llama-v3p1-8b-instruct", "key_url": "https://fireworks.ai/account/api-keys"},
        {"name": "Ollama (local, no key needed)", "base_url": "http://localhost:11434/v1", "model_id": "llama3.1", "key_url": "https://ollama.com"},
    ]
    cols = st.columns(2)
    for i, p in enumerate(popular):
        with cols[i % 2]:
            st.markdown(f"""<div style='background:#f1f5f9;border-radius:10px;padding:14px;margin-bottom:12px;'>
<b>{p['name']}</b><br>
<code style='font-size:11px'>{p['base_url']}</code><br>
Model ID: <code style='font-size:11px'>{p['model_id']}</code><br>
<a href='{p['key_url']}' target='_blank'>Get API key →</a>
</div>""", unsafe_allow_html=True)

elif page == "Model Comparison":
    _hero(
        "Model Comparison",
        "Side-by-side performance comparison of any two models across all benchmark domains.",
    )

    ALL_MODELS = ["gpt", "claude", "gemini", "deepseek", "llama", "mistral"]
    ALL_CATEGORIES = ["coding", "sql", "support", "summarization", "reasoning"]

    col1, col2 = st.columns(2)
    model_a = col1.selectbox("Model A", ALL_MODELS, index=0)
    model_b = col2.selectbox("Model B", ALL_MODELS, index=1)
    compare_category = st.selectbox("Domain to compare", ALL_CATEGORIES)

    if st.button("📊 Run Comparison", type="primary", use_container_width=True):
        with st.spinner("Running benchmark for both models…"):
            comp_run = benchmark_engine.run(
                category=compare_category,
                selected_models=[model_a, model_b],
                daily_requests=10_000,
            )
        st.session_state["last_run"] = comp_run
        last_run = comp_run
        st.success("Comparison complete!")

    if last_run and len(last_run["leaderboard"]) >= 2:
        lb = last_run["leaderboard"]
        a_data = next((r for r in lb if model_a in r["model"].lower()), None)
        b_data = next((r for r in lb if model_b in r["model"].lower()), None)

        if a_data and b_data:
            comparison_rows = [
                {
                    "Metric": "Accuracy (%)",
                    a_data["model"]: f"{a_data['accuracy']:.1f}",
                    b_data["model"]: f"{b_data['accuracy']:.1f}",
                    "Winner": a_data["model"] if a_data["accuracy"] >= b_data["accuracy"] else b_data["model"],
                },
                {
                    "Metric": "Validation Success (%)",
                    a_data["model"]: f"{a_data['validation_success_rate']:.1f}",
                    b_data["model"]: f"{b_data['validation_success_rate']:.1f}",
                    "Winner": a_data["model"] if a_data["validation_success_rate"] >= b_data["validation_success_rate"] else b_data["model"],
                },
                {
                    "Metric": "Avg Latency (ms)",
                    a_data["model"]: f"{a_data['average_latency_ms']:.1f}",
                    b_data["model"]: f"{b_data['average_latency_ms']:.1f}",
                    "Winner": a_data["model"] if a_data["average_latency_ms"] <= b_data["average_latency_ms"] else b_data["model"],
                },
                {
                    "Metric": "Avg Cost/Request ($)",
                    a_data["model"]: f"{a_data['average_cost']:.8f}",
                    b_data["model"]: f"{b_data['average_cost']:.8f}",
                    "Winner": a_data["model"] if a_data["average_cost"] <= b_data["average_cost"] else b_data["model"],
                },
                {
                    "Metric": "Avg Reliability",
                    a_data["model"]: f"{a_data.get('average_reliability', 0):.4f}",
                    b_data["model"]: f"{b_data.get('average_reliability', 0):.4f}",
                    "Winner": a_data["model"] if a_data.get("average_reliability", 0) >= b_data.get("average_reliability", 0) else b_data["model"],
                },
            ]
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

            # Migration analysis
            st.markdown("### Migration Decision")
            report = migration_advisor.compare_from_leaderboard(
                current_model_key=a_data["model"],
                candidate_model_key=b_data["model"],
                leaderboard=lb,
            )
            if "error" not in report:
                verdict = report["verdict"]
                st.markdown(
                    f"**Migrating from {a_data['model']} → {b_data['model']}:** {_verdict_html(verdict)}",
                    unsafe_allow_html=True,
                )
                for point in report.get("reasoning", []):
                    st.markdown(f"- {point}")

    elif last_run:
        st.dataframe(pd.DataFrame(last_run["leaderboard"]), use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Run Comparison' to compare models side by side.")


# ---------------------------------------------------------------------------
# PAGE: Cost Optimizer (alias handled above)
# ---------------------------------------------------------------------------
