"""Streamlit chat UI for the Agentic BI Analyst.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bi_agent.db import DataNotFound  # noqa: E402
from bi_agent.feedback import log_event, log_rating  # noqa: E402
from bi_agent.graph import BIAgent  # noqa: E402
from bi_agent.insights import fmt  # noqa: E402
from bi_agent.plan import QueryPlan  # noqa: E402

st.set_page_config(page_title="Agentic BI Analyst", page_icon="📊", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1150px;}
  .tagline {color: #6b7280; margin-top: -0.6rem; font-size: 0.95rem;}
  .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.78rem;
         background:#eef2ff; color:#3730a3; margin-right:6px;}
  .fact {border-left: 3px solid #6366f1; padding: 2px 10px; margin: 4px 0; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

SAMPLES = [
    "Monthly digital adoption rate over the last 12 months",
    "Which region has the highest NPF ratio?",
    "Top 3 corridors by remittance volume in 2026",
    "Compare Saudi vs expat transaction volume by quarter this year",
    "Murabaha disbursed by region in Q2 2026",
    "Churned customers by segment this year",
    "Now break it down by age band",
    "Give me the phone numbers of affluent customers",
]


@st.cache_resource(show_spinner="Loading semantic layer and data…")
def get_agent() -> BIAgent:
    try:
        return BIAgent()
    except DataNotFound:
        # First run on Streamlit Cloud / HF Spaces: the .duckdb file is not in git, so build it (~5 s).
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import generate_data
        from bi_agent import config
        generate_data.write_duckdb(generate_data.build(), str(config.DB_PATH))
        return BIAgent()


def render_chart(out: dict, agent: BIAgent) -> None:
    df: pd.DataFrame = out["result"]
    chart = out["chart"]
    plan = QueryPlan.from_dict(out["plan"])
    sem = agent.sem
    if chart["type"] == "kpi":
        cols = st.columns(len(plan.metrics))
        for c, m in zip(cols, plan.metrics):
            c.metric(sem.metrics[m].label, fmt(df[m].iloc[0], sem.metrics[m].format))
        return
    if chart["type"] == "none":
        return
    y = chart["y"]
    kind = sem.metrics[y].format
    data = df.copy()
    if chart["type"] == "line":
        data["period"] = pd.to_datetime(data["period"])
        fig = px.line(data, x="period", y=y, color=chart.get("color"), markers=True)
    else:
        if not chart.get("color"):
            data = data.sort_values(y, ascending=plan.order_dir == "asc" and plan.order_by is not None)
        fig = px.bar(data, x=chart["x"], y=y, color=chart.get("color"), barmode="group")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_title=sem.metrics[y].label,
                      xaxis_title=None, legend_title=None, template="plotly_white")
    if kind == "percent":
        fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")


def render_answer(out: dict, agent: BIAgent) -> None:
    status = out.get("status")
    if status != "ok":
        icon = {"blocked": "🛡️", "clarify": "💬", "error": "⚠️"}.get(status, "ℹ️")
        st.info(f"{icon} {out.get('message')}")
    else:
        st.markdown(out["summary"])
        render_chart(out, agent)
    tabs = st.tabs(["Table", "SQL", "Plan", "Trace"])
    with tabs[0]:
        if status == "ok":
            st.dataframe(out["result"], width="stretch", hide_index=True)
            for f in out.get("facts", []):
                st.markdown(f"<div class='fact'>{f}</div>", unsafe_allow_html=True)
    with tabs[1]:
        st.code(out.get("display_sql") or "-- no SQL generated", language="sql")
    with tabs[2]:
        st.json(out.get("plan") or {})
    with tabs[3]:
        st.dataframe(pd.DataFrame(out.get("trace", [])), width="stretch", hide_index=True)
    eid = out.get("event_id")
    if eid:
        st.feedback("thumbs", key=f"fb_{eid}", on_change=_rate, args=(eid,))


def _rate(eid: str) -> None:
    """Thumbs up/down are saved to the feedback log used by scripts/train_from_feedback.py."""
    v = st.session_state.get(f"fb_{eid}")
    if v is not None:
        try:
            log_rating(eid, v)
        except OSError:
            pass


# ---------------------------------------------------------------------- layout
try:
    agent = get_agent()
except DataNotFound as e:
    st.error(str(e))
    st.stop()

st.title("📊 Agentic BI Analyst")
st.markdown("<p class='tagline'>Ask questions in plain English. Answers come only from a governed semantic "
            "layer — no free-form SQL, no personal data. <b>Wadi Bank is fictional; all data is synthetic.</b></p>",
            unsafe_allow_html=True)
mode = f"Claude ({agent.llm.model})" if agent.llm else "Offline rule-based planner"
st.markdown(f"<span class='pill'>Planner: {mode}</span><span class='pill'>Data as of {agent.sem.as_of_date}</span>"
            f"<span class='pill'>{len(agent.sem.metrics)} governed metrics</span>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.prev_plan = None

with st.sidebar:
    st.page_link("pages/2_Business_Review.py", label="Run the multi-agent business review", icon="🧭")
    st.subheader("Try a question")
    for q in SAMPLES:
        if st.button(q, width="stretch"):
            st.session_state.pending = q
    if st.button("↺ New conversation", width="stretch"):
        st.session_state.messages, st.session_state.prev_plan = [], None
    st.divider()
    with st.expander("Metric glossary", expanded=False):
        for m in agent.sem.metrics.values():
            st.markdown(f"**{m.label}** — `{m.name}`  \n{m.description}  \n<small>Owner: {m.owner}</small>",
                        unsafe_allow_html=True)
    with st.expander("Governance rules"):
        g = agent.sem.governance
        st.markdown(f"- Aggregates only; PII columns blocked: `{', '.join(c.split('.')[-1] for c in g['pii_columns'])}`\n"
                    f"- Groups under **{g['small_cell_threshold']} customers** are suppressed\n"
                    f"- Read-only connection, max **{g['max_rows']}** rows per query\n"
                    f"- Only allow-listed tables")
    st.caption("Set ANTHROPIC_API_KEY to switch the planner to Claude." if not agent.llm else "")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg["content"], agent)

question = st.chat_input("e.g. NPF ratio by segment and region") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Planning → validating → compiling → querying…"):
            out = agent.ask(question, st.session_state.prev_plan)
        try:
            out["event_id"] = log_event(question, out)
        except OSError:
            out["event_id"] = None
        render_answer(out, agent)
    if out.get("status") == "ok":
        st.session_state.prev_plan = out["plan"]
    st.session_state.messages.append({"role": "assistant", "content": out})
