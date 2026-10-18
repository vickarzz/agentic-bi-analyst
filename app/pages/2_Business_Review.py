"""Multi-agent cross-business review page."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bi_agent.db import DataNotFound  # noqa: E402
from bi_agent.orchestrator.domains import DOMAINS  # noqa: E402
from bi_agent.orchestrator.graph import BusinessReviewOrchestrator  # noqa: E402

st.set_page_config(page_title="Business Review", page_icon="🧭", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1200px;}
  .tagline {color:#6b7280; margin-top:-0.6rem; font-size:0.95rem;}
  .card {border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px; margin-bottom:8px;}
  .muted {color:#6b7280; font-size:0.85rem;}
</style>""", unsafe_allow_html=True)

VERDICT_COLOR = {"confirmed": "#0E6E69", "not detected": "#9CA3AF", "contradicted": "#B9472F",
                 "control passed": "#C8902E", "control FAILED": "#B9472F"}


def ensure_data():
    from bi_agent.data_setup import ensure_current_data
    ensure_current_data()          # builds the data on first start, or rebuilds data from an older version


@st.cache_data(show_spinner=False)
def run_review(request: str) -> dict:
    ensure_data()
    out = BusinessReviewOrchestrator().run(request)
    out["heatmap"] = out["heatmap"].to_dict()
    out["panel"] = None
    return out


def impact_map(links: list[dict]) -> go.Figure:
    names = list(DOMAINS)
    pos = {d: (math.cos(math.pi / 2 + 2 * math.pi * i / len(names)), math.sin(math.pi / 2 + 2 * math.pi * i / len(names)))
           for i, d in enumerate(names)}
    fig = go.Figure()
    for l in links:
        a, b = l["from_domain"], l["to_domain"]
        color = VERDICT_COLOR.get(l["verdict"], "#9CA3AF")
        dash = "solid" if l["verdict"] == "confirmed" else "dot"
        if a == b:
            x, y = pos[a]
            fig.add_trace(go.Scatter(x=[x * 1.28], y=[y * 1.28], mode="markers", marker=dict(size=12, color=color, symbol="circle-open-dot"),
                                     hovertext=f"{l['title']}<br>{l['verdict']}", hoverinfo="text", showlegend=False))
            continue
        (x0, y0), (x1, y1) = pos[a], pos[b]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=color, width=4 if dash == "solid" else 2, dash=dash),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="markers", marker=dict(size=10, color=color),
                                 hovertext=f"<b>{l['title']}</b><br>{l['verdict']}<br>{l['summary'][:140]}", hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[pos[d][0] for d in names], y=[pos[d][1] for d in names], mode="markers+text",
                             marker=dict(size=46, color="#12262E"), text=[DOMAINS[d]["label"].replace(" & ", "<br>& ") for d in names],
                             textposition="bottom center", textfont=dict(size=12), hoverinfo="skip", showlegend=False))
    for v, c in VERDICT_COLOR.items():
        if v != "control FAILED":
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=c, width=3), name=v))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                      xaxis=dict(visible=False, range=[-1.6, 1.6]), yaxis=dict(visible=False, range=[-1.55, 1.5], scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.02))
    return fig


st.title("🧭 Cross-business review")
st.markdown("<p class='tagline'>A supervisor agent dispatches five domain agents, an impact analyst, a discovery agent "
            "and a data scientist in parallel, then a synthesizer writes the briefing and a verifier checks every number. "
            "Governed metrics only · synthetic data.</p>", unsafe_allow_html=True)

request = st.text_input("What should the agents review?",
                        "Review the whole business and how each line affects the others")
if st.button("Run review", type="primary") or "review" not in st.session_state:
    with st.spinner("Supervisor → 8 agents in parallel → synthesizer → verifier (first run trains the models, ~30 s)…"):
        try:
            st.session_state.review = run_review(request)
        except DataNotFound as e:
            st.error(str(e))
            st.stop()
out = st.session_state.review
b = out["briefing"]
links = b["links"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Links confirmed", f"{sum(l['verdict'] == 'confirmed' for l in links)} / {len(links)}")
c2.metric("Risk alerts", len(b["alerts"]))
c3.metric("Watch items", len(b["watch"]))
v = b.get("verification", {})
c4.metric("Verifier", "Passed" if v.get("passed") else "Failed", f"{v.get('numbers_checked', 0)} numbers checked",
          delta_color="off")

left, right = st.columns([1.05, 1])
with left:
    st.subheader("Executive summary")
    for s_ in b["executive_summary"]:
        st.markdown(f"- {s_}")
with right:
    st.subheader("Impact map")
    st.plotly_chart(impact_map(links), width="stretch")
    st.caption("Lines connect business lines whose link was tested. Hover the dots for evidence. "
               "Ring markers are links inside one business line.")

tabs = st.tabs(["Impact tests", "Actions", "Risks", "Domain scorecards", "Discovery", "Data science", "Agent trace",
                "Full briefing"])
with tabs[0]:
    df = pd.DataFrame([{"Link": l["title"], "Verdict": l["verdict"], "Test": l["test_type"],
                        "Effect": l["effect"], "p-value": l["p_value"], "Evidence": l["summary"]} for l in links])
    st.dataframe(df, width="stretch", hide_index=True)
    pick = st.selectbox("Inspect a link", [l["title"] for l in links])
    l = next(x for x in links if x["title"] == pick)
    st.markdown(f"**Hypothesis:** {l['hypothesis']}  \n**Owner action:** {l['action']}")
    d = l["details"]
    if l["test_type"] == "timeseries" and d.get("lags"):
        fig = px.bar(pd.DataFrame(d["lags"]), x="lag", y="r", text="r", labels={"lag": "Lag (months)", "r": "Correlation r"})
        fig.update_layout(height=280, template="plotly_white", yaxis_range=[-1, 1])
        st.plotly_chart(fig, width="stretch")
    elif l["test_type"] == "cross_section" and d.get("points"):
        fig = px.scatter(pd.DataFrame(d["points"]), x="x", y="y", hover_name="group", trendline=None)
        fig.update_layout(height=320, template="plotly_white")
        st.plotly_chart(fig, width="stretch")
    elif l["test_type"] == "cohort" and d:
        fig = px.bar(x=[d["cohort_a"], d["cohort_b"]], y=[d["value_a"], d["value_b"]], text_auto=".1%",
                     labels={"x": "Cohort", "y": "NPF ratio"})
        fig.update_layout(height=300, template="plotly_white", yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    elif l["test_type"] == "event" and d.get("ok"):
        fig = px.bar(x=["6 months before", "6 months after"], y=[d["before"], d["after"]], text_auto=".3s",
                     labels={"x": "", "y": "Average"})
        fig.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig, width="stretch")
with tabs[1]:
    for i, a in enumerate(b["actions"], 1):
        st.markdown(f"<div class='card'><b>{i}. {a['link']}</b> <span class='muted'>({a['verdict']})</span><br>{a['action']}</div>",
                    unsafe_allow_html=True)
with tabs[2]:
    for a in b["alerts"]:
        st.error(a)
    for w in b["watch"]:
        st.warning(w)
with tabs[3]:
    icon = {"alert": "🔴", "watch": "🟠", "positive": "🟢", "info": "⚪"}
    for dname, dv in b["domains"].items():
        with st.expander(dv["label"], expanded=dname == "credit"):
            for f in dv["findings"]:
                st.markdown(f"{icon.get(f['severity'], '')} {f['text']}")
with tabs[4]:
    hm = pd.DataFrame(out["heatmap"])
    fig = px.imshow(hm, color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".1f", aspect="auto")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Correlation of month-over-month growth between governed KPIs.")
    for dsc in b["discoveries"]:
        st.info(dsc["text"])
    if not b["discoveries"]:
        st.write("No new cross-domain candidates beyond the curated hypotheses.")
with tabs[5]:
    ds = b.get("data_science", [])
    if ds:
        for f in ds:
            st.markdown(f"<div class='card'>{f['text']}</div>", unsafe_allow_html=True)
        try:
            st.page_link("pages/3_Data_Science_Lab.py", label="Open the Data Science Lab for charts and the model card", icon="🧪")
        except Exception:  # noqa: BLE001 - page links only resolve when launched from the main app
            st.caption("Open the Data Science Lab page for charts and the model card.")
    else:
        st.write("The data scientist was not in scope for this request.")
with tabs[6]:
    tr = pd.DataFrame(out["trace"])
    st.dataframe(tr, width="stretch", hide_index=True)
    fig = px.bar(tr, x="ms", y="agent", color="group", orientation="h", labels={"ms": "milliseconds", "agent": ""},
                 color_discrete_map={"parallel": "#0E6E69", "": "#9CA3AF"})
    fig.update_layout(height=380, template="plotly_white", yaxis=dict(autorange="reversed"), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(out['tool_calls'])} governed tool calls. Teal bars ran in parallel.")
with tabs[7]:
    st.markdown(out["markdown"])
    st.download_button("Download briefing (.md)", out["markdown"], "business_review.md")
    st.download_button("Download briefing (.json)", json.dumps(b, indent=2, default=str), "business_review.json")
