"""Data Science Lab page: causal inference, churn model, forecast, A/B test, segmentation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bi_agent.ds import churn as churn_mod  # noqa: E402
from bi_agent.ds import experiment as exp_mod  # noqa: E402
from bi_agent.ds.common import pct, sar  # noqa: E402
from bi_agent.ds.lab import run_lab, validate  # noqa: E402

st.set_page_config(page_title="Data Science Lab", page_icon="🧪", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1200px;}
  .tagline {color:#6b7280; margin-top:-0.6rem; font-size:0.95rem;}
  .fact {border-left: 3px solid #0E6E69; padding: 2px 10px; margin: 6px 0; font-size: 0.95rem;}
</style>""", unsafe_allow_html=True)
TEAL, GOLD, RUST, GREY = "#0E6E69", "#C8902E", "#B9472F", "#9CA3AF"
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), height=360, legend_title=None)


def lay(fig, **kw):
    fig.update_layout(**{**LAYOUT, **kw})


def ensure_data():
    from bi_agent.data_setup import ensure_current_data
    ensure_current_data()          # builds the data on first start, or rebuilds data from an older version


@st.cache_resource(show_spinner="Running the Data Science Lab (about 20 seconds the first time)…")
def lab():
    ensure_data()
    r = run_lab()
    return r, validate(r)


r, val = lab()
st.title("🧪 Data Science Lab")
st.markdown("<p class='tagline'>Causal inference, prediction, forecasting, experimentation and segmentation on the same "
            "governed data as the agents. Every estimate is checked against the truth planted in the synthetic data.</p>",
            unsafe_allow_html=True)
passed = int((val["result"] == "PASS").sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Validation vs planted truth", f"{passed} / {len(val)} checks")
if "causal" in r:
    c2.metric("Campaign effect (DiD)", pct(r["causal"]["campaign"]["did"], sign=True),
              f"naive said {pct(r['causal']['campaign']['naive'], sign=True)}", delta_color="off")
if "churn" in r:
    c3.metric("Churn model ROC-AUC (out of time)", f"{r['churn']['metrics']['roc_auc']:.2f}",
              f"top 10% hold {pct(r['churn']['metrics']['top_decile_capture'], 0)} of churners", delta_color="off")
if "forecast" in r:
    s = r["forecast"]["backtest_summary"]
    c4.metric("Forecast error (WAPE)", pct(s["Calendar model (Hijri-aware)"]),
              f"vs {pct(s['Seasonal naive (same week last year)'])} seasonal naive", delta_color="off")
if r.get("errors"):
    st.warning("Some modules did not run: " + "; ".join(f"{k}: {v}" for k, v in r["errors"].items()))

tabs = st.tabs(["Overview", "Causal impact", "Churn model", "Forecast", "A/B test", "Segments", "Validation & governance"])

# ------------------------------------------------------------------------------------------- overview
with tabs[0]:
    st.subheader("What the models say")
    for f in r["findings"]:
        st.markdown(f"<div class='fact'>{f['text']}</div>", unsafe_allow_html=True)
    st.caption("The same findings feed the data-scientist agent in the Business Review, where the verifier checks every number.")

# ------------------------------------------------------------------------------------------- causal
with tabs[1]:
    if "causal" in r:
        c = r["causal"]["campaign"]
        st.subheader("Did the Western cashback campaign work?")
        st.markdown(f"Treated: {', '.join(c['treated_cities'])}. Controls: the other {len(c['control_cities'])} cities. "
                    f"Outcome: {c['outcome']}. Campaign: {c['event']['start']} to {c['event']['end']}.")
        est = pd.DataFrame([
            {"method": "Naive before/after", "effect": c["naive"], "lo": np.nan, "hi": np.nan},
            {"method": "Difference-in-differences", "effect": c["did"], "lo": c["did_ci"][0], "hi": c["did_ci"][1]},
            {"method": "DiD + group trend", "effect": c["trend_adjusted"], "lo": np.nan, "hi": np.nan},
            {"method": "Synthetic control", "effect": c["scm"]["effect"], "lo": np.nan, "hi": np.nan}])
        fig = go.Figure(go.Bar(x=est["effect"], y=est["method"], orientation="h", marker_color=[GREY, TEAL, TEAL, GOLD],
                               error_x=dict(type="data", symmetric=False, array=(est["hi"] - est["effect"]).fillna(0),
                                            arrayminus=(est["effect"] - est["lo"]).fillna(0)),
                               text=[pct(v, sign=True) for v in est["effect"]], textposition="inside",
                               insidetextanchor="start", textfont=dict(color="white")))
        lay(fig, title="Estimated effect by method", xaxis_tickformat=".0%", height=280)
        st.plotly_chart(fig, width="stretch")

        a, b = st.columns(2)
        with a:
            e = c["event_study"]
            fig = go.Figure()
            fig.add_vrect(x0=-0.5, x1=2.5, fillcolor=GOLD, opacity=0.12, line_width=0)
            fig.add_trace(go.Scatter(x=e["rel_month"], y=e["effect"], mode="markers+lines", line_color=TEAL,
                                     error_y=dict(type="data", symmetric=False, array=e["hi"] - e["effect"],
                                                  arrayminus=e["effect"] - e["lo"]), name="effect"))
            fig.add_hline(y=0, line_color=GREY)
            lay(fig, title=f"Event study (pre-trend test p = {c['pretrend_p']:.3f})",
                              xaxis_title="months since campaign start", yaxis_tickformat=".0%", showlegend=False)
            st.plotly_chart(fig, width="stretch")
            st.caption("Flat before the shaded campaign months, a jump during them, and back to zero after: the effect "
                       "is real and temporary.")
        with b:
            sc = c["scm"]["series"]
            fig = go.Figure()
            fig.add_vrect(x0=c["event"]["start"], x1=c["event"]["end"], fillcolor=GOLD, opacity=0.12, line_width=0)
            fig.add_trace(go.Scatter(x=sc["month"], y=sc["actual"], name="Western region", line_color=TEAL))
            fig.add_trace(go.Scatter(x=sc["month"], y=sc["synthetic"], name="Synthetic Western", line=dict(color=GREY, dash="dash")))
            lay(fig, title=f"Synthetic control: {pct(c['scm']['effect'], sign=True)} "
                                              f"(rank {c['scm']['rank']} of {c['scm']['units']})",
                              yaxis_title="card transactions per customer")
            st.plotly_chart(fig, width="stretch")
            st.caption("Weights: " + ", ".join(f"{k} {v:.0%}" for k, v in c["scm"]["weights"].items()))
        fig = px.histogram(x=c["ri_draws"], nbins=60, color_discrete_sequence=[GREY])
        fig.add_vline(x=c["ri_obs"], line_color=RUST, annotation_text="actual Western assignment")
        lay(fig, title=f"Randomisation inference: every way to pick {len(c['treated_cities'])} of "
                                          f"{len(c['treated_cities']) + len(c['control_cities'])} cities (p = {c['ri_p']:.4f})",
                          xaxis_tickformat=".0%", xaxis_title="DiD estimate", yaxis_title="assignments", height=300)
        st.plotly_chart(fig, width="stretch")

        f = r["causal"]["fee_cut"]
        st.subheader("Did the digital fee cut move customers to digital?")
        se = f["series"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=se["month"], y=se["share"], mode="markers", marker_color=TEAL, name="digital share"))
        fig.add_trace(go.Scatter(x=se["month"], y=se["fitted"], line_color=TEAL, name="model"))
        fig.add_trace(go.Scatter(x=se["month"], y=se["counterfactual"], line=dict(color=GREY, dash="dash"), name="no-cut counterfactual"))
        fig.add_vline(x=f["event"]["start"], line_color=RUST, line_dash="dash")
        lay(fig, title="Digital share of remittances: the trend was there before the cut", yaxis_tickformat=".0%", height=320)
        st.plotly_chart(fig, width="stretch")
        fits = pd.DataFrame([{"model": k + (" ✓ chosen by AIC" if k == f["chosen"] else ""),
                              "shift at the cut": f"{v['effect_pts']:+.2f} pts",
                              "95% CI": f"{v['lo_pts']:+.2f} to {v['hi_pts']:+.2f} pts", "p": round(v["p"], 3),
                              "AIC": round(v["aic"], 1)} for k, v in f["fits"].items()])
        st.dataframe(fits, hide_index=True, width="stretch")
        st.caption(f"Naive before/after: digital remittances {pct(f['naive'], sign=True)}. Pre-cut trend "
                   f"{f['pretrend_pts']:+.2f} pts a month. Placebo dates: {f['placebo']['min']:+.2f} to "
                   f"{f['placebo']['max']:+.2f} pts. The fee itself fell by SAR {abs(f['fee_did_sar']):.2f}.")

# ------------------------------------------------------------------------------------------- churn
with tabs[2]:
    if "churn" in r:
        ch, m = r["churn"], r["churn"]["metrics"]
        st.subheader(f"90-day churn model · {ch['model_name']}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ROC-AUC (out of time)", f"{m['roc_auc']:.2f}")
        k2.metric("PR-AUC", f"{m['pr_auc']:.2f}", f"base rate {pct(m['base_rate'])}", delta_color="off")
        k3.metric("Top-decile lift", f"{m['top_decile_lift']:.1f}x")
        k4.metric("Brier (calibrated)", f"{m['brier_calibrated']:.4f}", f"base rate {m['brier_base_rate']:.4f}", delta_color="off")
        a, b = st.columns(2)
        with a:
            lt = ch["lift"]
            fig = go.Figure(go.Bar(x=lt["decile"], y=lt["captured"], marker_color=TEAL,
                                   text=[pct(v, 0) for v in lt["captured"]], textposition="outside"))
            lay(fig, title="Cumulative share of churners captured by risk decile",
                              xaxis_title="risk decile (1 = riskiest 10%)", yaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
        with b:
            cal = ch["calibration"]
            fig = go.Figure()
            mx = max(cal["predicted"].max(), cal["observed"].max()) * 1.1
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=GREY, dash="dash"), name="perfect"))
            fig.add_trace(go.Scatter(x=cal["predicted"], y=cal["observed"], mode="markers+lines", line_color=TEAL, name="model"))
            lay(fig, title="Calibration on the test set", xaxis_title="predicted risk",
                              yaxis_title="observed churn", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
            st.plotly_chart(fig, width="stretch")
        imp = ch["importance"].head(8).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["label"], orientation="h", marker_color=TEAL))
        lay(fig, title="Permutation importance (drop in test ROC-AUC)", height=320)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Should we call them? Retention economics")
        econ = ch["economics"]
        e1, e2 = st.columns(2)
        cost = e1.slider("Cost per retention call (SAR)", 5, 200, int(econ["contact_cost_sar"]), 5)
        save = e2.slider("Share of would-be churners a call saves", 0.05, 0.6, float(econ["save_rate"]), 0.05)
        rec = churn_mod.recommend(ch["_scored"], cost, save)
        x1, x2, x3 = st.columns(3)
        x1.metric("Customers to call", f"{rec['contacts']:,}")
        x2.metric("Expected customers saved", f"{rec['expected_saves']:.1f}")
        x3.metric("Expected net value", sar(rec["expected_net_value"]))
        st.caption("Call a customer when risk × save rate × annual value > call cost. Values per segment: "
                   + ", ".join(f"{k} SAR {v:,}" for k, v in econ["annual_value_sar"].items()) + ".")
        a, b = st.columns(2)
        with a:
            seg = ch["scoring"]["by_segment"].copy()
            seg["expected_churners"] = seg["expected_churners"].round(0)
            seg["expected_net_value"] = seg["expected_net_value"].map(sar)
            st.markdown("**Next quarter by segment** (default assumptions)")
            st.dataframe(seg, hide_index=True, width="stretch")
        with b:
            rc = ch["scoring"]["reason_codes"]
            if len(rc):
                fig = go.Figure(go.Bar(x=rc["customers"][::-1], y=rc["reason"][::-1], orientation="h", marker_color=GOLD))
                lay(fig, title="Most common reasons among customers to call", height=300)
                st.plotly_chart(fig, width="stretch")
        st.subheader("Fairness audit")
        fair = ch["fairness"].copy()
        for col in ("observed", "predicted", "contact_rate", "recall"):
            fair[col] = fair[col].map(lambda v: pct(v, 2))
        fair["calibration_gap_pts"] = fair["calibration_gap_pts"].round(2)
        fair["roc_auc"] = fair["roc_auc"].round(3)
        st.dataframe(fair, hide_index=True, width="stretch")
        st.caption("Gender and nationality are not model inputs; they are used only to check that the model is equally "
                   "calibrated and finds churners equally well across groups. "
                   + ("Flags: " + "; ".join(ch["fairness_flags"]) if ch["fairness_flags"] else "No flags."))
        st.caption(f"Drift: score PSI between training and today = {ch['drift']['score_psi']:.3f} (alert above 0.2).")

# ------------------------------------------------------------------------------------------- forecast
with tabs[3]:
    if "forecast" in r:
        fo = r["forecast"]
        st.subheader("Weekly remittance volume, Hijri-aware forecast")
        s, fc = fo["series"], fo["forecast"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s["week"], y=s["actual"], name="actual", line_color=GREY))
        fig.add_trace(go.Scatter(x=s["week"], y=s["fitted"], name="model fit", line=dict(color=TEAL, width=1)))
        fig.add_trace(go.Scatter(x=list(fc["week"]) + list(fc["week"][::-1]), y=list(fc["hi"]) + list(fc["lo"][::-1]),
                                 fill="toself", fillcolor="rgba(14,110,105,0.15)", line_width=0, name="80% interval"))
        fig.add_trace(go.Scatter(x=fc["week"], y=fc["forecast"], name="forecast", line=dict(color=TEAL, width=3)))
        lay(fig, yaxis_title="SAR per week", height=380)
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"Next {fo['horizon_weeks']} weeks: **{sar(fo['total'])}** (80% interval {sar(fo['total_lo'])} to "
                    f"{sar(fo['total_hi'])}).")
        a, b = st.columns(2)
        with a:
            rb = fo["ramadan_backtest"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rb["week"], y=rb["actual"], name="actual", line=dict(color="black", width=3)))
            for name, col in (("Calendar model (Hijri-aware)", TEAL), ("Seasonal naive (same week last year)", RUST),
                              ("Moving average (last 4 weeks)", GREY)):
                fig.add_trace(go.Scatter(x=rb["week"], y=rb[name], name=name.split(" (")[0], line_color=col))
            lay(fig, title=f"Backtest from {fo['ramadan_origin']}: Ramadan 2026 came 11 days earlier",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
        with b:
            bs = fo["backtest_summary"].reset_index()
            bs.columns = ["model", "wape"]
            fig = go.Figure(go.Bar(x=bs["wape"], y=bs["model"], orientation="h", marker_color=[TEAL, GREY, RUST][: len(bs)],
                                   text=[pct(v) for v in bs["wape"]], textposition="outside"))
            lay(fig, title=f"Average error over {fo['n_origins']} rolling backtests (WAPE)",
                              xaxis_tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame([{"calendar effect": k, "multiplier": pct(v, sign=True)} for k, v in fo["effects"].items()]),
                     hide_index=True)

# ------------------------------------------------------------------------------------------- experiment
with tabs[4]:
    if "experiment" in r:
        e = r["experiment"]
        st.subheader("A/B test: one-tap remittance flow")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Customers", f"{e['n']:,}", f"{e['n_treatment']:,} vs {e['n_control']:,}", delta_color="off")
        k2.metric("Sample ratio check", f"p = {e['srm_p']:.2f}")
        k3.metric("Lift (CUPED)", pct(e["cuped"]["lift"], sign=True), f"p = {e['cuped']['p']:.3f}", delta_color="off")
        k4.metric("Decision", "Ship" if e["ship"] else "Hold")
        forest = pd.DataFrame([
            {"label": "Overall, no CUPED", "lift": e["raw"]["lift"], "lo": e["raw"]["lo"], "hi": e["raw"]["hi"]},
            {"label": "Overall, CUPED", "lift": e["cuped"]["lift"], "lo": e["cuped"]["lo"], "hi": e["cuped"]["hi"]}]
            + [{"label": f"Age {h['age_band']} (Bonferroni)", "lift": h["lift"], "lo": h["lo"], "hi": h["hi"]}
               for _, h in e["hte"].iterrows()])
        fig = go.Figure(go.Scatter(x=forest["lift"], y=forest["label"], mode="markers", marker=dict(size=10, color=TEAL),
                                   error_x=dict(type="data", symmetric=False, array=forest["hi"] - forest["lift"],
                                                arrayminus=forest["lift"] - forest["lo"])))
        fig.add_vline(x=0, line_color=GREY)
        lay(fig, title=f"Lift in remittances per customer with 95% CIs (CUPED cut variance by "
                                          f"{pct(e['variance_reduction'], 0)})", xaxis_tickformat=".0%",
                          yaxis=dict(autorange="reversed"), height=380)
        st.plotly_chart(fig, width="stretch")
        g = e["guardrail"]
        st.markdown(f"**Guardrail:** failure rate per transfer {g['control_rate'] * 100:.2f}% (control) vs "
                    f"{g['treatment_rate'] * 100:.2f}% (treatment); one-sided upper bound {g['upper_pts']:+.2f} pts vs a "
                    f"{g['margin_pts']:.1f}-pt margin → {'OK' if g['ok'] else 'breached'}. "
                    f"**A/A check:** {pct(e['aa']['false_positive_rate'])} of {e['aa']['runs']} random splits were significant.")
        st.subheader("Plan the next test")
        p = e["power"]
        target = st.slider("Smallest lift worth detecting", 0.01, 0.15, 0.03, 0.01, format="%.2f")
        n_raw = exp_mod.sample_size(p["sd_raw"], p["base"], target)
        n_cup = exp_mod.sample_size(p["sd_cuped"], p["base"], target)
        q1, q2 = st.columns(2)
        q1.metric("Customers per arm, no CUPED", f"{n_raw:,}")
        q2.metric("Customers per arm, with CUPED", f"{n_cup:,}", f"{pct(1 - n_cup / n_raw, 0)} fewer", delta_color="off")

# ------------------------------------------------------------------------------------------- segments
with tabs[5]:
    if "segments" in r:
        sg = r["segments"]
        st.subheader(f"Remittance senders, last {sg['lookback_months']} months")
        a, b = st.columns([1, 1.4])
        with a:
            sil = sg["silhouette"]
            fig = go.Figure(go.Scatter(x=sil["k"], y=sil["silhouette"], mode="lines+markers", line_color=TEAL))
            lay(fig, title=f"Silhouette by k (chosen k = {sg['k']})", xaxis_title="k", height=300)
            st.plotly_chart(fig, width="stretch")
            st.metric("Bootstrap stability (ARI)", f"{sg['stability_ari']:.2f}", f"min {sg['stability_min']:.2f}", delta_color="off")
        with b:
            prof = sg["profiles"].copy()
            show = pd.DataFrame({"segment": prof["name"], "senders": prof["customers"],
                                 "share of senders": prof["share_customers"].map(lambda v: pct(v, 0)),
                                 "share of volume": prof["share_volume"].map(lambda v: pct(v, 0)),
                                 "transfers / year": prof["remittances"].round(1),
                                 "avg transfer": prof["avg_amount"].map(sar),
                                 "days since last": prof["recency_days"].round(0),
                                 "digital share": prof["digital_share"].map(lambda v: pct(v, 0))})
            st.dataframe(show, hide_index=True, width="stretch")
        cm = sg["corridor_mix"]
        fig = px.bar(cm, x="customers", y="name", color="corridor", orientation="h", barmode="stack")
        lay(fig, title="Corridor mix per segment", yaxis_title=None, height=280)
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------------------------------- validation
with tabs[6]:
    st.subheader("Does each method recover the truth planted in the data?")
    st.dataframe(val, hide_index=True, width="stretch")
    st.caption("The data generator plants known effects (a +18% campaign, a zero-effect fee cut, a +35% Ramadan week, "
               "disengagement before churn, an age-dependent A/B lift). This table is the data-science equivalent of the "
               "agents' golden questions, and CI fails if any check fails.")
    if "churn" in r:
        with st.expander("Model card: 90-day churn model", expanded=False):
            st.markdown(churn_mod.model_card(r["churn"], r["fingerprint"]))
    st.caption(f"Data snapshot {r['fingerprint']} · module timings (s): " + ", ".join(f"{k} {v}" for k, v in r["timings"].items()))
