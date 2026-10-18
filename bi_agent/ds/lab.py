"""Run the whole Data Science Lab, cache it per data snapshot, validate it against planted truth,
and write the reports (lab report, model card, validation)."""
from __future__ import annotations

import pickle
import time
import traceback
from pathlib import Path

import pandas as pd

from .. import config
from . import causal, churn, experiment, forecast, segments
from .common import fingerprint, ground_truth, pct, sar, template, write_report

MODULES = ["churn", "causal", "forecast", "experiment", "segments"]
LAB_VERSION = "2"          # bump when module logic changes, so cached runs are recomputed


def _cache_path(db_path: str | None, fp: str) -> Path:
    base = Path(db_path).parent if db_path else config.DB_PATH.parent
    return base / "ds_cache" / f"lab_v{LAB_VERSION}_{fp}.pkl"


def run_lab(db_path: str | None = None, use_cache: bool = True, modules: list[str] | None = None) -> dict:
    """Run every module (or a subset). A failing module is reported, not fatal."""
    try:
        ground_truth(db_path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("This database was built by an older version of the project. "
                           "Rebuild it with: python scripts/generate_data.py") from e
    fp = fingerprint(db_path)
    path = _cache_path(db_path, fp)
    if use_cache and modules is None and path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    out: dict = {"fingerprint": fp, "errors": {}, "timings": {}}
    runners = {
        "churn": lambda: churn.train_and_evaluate(db_path),
        "causal": lambda: {"campaign": causal.campaign_effect(db_path), "fee_cut": causal.fee_cut_effect(db_path)},
        "forecast": lambda: forecast.run(db_path),
        "experiment": lambda: experiment.run(db_path),
        "segments": lambda: segments.run(db_path),
    }
    for name in modules or MODULES:
        t0 = time.perf_counter()
        try:
            out[name] = runners[name]()
        except Exception as e:  # noqa: BLE001
            out["errors"][name] = f"{type(e).__name__}: {e}"
            out[f"{name}_trace"] = traceback.format_exc()
        out["timings"][name] = round(time.perf_counter() - t0, 2)
    out["findings"] = lab_findings(out)
    if modules is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(out, f)
    return out


def lab_findings(r: dict) -> list[dict]:
    """Findings in the orchestrator's format (domain 'data_science')."""
    rows: list[dict] = []
    add = lambda kind, texts, sev="info": rows.extend(  # noqa: E731
        {"domain": "data_science", "kind": kind, "text": t, "severity": sev, "metric": None, "facts": {}} for t in texts)
    if "causal" in r:
        add("causal", causal.findings(r["causal"]["campaign"], r["causal"]["fee_cut"]), "positive")
    if "churn" in r:
        add("prediction", churn.findings(r["churn"]), "info")
    if "forecast" in r:
        add("forecast", forecast.findings(r["forecast"]))
    if "experiment" in r:
        add("experiment", experiment.findings(r["experiment"]), "positive" if r["experiment"]["ship"] else "watch")
    if "segments" in r:
        add("segmentation", segments.findings(r["segments"]))
    for name, err in r.get("errors", {}).items():
        add("error", [f"The {name} module did not run: {err.split(':')[0]}."], "watch")
    return rows


# ------------------------------------------------------------------------------------- validation
def _planted_experiment_lift(db_path: str | None, truth: dict) -> float:
    df = template("""
        SELECT e.variant, e.pre_remittances_28d AS pre, c.age_band
        FROM fact_experiment e JOIN dim_customer c ON c.customer_id = e.customer_id
        WHERE e.variant = 'treatment'
        LIMIT 1000000""", db_path=db_path, name="experiment_truth")
    lam = 0.45 + 0.8 * df["pre"].to_numpy()
    lift = df["age_band"].map(truth["experiment"]["lift_by_age"]).to_numpy()
    return float((lam * lift).sum() / lam.sum())


def validate(r: dict, db_path: str | None = None) -> pd.DataFrame:
    """Compare every estimate with the truth planted by the data generator."""
    truth = ground_truth(db_path)
    rows = []

    def check(module, what, truth_txt, estimate_txt, ok, note=""):
        rows.append({"module": module, "check": what, "planted truth": truth_txt, "estimate": estimate_txt,
                     "result": "PASS" if ok else "FAIL", "note": note})

    if "causal" in r:
        c, f = r["causal"]["campaign"], r["causal"]["fee_cut"]
        u = truth["campaign"]["uplift"]
        check("Causal", "Campaign effect (DiD 95% CI covers truth)", pct(u, sign=True),
              f"{pct(c['did'], sign=True)} ({pct(c['did_ci'][0], sign=True)} to {pct(c['did_ci'][1], sign=True)})",
              c["did_ci"][0] <= u <= c["did_ci"][1])
        check("Causal", "Campaign effect is significant (randomisation inference)", "non-zero",
              f"p = {c['ri_p']:.4f}", c["ri_p"] < 0.05)
        check("Causal", "Synthetic control close to truth (±6 pts)", pct(u, sign=True), pct(c["scm"]["effect"], sign=True),
              abs(c["scm"]["effect"] - u) <= 0.06)
        ch = f["fits"][f["chosen"]]
        check("Causal", "Fee cut: interrupted time series CI covers zero", "0 pts",
              f"{ch['effect_pts']:+.2f} pts ({ch['lo_pts']:+.2f} to {ch['hi_pts']:+.2f})", ch["lo_pts"] <= 0 <= ch["hi_pts"])
        check("Causal", "Fee cut: naive before/after is biased (shows why the method matters)", "0%",
              pct(f["naive"], sign=True), abs(f["naive"]) > 0.03, "expected to differ from truth")
    if "churn" in r:
        m = r["churn"]["metrics"]
        check("Churn", "Out-of-time ROC-AUC ≥ 0.75", "disengagement before churn", f"{m['roc_auc']:.2f}", m["roc_auc"] >= 0.75)
        check("Churn", "Top decile captures ≥ 40% of churners", "", pct(m["top_decile_capture"], 0), m["top_decile_capture"] >= 0.4)
        check("Churn", "Calibration beats the base rate (Brier)", "", f"{m['brier_calibrated']:.4f} vs {m['brier_base_rate']:.4f}",
              m["brier_calibrated"] <= m["brier_base_rate"])
        check("Churn", "No fairness flags", "gender/nationality not used", "none" if not r["churn"]["fairness_flags"]
              else "; ".join(r["churn"]["fairness_flags"]), not r["churn"]["fairness_flags"])
    if "forecast" in r:
        fo = r["forecast"]
        check("Forecast", "Ramadan effect recovered (±10 pts)", "+35%", pct(fo["effects"]["Ramadan week"], sign=True),
              abs(fo["effects"]["Ramadan week"] - 0.35) <= 0.10)
        check("Forecast", "Pre-Eid effect recovered (±10 pts)", "+25%", pct(fo["effects"]["Pre-Eid week"], sign=True),
              abs(fo["effects"]["Pre-Eid week"] - 0.25) <= 0.10)
        s = fo["backtest_summary"]
        check("Forecast", "Beats 'same week last year' in backtests", "", f"WAPE {pct(s['Calendar model (Hijri-aware)'])} vs "
              f"{pct(s['Seasonal naive (same week last year)'])}",
              s["Calendar model (Hijri-aware)"] < s["Seasonal naive (same week last year)"])
    if "experiment" in r:
        e = r["experiment"]
        planted = _planted_experiment_lift(db_path, truth)
        check("A/B test", "CUPED 95% CI covers the planted lift", pct(planted, sign=True),
              f"{pct(e['cuped']['lift'], sign=True)} ({pct(e['cuped']['lo'], sign=True)} to {pct(e['cuped']['hi'], sign=True)})",
              e["cuped"]["lo"] <= planted <= e["cuped"]["hi"])
        check("A/B test", "A/A false-positive rate near 5%", "5%", pct(e["aa"]["false_positive_rate"]),
              0.02 <= e["aa"]["false_positive_rate"] <= 0.08)
        check("A/B test", "No sample ratio mismatch", "50/50", f"p = {e['srm_p']:.2f}", e["srm_p"] > 0.001)
        check("A/B test", "Guardrail unchanged", "no effect on failures",
              f"{e['guardrail']['diff_pts']:+.2f} pts", e["guardrail"]["ok"])
    if "segments" in r:
        s = r["segments"]
        check("Segments", "Stable under bootstrap (ARI ≥ 0.8)", "no planted segments", f"{s['stability_ari']:.2f}",
              s["stability_ari"] >= 0.8)
    for name, err in r.get("errors", {}).items():
        check(name, "Module ran", "", err, False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- reports
def lab_markdown(r: dict, val: pd.DataFrame) -> str:
    L = ["# Data Science Lab report", "",
         f"*Data snapshot `{r['fingerprint']}` · synthetic data (Wadi Bank is fictional) · "
         f"{(val['result'] == 'PASS').sum()}/{len(val)} validation checks passed*", "", "## Headlines", ""]
    L += [f"- {f['text']}" for f in r["findings"]] + [""]
    if "causal" in r:
        c = r["causal"]["campaign"]
        L += ["## Causal inference: Western cashback campaign", "",
              "| Method | Estimate | Uncertainty |", "|---|---|---|",
              f"| Naive before/after ({c['naive_window']}) | {pct(c['naive'], sign=True)} | none |",
              f"| Difference-in-differences (city and month fixed effects) | {pct(c['did'], sign=True)} | 95% CI "
              f"{pct(c['did_ci'][0], sign=True)} to {pct(c['did_ci'][1], sign=True)}; randomisation p = {c['ri_p']:.4f} |",
              f"| DiD + treated-group trend (robustness) | {pct(c['trend_adjusted'], sign=True)} | |",
              f"| Synthetic control (Western region) | {pct(c['scm']['effect'], sign=True)} | rank {c['scm']['rank']} of "
              f"{c['scm']['units']} placebos (p = {c['scm']['p']:.3f}) |", "",
              f"Pre-trend check: treated vs control slope difference {pct(c['pretrend_slope'], 2, sign=True)} a month "
              f"(p = {c['pretrend_p']:.3f}). Spend per customer (noisier): {pct(c['spend_effect'], sign=True)}.", ""]
        f = r["causal"]["fee_cut"]
        L += ["## Causal inference: digital fee cut", "", "| Interrupted time-series model (digital share) | Shift at the cut | 95% CI | p | AIC |",
              "|---|---|---|---|---|"]
        for k, v in f["fits"].items():
            L.append(f"| {k}{' (chosen)' if k == f['chosen'] else ''} | {v['effect_pts']:+.2f} pts | "
                     f"{v['lo_pts']:+.2f} to {v['hi_pts']:+.2f} | {v['p']:.3f} | {v['aic']:.1f} |")
        L += ["", f"Naive: digital remittances {pct(f['naive'], sign=True)}, digital share {f['naive_share_pts']:+.1f} pts "
                  f"(6 months after vs before). Pre-period trend: {f['pretrend_pts']:+.2f} pts a month "
                  f"(p = {f['pretrend_p']:.3g}). Placebo dates: {f['placebo']['min']:+.2f} to {f['placebo']['max']:+.2f} pts. "
                  f"Fee change, digital vs branch: SAR {f['fee_did_sar']:.2f}.", ""]
    if "churn" in r:
        ch = r["churn"]
        L += ["## Churn model", "", f"Selected: {ch['model_name']}. See `churn_model_card.md` for the full model card.", "",
              "| Decile | Customers | Predicted | Observed | Lift | Cumulative capture |", "|---|---|---|---|---|---|"]
        for _, d in ch["lift"].iterrows():
            L.append(f"| {int(d['decile'])} | {int(d['customers']):,} | {pct(d['predicted'], 2)} | {pct(d['observed'], 2)} | "
                     f"{d['lift']:.1f}x | {pct(d['captured'], 0)} |")
        s = ch["scoring"]
        L += ["", "Next-quarter retention plan by segment:", "", "| Segment | Customers | Expected churners | Contact | Expected net value |",
              "|---|---|---|---|---|"]
        for _, b in s["by_segment"].iterrows():
            L.append(f"| {b['segment']} | {int(b['customers']):,} | {b['expected_churners']:.0f} | {int(b['contact']):,} | "
                     f"{sar(b['expected_net_value'])} |")
        L.append("")
    if "forecast" in r:
        fo = r["forecast"]
        L += ["## Forecast", "", "| Model | Mean WAPE (rolling backtests) | WAPE around Ramadan 2026 |", "|---|---|---|"]
        for k, v in fo["backtest_summary"].items():
            L.append(f"| {k} | {pct(v)} | {pct(fo['ramadan_wape'][k])} |")
        L += ["", "Calendar effects learned by the model: " +
              ", ".join(f"{k} {pct(v, sign=True)}" for k, v in fo["effects"].items()) + ".", ""]
    if "experiment" in r:
        e = r["experiment"]
        L += ["## A/B test", "", f"- Power: minimum detectable effect {pct(e['power']['mde_raw'])} without CUPED, "
              f"{pct(e['power']['mde_cuped'])} with CUPED, at {e['power']['n_per_arm']:,} customers per arm. A 3% effect needs "
              f"{e['power']['n_for_3pct_raw']:,} per arm without CUPED or {e['power']['n_for_3pct_cuped']:,} with it.",
              f"- A/A check: {pct(e['aa']['false_positive_rate'])} of {e['aa']['runs']} random splits were 'significant'.", "",
              "| Age band | Customers | Lift | Bonferroni CI |", "|---|---|---|---|"]
        for _, h in e["hte"].iterrows():
            L.append(f"| {h['age_band']} | {int(h['customers']):,} | {pct(h['lift'], sign=True)} | "
                     f"{pct(h['lo'], sign=True)} to {pct(h['hi'], sign=True)} |")
        L.append("")
    if "segments" in r:
        s = r["segments"]
        L += ["## Segmentation", "", "| Segment | Senders | Share of volume | Transfers / year | Avg transfer | Days since last |",
              "|---|---|---|---|---|---|"]
        for _, p in s["profiles"].iterrows():
            L.append(f"| {p['name']} | {int(p['customers']):,} | {pct(p['share_volume'], 0)} | {p['remittances']:.1f} | "
                     f"{sar(p['avg_amount'])} | {p['recency_days']:.0f} |")
        L.append("")
    L += ["## Validation against planted truth", "", "| Module | Check | Planted truth | Estimate | Result |", "|---|---|---|---|---|"]
    for _, v in val.iterrows():
        L.append(f"| {v['module']} | {v['check']} | {v['planted truth']} | {v['estimate']} | {v['result']} |")
    L += ["", "Timings (seconds): " + ", ".join(f"{k} {v}" for k, v in r["timings"].items()) + "."]
    return "\n".join(L) + "\n"


def write_reports(r: dict, db_path: str | None = None) -> pd.DataFrame:
    val = validate(r, db_path)
    write_report("lab_report.md", lab_markdown(r, val))
    if "churn" in r:
        write_report("churn_model_card.md", churn.model_card(r["churn"], r["fingerprint"]))
    rows = ["| " + " | ".join(val.columns) + " |", "|" + "---|" * len(val.columns)]
    rows += ["| " + " | ".join(str(v) for v in r_) + " |" for r_ in val.itertuples(index=False)]
    write_report("validation.md", "# Validation against planted truth\n\n" + "\n".join(rows) + "\n")
    return val
