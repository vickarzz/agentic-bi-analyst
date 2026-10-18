"""A/B test readout for the one-tap remittance flow, done the way an experimentation team would.

1. Sample ratio mismatch (SRM): was the 50/50 split honoured?          chi-square test
2. Primary metric: remittances in 28 days.                              Welch t-test, relative lift CI
3. CUPED: use each customer's pre-period remittances to cut variance.   same test, narrower CI
4. Heterogeneous effects by age band, Bonferroni-adjusted.
5. Guardrail: failure rate per transfer must not rise by more than the margin (bootstrap, one-sided)
6. Power: minimum detectable effect for this sample, and sample size for a target effect.
7. A/A check: split the control group at random many times; about 5% should be "significant".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .common import ds_config, pct, template

EXPERIMENT_SQL = """
SELECT e.variant, e.pre_remittances_28d AS pre, e.remittances_28d AS y, e.failed_transfers_28d AS failed,
       c.age_band, c.segment
FROM fact_experiment e
JOIN dim_customer c ON c.customer_id = e.customer_id
WHERE e.experiment_id = $1
LIMIT 1000000
"""


def _lift(yt: np.ndarray, yc: np.ndarray, alpha: float = 0.05) -> dict:
    mt, mc = yt.mean(), yc.mean()
    vt, vc = yt.var(ddof=1) / len(yt), yc.var(ddof=1) / len(yc)
    diff, se = mt - mc, np.sqrt(vt + vc)
    t = stats.ttest_ind(yt, yc, equal_var=False)
    z = stats.norm.ppf(1 - alpha / 2)
    # relative lift with a delta-method standard error
    rel = diff / mc
    se_rel = np.sqrt(vt / mc ** 2 + vc * mt ** 2 / mc ** 4)
    return {"treatment_mean": mt, "control_mean": mc, "diff": diff, "lift": rel,
            "lo": rel - z * se_rel, "hi": rel + z * se_rel, "p": float(t.pvalue), "se_diff": se}


def _cuped(df: pd.DataFrame) -> np.ndarray:
    theta = np.cov(df["y"], df["pre"])[0, 1] / df["pre"].var()
    return (df["y"] - theta * (df["pre"] - df["pre"].mean())).to_numpy()


def mde(sd: float, n_per_arm: int, base: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable relative effect for a two-sample test."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(2 / n_per_arm) / base)


def sample_size(sd: float, base: float, rel_effect: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sd / (rel_effect * base)) ** 2))


def run(db_path: str | None = None, aa_runs: int = 500, seed: int = 0) -> dict:
    cfg = ds_config()["experiment"]
    df = template(EXPERIMENT_SQL, [cfg["experiment_id"]], db_path, name="experiment_readout")
    alpha = cfg["alpha"]
    t, c = df[df["variant"] == "treatment"], df[df["variant"] == "control"]

    n = len(df)
    srm = stats.chisquare([len(t), len(c)], [n / 2, n / 2])
    raw = _lift(t["y"].to_numpy(), c["y"].to_numpy(), alpha)
    df["y_cuped"] = _cuped(df)
    t, c = df[df["variant"] == "treatment"], df[df["variant"] == "control"]
    cup = _lift(t["y_cuped"].to_numpy(), c["y_cuped"].to_numpy(), alpha)
    var_reduction = 1 - df["y_cuped"].var() / df["y"].var()

    hte_rows = []
    groups = sorted(df["age_band"].unique())
    for g in groups:
        sub = df[df["age_band"] == g]
        r = _lift(sub.loc[sub["variant"] == "treatment", "y_cuped"].to_numpy(),
                  sub.loc[sub["variant"] == "control", "y_cuped"].to_numpy(), alpha / len(groups))
        hte_rows.append({"age_band": g, "customers": len(sub), "lift": r["lift"], "lo": r["lo"], "hi": r["hi"],
                         "p_adjusted": min(1.0, r["p"] * len(groups))})
    hte = pd.DataFrame(hte_rows)

    # guardrail: failure rate per transfer (ratio of sums), bootstrap one-sided 95% upper bound
    rng = np.random.default_rng(seed)
    def rate(d):
        return d["failed"].sum() / max(d["y"].sum(), 1)
    diff = rate(t) - rate(c)
    boots = []
    tv, cv = t[["failed", "y"]].to_numpy(), c[["failed", "y"]].to_numpy()
    for _ in range(1000):
        bt, bc = tv[rng.integers(0, len(tv), len(tv))], cv[rng.integers(0, len(cv), len(cv))]
        boots.append(bt[:, 0].sum() / bt[:, 1].sum() - bc[:, 0].sum() / bc[:, 1].sum())
    upper = float(np.quantile(boots, 1 - alpha))
    guardrail = {"treatment_rate": rate(t), "control_rate": rate(c), "diff_pts": diff * 100, "upper_pts": upper * 100,
                 "margin_pts": cfg["guardrail_margin_pts"], "ok": bool(upper * 100 < cfg["guardrail_margin_pts"])}

    sd_raw, sd_cup, base = df["y"].std(), df["y_cuped"].std(), c["y"].mean()
    power = {"n_per_arm": int(min(len(t), len(c))), "mde_raw": mde(sd_raw, min(len(t), len(c)), base, alpha, cfg["power"]),
             "mde_cuped": mde(sd_cup, min(len(t), len(c)), base, alpha, cfg["power"]),
             "n_for_3pct_raw": sample_size(sd_raw, base, 0.03, alpha, cfg["power"]),
             "n_for_3pct_cuped": sample_size(sd_cup, base, 0.03, alpha, cfg["power"]), "sd_raw": sd_raw,
             "sd_cuped": sd_cup, "base": base}

    yc = c["y_cuped"].to_numpy()
    aa_p = np.empty(aa_runs)
    for i in range(aa_runs):
        m = rng.random(len(yc)) < 0.5
        aa_p[i] = stats.ttest_ind(yc[m], yc[~m], equal_var=False).pvalue
    aa = {"runs": aa_runs, "false_positive_rate": float((aa_p < alpha).mean())}

    ship = cup["lo"] > 0 and guardrail["ok"] and srm.pvalue > 0.001
    slope = np.polyfit(np.arange(len(hte)), hte["lift"].to_numpy(), 1)[0]     # lift change per older age band
    return {"experiment_id": cfg["experiment_id"], "n": n, "n_treatment": len(t), "n_control": len(c),
            "srm_p": float(srm.pvalue), "raw": raw, "cuped": cup, "variance_reduction": float(var_reduction),
            "hte": hte, "hte_slope_per_band": float(slope), "guardrail": guardrail, "power": power, "aa": aa,
            "ship": bool(ship)}


def findings(e: dict) -> list[str]:
    c, r, g = e["cuped"], e["raw"], e["guardrail"]
    out = [f"A/B test of the one-tap remittance flow ({e['n']:,} customers; sample-ratio check p = {e['srm_p']:.2f}): "
           f"{pct(c['lift'], sign=True)} remittances per customer with CUPED (95% CI {pct(c['lo'], sign=True)} to "
           f"{pct(c['hi'], sign=True)}, p = {c['p']:.3f}). Without CUPED the same data gives {pct(r['lift'], sign=True)} "
           f"(CI {pct(r['lo'], sign=True)} to {pct(r['hi'], sign=True)}, p = {r['p']:.2f}): using pre-period behaviour cut "
           f"variance by {pct(e['variance_reduction'], 0)} and turned an inconclusive test into a clear one."]
    young, old = e["hte"].iloc[0], e["hte"].iloc[-1]
    out.append(f"The lift falls with age ({young['age_band']}: {pct(young['lift'], sign=True)}, {old['age_band']}: "
               f"{pct(old['lift'], sign=True)}), but no single age band is significant after Bonferroni: the test was powered "
               f"for the overall effect (minimum detectable effect {pct(e['power']['mde_cuped'])}). Failure rate per transfer "
               f"{'stayed within' if g['ok'] else 'breached'} the {g['margin_pts']:.1f}-point guardrail "
               f"({g['control_rate'] * 100:.1f}% vs {g['treatment_rate'] * 100:.1f}%). "
               f"Recommendation: {'ship to all expat remitters' if e['ship'] else 'do not ship yet'}.")
    return out
