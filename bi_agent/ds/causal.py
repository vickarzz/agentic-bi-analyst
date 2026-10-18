"""Causal inference: did a business action actually change behaviour?

Case 1 - Western card cashback campaign (Nov 2025 - Jan 2026)
    Treated: the 4 Western cities. Controls: the other 13 cities.
    Outcome: card transactions per customer (spend per customer is reported too; it is noisier).
    * Naive before/after (what a dashboard would say)
    * Difference-in-differences with city and month fixed effects (TWFE), cluster-robust SEs
    * Randomisation inference: re-run the DiD for all 2,380 ways to pick 4 of 17 cities
    * Event study (leads and lags) + joint pre-trend test
    * Synthetic control for the Western region + placebo-in-space ranking

Case 2 - Digital remittance fee cut (25 Aug 2025)
    Outcome: digital share of remittances (branch fees did not change). The share was already rising,
    so before/after comparisons mistake the trend for an effect. Interrupted time series with the
    trend modelled (spec chosen by AIC) and placebo dates in the pre-period.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize

from .common import ds_config, pct, template

CARD_SQL = """
SELECT b.city, b.region, CAST(date_trunc('month', t.txn_date) AS DATE) AS month,
       SUM(t.amount_sar) AS card_spend, COUNT(*) AS card_txns
FROM fact_transaction t
JOIN dim_customer c ON c.customer_id = t.customer_id
JOIN dim_branch b ON b.branch_id = c.branch_id
WHERE t.txn_type IN ('Card Purchase', 'E-commerce')
GROUP BY ALL
LIMIT 100000
"""
BOOK_SQL = """
SELECT b.city, CAST(date_trunc('month', c.join_date) AS DATE) AS join_month,
       CAST(date_trunc('month', c.churn_date) AS DATE) AS churn_month, COUNT(*) AS customers
FROM dim_customer c
JOIN dim_branch b ON b.branch_id = c.branch_id
GROUP BY ALL
LIMIT 100000
"""
REMIT_SQL = """
SELECT CAST(date_trunc('month', r.remittance_date) AS DATE) AS month,
       CASE WHEN r.channel IN ('Mobile App', 'Internet Banking') THEN 'digital' ELSE 'branch' END AS grp,
       COUNT(*) AS remittances, AVG(r.fee_sar) AS fee
FROM fact_remittance r
GROUP BY ALL
LIMIT 100000
"""


def _events() -> dict:
    import yaml
    from .. import config
    return {e["id"]: e for e in yaml.safe_load((config.ROOT / "semantic" / "business_calendar.yaml").read_text())["events"]}


# ----------------------------------------------------------------------------------- campaign
def campaign_panel(db_path: str | None = None) -> pd.DataFrame:
    card = template(CARD_SQL, db_path=db_path, name="campaign_card_spend")
    book = template(BOOK_SQL, db_path=db_path, name="campaign_customer_book")
    card["month"] = pd.to_datetime(card["month"])
    months = sorted(card["month"].unique())
    book["join_month"] = pd.to_datetime(book["join_month"])
    book["churn_month"] = pd.to_datetime(book["churn_month"])
    rows = []
    for m in months:  # customers on the book in month m: joined by m and not churned before m
        on = book[(book["join_month"] <= m) & (book["churn_month"].isna() | (book["churn_month"] >= m))]
        rows.append(on.groupby("city")["customers"].sum().rename(m))
    on_book = pd.DataFrame(rows).stack().rename("customers").reset_index().rename(columns={"level_0": "month"})
    p = card.merge(on_book, on=["city", "month"])
    p["spend_pc"] = p["card_spend"] / p["customers"]
    p["txns_pc"] = p["card_txns"] / p["customers"]
    p["y"] = np.log(p["txns_pc"])            # primary outcome: card transactions per customer
    p["y_spend"] = np.log(p["spend_pc"])     # secondary outcome: card spend per customer
    return p.sort_values(["city", "month"]).reset_index(drop=True)


def _did_means(a: pd.Series, treated: list[str]) -> float:
    return a[a.index.isin(treated)].mean() - a[~a.index.isin(treated)].mean()


def _synth_weights(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    k = donors.shape[1]
    obj = lambda w: np.mean((target - donors @ w) ** 2)  # noqa: E731
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 500})
    return res.x


def campaign_effect(db_path: str | None = None) -> dict:
    cfg = ds_config()["causal"]
    ev = _events()[cfg["campaign_event"]]
    start, end = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
    p = campaign_panel(db_path)
    treated = sorted(p.loc[p["region"] == cfg["treated_region"], "city"].unique())
    p["treated"] = p["city"].isin(treated).astype(int)
    p["window"] = p["month"].between(start, end).astype(int)
    p["D"] = p["treated"] * p["window"]

    # naive: treated cities, campaign months vs the 3 months before
    w_sum = p[p["treated"] == 1].groupby("month")[["card_txns", "customers"]].sum()
    w_pc = w_sum["card_txns"] / w_sum["customers"]
    during = w_pc[(w_pc.index >= start) & (w_pc.index <= end)].mean()
    before = w_pc[(w_pc.index < start) & (w_pc.index >= start - pd.DateOffset(months=3))].mean()
    naive = during / before - 1

    # DiD with two-way fixed effects, clustered by city
    twfe = smf.ols("y ~ D + C(city) + C(month)", data=p).fit(cov_type="cluster", cov_kwds={"groups": p["city"]}, use_t=True)
    beta = twfe.params["D"]
    ci = tuple(twfe.conf_int().loc["D"])                       # t-based, 16 degrees of freedom
    p["t"] = (p["month"].dt.year - 2024) * 12 + p["month"].dt.month
    robust = smf.ols("y ~ D + treated:t + C(city) + C(month)", data=p).fit(
        cov_type="cluster", cov_kwds={"groups": p["city"]}, use_t=True)
    trend_adjusted = np.exp(robust.params["D"]) - 1
    twfe_spend = smf.ols("y_spend ~ D + C(city) + C(month)", data=p).fit(cov_type="cluster", cov_kwds={"groups": p["city"]}, use_t=True)
    spend_effect = np.exp(twfe_spend.params["D"]) - 1

    # randomisation inference over every possible set of 4 treated cities
    a = p.groupby("city").apply(lambda g: g.loc[g["window"] == 1, "y"].mean() - g.loc[g["window"] == 0, "y"].mean(),
                                include_groups=False)
    cities = list(a.index)
    obs = _did_means(a, treated)
    draws = np.array([_did_means(a, list(s)) for s in itertools.combinations(cities, len(treated))])
    ri_p = float(np.mean(np.abs(draws) >= abs(obs) - 1e-12))

    # event study: leads and lags around the start month (endpoints binned), reference = -1
    lo, hi = cfg["event_window"]
    p["rel"] = ((p["month"].dt.year - start.year) * 12 + p["month"].dt.month - start.month).clip(lo, hi)
    terms = []
    for k in range(lo, hi + 1):
        if k == -1:
            continue
        col = f"e_m{abs(k)}" if k < 0 else f"e_p{k}"
        p[col] = ((p["rel"] == k) & (p["treated"] == 1)).astype(int)
        terms.append((k, col))
    es = smf.ols("y ~ " + " + ".join(c for _, c in terms) + " + C(city) + C(month)", data=p).fit(
        cov_type="cluster", cov_kwds={"groups": p["city"]}, use_t=True)
    event = pd.DataFrame([{"rel_month": k, "effect": np.exp(es.params[c]) - 1,
                           "lo": np.exp(es.params[c] - 1.96 * es.bse[c]) - 1, "hi": np.exp(es.params[c] + 1.96 * es.bse[c]) - 1}
                          for k, c in terms] + [{"rel_month": -1, "effect": 0.0, "lo": 0.0, "hi": 0.0}]).sort_values("rel_month")
    # pre-trend test: do treated and control cities follow different slopes before the campaign?
    pre_df = p[p["month"] < start].copy()
    pre_df["t"] = (pre_df["month"].dt.year - 2024) * 12 + pre_df["month"].dt.month
    pt = smf.ols("y ~ treated:t + C(city) + C(month)", data=pre_df).fit(cov_type="cluster", cov_kwds={"groups": pre_df["city"]}, use_t=True)
    pre_p = float(pt.pvalues["treated:t"])
    pre_slope = float(np.exp(pt.params["treated:t"]) - 1)

    # synthetic control for the Western region (levels: spend per customer)
    agg = p.copy()
    agg["unit"] = np.where(agg["treated"] == 1, "Western region", agg["city"])
    u = agg.groupby(["unit", "month"])[["card_txns", "customers"]].sum()
    wide = (u["card_txns"] / u["customers"]).unstack("unit").sort_index()
    pre_idx, win_idx = wide.index < start, (wide.index >= start) & (wide.index <= end)

    def fit_unit(unit, donors):
        w = _synth_weights(wide.loc[pre_idx, unit].to_numpy(), wide.loc[pre_idx, donors].to_numpy())
        synth = wide[donors].to_numpy() @ w
        gap = wide[unit].to_numpy() - synth
        rmspe_pre = np.sqrt(np.mean(gap[pre_idx] ** 2))
        rmspe_post = np.sqrt(np.mean(gap[win_idx] ** 2))
        return w, synth, gap, rmspe_post / rmspe_pre

    donors = [c for c in wide.columns if c != "Western region"]
    w, synth, gap, ratio = fit_unit("Western region", donors)
    scm_effect = gap[win_idx].mean() / synth[win_idx].mean()
    placebo_ratios = {}
    placebo_gaps = {}
    for d in donors:
        _, s_d, g_d, r_d = fit_unit(d, [x for x in donors if x != d])
        placebo_ratios[d] = r_d
        placebo_gaps[d] = g_d / s_d
    rank = 1 + sum(r >= ratio for r in placebo_ratios.values())
    scm_p = rank / (len(donors) + 1)

    return {
        "event": ev, "treated_cities": treated, "control_cities": [c for c in cities if c not in treated],
        "naive": naive, "did": np.exp(beta) - 1, "did_ci": (np.exp(ci[0]) - 1, np.exp(ci[1]) - 1),
        "did_cluster_p": float(twfe.pvalues["D"]), "ri_p": ri_p, "ri_draws": np.exp(draws) - 1, "ri_obs": np.exp(obs) - 1,
        "n_assignments": len(draws), "event_study": event, "pretrend_p": pre_p, "pretrend_slope": pre_slope,
        "spend_effect": spend_effect, "trend_adjusted": trend_adjusted, "outcome": "card transactions per customer",
        "naive_window": "campaign months vs the 3 months before",
        "scm": {"effect": scm_effect, "weights": dict(sorted(((d, float(x)) for d, x in zip(donors, w) if x > 0.01),
                                                             key=lambda kv: -kv[1])),
                "series": pd.DataFrame({"month": wide.index, "actual": wide["Western region"].to_numpy(), "synthetic": synth}),
                "rank": rank, "units": len(donors) + 1, "p": scm_p,
                "placebo_gaps": pd.DataFrame(placebo_gaps, index=wide.index),
                "treated_gap": pd.Series(gap / synth, index=wide.index)},
    }


# ----------------------------------------------------------------------------------- fee cut
def fee_cut_effect(db_path: str | None = None, months: int = 6) -> dict:
    """Interrupted time series on the digital share of remittances.

    The fee cut only applied to digital channels, so if it worked, customers should have switched
    to digital faster than the existing trend. The share was already rising every month, so a naive
    before/after mistakes the trend for an effect. Models: level shift with a linear trend, plus a
    slope change, plus a quadratic trend; the simplest adequate one is chosen by AIC.
    """
    ev = _events()[ds_config()["causal"]["fee_cut_event"]]
    t0 = pd.Timestamp(ev["start"]).normalize().replace(day=1)          # Aug 2025 is a mixed month, dropped
    r = template(REMIT_SQL, db_path=db_path, name="fee_cut_remittances")
    r["month"] = pd.to_datetime(r["month"])
    r = r[r["month"] != t0]
    w = r.pivot(index="month", columns="grp", values="remittances").sort_index()
    fee = r.pivot(index="month", columns="grp", values="fee").sort_index()
    d = pd.DataFrame({"share": w["digital"] / (w["digital"] + w["branch"]), "digital": w["digital"]})
    d["t"] = (d.index.year - 2024) * 12 + d.index.month
    cut_t = (t0.year - 2024) * 12 + t0.month
    d["post"] = (d.index > t0).astype(int)
    d["slope_change"] = d["post"] * (d["t"] - cut_t)

    win_before = d[(d.index < t0) & (d.index >= t0 - pd.DateOffset(months=months))]
    win_after = d[(d.index > t0) & (d.index <= t0 + pd.DateOffset(months=months))]
    naive = win_after["digital"].mean() / win_before["digital"].mean() - 1
    naive_share_pts = (win_after["share"].mean() - win_before["share"].mean()) * 100

    specs = {
        "Level shift + linear trend": "share ~ t + post",
        "+ slope change": "share ~ t + post + slope_change",
        "+ quadratic trend": "share ~ t + I(t**2) + post",
    }
    fits = {}
    share_at_cut = d.loc[d.index < t0, "share"].iloc[-1]
    for name, f in specs.items():
        m = smf.ols(f, data=d).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
        lo, hi = m.conf_int().loc["post"]
        b = m.params["post"]
        fits[name] = {"effect_pts": b * 100, "lo_pts": lo * 100, "hi_pts": hi * 100, "p": float(m.pvalues["post"]),
                      "aic": float(m.aic), "effect": b / share_at_cut, "lo": lo / share_at_cut, "hi": hi / share_at_cut}
    chosen = min(fits, key=lambda k: fits[k]["aic"])

    pre = d[d.index < t0]
    pt = smf.ols("share ~ t", data=pre).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    pretrend_pts, pretrend_p = float(pt.params["t"] * 100), float(pt.pvalues["t"])

    # placebo in time: the same model with fake cut dates inside the pre-period
    rows = []
    for fake in pre.index[5:-5]:
        q = pre.copy()
        q["post"] = (q.index > fake).astype(int)
        q["slope_change"] = q["post"] * (q["t"] - ((fake.year - 2024) * 12 + fake.month))
        rows.append({"fake_date": fake.date().isoformat(), "effect_pts": smf.ols(specs[chosen], data=q).fit().params["post"] * 100})
    placebo_df = pd.DataFrame(rows)
    placebo = {"n": len(placebo_df), "min": float(placebo_df["effect_pts"].min()), "max": float(placebo_df["effect_pts"].max()),
               "dates": placebo_df}

    fee_change = (fee.loc[fee.index > t0, "digital"].mean() - fee.loc[fee.index < t0, "digital"].mean()) - \
                 (fee.loc[fee.index > t0, "branch"].mean() - fee.loc[fee.index < t0, "branch"].mean())
    series = d.reset_index()[["month", "share"]]
    m = smf.ols(specs[chosen], data=d).fit()
    series["fitted"] = m.fittedvalues.to_numpy()
    series["counterfactual"] = (m.fittedvalues - m.params["post"] * d["post"]).to_numpy()
    return {"event": ev, "naive": naive, "naive_share_pts": naive_share_pts, "fits": fits, "chosen": chosen,
            "pretrend_pts": pretrend_pts, "pretrend_p": pretrend_p, "series": series, "placebo": placebo,
            "fee_did_sar": float(fee_change), "share_at_cut": float(share_at_cut)}


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def findings(c: dict, f: dict) -> list[str]:
    ch = f["fits"][f["chosen"]]
    return [
        f"Cashback campaign (Western, Nov 2025 to Jan 2026): difference-in-differences estimates "
        f"{pct(c['did'], sign=True)} card transactions per customer (95% CI {pct(c['did_ci'][0], sign=True)} to "
        f"{pct(c['did_ci'][1], sign=True)}; randomisation {_p(c['ri_p'])} over {c['n_assignments']:,} "
        f"city assignments). "
        f"A naive before/after says {pct(c['naive'], sign=True)}; synthetic control says {pct(c['scm']['effect'], sign=True)}.",
        f"Fee cut: digital remittances rose {pct(f['naive'], sign=True)} in the 6 months after the cut, but digital's share "
        f"was already climbing {f['pretrend_pts']:+.1f} pts a month. An interrupted time series ({f['chosen'].lower()}) finds "
        f"a shift of {ch['effect_pts']:+.1f} pts (95% CI {ch['lo_pts']:+.1f} to {ch['hi_pts']:+.1f}), within the range of "
        f"{f['placebo']['n']} placebo dates ({f['placebo']['min']:+.1f} to {f['placebo']['max']:+.1f} pts): no measurable effect.",
    ]
