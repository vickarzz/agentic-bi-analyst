"""Impact Analyst: tests the curated cross-business hypotheses in semantic/impact_links.yaml.

Each result carries a verdict, an effect size, a p-value, the evidence numbers and the
recommended action the business owner wrote for that outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .. import config
from ..insights import fmt
from ..semantic import SemanticLayer
from . import stats
from .tools import ToolLog, approved_analysis, q

ALPHA = 0.05
MIN_R = 0.3
AGE_RANK = {"18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55+": 5}


@dataclass
class LinkResult:
    id: str
    title: str
    from_domain: str
    to_domain: str
    hypothesis: str
    expected: str
    test_type: str
    verdict: str                 # confirmed | contradicted | not detected | control passed | control FAILED
    effect: float | None         # r for correlations, relative change for cohort/event
    p_value: float | None
    summary: str                 # one sentence with the evidence numbers
    action: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def load_links(path: str | Path | None = None) -> list[dict]:
    p = Path(path or config.ROOT / "semantic" / "impact_links.yaml")
    return yaml.safe_load(p.read_text())["links"]


def load_calendar(path: str | Path | None = None) -> list[dict]:
    p = Path(path or config.ROOT / "semantic" / "business_calendar.yaml")
    return yaml.safe_load(p.read_text())["events"]


def _verdict(expected: str, effect: float, p: float, min_effect: float) -> str:
    if expected == "none":
        return "control passed" if (np.isnan(p) or p >= ALPHA) else "control FAILED"
    if np.isnan(effect) or np.isnan(p) or p >= ALPHA or abs(effect) < min_effect:
        return "not detected"
    sign_ok = (effect > 0) == (expected == "positive")
    return "confirmed" if sign_ok else "contradicted"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


class ImpactAnalyst:
    def __init__(self, sem: SemanticLayer, panel: pd.DataFrame, db_path: str | None = None,
                 log: ToolLog | None = None):
        self.sem, self.panel, self.db_path = sem, panel, db_path
        self.log = log or ToolLog()
        self.events = {e["id"]: e for e in load_calendar()}

    # ------------------------------------------------------------------ series helpers
    def series(self, name: str) -> pd.Series:
        p = self.panel
        if name in p:
            return p[name]
        if name == "fee_per_remittance":
            return p["remittance_fee_income"] / p["remittance_count"]
        if name == "remittance_count_yoy":
            return p["remittance_count"].pct_change(12)
        raise KeyError(name)

    def cross_section(self, name: str, dims: list[str]) -> pd.Series:
        if name == "churn_rate":
            df = q(self.sem, ["churned_customers", "total_customers"], dims, db_path=self.db_path, log=self.log)
            s = df["churned_customers"] / df["total_customers"]
        elif name == "age_rank":
            df = q(self.sem, ["total_customers"], dims, db_path=self.db_path, log=self.log)
            s = df["age_band"].map(AGE_RANK).astype(float)
        else:
            df = q(self.sem, [name], dims, db_path=self.db_path, log=self.log)
            s = df[name]
        return pd.Series(s.to_numpy(), index=pd.MultiIndex.from_frame(df[dims].astype(str)))

    # ------------------------------------------------------------------ tests
    def run(self, link: dict) -> LinkResult:
        t = link["test"]
        fn = {"timeseries": self._timeseries, "cross_section": self._cross_section,
              "cohort": self._cohort, "event": self._event}[t["type"]]
        verdict, effect, p, summary, details = fn(link, t)
        action = link.get("if_confirmed") if verdict == "confirmed" else link.get("if_not_detected", "")
        if verdict == "contradicted":
            action = "Evidence points the opposite way to the hypothesis; review with the business owner before acting."
        if verdict == "control FAILED":
            action = link.get("if_confirmed", "")
        return LinkResult(id=link["id"], title=link["title"], from_domain=link["from"]["domain"],
                          to_domain=link["to"]["domain"], hypothesis=link["hypothesis"],
                          expected=link["expected"], test_type=t["type"], verdict=verdict,
                          effect=None if effect is None or np.isnan(effect) else round(float(effect), 4),
                          p_value=None if p is None or np.isnan(p) else round(float(p), 4),
                          summary=summary, action=action, details=details)

    def _timeseries(self, link, t):
        xm, ym = link["from"]["metric"], link["to"]["metric"]
        x = self.series(xm)
        y = self.series(ym)
        x = x if t.get("x_transform") == "level" else stats.growth(x)
        y = stats.growth(y) if t.get("transform") == "growth" else y
        lags = stats.lagged_correlation(x, y, t.get("max_lag", 2))
        best = max(lags, key=lambda r: -1 if np.isnan(r.r) else abs(r.r))
        p_adj = min(1.0, best.p * len(lags))            # Bonferroni across the lags tried
        verdict = _verdict(link["expected"], best.r, p_adj, MIN_R)
        lag_txt = "the same month" if best.lag == 0 else f"{best.lag} month(s) later"
        xl, yl = self.sem.metrics[xm].label, self.sem.metrics[ym].label
        x_desc = f"{xl} (monthly count)" if t.get("x_transform") == "level" else f"Monthly growth in {xl}"
        summary = (f"{x_desc} vs growth in {yl} {lag_txt}: r = {best.r:+.2f} "
                   f"({stats.strength(best.r)}, {_p(p_adj)}, n = {best.n} months).")
        details = {"lags": [{"lag": r.lag, "r": round(r.r, 3), "p": round(r.p, 4), "n": r.n} for r in lags],
                   "best_lag": best.lag}
        if verdict == "confirmed" and t.get("transform") == "growth" and t.get("x_transform") != "level":
            xx, yy = x.iloc[: len(x) - best.lag].to_numpy(), y.iloc[best.lag:].to_numpy()
            slope, lo, hi = stats.ols_slope(xx, yy)
            details["elasticity"] = {"slope": round(slope, 3), "lo90": round(lo, 3), "hi90": round(hi, 3)}
            summary += (f" A 1% change in {xl} is associated with a {slope:+.2f}% change in {yl} "
                        f"(90% interval {lo:+.2f} to {hi:+.2f}).")
        return verdict, best.r, p_adj, summary, details

    def _cross_section(self, link, t):
        dims = t["dimensions"]
        xs, ys = self.cross_section(t["x"], dims), self.cross_section(t["y"], dims)
        df = pd.concat([xs.rename("x"), ys.rename("y")], axis=1).dropna()
        r, p = stats.pearson(df["x"], df["y"]), stats.perm_pvalue(df["x"], df["y"])
        verdict = _verdict(link["expected"], r, p, MIN_R)
        slope, lo, hi = stats.ols_slope(df["x"], df["y"])
        summary = (f"Across {len(df)} {' x '.join(dims)} groups, {t['x'].replace('_', ' ')} vs "
                   f"{t['y'].replace('_', ' ')}: r = {r:+.2f} ({stats.strength(r)}, {_p(p)}).")
        details = {"n_groups": len(df), "slope": round(slope, 4), "lo90": round(lo, 4), "hi90": round(hi, 4),
                   "points": [{"group": " / ".join(k) if isinstance(k, tuple) else str(k),
                               "x": round(float(a), 4), "y": round(float(b), 4)} for k, a, b in
                              zip(df.index, df["x"], df["y"])]}
        if t["x"] == "churn_rate" and verdict == "confirmed":
            summary += f" Each +1 pt of churn rate is associated with {slope:+.2f} pts of NPF ratio."
        return verdict, r, p, summary, details

    def _cohort(self, link, t, n_perm: int = 2000):
        rows = approved_analysis(self.sem, link["id"], t["sql"], self.db_path, self.log)
        a_name, b_name = t["cohorts"]
        npl = (rows["status"] == "90+ DPD").to_numpy() * rows["outstanding_sar"].to_numpy()
        out = rows["outstanding_sar"].to_numpy()
        lab = (rows["cohort"] == a_name).to_numpy()
        keep = rows["cohort"].isin([a_name, b_name]).to_numpy()
        npl, out, lab = npl[keep], out[keep], lab[keep]

        def ratio(mask):
            return npl[mask].sum() / out[mask].sum() if out[mask].sum() else np.nan

        ra, rb = ratio(lab), ratio(~lab)
        diff = ra - rb
        rng = np.random.default_rng(stats.RNG_SEED)
        perms = np.empty(n_perm)
        for i in range(n_perm):
            m = rng.permutation(lab)
            perms[i] = ratio(m) - ratio(~m)
        p = float((np.sum(np.abs(perms) >= abs(diff) - 1e-12) + 1) / (n_perm + 1))
        rel = ra / rb - 1 if rb else np.nan
        verdict = _verdict(link["expected"], rel, p, 0.10)
        summary = (f"{t['outcome'].split(' (')[0]}: {a_name} {fmt(ra, 'percent')} vs {b_name} {fmt(rb, 'percent')} "
                   f"({ra / rb:.1f}x, {_p(p)}; {lab.sum():,} vs {(~lab).sum():,} open contracts).")
        details = {"cohort_a": a_name, "cohort_b": b_name, "value_a": round(float(ra), 4),
                   "value_b": round(float(rb), 4), "n_a": int(lab.sum()), "n_b": int((~lab).sum())}
        return verdict, rel, p, summary, details

    def _event(self, link, t):
        ev = self.events[t["event"]]
        s = self.series(t["series"])
        res = stats.event_test(s, ev["start"], t.get("window", 6))
        if not res.get("ok"):
            return "not detected", float("nan"), float("nan"), "Not enough data around the event.", {}
        verdict = _verdict(link["expected"], res["pct"] if t["series"] != "remittance_count_yoy" else res["delta"],
                           res["p"], 0.02)
        if t["series"] == "remittance_count_yoy":
            summary = (f"Year-over-year growth in remittances: {res['before'] * 100:.1f}% in the {res['n_before']} "
                       f"months before {ev['name']} vs {res['after'] * 100:.1f}% after ({_p(res['p'])}).")
            effect = res["delta"]
        else:
            summary = (f"Fee income per remittance: SAR {res['before']:.2f} before vs SAR {res['after']:.2f} after "
                       f"{ev['name']} ({res['pct'] * 100:+.1f}%, {_p(res['p'])}).")
            effect = res["pct"]
        return verdict, effect, res["p"], summary, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}


def run_all(sem: SemanticLayer, panel: pd.DataFrame, db_path: str | None = None,
            log: ToolLog | None = None) -> list[LinkResult]:
    analyst = ImpactAnalyst(sem, panel, db_path, log)
    return [analyst.run(link) for link in load_links()]
