"""Domain analyst agents. Each one reviews one line of business with governed metrics only.

Every finding carries the exact numbers it states (`facts`), so the Verifier can later check
that the executive briefing does not contain any figure that was not computed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from ..insights import fmt
from ..semantic import SemanticLayer
from . import stats
from .tools import ToolLog, q

DOMAINS: dict[str, dict] = {
    "retail": {
        "label": "Retail & cards",
        "flow": ["transaction_volume", "transaction_count", "avg_ticket", "active_customers"],
        "breakdowns": {"transaction_volume": ["segment", "region"]},
    },
    "digital": {
        "label": "Digital banking",
        "flow": ["digital_adoption_rate"],
        "breakdowns": {"digital_adoption_rate": ["age_band", "region"]},
    },
    "credit": {
        "label": "Islamic finance & credit risk",
        "flow": ["finance_disbursed", "finance_contracts"],
        "portfolio": ["outstanding_portfolio", "npf_ratio"],
        "breakdowns": {"npf_ratio": ["segment", "product", "region"]},
        "alerts": {"npf_ratio": 0.05},
    },
    "payments": {
        "label": "Remittances & payments",
        "flow": ["remittance_volume", "remittance_count", "remittance_fee_income"],
        "breakdowns": {"remittance_volume": ["corridor", "channel"]},
    },
    "customer": {
        "label": "Customer base",
        "flow": ["new_customers", "churned_customers"],
        "breakdowns": {"churned_customers": ["segment"]},
    },
}


@dataclass
class Finding:
    domain: str
    kind: str              # kpi | trend | breakdown | anomaly | risk
    text: str
    severity: str = "info" # info | watch | alert | positive
    metric: str | None = None
    facts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def quarter_bounds(as_of: date, back: int) -> tuple[date, date]:
    """Start/end of the complete quarter `back` quarters before the latest complete one."""
    q_end_month = ((as_of.month - 1) // 3) * 3 + 3
    end = pd.Timestamp(as_of.year, q_end_month, 1) + pd.offsets.MonthEnd(0)
    if end.date() > as_of:
        end = (pd.Timestamp(as_of.year, q_end_month, 1) - pd.offsets.QuarterEnd(1))
    end = end - pd.offsets.QuarterEnd(back) if back else end
    start = (end - pd.offsets.QuarterBegin(1, startingMonth=1)).normalize()
    start = pd.Timestamp(end.year, end.month - 2, 1)
    return start.date(), end.date()


def qlabel(d: date) -> str:
    return f"Q{(d.month - 1) // 3 + 1} {d.year}"


def _change(new, old, kind) -> tuple[str, float]:
    if old is None or new is None or any(isinstance(v, float) and np.isnan(v) for v in (new, old)):
        return "n/a", float("nan")
    if kind == "percent":
        d = (new - old) * 100
        return f"{d:+.1f} pts", d
    if not old:
        return "n/a", float("nan")
    d = (new - old) / abs(old) * 100
    return f"{d:+.1f}%", d


class DomainAnalyst:
    def __init__(self, sem: SemanticLayer, domain: str, panel: pd.DataFrame, calendar: list[dict],
                 db_path: str | None = None, log: ToolLog | None = None):
        self.sem, self.domain, self.spec = sem, domain, DOMAINS[domain]
        self.panel, self.calendar, self.db_path = panel, calendar, db_path
        self.log = log or ToolLog()
        self.as_of = date.fromisoformat(sem.as_of_date)

    def run(self) -> list[Finding]:
        out: list[Finding] = []
        out += self.quarter_kpis()
        out += self.portfolio_kpis()
        out += self.breakdowns()
        out += self.anomalies()
        return out

    # -------------------------------------------------------------- KPIs
    def quarter_kpis(self) -> list[Finding]:
        flow = self.spec.get("flow", [])
        if not flow:
            return []
        cur_s, cur_e = quarter_bounds(self.as_of, 0)
        prev_s, prev_e = quarter_bounds(self.as_of, 1)
        ly_s, ly_e = quarter_bounds(self.as_of, 4)
        df = q(self.sem, flow, grain="quarter", start=ly_s.isoformat(), end=cur_e.isoformat(),
               db_path=self.db_path, log=self.log)
        df["period"] = pd.to_datetime(df["period"]).dt.date
        get = lambda d, m: df.loc[df["period"] == d, m].squeeze() if (df["period"] == d).any() else None  # noqa: E731
        out = []
        for m in flow:
            meta = self.sem.metrics[m]
            cur, prev, ly = get(cur_s, m), get(prev_s, m), get(ly_s, m)
            qoq, qoq_v = _change(cur, prev, meta.format)
            yoy, yoy_v = _change(cur, ly, meta.format)
            sev = "info"
            if m == "churned_customers" and yoy_v > 10:
                sev = "watch"
            elif m in ("transaction_volume", "remittance_volume", "digital_adoption_rate", "new_customers") and yoy_v > 5:
                sev = "positive"
            elif yoy_v < -5 and m not in ("churned_customers",):
                sev = "watch"
            text = (f"{meta.label} in {qlabel(cur_s)}: {fmt(cur, meta.format)} "
                    f"({qoq} vs {qlabel(prev_s)}, {yoy} vs {qlabel(ly_s)}).")
            out.append(Finding(self.domain, "kpi", text, sev, m,
                               {"value": fmt(cur, meta.format), "qoq": qoq, "yoy": yoy}))
        return out

    def portfolio_kpis(self) -> list[Finding]:
        ms = self.spec.get("portfolio", [])
        if not ms:
            return []
        df = q(self.sem, ms, db_path=self.db_path, log=self.log)
        out = []
        for m in ms:
            meta = self.sem.metrics[m]
            v = df[m].iloc[0]
            sev = "alert" if m in self.spec.get("alerts", {}) and v > self.spec["alerts"][m] else "info"
            out.append(Finding(self.domain, "kpi", f"{meta.label} (portfolio, as of {self.sem.as_of_date}): "
                               f"{fmt(v, meta.format)}.", sev, m, {"value": fmt(v, meta.format)}))
        return out

    # -------------------------------------------------------------- breakdowns
    def breakdowns(self) -> list[Finding]:
        out = []
        start = (pd.Timestamp(self.as_of) - pd.DateOffset(months=12) + pd.Timedelta(days=1)).date().isoformat()
        for m, dims in self.spec.get("breakdowns", {}).items():
            meta = self.sem.metrics[m]
            is_portfolio = m in self.spec.get("portfolio", [])
            for d in dims:
                df = q(self.sem, [m], [d], start=None if is_portfolio else start,
                       end=None if is_portfolio else self.as_of.isoformat(), db_path=self.db_path, log=self.log)
                df = df.dropna(subset=[m]).sort_values(m, ascending=False)
                if len(df) < 2:
                    continue
                top, bot = df.iloc[0], df.iloc[-1]
                window = "portfolio" if is_portfolio else "last 12 months"
                text = (f"{meta.label} by {d.replace('_', ' ')} ({window}): highest {top[d]} "
                        f"{fmt(top[m], meta.format)}, lowest {bot[d]} {fmt(bot[m], meta.format)}.")
                facts = {"top": str(top[d]), "top_value": fmt(top[m], meta.format),
                         "bottom": str(bot[d]), "bottom_value": fmt(bot[m], meta.format)}
                sev = "info"
                if meta.additive and df[m].sum() > 0:
                    share = top[m] / df[m].sum() * 100
                    text = text[:-1] + f"; {top[d]} is {share:.1f}% of the total."
                    facts["top_share"] = f"{share:.1f}%"
                limit = self.spec.get("alerts", {}).get(m)
                if limit is not None:
                    over = df[df[m] > limit]
                    if len(over):
                        sev = "alert"
                        text += (f" {len(over)} {d.replace('_', ' ')} group(s) above the "
                                 f"{fmt(limit, meta.format)} alert level: {', '.join(map(str, over[d]))}.")
                out.append(Finding(self.domain, "breakdown", text, sev, m, facts))
        return out

    # -------------------------------------------------------------- anomalies
    def anomalies(self, z_cut: float = 3.0) -> list[Finding]:
        out = []
        cutoff = pd.Timestamp(self.as_of) - pd.DateOffset(months=12)
        for m in self.spec.get("flow", []):
            if m not in self.panel:
                continue
            g = stats.growth(self.panel[m]).dropna()
            z = stats.robust_z(g)
            hits = z[(z.abs() >= z_cut) & (z.index > cutoff)]
            for ts, zv in hits.items():
                month_end = ts + pd.offsets.MonthEnd(0)
                prev_start = ts - pd.DateOffset(months=1)
                ev = [e["name"] for e in self.calendar
                      if pd.Timestamp(e["start"]) <= month_end and pd.Timestamp(e["end"]) >= ts]
                ev_prev = [e["name"] for e in self.calendar
                           if pd.Timestamp(e["start"]) <= ts - pd.Timedelta(days=1) and pd.Timestamp(e["end"]) >= prev_start]
                meta = self.sem.metrics[m]
                if ev:
                    why = f" It coincides with {', '.join(dict.fromkeys(ev))}."
                elif ev_prev:
                    why = f" It follows {', '.join(dict.fromkeys(ev_prev))} (normalisation after a seasonal peak)."
                    ev = ev_prev
                else:
                    why = " No calendar event explains it; worth a closer look."
                text = (f"Unusual month: {meta.label} changed {g[ts] * 100:+.1f}% in {ts:%b %Y} "
                        f"(robust z = {zv:+.1f}).{why}")
                out.append(Finding(self.domain, "anomaly", text, "info" if ev else "watch", m,
                                   {"change": f"{g[ts] * 100:+.1f}%", "z": f"{zv:+.1f}"}))
        return out
