"""Deterministic insight generation + chart selection.

All numbers in the narrative are computed here from the result DataFrame, so the agent
cannot "hallucinate" a figure. (An LLM may optionally rephrase these facts, nothing more.)
"""
from __future__ import annotations

import math

import pandas as pd

from .plan import QueryPlan
from .semantic import SemanticLayer


def fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a (suppressed)"
    if kind == "percent":
        return f"{value * 100:.1f}%"
    if kind == "currency":
        a = abs(value)
        for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= div:
                return f"SAR {value / div:,.2f}{suf}"
        return f"SAR {value:,.2f}"
    if kind == "integer":
        return f"{int(round(value)):,}"
    return f"{value:,.2f}"


def _pct_change(new, old, kind: str = "number") -> str | None:
    """Relative change, or percentage-point change for rates (75% vs 70% is +5.0 pts, not +7.1%)."""
    if old is None or any(isinstance(x, float) and math.isnan(x) for x in (new, old)):
        return None
    if kind == "percent":
        return f"{(new - old) * 100:+.1f} pts"
    if old == 0:
        return None
    return f"{(new - old) / abs(old) * 100:+.1f}%"


def _period_label(p, grain: str | None) -> str:
    ts = pd.Timestamp(p)
    return {"year": f"{ts.year}", "quarter": f"Q{ts.quarter} {ts.year}", "month": ts.strftime("%b %Y"),
            "week": f"week of {ts:%d %b %Y}"}.get(grain or "", ts.strftime("%d %b %Y"))


def build_facts(df: pd.DataFrame, plan: QueryPlan, sem: SemanticLayer) -> list[str]:
    facts: list[str] = []
    if df.empty:
        return ["No rows matched the question's filters."]
    when = f" ({plan.time_label})" if plan.time_label else ""
    m0 = plan.metrics[0]
    meta = sem.metrics[m0]

    # KPI
    if not plan.dimensions and not plan.time_grain:
        for m in plan.metrics:
            facts.append(f"{sem.metrics[m].label}{when}: {fmt(df[m].iloc[0], sem.metrics[m].format)}.")
        return facts

    # Time series without breakdown
    if plan.time_grain and not plan.dimensions:
        s = df.dropna(subset=[m0]).sort_values("period")
        if len(s) >= 1:
            last = s.iloc[-1]
            facts.append(f"Latest {plan.time_grain} ({_period_label(last['period'], plan.time_grain)}): "
                         f"{meta.label} = {fmt(last[m0], meta.format)}.")
        if len(s) >= 2:
            prev = s.iloc[-2]
            ch = _pct_change(last[m0], prev[m0], meta.format)
            if ch:
                facts.append(f"Change vs previous {plan.time_grain}: {ch}.")
            first = s.iloc[0]
            ch_all = _pct_change(last[m0], first[m0], meta.format)
            if ch_all and len(s) > 2:
                facts.append(f"Change since {_period_label(first['period'], plan.time_grain)}: {ch_all}.")
            peak = s.loc[s[m0].idxmax()]
            facts.append(f"Peak: {_period_label(peak['period'], plan.time_grain)} at {fmt(peak[m0], meta.format)}.")
        for m in plan.metrics[1:]:
            facts.append(f"Latest {sem.metrics[m].label}: {fmt(s.iloc[-1][m], sem.metrics[m].format)}.")
        return facts

    d0 = plan.dimensions[0]
    # Breakdown without time
    if not plan.time_grain:
        if len(plan.dimensions) == 1:
            g = df.set_index(d0)[m0]
        elif meta.additive:
            g = df.groupby(d0)[m0].sum(min_count=1)
        else:
            g = None
        if g is not None:
            g = g.dropna().sort_values(ascending=False)
            if len(g):
                facts.append(f"Highest {meta.label}{when}: {g.index[0]} at {fmt(g.iloc[0], meta.format)}.")
            if len(g) > 1:
                facts.append(f"Lowest: {g.index[-1]} at {fmt(g.iloc[-1], meta.format)}.")
            if meta.additive and len(g) > 1 and g.sum() > 0 and not plan.limit:
                facts.append(f"{g.index[0]} accounts for {g.iloc[0] / g.sum() * 100:.1f}% of the total "
                             f"({fmt(g.sum(), meta.format)}).")
            elif len(g) > 1 and g.iloc[-1] not in (0, None):
                facts.append(f"{g.index[0]} is {g.iloc[0] / g.iloc[-1]:.1f}x {g.index[-1]}.")
        else:
            facts.append(f"{len(df)} combinations of {', '.join(plan.dimensions)} returned.")
        return facts

    # Breakdown over time: growth by group between first and last period
    s = df.dropna(subset=[m0])
    periods = sorted(s["period"].unique())
    if len(periods) >= 2:
        first, last = periods[0], periods[-1]
        a = s[s["period"] == first].groupby(d0)[m0].sum()
        b = s[s["period"] == last].groupby(d0)[m0].sum()
        growth = ((b - a) if meta.format == "percent" else (b - a) / a.abs()).dropna().sort_values(ascending=False)
        unit = " pts" if meta.format == "percent" else "%"
        if len(growth):
            facts.append(f"Strongest change {_period_label(first, plan.time_grain)} → {_period_label(last, plan.time_grain)}: "
                         f"{growth.index[0]} ({growth.iloc[0] * 100:+.1f}{unit}).")
            if len(growth) > 1:
                facts.append(f"Weakest change: {growth.index[-1]} ({growth.iloc[-1] * 100:+.1f}{unit}).")
        top_last = b.sort_values(ascending=False)
        facts.append(f"Largest in {_period_label(last, plan.time_grain)}: {top_last.index[0]} "
                     f"({fmt(top_last.iloc[0], meta.format)}).")
    return facts


def choose_chart(plan: QueryPlan, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"type": "none"}
    if not plan.dimensions and not plan.time_grain:
        return {"type": "kpi", "metrics": plan.metrics}
    y = plan.metrics[0]
    if plan.time_grain:
        return {"type": "line", "x": "period", "y": y,
                "color": plan.dimensions[0] if plan.dimensions else None}
    return {"type": "bar", "x": plan.dimensions[0], "y": y,
            "color": plan.dimensions[1] if len(plan.dimensions) > 1 else None}
