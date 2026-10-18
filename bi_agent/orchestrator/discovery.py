"""Discovery agent: scans every pair of governed monthly KPIs for co-movement that nobody
has written a hypothesis for yet.

It works on month-over-month growth (not levels), so shared upward trends do not create fake
links, and it only proposes candidates. Candidates go to a human for review; they are never
reported as confirmed.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..semantic import SemanticLayer
from . import stats
from .domains import DOMAINS

METRIC_DOMAIN = {m: d for d, spec in DOMAINS.items() for m in spec.get("flow", [])}
TRIVIAL = {frozenset(p) for p in [
    ("transaction_volume", "transaction_count"), ("transaction_volume", "avg_ticket"),
    ("transaction_count", "active_customers"), ("remittance_volume", "remittance_count"),
    ("remittance_count", "remittance_fee_income"), ("remittance_volume", "remittance_fee_income"),
    ("finance_disbursed", "finance_contracts"),
]}


def growth_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.apply(stats.growth).iloc[1:]


def correlation_heatmap(panel: pd.DataFrame) -> pd.DataFrame:
    return growth_matrix(panel).corr().round(3)


def discover(panel: pd.DataFrame, sem: SemanticLayer, known_pairs: set[frozenset],
             min_r: float = 0.5, alpha: float = 0.05, top: int = 6) -> tuple[list[dict], int]:
    g = growth_matrix(panel)
    cols = [c for c in g.columns if c in METRIC_DOMAIN]
    pairs = list(combinations(cols, 2))
    out = []
    for a, b in pairs:
        if frozenset((a, b)) in TRIVIAL or METRIC_DOMAIN[a] == METRIC_DOMAIN[b]:
            continue
        r = stats.pearson(g[a], g[b])
        if np.isnan(r) or abs(r) < min_r:
            continue
        p = stats.perm_pvalue(g[a], g[b])
        p_adj = min(1.0, p * len(pairs))  # Bonferroni across every pair scanned
        if p_adj >= alpha:
            continue
        out.append({
            "a": a, "b": b, "domain_a": METRIC_DOMAIN[a], "domain_b": METRIC_DOMAIN[b],
            "r": round(r, 3), "p_adj": round(p_adj, 4),
            "already_hypothesised": frozenset((a, b)) in known_pairs,
            "text": (f"{sem.metrics[a].label} and {sem.metrics[b].label} move together month to month "
                     f"(r = {r:+.2f}, adjusted p = {p_adj:.3f}). Candidate link for analyst review."),
        })
    out.sort(key=lambda d: -abs(d["r"]))
    # keep the strongest NEW candidate per pair of domains; hypothesised links are tested elsewhere
    best: dict[frozenset, dict] = {}
    for d in out:
        key = frozenset((d["domain_a"], d["domain_b"]))
        if not d["already_hypothesised"] and key not in best:
            best[key] = d
    return list(best.values())[:top], len(pairs)
