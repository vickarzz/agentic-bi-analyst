"""Small, dependency-free statistics toolkit used by the orchestrator agents.

Everything is NumPy only and seeded, so results are reproducible in tests and CI.
Significance is estimated with permutation tests / bootstrap instead of distributional
assumptions, which is easier to explain to business stakeholders ("how often would we see
a link this strong if the months were shuffled at random?").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RNG_SEED = 7


def _clean_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok], y[ok]


def pearson(x, y) -> float:
    x, y = _clean_xy(x, y)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def perm_pvalue(x, y, n: int = 2000, seed: int = RNG_SEED) -> float:
    """Two-sided permutation p-value for Pearson r."""
    x, y = _clean_xy(x, y)
    r0 = pearson(x, y)
    if np.isnan(r0):
        return float("nan")
    rng = np.random.default_rng(seed)
    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()
    perms = np.array([np.mean(xs * rng.permutation(ys)) for _ in range(n)])
    return float((np.sum(np.abs(perms) >= abs(r0) - 1e-12) + 1) / (n + 1))


@dataclass
class LagResult:
    lag: int          # months that X leads Y (0 = same month)
    r: float
    p: float
    n: int


def lagged_correlation(x: pd.Series, y: pd.Series, max_lag: int = 3) -> list[LagResult]:
    """Correlate X(t) with Y(t+lag) for lag = 0..max_lag on already-transformed series."""
    out = []
    for lag in range(max_lag + 1):
        xx, yy = x.iloc[: len(x) - lag].to_numpy(), y.iloc[lag:].to_numpy()
        xx, yy = _clean_xy(xx, yy)
        out.append(LagResult(lag, pearson(xx, yy), perm_pvalue(xx, yy), len(xx)))
    return out


def growth(s: pd.Series) -> pd.Series:
    """Month-over-month % change. Removes the shared upward trend that makes almost any two
    growing business lines look 'correlated' (the classic spurious-correlation trap)."""
    return s.pct_change().replace([np.inf, -np.inf], np.nan)


def diff(s: pd.Series) -> pd.Series:
    return s.diff()


def ols_slope(x, y, n_boot: int = 1000, seed: int = RNG_SEED) -> tuple[float, float, float]:
    """Slope of y on x with a bootstrap 90% interval."""
    x, y = _clean_xy(x, y)
    if len(x) < 3 or x.std() == 0:
        return float("nan"), float("nan"), float("nan")
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if x[i].std() > 0:
            boots.append(np.polyfit(x[i], y[i], 1)[0])
    lo, hi = np.percentile(boots, [5, 95]) if boots else (float("nan"), float("nan"))
    return slope, float(lo), float(hi)


def event_test(series: pd.Series, event_date: str, window: int = 6, n: int = 2000,
               seed: int = RNG_SEED) -> dict:
    """Compare the mean of a monthly series in `window` months before vs after an event.
    p-value: permutation of the before/after labels."""
    s = series.dropna()
    t = pd.Timestamp(event_date)
    event_month = t.normalize().replace(day=1)
    before = s[s.index < event_month].tail(window)          # the event month itself is mixed, so skip it
    after = s[s.index > event_month].head(window)
    if len(before) < 3 or len(after) < 3:
        return {"ok": False}
    delta = after.mean() - before.mean()
    pooled = np.concatenate([before.to_numpy(), after.to_numpy()])
    rng = np.random.default_rng(seed)
    k = len(before)
    perms = []
    for _ in range(n):
        p = rng.permutation(pooled)
        perms.append(p[k:].mean() - p[:k].mean())
    pval = float((np.sum(np.abs(perms) >= abs(delta) - 1e-12) + 1) / (n + 1))
    return {"ok": True, "before": float(before.mean()), "after": float(after.mean()),
            "delta": float(delta), "pct": float(delta / before.mean()) if before.mean() else float("nan"),
            "p": pval, "n_before": len(before), "n_after": len(after)}


def robust_z(s: pd.Series) -> pd.Series:
    """Robust z-score (median / MAD) — flags unusual months without being fooled by outliers."""
    med = s.median()
    mad = (s - med).abs().median()
    if not mad:
        return s * 0
    return 0.6745 * (s - med) / mad


def strength(r: float) -> str:
    a = abs(r)
    if np.isnan(a):
        return "n/a"
    return "strong" if a >= 0.7 else "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "none"
