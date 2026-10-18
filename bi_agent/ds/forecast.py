"""Remittance forecasting with the Hijri calendar.

Ramadan and Eid move about 11 days earlier every Gregorian year, so "same week last year"
(seasonal naive) lines up with the wrong days. This model uses calendar features instead:
the share of each week that falls in Ramadan or the pre-Eid window, the share of payday
days (25th onwards), annual Fourier terms and a trend, fitted on log volume.

Validation: rolling-origin backtests (train on everything before the origin, forecast 13 weeks),
compared with two baselines. Prediction intervals come from the backtest errors at each horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from .. import config
from .common import ds_config, pct, sar, template

DAILY_SQL = """
SELECT CAST(r.remittance_date AS DATE) AS day, SUM(r.amount_sar) AS volume, COUNT(*) AS remittances
FROM fact_remittance r
GROUP BY ALL
LIMIT 10000
"""


def _calendar_windows() -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    events = yaml.safe_load((config.ROOT / "semantic" / "business_calendar.yaml").read_text())["events"]
    events = [e for e in events if e.get("type") == "seasonal"] + ds_config()["forecast"]["future_events"]
    out: dict[str, list] = {"Ramadan": [], "Pre-Eid al-Adha": []}
    for e in events:
        if e["name"] in out:
            out[e["name"]].append((pd.Timestamp(e["start"]), pd.Timestamp(e["end"])))
    return out


def weekly_series(db_path: str | None = None) -> pd.Series:
    d = template(DAILY_SQL, db_path=db_path, name="daily_remittances")
    d["day"] = pd.to_datetime(d["day"])
    s = d.set_index("day")["volume"].asfreq("D", fill_value=0)
    w = s.resample("W-SUN").sum()
    counts = s.resample("W-SUN").size()
    return w[counts == 7]                              # full weeks only


def features(week_ends: pd.DatetimeIndex, t0: pd.Timestamp) -> pd.DataFrame:
    wins = _calendar_windows()
    rows = []
    for we in week_ends:
        days = pd.date_range(we - pd.Timedelta(days=6), we)
        r = {"trend": (we - t0).days / 365.25}
        for name, key in (("Ramadan", "ramadan"), ("Pre-Eid al-Adha", "pre_eid")):
            r[key] = sum(any(a <= d <= b for a, b in wins[name]) for d in days) / 7
        r["payday"] = sum(d.day >= 25 for d in days) / 7
        doy = (we.dayofyear - 3) / 365.25 * 2 * np.pi
        for k in (1, 2):
            r[f"sin{k}"], r[f"cos{k}"] = np.sin(k * doy), np.cos(k * doy)
        rows.append(r)
    return pd.DataFrame(rows, index=week_ends)


class CalendarModel:
    def __init__(self, ridge: float = 0.1):
        self.ridge = ridge

    def fit(self, y: pd.Series, t0: pd.Timestamp):
        self.t0 = t0
        x = features(y.index, t0)
        self.cols = list(x.columns)
        X = np.column_stack([np.ones(len(x)), x.to_numpy()])
        pen = self.ridge * np.eye(X.shape[1])
        pen[0, 0] = 0
        self.coef = np.linalg.solve(X.T @ X + pen, X.T @ np.log(y.to_numpy()))
        return self

    def predict(self, week_ends: pd.DatetimeIndex) -> np.ndarray:
        x = features(week_ends, self.t0)
        return np.exp(np.column_stack([np.ones(len(x)), x.to_numpy()]) @ self.coef)

    def effect(self, name: str) -> float:
        """Multiplicative effect of a week fully inside the event, e.g. Ramadan."""
        return float(np.exp(self.coef[1 + self.cols.index(name)]) - 1)


def _baselines(hist: pd.Series, future: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    seas = np.array([hist.get(w - pd.Timedelta(weeks=52), np.nan) for w in future])
    return {"Seasonal naive (same week last year)": seas,
            "Moving average (last 4 weeks)": np.full(len(future), hist.tail(4).mean())}


def _wape(actual: np.ndarray, pred: np.ndarray) -> float:
    ok = ~np.isnan(pred)
    return float(np.abs(actual[ok] - pred[ok]).sum() / actual[ok].sum())


def run(db_path: str | None = None) -> dict:
    cfg = ds_config()["forecast"]
    H, step = cfg["horizon_weeks"], cfg["backtest_step_weeks"]
    y = weekly_series(db_path)
    t0 = y.index[0]
    origins = [o for o in y.index if o >= pd.Timestamp("2025-01-05") and o <= y.index[-H - 1]][::step]
    rows, errors = [], []
    for o in origins:
        hist, fut = y[y.index <= o], y[(y.index > o)].head(H)
        preds = {"Calendar model (Hijri-aware)": CalendarModel().fit(hist, t0).predict(fut.index)}
        preds.update(_baselines(hist, fut.index))
        for name, p in preds.items():
            rows.append({"origin": o, "model": name, "wape": _wape(fut.to_numpy(), p)})
        cm = preds["Calendar model (Hijri-aware)"]
        errors += [{"h": h + 1, "log_err": np.log(a / p)} for h, (a, p) in enumerate(zip(fut.to_numpy(), cm))]
    bt = pd.DataFrame(rows)
    summary = bt.groupby("model")["wape"].mean().sort_values()

    # the backtest that crosses Ramadan 2026 (moved 11 days earlier than 2025)
    ram_origin = y.index[y.index <= pd.Timestamp("2026-01-25")][-1]
    hist, fut = y[y.index <= ram_origin], y[y.index > ram_origin].head(H)
    ram_preds = {"Calendar model (Hijri-aware)": CalendarModel().fit(hist, t0).predict(fut.index)}
    ram_preds.update(_baselines(hist, fut.index))
    ram = pd.DataFrame({"week": fut.index, "actual": fut.to_numpy(), **ram_preds})
    ram_wape = {k: _wape(fut.to_numpy(), v) for k, v in ram_preds.items()}

    # final model and forecast with empirical intervals by horizon
    model = CalendarModel().fit(y, t0)
    future = pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=H, freq="W-SUN")
    point = model.predict(future)
    err = pd.DataFrame(errors)
    q = (1 - cfg["interval"]) / 2
    qs = err.groupby("h")["log_err"].quantile([q, 1 - q]).unstack()
    fc = pd.DataFrame({"week": future, "forecast": point, "lo": point * np.exp(qs[q].to_numpy()),
                       "hi": point * np.exp(qs[1 - q].to_numpy())})
    fitted = pd.DataFrame({"week": y.index, "actual": y.to_numpy(), "fitted": model.predict(y.index)})
    total = fc["forecast"].sum()
    # interval for the 13-week total: simulate by resampling whole backtest error paths
    paths = err.assign(run=np.repeat(np.arange(len(err) // H), H)[: len(err)]).pivot(index="run", columns="h", values="log_err")
    totals = (point * np.exp(paths.to_numpy())).sum(axis=1)
    return {
        "series": fitted, "backtest": bt, "backtest_summary": summary, "n_origins": len(origins),
        "ramadan_backtest": ram, "ramadan_wape": ram_wape, "ramadan_origin": ram_origin.date().isoformat(),
        "forecast": fc, "total": total, "total_lo": float(np.quantile(totals, q)), "total_hi": float(np.quantile(totals, 1 - q)),
        "effects": {"Ramadan week": model.effect("ramadan"), "Pre-Eid week": model.effect("pre_eid"),
                    "Payday week": model.effect("payday"), "Growth per year": model.effect("trend")},
        "horizon_weeks": H, "last_actual_week": y.index[-1].date().isoformat(),
    }


def findings(f: dict) -> list[str]:
    s = f["backtest_summary"]
    cal, sn = s["Calendar model (Hijri-aware)"], s["Seasonal naive (same week last year)"]
    rw = f["ramadan_wape"]
    return [
        f"Remittance forecast for the next {f['horizon_weeks']} weeks: {sar(f['total'])} "
        f"(80% interval {sar(f['total_lo'])} to {sar(f['total_hi'])}).",
        f"Across {f['n_origins']} rolling backtests the Hijri-aware model's error (WAPE) is {pct(cal)} vs {pct(sn)} for "
        f"'same week last year'; around Ramadan 2026 it is {pct(rw['Calendar model (Hijri-aware)'])} vs "
        f"{pct(rw['Seasonal naive (same week last year)'])}, because Ramadan moved 11 days earlier. "
        f"A full Ramadan week carries about {pct(f['effects']['Ramadan week'], 0)} more remittance volume.",
    ]
