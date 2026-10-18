"""Shared helpers for the Data Science Lab: governed data access, config, fingerprints, PSI."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .. import config, db
from ..guardrails import check_sql
from ..semantic import SemanticLayer, default as default_semantic

DS_CONFIG = config.ROOT / "semantic" / "ds_config.yaml"
REPORT_DIR = config.ROOT / "reports" / "ds"


class GovernanceError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ds_config() -> dict:
    return yaml.safe_load(DS_CONFIG.read_text())


def template(sql: str, params: list | None = None, db_path: str | None = None,
             sem: SemanticLayer | None = None, name: str = "template") -> pd.DataFrame:
    """Run a reviewed SQL template. It must pass the same SQL guard as the agents."""
    sem = sem or default_semantic()
    problems = check_sql(sql, sem)
    if problems:
        raise GovernanceError(f"{name}: " + "; ".join(problems))
    return db.run(sql, params or [], db_path)


def fingerprint(db_path: str | None = None) -> str:
    """Identifies the data snapshot (as-of date + row counts), used to version models and caches."""
    con = db.connect(db_path)
    parts = [str(con.execute("SELECT as_of_date FROM _metadata").fetchone()[0])]
    for t in ("dim_customer", "fact_transaction", "fact_finance", "fact_remittance", "fact_experiment"):
        try:
            parts.append(f"{t}:{con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
        except Exception:  # noqa: BLE001 - older databases have no experiment table
            parts.append(f"{t}:0")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def ground_truth(db_path: str | None = None) -> dict:
    """Planted truth. Only the validation step reads this, never the models."""
    row = db.connect(db_path).execute("SELECT truth_json FROM _ground_truth").fetchone()
    return json.loads(row[0])


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index between two score distributions (quantile bins of `expected`)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def pct(x: float, digits: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def sar(x: float) -> str:
    a = abs(x)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"SAR {x / div:,.2f}{suf}"
    return f"SAR {x:,.0f}"


def write_report(name: str, text: str) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORT_DIR / name
    p.write_text(text)
    return p


def to_jsonable(obj):
    """Make nested results JSON-serialisable (numpy, pandas, NaN)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.date().isoformat()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj
