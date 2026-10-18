"""Read-only DuckDB access."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from . import config
from .semantic import SemanticLayer


class DataNotFound(RuntimeError):
    pass


@lru_cache(maxsize=4)
def connect(path: str | None = None) -> duckdb.DuckDBPyConnection:
    p = Path(path or config.DB_PATH)
    if not p.exists():
        raise DataNotFound(f"{p} not found. Run `python scripts/generate_data.py` first.")
    return duckdb.connect(str(p), read_only=True)   # read-only at the engine level too


def run(sql: str, params: list | None = None, path: str | None = None) -> pd.DataFrame:
    con = connect(path).cursor()
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


_VALUES_CACHE: dict[str, dict[str, list[str]]] = {}


def dimension_values(sem: SemanticLayer, path: str | None = None) -> dict[str, list[str]]:
    """Distinct values per dimension — lets the planner recognise 'Riyadh', 'Affluent', 'Mobile App'..."""
    key = str(path or config.DB_PATH)
    if key in _VALUES_CACHE:
        return _VALUES_CACHE[key]
    out: dict[str, list[str]] = {}
    for name, dim in sem.dimensions.items():
        model = next(m for m in sem.models.values() if sem.dimension_available(name, m.name))
        joins = []
        needed = set(dim.requires)
        # resolve required joins in dependency order
        for alias in ["c", "p", "b"]:
            if alias in needed and alias != model.alias:
                joins.append(model.joins[alias].sql)
        sql = (f"SELECT DISTINCT CAST({dim.expr_for(model)} AS VARCHAR) v FROM {model.table} {model.alias} "
               f"{' '.join(joins)} WHERE {dim.expr_for(model)} IS NOT NULL ORDER BY 1 LIMIT 200")
        out[name] = run(sql, path=path)["v"].tolist()
    _VALUES_CACHE[key] = out
    return out
