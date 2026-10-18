"""The ONLY data tools the orchestrator agents may use.

`governed_query` takes a QueryPlan (semantic-layer metric names), so every number in the
business review comes from the same governed definitions as the chat agent, and passes the
same SQL guard and small-cell suppression.

`approved_analysis` runs a reviewed SQL template from semantic/impact_links.yaml (cohort
comparisons that need joins across fact tables). Templates are owned and versioned like
metrics, and still go through the SQL guard; agents cannot write new SQL.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import db
from ..compiler import compile_plan
from ..guardrails import check_sql, suppress_small_cells
from ..plan import Filter, QueryPlan, validate_plan
from ..semantic import SemanticLayer


class GovernanceError(RuntimeError):
    pass


@dataclass
class ToolLog:
    calls: list[dict] = field(default_factory=list)


def governed_query(sem: SemanticLayer, plan: QueryPlan, db_path: str | None = None,
                   log: ToolLog | None = None) -> pd.DataFrame:
    errs = validate_plan(plan, sem)
    if errs:
        raise GovernanceError("; ".join(errs))
    cq = compile_plan(plan, sem)
    problems = check_sql(cq.sql, sem)
    if problems:
        raise GovernanceError("; ".join(problems))
    df = db.run(cq.sql, cq.params, db_path)
    df, masked = suppress_small_cells(df, cq.helper_columns, int(sem.governance.get("small_cell_threshold", 10)))
    if log is not None:
        log.calls.append({"tool": "governed_query", "metrics": plan.metrics, "dimensions": plan.dimensions,
                          "grain": plan.time_grain, "rows": len(df), "suppressed": masked})
    return df


def approved_analysis(sem: SemanticLayer, name: str, sql: str, db_path: str | None = None,
                      log: ToolLog | None = None) -> pd.DataFrame:
    problems = check_sql(sql, sem)
    if problems:
        raise GovernanceError(f"template {name}: " + "; ".join(problems))
    df = db.run(sql, [], db_path)
    if log is not None:
        log.calls.append({"tool": "approved_analysis", "template": name, "rows": len(df)})
    return df


def monthly_panel(sem: SemanticLayer, metrics: list[str], start: str, end: str,
                  db_path: str | None = None, log: ToolLog | None = None) -> pd.DataFrame:
    """One governed monthly time series per metric, joined into a single panel (index = month)."""
    frames = []
    by_model: dict[str, list[str]] = {}
    for m in metrics:
        by_model.setdefault(sem.metrics[m].model, []).append(m)
    for ms in by_model.values():
        df = governed_query(sem, QueryPlan(metrics=ms, time_grain="month", start_date=start, end_date=end),
                            db_path, log)
        frames.append(df.assign(period=pd.to_datetime(df["period"])).set_index("period")[ms])
    panel = pd.concat(frames, axis=1).sort_index()
    return panel


def q(sem, metrics, dims=None, grain=None, start=None, end=None, filters=None, db_path=None, log=None):
    """Short helper for agents."""
    return governed_query(sem, QueryPlan(metrics=metrics, dimensions=dims or [], time_grain=grain,
                                         start_date=start, end_date=end,
                                         filters=[Filter(**f) if isinstance(f, dict) else f for f in (filters or [])]),
                          db_path, log)
