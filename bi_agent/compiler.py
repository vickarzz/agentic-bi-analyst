"""Deterministic semantic-layer compiler: QueryPlan -> parameterised DuckDB SQL.

Why a compiler instead of letting the LLM write SQL?
  * Every number comes from the governed metric definition (no "creative" SQL).
  * Joins are pre-approved; PII columns are simply not reachable.
  * The same plan always produces the same SQL, which makes evals reproducible.

Metrics from different models (e.g. transactions + remittances) are computed in
separate CTEs and FULL OUTER JOINed on the shared dimensions — this avoids the
classic fan-out / double-counting bug of joining two fact tables directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .plan import QueryPlan
from .semantic import Model, SemanticLayer


@dataclass
class CompiledQuery:
    sql: str
    params: list
    group_columns: list[str]                 # output columns that are dimensions/period
    metric_columns: list[str]
    helper_columns: dict[str, list[str]] = field(default_factory=dict)  # __n_x -> metrics it guards


def _resolve_joins(model: Model, needed: set[str]) -> list[str]:
    """Return join SQL in dependency order for the aliases a query needs."""
    ordered: list[str] = []

    def visit(alias: str) -> None:
        if alias == model.alias or alias in ordered:
            return
        join = model.joins[alias]
        for r in join.requires:
            visit(r)
        ordered.append(alias)

    for a in sorted(needed):
        visit(a)
    return [model.joins[a].sql for a in ordered]


def _model_select(plan: QueryPlan, sem: SemanticLayer, model: Model, metrics: list[str],
                  helper: str) -> tuple[str, list]:
    params: list = []
    needed: set[str] = set()
    select: list[str] = []

    if plan.time_grain:
        select.append(f"CAST(date_trunc('{plan.time_grain}', {model.time_column}) AS DATE) AS period")
    for d in plan.dimensions:
        dim = sem.dimensions[d]
        needed |= set(dim.requires)
        select.append(f"{dim.expr_for(model)} AS {d}")
    for m in metrics:
        select.append(f"{sem.metrics[m].expr} AS {m}")
    select.append(f"COUNT(DISTINCT {model.customer_key}) AS {helper}")

    where: list[str] = []
    if model.where:
        where.append(model.where)
    if plan.start_date:
        where.append(f"CAST({model.time_column} AS DATE) >= ?")
        params.append(plan.start_date)
    if plan.end_date:
        where.append(f"CAST({model.time_column} AS DATE) <= ?")
        params.append(plan.end_date)
    for f in plan.filters:
        dim = sem.dimensions[f.dimension]
        needed |= set(dim.requires)
        placeholders = ", ".join("?" for _ in f.values)
        neg = "NOT " if f.op == "not_in" else ""
        where.append(f"{dim.expr_for(model)} {neg}IN ({placeholders})")
        params.extend(f.values)

    sql = f"SELECT\n    " + ",\n    ".join(select)
    sql += f"\nFROM {model.table} {model.alias}"
    for j in _resolve_joins(model, needed):
        sql += f"\n{j}"
    if where:
        sql += "\nWHERE " + "\n  AND ".join(where)
    if plan.time_grain or plan.dimensions:
        sql += "\nGROUP BY ALL"
    return sql, params


def compile_plan(plan: QueryPlan, sem: SemanticLayer) -> CompiledQuery:
    max_rows = int(sem.governance.get("max_rows", 500))
    limit = min(plan.limit or max_rows, max_rows)

    by_model: dict[str, list[str]] = {}
    for m in plan.metrics:
        by_model.setdefault(sem.metrics[m].model, []).append(m)

    group_cols = (["period"] if plan.time_grain else []) + list(plan.dimensions)
    helpers: dict[str, list[str]] = {}
    parts: list[tuple[str, str, list]] = []
    for i, (model_name, metrics) in enumerate(by_model.items()):
        helper = f"__n_{model_name}"
        helpers[helper] = metrics
        sql, params = _model_select(plan, sem, sem.models[model_name], metrics, helper)
        parts.append((f"m{i}", sql, params))

    if plan.order_by:
        order = f"{plan.order_by} {plan.order_dir.upper()} NULLS LAST"
    elif plan.time_grain:
        order = "period ASC" + "".join(f", {d}" for d in plan.dimensions)
    elif plan.dimensions:
        order = f"{plan.metrics[0]} DESC NULLS LAST"
    else:
        order = None

    if len(parts) == 1:
        sql = parts[0][1]
        params = parts[0][2]
    else:
        ctes = ",\n".join(f"{name} AS (\n{body}\n)" for name, body, _ in parts)
        params = [p for _, _, ps in parts for p in ps]
        out_cols = group_cols + [m for m in plan.metrics] + list(helpers)
        if group_cols:
            using = ", ".join(group_cols)
            frm = parts[0][0] + "".join(f"\nFULL OUTER JOIN {n} USING ({using})" for n, _, _ in parts[1:])
        else:
            frm = parts[0][0] + "".join(f"\nCROSS JOIN {n}" for n, _, _ in parts[1:])
        sql = f"WITH {ctes}\nSELECT {', '.join(out_cols)}\nFROM {frm}"

    if order:
        sql += f"\nORDER BY {order}"
    sql += f"\nLIMIT {limit}"
    return CompiledQuery(sql=sql, params=params, group_columns=group_cols,
                         metric_columns=list(plan.metrics), helper_columns=helpers)


def render_sql(cq: CompiledQuery) -> str:
    """SQL with parameters inlined — for display only, never executed."""
    out = cq.sql
    for p in cq.params:
        out = out.replace("?", "'" + str(p).replace("'", "''") + "'", 1)
    return out
