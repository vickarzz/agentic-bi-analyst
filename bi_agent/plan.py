"""QueryPlan: the ONLY thing a planner (rules or LLM) is allowed to produce.

A plan is a small, structured description of a question in semantic-layer terms.
It is validated against the catalog before any SQL is generated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

from .semantic import SemanticLayer


@dataclass
class Filter:
    dimension: str
    values: list[str]
    op: str = "in"           # "in" | "not_in"


@dataclass
class QueryPlan:
    metrics: list[str]
    dimensions: list[str] = field(default_factory=list)
    time_grain: str | None = None
    start_date: str | None = None    # ISO yyyy-mm-dd, inclusive
    end_date: str | None = None      # ISO yyyy-mm-dd, inclusive
    filters: list[Filter] = field(default_factory=list)
    order_by: str | None = None      # a metric name
    order_dir: str = "desc"
    limit: int | None = None
    time_label: str | None = None    # human text, e.g. "last 12 months"
    clarification: str | None = None # set when the planner cannot answer

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QueryPlan":
        d = dict(d)
        d["filters"] = [f if isinstance(f, Filter) else Filter(**f) for f in d.get("filters") or []]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def validate_plan(plan: QueryPlan, sem: SemanticLayer) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errs: list[str] = []
    if plan.clarification:
        return errs
    if not plan.metrics:
        errs.append("No metric selected. Choose at least one metric from the catalog.")
    unknown = [m for m in plan.metrics if m not in sem.metrics]
    if unknown:
        errs.append(f"Unknown metric(s): {unknown}. Valid: {sorted(sem.metrics)}")
    valid_metrics = [m for m in plan.metrics if m in sem.metrics]

    used_dims = list(plan.dimensions) + [f.dimension for f in plan.filters]
    for d in used_dims:
        if d not in sem.dimensions:
            errs.append(f"Unknown dimension '{d}'. Valid: {sorted(sem.dimensions)}")
            continue
        for m in valid_metrics:
            model = sem.metrics[m].model
            if not sem.dimension_available(d, model):
                errs.append(f"Dimension '{d}' cannot be used with metric '{m}' (model '{model}'). "
                            f"Compatible dimensions: {sem.compatible_dimensions([m])}")
    if plan.time_grain and plan.time_grain not in sem.time_grains:
        errs.append(f"Unknown time grain '{plan.time_grain}'. Valid: {sem.time_grains}")
    for label, val in (("start_date", plan.start_date), ("end_date", plan.end_date)):
        if val:
            try:
                date.fromisoformat(val)
            except ValueError:
                errs.append(f"{label} must be ISO yyyy-mm-dd, got '{val}'")
    if plan.start_date and plan.end_date and plan.start_date > plan.end_date:
        errs.append("start_date is after end_date")
    if plan.order_by and plan.order_by not in plan.metrics:
        errs.append(f"order_by '{plan.order_by}' must be one of the selected metrics {plan.metrics}")
    if plan.order_dir not in ("asc", "desc"):
        errs.append("order_dir must be 'asc' or 'desc'")
    if plan.limit is not None and (not isinstance(plan.limit, int) or plan.limit < 1):
        errs.append("limit must be a positive integer")
    for f in plan.filters:
        if f.op not in ("in", "not_in") or not f.values:
            errs.append(f"Filter on '{f.dimension}' needs op in/not_in and at least one value")
    if len(plan.dimensions) > 3:
        errs.append("At most 3 dimensions per question keeps results readable")
    return errs
