"""Load, validate and describe the semantic layer (semantic/semantic_layer.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from . import config


@dataclass
class Join:
    alias: str
    sql: str
    requires: list[str] = field(default_factory=list)


@dataclass
class Model:
    name: str
    table: str
    alias: str
    time_column: str
    customer_key: str
    joins: dict[str, Join]
    where: str | None = None


@dataclass
class Dimension:
    name: str
    expr: str
    description: str
    requires: list[str] = field(default_factory=list)
    models: list[str] | None = None          # None = any model that can satisfy `requires`
    synonyms: list[str] = field(default_factory=list)

    def expr_for(self, model: Model) -> str:
        return self.expr.replace("{alias}", model.alias)


@dataclass
class Metric:
    name: str
    model: str
    label: str
    expr: str
    format: str
    owner: str
    description: str
    synonyms: list[str] = field(default_factory=list)
    additive: bool = True       # can values be summed across groups (shares of total)?


@dataclass
class SemanticLayer:
    as_of_date: str
    governance: dict
    models: dict[str, Model]
    dimensions: dict[str, Dimension]
    metrics: dict[str, Metric]
    time_grains: list[str]
    value_aliases: dict = field(default_factory=dict)   # business word -> {dimension, value}

    # ------------------------------------------------------------------ helpers
    def dimension_available(self, dim: str, model_name: str) -> bool:
        d, m = self.dimensions[dim], self.models[model_name]
        if d.models is not None and model_name not in d.models:
            return False
        return all(a == m.alias or a in m.joins for a in d.requires)

    def compatible_dimensions(self, metric_names: list[str]) -> list[str]:
        models = {self.metrics[m].model for m in metric_names}
        return [d for d in self.dimensions if all(self.dimension_available(d, mo) for mo in models)]

    def catalog_text(self) -> str:
        """Compact catalog given to the LLM planner (definitions only, no data)."""
        lines = [f"As-of date of the data: {self.as_of_date}", "", "METRICS (name | model | definition):"]
        for m in self.metrics.values():
            lines.append(f"- {m.name} | {m.model} | {m.description}")
        lines += ["", "DIMENSIONS (name | definition | usable with models):"]
        for d in self.dimensions.values():
            usable = [mo for mo in self.models if self.dimension_available(d.name, mo)]
            lines.append(f"- {d.name} | {d.description} | {', '.join(usable)}")
        lines += ["", f"TIME GRAINS: {', '.join(self.time_grains)}"]
        return "\n".join(lines)


def _validate(raw: dict, sem: SemanticLayer) -> None:
    errs = []
    for m in sem.metrics.values():
        if m.model not in sem.models:
            errs.append(f"metric {m.name}: unknown model {m.model}")
    for d in sem.dimensions.values():
        for mo in d.models or []:
            if mo not in sem.models:
                errs.append(f"dimension {d.name}: unknown model {mo}")
        if not any(sem.dimension_available(d.name, mo) for mo in sem.models):
            errs.append(f"dimension {d.name}: not reachable from any model")
    for mo in sem.models.values():
        for j in mo.joins.values():
            for r in j.requires:
                if r not in mo.joins:
                    errs.append(f"model {mo.name}: join {j.alias} requires unknown join {r}")
    for phrase, v in sem.value_aliases.items():
        if not isinstance(v, dict) or v.get("dimension") not in sem.dimensions or not v.get("value"):
            errs.append(f"value alias '{phrase}': needs a known dimension and a value")
    names = list(sem.metrics) + list(sem.dimensions)
    if len(names) != len(set(names)):
        errs.append("metric and dimension names must be unique")
    if errs:
        raise ValueError("Invalid semantic layer:\n  " + "\n  ".join(errs))


def load(path: str | Path | None = None) -> SemanticLayer:
    raw = yaml.safe_load(Path(path or config.SEMANTIC_PATH).read_text())
    models = {
        name: Model(
            name=name, table=m["table"], alias=m["alias"], time_column=m["time_column"],
            customer_key=m["customer_key"], where=m.get("where"),
            joins={a: Join(a, j["sql"], j.get("requires", [])) for a, j in (m.get("joins") or {}).items()},
        )
        for name, m in raw["models"].items()
    }
    dims = {
        name: Dimension(name=name, expr=d["expr"], description=d.get("description", ""),
                        requires=d.get("requires", []), models=d.get("models"),
                        synonyms=d.get("synonyms", []))
        for name, d in raw["dimensions"].items()
    }
    metrics = {
        name: Metric(name=name, model=m["model"], label=m.get("label", name), expr=m["expr"],
                     format=m.get("format", "number"), owner=m.get("owner", ""),
                     description=m.get("description", ""), synonyms=m.get("synonyms", []),
                     additive=m.get("additive", True))
        for name, m in raw["metrics"].items()
    }
    sem = SemanticLayer(as_of_date=str(raw["as_of_date"]), governance=raw.get("governance", {}),
                        models=models, dimensions=dims, metrics=metrics,
                        time_grains=raw.get("time_grains", ["month"]),
                        value_aliases={k.lower(): v for k, v in (raw.get("value_aliases") or {}).items()})
    _validate(raw, sem)
    return sem


@lru_cache(maxsize=1)
def default() -> SemanticLayer:
    return load()
