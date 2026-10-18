"""Claude-powered planner.

The LLM never sees raw data and never writes SQL. It receives the semantic catalog and must
call ONE tool, `submit_query_plan`, whose JSON schema only allows catalog names (enums).
Validation errors from the previous attempt are fed back so the model can self-correct.
"""
from __future__ import annotations

import json

from . import config
from .plan import QueryPlan
from .semantic import SemanticLayer

SYSTEM = """You are the planning step of a governed BI agent for Wadi Bank (a fictional Saudi retail bank).
Translate the user's question into a query plan by calling the `submit_query_plan` tool exactly once.

Rules:
- Use ONLY metric and dimension names from the catalog. Never invent metrics or SQL.
- Resolve relative dates against the data as-of date ({as_of}). "Last 12 months" = the 12 full months ending on the as-of month.
  Dates are inclusive ISO yyyy-mm-dd. Put a short human label in time_label (e.g. "last 12 months", "Q2 2026").
- Use time_grain for trends ("monthly", "over time", "trend"); use dimensions for breakdowns ("by region").
- Filters must use exact values from the KNOWN VALUES list.
- "Top N"/"highest" -> order_by the main metric, order_dir desc, limit N. "Lowest/bottom" -> asc.
- For a follow-up ("and by region?", "what about expats?") reuse the previous plan and change only what the user changed.
- If the question cannot be answered with this catalog (or asks for personal data), set `clarification` to a short,
  helpful message and leave metrics empty.

CATALOG
{catalog}

KNOWN VALUES (dimension: values)
{values}
"""


def _tool_schema(sem: SemanticLayer) -> dict:
    return {
        "name": "submit_query_plan",
        "description": "Submit the structured query plan for the user's question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metrics": {"type": "array", "items": {"type": "string", "enum": list(sem.metrics)}},
                "dimensions": {"type": "array", "items": {"type": "string", "enum": list(sem.dimensions)}, "maxItems": 3},
                "time_grain": {"type": ["string", "null"], "enum": [*sem.time_grains, None]},
                "start_date": {"type": ["string", "null"], "description": "inclusive yyyy-mm-dd"},
                "end_date": {"type": ["string", "null"], "description": "inclusive yyyy-mm-dd"},
                "time_label": {"type": ["string", "null"]},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dimension": {"type": "string", "enum": list(sem.dimensions)},
                            "op": {"type": "string", "enum": ["in", "not_in"]},
                            "values": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        },
                        "required": ["dimension", "op", "values"],
                    },
                },
                "order_by": {"type": ["string", "null"], "enum": [*sem.metrics, None]},
                "order_dir": {"type": "string", "enum": ["asc", "desc"]},
                "limit": {"type": ["integer", "null"], "minimum": 1},
                "clarification": {"type": ["string", "null"]},
            },
            "required": ["metrics"],
        },
    }


class LLMPlanner:
    name = "llm"

    def __init__(self, sem: SemanticLayer, dim_values: dict[str, list[str]], client=None, model: str | None = None):
        self.sem = sem
        self.model = model or config.LLM_MODEL
        if client is None:
            import anthropic  # imported lazily so the offline mode needs no key
            client = anthropic.Anthropic()
        self.client = client
        values = "\n".join(f"- {d}: {', '.join(v[:30])}" for d, v in dim_values.items())
        self.system = SYSTEM.format(as_of=sem.as_of_date, catalog=sem.catalog_text(), values=values)
        self.tool = _tool_schema(sem)

    def plan(self, question: str, previous: QueryPlan | None = None,
             feedback: list[str] | None = None) -> QueryPlan:
        content = f"Question: {question}"
        if previous:
            content += f"\n\nPrevious plan (for follow-ups):\n{json.dumps(previous.to_dict())}"
        if feedback:
            content += "\n\nYour previous plan failed validation. Fix these problems:\n- " + "\n- ".join(feedback)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.system,
            tools=[self.tool],
            tool_choice={"type": "tool", "name": "submit_query_plan"},
            messages=[{"role": "user", "content": content}],
        )
        block = next(b for b in resp.content if getattr(b, "type", None) == "tool_use")
        data = dict(block.input)
        data.setdefault("order_dir", "desc")
        return QueryPlan.from_dict(data)


def summarize(facts: list[str], table_preview: str, question: str, client=None, model: str | None = None) -> str:
    """Optional: turn deterministic facts into a short executive summary. Numbers come only from `facts`."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    prompt = (f"Question: {question}\n\nVerified facts:\n- " + "\n- ".join(facts) +
              f"\n\nResult preview:\n{table_preview}\n\n"
              "Write a 2-3 sentence executive summary for a bank manager. Use ONLY numbers that appear in the "
              "verified facts or the preview; do not speculate about causes you cannot see in the data.")
    resp = client.messages.create(model=model or config.LLM_MODEL, max_tokens=300,
                                  messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") for b in resp.content).strip()
