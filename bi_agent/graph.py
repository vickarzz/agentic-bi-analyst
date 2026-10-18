"""The agent: a LangGraph state machine.

    guard_input ──blocked──────────────────────────────────────────────┐
        │ ok                                                            │
        ▼                                                               │
      plan ◄──── retry with validator feedback (max N attempts) ───┐    │
        │                                                          │    │
        ▼                                                          │    │
    validate ──errors──────────────────────────────────────────────┘    │
        │ ok            └─ attempts exhausted / needs clarification ─┐  │
        ▼                                                            │  │
     compile ──► check_sql ──violation────────────────────────────┐  │  │
                    │ ok                                          │  │  │
                    ▼                                             ▼  ▼  ▼
                 execute ──► explain ─────────────────────────►  respond

Each node appends to `trace`, which the UI shows so users (and auditors) can see
exactly how an answer was produced.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from . import config, db
from .compiler import compile_plan, render_sql
from .guardrails import check_question, check_sql, suppress_small_cells
from .insights import build_facts, choose_chart, fmt
from .plan import QueryPlan, validate_plan
from .planner_rules import RulePlanner
from .semantic import SemanticLayer, default as default_semantic


class AgentState(TypedDict, total=False):
    question: str
    previous_plan: dict | None
    planner: str
    plan: dict | None
    errors: list[str]
    attempts: int
    sql: str
    params: list
    display_sql: str
    helpers: dict
    notes: list[str]
    result: Any                 # pandas DataFrame
    suppressed_cells: int
    facts: list[str]
    summary: str
    chart: dict
    status: str                 # ok | blocked | clarify | error
    message: str
    trace: Annotated[list[dict], operator.add]


def _t(step: str, detail: str, started: float) -> list[dict]:
    return [{"step": step, "detail": detail, "ms": round((time.perf_counter() - started) * 1000, 1)}]


class BIAgent:
    def __init__(self, sem: SemanticLayer | None = None, db_path: str | None = None,
                 planner: str | None = None, llm_client=None):
        self.sem = sem or default_semantic()
        self.db_path = db_path
        self.values = db.dimension_values(self.sem, db_path)
        self.rules = RulePlanner(self.sem, self.values)
        self.llm = None
        self.llm_client = llm_client
        want_llm = (planner == "llm") or (planner is None and config.use_llm())
        if want_llm:
            from .planner_llm import LLMPlanner
            self.llm = LLMPlanner(self.sem, self.values, client=llm_client)
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def guard_input(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        ok, reason = check_question(s["question"])
        if not ok:
            return {"status": "blocked", "message": reason, "trace": _t("guard_input", "BLOCKED: " + reason, t0)}
        return {"status": "ok", "attempts": 0, "errors": [], "trace": _t("guard_input", "passed", t0)}

    def plan(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        prev = QueryPlan.from_dict(s["previous_plan"]) if s.get("previous_plan") else None
        feedback = s.get("errors") or None
        used = "rules"
        if self.llm is not None:
            try:
                p = self.llm.plan(s["question"], prev, feedback)
                used = f"llm:{self.llm.model}"
            except Exception as e:  # network, quota, bad output -> deterministic fallback
                p = self.rules.plan(s["question"], prev, feedback)
                used = f"rules (LLM failed: {type(e).__name__})"
        else:
            p = self.rules.plan(s["question"], prev, feedback)
        attempt = s.get("attempts", 0) + 1
        return {"plan": p.to_dict(), "attempts": attempt, "planner": used,
                "trace": _t("plan", f"attempt {attempt} via {used}: {p.to_dict()}", t0)}

    def validate(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        p = QueryPlan.from_dict(s["plan"])
        if p.clarification:
            return {"status": "clarify", "message": p.clarification, "errors": [],
                    "trace": _t("validate", "planner asked for clarification", t0)}
        errs = validate_plan(p, self.sem)
        if errs and s["attempts"] >= config.MAX_PLAN_ATTEMPTS:
            return {"status": "clarify", "errors": errs,
                    "message": "I couldn't build a valid query for that: " + " ".join(errs),
                    "trace": _t("validate", f"giving up after {s['attempts']} attempts", t0)}
        notes = list(s.get("notes") or []) + errs
        return {"errors": errs, "notes": notes,
                "trace": _t("validate", "ok" if not errs else f"{len(errs)} problem(s): {errs}", t0)}

    def compile(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        cq = compile_plan(QueryPlan.from_dict(s["plan"]), self.sem)
        return {"sql": cq.sql, "helpers": cq.helper_columns, "params": cq.params, "display_sql": render_sql(cq),
                "trace": _t("compile", f"{len(cq.sql.splitlines())} lines of SQL, {len(cq.params)} bound parameter(s)", t0)}

    def check_sql(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        problems = check_sql(s["sql"], self.sem)
        if problems:
            return {"status": "error", "message": "SQL guard rejected the query: " + "; ".join(problems),
                    "trace": _t("check_sql", f"REJECTED: {problems}", t0)}
        return {"trace": _t("check_sql", "read-only, allow-listed tables, no PII, LIMIT present", t0)}

    def execute(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        try:
            df = db.run(s["sql"], s["params"], self.db_path)
        except Exception as e:
            return {"status": "error", "message": f"Query failed: {e}", "trace": _t("execute", f"ERROR {e}", t0)}
        k = int(self.sem.governance.get("small_cell_threshold", 10))
        df, masked = suppress_small_cells(df, s["helpers"], k)
        return {"result": df, "suppressed_cells": masked,
                "trace": _t("execute", f"{len(df)} row(s); {masked} cell(s) suppressed (<{k} customers)", t0)}

    def explain(self, s: AgentState) -> dict:
        t0 = time.perf_counter()
        p = QueryPlan.from_dict(s["plan"])
        df: pd.DataFrame = s["result"]
        facts = build_facts(df, p, self.sem)
        for n in s.get("notes") or []:
            facts.insert(0, "Adjusted the question: " + n.split(". Compatible")[0] + ".")
        if s.get("suppressed_cells"):
            facts.append(f"{s['suppressed_cells']} value(s) hidden because they describe fewer than "
                         f"{self.sem.governance.get('small_cell_threshold', 10)} customers.")
        summary = " ".join(facts)
        if self.llm is not None and not df.empty:
            try:
                from .planner_llm import summarize
                summary = summarize(facts, df.head(20).to_string(index=False), s["question"],
                                    client=self.llm.client, model=self.llm.model)
            except Exception:
                pass
        return {"facts": facts, "summary": summary, "chart": choose_chart(p, df), "status": "ok",
                "trace": _t("explain", f"{len(facts)} verified fact(s)", t0)}

    def respond(self, s: AgentState) -> dict:
        return {"trace": _t("respond", s.get("status", "ok"), time.perf_counter())}

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        for name in ("guard_input", "plan", "validate", "compile", "check_sql", "execute", "explain", "respond"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("guard_input")
        g.add_conditional_edges("guard_input", lambda s: "respond" if s["status"] == "blocked" else "plan")
        g.add_edge("plan", "validate")
        g.add_conditional_edges("validate", lambda s: "respond" if s.get("status") == "clarify"
                                else ("plan" if s.get("errors") else "compile"))
        g.add_edge("compile", "check_sql")
        g.add_conditional_edges("check_sql", lambda s: "respond" if s.get("status") == "error" else "execute")
        g.add_conditional_edges("execute", lambda s: "respond" if s.get("status") == "error" else "explain")
        g.add_edge("explain", "respond")
        g.add_edge("respond", END)
        return g.compile()

    # ------------------------------------------------------------------ public API
    def ask(self, question: str, previous_plan: dict | None = None) -> dict:
        state = self.graph.invoke({"question": question, "previous_plan": previous_plan, "trace": []})
        return dict(state)

    def format_answer(self, out: dict) -> str:
        if out.get("status") != "ok":
            return out.get("message", "Something went wrong.")
        lines = [out["summary"]]
        df = out["result"]
        if not df.empty and len(df) <= 30:
            show = df.copy()
            p = QueryPlan.from_dict(out["plan"])
            for m in p.metrics:
                show[m] = show[m].map(lambda v, k=self.sem.metrics[m].format: fmt(v, k))
            if "period" in show:
                show["period"] = pd.to_datetime(show["period"]).dt.date
            lines += ["", show.to_string(index=False)]
        return "\n".join(lines)
