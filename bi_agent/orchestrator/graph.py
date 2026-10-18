"""Business Review Orchestrator: a supervisor + parallel specialist agents (LangGraph).

                          ┌─► Domain agent: Retail & cards ─────┐
                          ├─► Domain agent: Digital banking ────┤
  supervisor ─► build_panel ├─► Domain agent: Islamic finance ────┤
  (scope)     (governed     ├─► Domain agent: Remittances ────────┼─► synthesizer ─► verifier ─► respond
               monthly KPIs)├─► Domain agent: Customer base ──────┤        ▲              │
                          ├─► Impact analyst (curated links) ───┤        └── issues ────┘
                          └─► Discovery agent (pair scan) ──────┘     (max 2 rewrites)

* Fan-out uses LangGraph `Send`, so specialists run in parallel in one super-step.
* Every agent uses governed tools only (semantic-layer metrics or approved SQL templates).
* The verifier blocks ungrounded numbers, causal wording and weak "confirmed" claims.
"""
from __future__ import annotations

import json
import operator
import re
import time
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .. import config
from ..semantic import SemanticLayer, default as default_semantic
from .discovery import correlation_heatmap, discover
from .domains import DOMAINS, DomainAnalyst
from .links import ImpactAnalyst, load_calendar, load_links
from .report import briefing_text, build_briefing, to_markdown
from .tools import ToolLog, monthly_panel
from .verifier import verify

PANEL_START = "2024-01-01"
KEYWORDS = {
    "retail": ["retail", "card", "spend", "transaction", "ticket"],
    "digital": ["digital", "mobile", "app", "online", "channel"],
    "credit": ["credit", "finance", "financing", "murabaha", "ijarah", "npf", "npl", "loan", "risk", "portfolio"],
    "payments": ["remittance", "payment", "fee", "corridor", "transfer"],
    "customer": ["customer", "churn", "acquisition", "attrition", "onboard"],
}


class ReviewState(TypedDict, total=False):
    request: str
    scope: dict
    panel: Any
    heatmap: Any
    findings: Annotated[list[dict], operator.add]
    links: list[dict]
    discoveries: list[dict]
    n_pairs: int
    briefing: dict
    verification: dict
    attempts: int
    feedback: list[str]
    markdown: str
    writer: str
    trace: Annotated[list[dict], operator.add]
    tool_calls: Annotated[list[dict], operator.add]


def _trace(agent: str, detail: str, t0: float, group: str = "") -> list[dict]:
    return [{"agent": agent, "detail": detail, "ms": round((time.perf_counter() - t0) * 1000, 1), "group": group}]


class BusinessReviewOrchestrator:
    def __init__(self, sem: SemanticLayer | None = None, db_path: str | None = None,
                 writer: str | None = None, llm_client=None, max_rewrites: int = 2):
        self.sem = sem or default_semantic()
        self.db_path = db_path
        self.calendar = load_calendar()
        self.writer = writer or ("llm" if config.use_llm() else "deterministic")
        self.llm_client = llm_client
        self.max_rewrites = max_rewrites
        self.flow_metrics = sorted({m for d in DOMAINS.values() for m in d.get("flow", [])})
        self.graph = self._build()

    # ------------------------------------------------------------------ agents
    def supervisor(self, s: ReviewState) -> dict:
        t0 = time.perf_counter()
        text = (s.get("request") or "").lower()
        domains = [d for d, kws in KEYWORDS.items() if any(re.search(rf"\b{k}", text) for k in kws)]
        whole = not domains or re.search(r"\b(whole|all|entire|overall|full|every|seluruh|semua)\b", text)
        if whole:
            domains = list(DOMAINS)
        scope = {"domains": domains, "panel_start": PANEL_START, "as_of": self.sem.as_of_date,
                 "mode": "whole business" if whole else "focused"}
        return {"scope": scope, "attempts": 0, "feedback": [], "writer": self.writer,
                "trace": _trace("supervisor", f"scope = {scope['mode']}: {', '.join(domains)}; "
                                f"dispatching {len(domains)} domain agents + impact analyst + discovery agent", t0)}

    def build_panel(self, s: ReviewState) -> dict:
        t0, log = time.perf_counter(), ToolLog()
        panel = monthly_panel(self.sem, self.flow_metrics, s["scope"]["panel_start"], self.sem.as_of_date,
                              self.db_path, log)
        return {"panel": panel, "heatmap": correlation_heatmap(panel), "tool_calls": log.calls,
                "trace": _trace("data_steward", f"governed monthly panel: {panel.shape[1]} metrics x "
                                f"{panel.shape[0]} months", t0)}

    def fan_out(self, s: ReviewState) -> list[Send]:
        sends = [Send("domain_agent", {"domain": d, "panel": s["panel"]}) for d in s["scope"]["domains"]]
        sends.append(Send("impact_analyst", {"panel": s["panel"], "scope": s["scope"]}))
        sends.append(Send("discovery_agent", {"panel": s["panel"]}))
        return sends

    def domain_agent(self, payload: dict) -> dict:
        t0, log = time.perf_counter(), ToolLog()
        d = payload["domain"]
        findings = DomainAnalyst(self.sem, d, payload["panel"], self.calendar, self.db_path, log).run()
        sev = {k: sum(f.severity == k for f in findings) for k in ("alert", "watch", "positive")}
        return {"findings": [f.to_dict() for f in findings], "tool_calls": log.calls,
                "trace": _trace(f"domain:{d}", f"{len(findings)} findings ({sev['alert']} alert, "
                                f"{sev['watch']} watch, {sev['positive']} positive)", t0, "parallel")}

    def impact_analyst(self, payload: dict) -> dict:
        t0, log = time.perf_counter(), ToolLog()
        analyst = ImpactAnalyst(self.sem, payload["panel"], self.db_path, log)
        doms = set(payload["scope"]["domains"])
        links = [l for l in load_links() if l["from"]["domain"] in doms or l["to"]["domain"] in doms]
        results = [analyst.run(l).to_dict() for l in links]
        counts = pd.Series([r["verdict"] for r in results]).value_counts().to_dict()
        return {"links": results, "tool_calls": log.calls,
                "trace": _trace("impact_analyst", f"tested {len(results)} cross-business links: {counts}", t0, "parallel")}

    def discovery_agent(self, payload: dict) -> dict:
        t0 = time.perf_counter()
        known = {frozenset((l["from"].get("metric"), l["to"].get("metric"))) for l in load_links()}
        found, n_pairs = discover(payload["panel"], self.sem, known)
        return {"discoveries": found, "n_pairs": n_pairs,
                "trace": _trace("discovery_agent", f"scanned {n_pairs} metric pairs; {len(found)} candidate link(s)", t0, "parallel")}

    def synthesizer(self, s: ReviewState) -> dict:
        t0 = time.perf_counter()
        b = build_briefing(s["findings"], s.get("links", []), s.get("discoveries", []), s["scope"],
                           s.get("n_pairs", 0), self.sem.as_of_date)
        attempt = s.get("attempts", 0) + 1
        how = "deterministic writer"
        if self.writer == "llm":
            try:
                b["executive_summary"] = self._llm_summary(b, s.get("feedback") or [])
                how = f"LLM writer (attempt {attempt})"
            except Exception as e:  # noqa: BLE001
                how = f"deterministic writer (LLM failed: {type(e).__name__})"
        return {"briefing": b, "attempts": attempt,
                "trace": _trace("synthesizer", f"briefing with {len(b['executive_summary'])} summary points, "
                                f"{len(b['actions'])} actions via {how}", t0)}

    def verifier(self, s: ReviewState) -> dict:
        t0 = time.perf_counter()
        b = s["briefing"]
        allowed_extra = " ".join([b["as_of"]])
        v = verify(briefing_text(b), s["findings"], s.get("links", []), s.get("discoveries", []),
                   s["scope"]["domains"], allowed_extra)
        b = dict(b, verification=v)
        return {"verification": v, "briefing": b, "feedback": v["issues"],
                "trace": _trace("verifier", ("passed" if v["passed"] else f"FAILED: {v['issues'][:3]}")
                                + f" ({v['numbers_checked']} numbers checked)", t0)}

    def respond(self, s: ReviewState) -> dict:
        t0 = time.perf_counter()
        b = s["briefing"]
        if not s["verification"]["passed"] and self.writer == "llm":
            # fall back to the verified deterministic summary rather than ship ungrounded text
            fb = build_briefing(s["findings"], s.get("links", []), s.get("discoveries", []), s["scope"],
                                s.get("n_pairs", 0), self.sem.as_of_date)
            v = verify(briefing_text(fb), s["findings"], s.get("links", []), s.get("discoveries", []),
                       s["scope"]["domains"], fb["as_of"])
            b = dict(fb, verification=v)
        return {"briefing": b, "markdown": to_markdown(b),
                "trace": _trace("respond", "briefing ready", t0)}

    # ------------------------------------------------------------------ LLM writer (optional)
    def _llm_summary(self, b: dict, feedback: list[str]) -> list[str]:
        client = self.llm_client
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        facts = {"current_summary": b["executive_summary"], "alerts": b["alerts"], "watch": b["watch"],
                 "links": [{k: l[k] for k in ("title", "verdict", "summary")} for l in b["links"]]}
        prompt = ("Rewrite this executive summary for a bank's executive committee as 4-6 short bullet points. "
                  "Use ONLY numbers that appear verbatim in the facts. Describe links as associations "
                  "('is associated with'), never as causes. Return a JSON list of strings only.\n\n"
                  f"FACTS:\n{json.dumps(facts, ensure_ascii=False, indent=1)}")
        if feedback:
            prompt += "\n\nThe verifier rejected the previous draft. Fix:\n- " + "\n- ".join(feedback)
        resp = client.messages.create(model=config.LLM_MODEL, max_tokens=800,
                                      messages=[{"role": "user", "content": prompt}])
        text = "".join(getattr(c, "text", "") for c in resp.content)
        items = json.loads(text[text.index("["): text.rindex("]") + 1])
        return [str(i) for i in items][:6]

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(ReviewState)
        for n in ("supervisor", "build_panel", "domain_agent", "impact_analyst", "discovery_agent",
                  "synthesizer", "verifier", "respond"):
            g.add_node(n, getattr(self, n))
        g.set_entry_point("supervisor")
        g.add_edge("supervisor", "build_panel")
        g.add_conditional_edges("build_panel", self.fan_out, ["domain_agent", "impact_analyst", "discovery_agent"])
        g.add_edge(["domain_agent", "impact_analyst", "discovery_agent"], "synthesizer")
        g.add_edge("synthesizer", "verifier")
        g.add_conditional_edges("verifier", lambda s: "respond" if s["verification"]["passed"]
                                or s["attempts"] > self.max_rewrites or self.writer != "llm" else "synthesizer",
                                ["respond", "synthesizer"])
        g.add_edge("respond", END)
        return g.compile()

    def run(self, request: str = "Review the whole business and how each line affects the others") -> dict:
        return dict(self.graph.invoke({"request": request, "trace": [], "findings": [], "tool_calls": []}))
