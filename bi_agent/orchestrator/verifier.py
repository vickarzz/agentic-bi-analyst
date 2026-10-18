"""Verifier (critic) agent: checks the briefing before anyone sees it.

1. Grounding  — every number in the narrative must appear in a computed fact.
2. Language   — no causal claims ("drives", "causes", ...) for statistical associations.
3. Evidence   — anything called "confirmed" must have p < 0.05; a failed control blocks sign-off.
4. Coverage   — every domain in scope has at least one finding.

If it fails, the orchestrator sends the issues back to the writer for another attempt.
"""
from __future__ import annotations

import re

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s?(?:%|pts|x|[KMB](?![a-z]))?", re.I)
CAUSAL = re.compile(r"\b(causes?|caused|causing|drives?|driven by|driving|leads? to|led to|results? in|"
                    r"because of|due to|proves?)\b", re.I)


def _norm(tok: str) -> str:
    return tok.replace(",", "").replace(" ", "").lower().lstrip("+")


def _allowed_numbers(findings: list[dict], links: list[dict], discoveries: list[dict], extra: str = "") -> set[str]:
    corpus = [f["text"] for f in findings] + [l["summary"] for l in links] + [l["title"] for l in links]
    corpus += [l.get("action", "") for l in links] + [d["text"] for d in discoveries] + [extra]
    return {_norm(m.group()) for t in corpus for m in NUM.finditer(t)}


def verify(narrative: str, findings: list[dict], links: list[dict], discoveries: list[dict],
           domains: list[str], allowed_extra: str = "") -> dict:
    issues: list[str] = []
    allowed = _allowed_numbers(findings, links, discoveries, allowed_extra)
    tokens = [m.group() for m in NUM.finditer(narrative)]
    for tok in tokens:
        n = _norm(tok)
        if re.fullmatch(r"\d", n):
            continue  # single digits: list numbering, "3 links", etc.
        if n not in allowed:
            issues.append(f"Ungrounded number '{tok.strip()}' does not appear in any computed fact.")
    for m in CAUSAL.finditer(narrative):
        issues.append(f"Causal wording '{m.group()}' — the evidence is an association; use 'is associated with'.")
    for l in links:
        if l["verdict"] == "confirmed" and (l.get("p_value") is None or l["p_value"] >= 0.05):
            issues.append(f"Link '{l['id']}' is labelled confirmed without p < 0.05.")
        if l["verdict"] == "control FAILED":
            issues.append(f"Control test '{l['id']}' failed: the method is finding links in noise.")
    covered = {f["domain"] for f in findings}
    for d in domains:
        if d not in covered:
            issues.append(f"Domain '{d}' has no findings.")
    return {"passed": not issues, "issues": issues, "numbers_checked": len(tokens)}
