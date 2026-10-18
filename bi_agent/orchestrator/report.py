"""Synthesizer: turns domain findings, impact tests and discoveries into an executive briefing.

The default writer is deterministic (assembles verified sentences). If an LLM writer is
enabled, it may only rewrite the executive summary, and its text must pass the Verifier.
"""
from __future__ import annotations

from datetime import datetime

from .domains import DOMAINS

SEV_ORDER = {"alert": 0, "watch": 1, "positive": 2, "info": 3}
VERDICT_ICON = {"confirmed": "✅", "contradicted": "⚠️", "not detected": "➖",
                "control passed": "🧪", "control FAILED": "❌"}


def build_briefing(findings: list[dict], links: list[dict], discoveries: list[dict],
                   scope: dict, n_pairs: int, as_of: str) -> dict:
    ds = [f for f in findings if f["domain"] == "data_science"]
    findings = [f for f in findings if f["domain"] != "data_science"]
    by_domain: dict[str, list[dict]] = {}
    for f in findings:
        by_domain.setdefault(f["domain"], []).append(f)
    for d in by_domain:
        by_domain[d].sort(key=lambda f: SEV_ORDER.get(f["severity"], 9))

    confirmed = [l for l in links if l["verdict"] == "confirmed"]
    not_detected = [l for l in links if l["verdict"] == "not detected"]
    control = [l for l in links if l["verdict"].startswith("control")]
    alerts = [f for f in findings if f["severity"] == "alert"]
    watch = [f for f in findings if f["severity"] == "watch"]
    positives = [f for f in findings if f["severity"] == "positive"]

    summary: list[str] = []
    for f in alerts[:2]:
        summary.append(f["text"])
    for l in confirmed[:3]:
        summary.append(f"{l['title']}: {l['summary']}")
    for l in not_detected:
        if l["id"] == "fee_cut_volume_uplift":
            summary.append(f"{l['title']}: the expected lift is not supported by the data. {l['summary']}")
    for f in positives[:1]:
        summary.append(f["text"])
    summary = summary[:6]
    ds_lead = [f for f in ds if f["kind"] in ("causal", "prediction")][:1]
    summary += [f["text"] for f in ds_lead]

    actions = []
    for l in confirmed + not_detected:
        if l.get("action") and not l["id"].startswith("placebo"):
            actions.append({"link": l["title"], "verdict": l["verdict"], "action": l["action"]})
    kinds = {f["kind"] for f in ds}
    if "causal" in kinds:
        actions.append({"link": "Cashback campaign", "verdict": "causal estimate",
                        "action": "Budget the next campaign on the difference-in-differences effect, not the naive "
                                  "before/after, and plan for the lift to stop when the campaign ends."})
    if "prediction" in kinds:
        actions.append({"link": "Churn model", "verdict": "out-of-time validated",
                        "action": "Give the retention team the model's quarterly contact plan and re-run the fairness "
                                  "audit at every retrain."})
    if any(f["kind"] == "experiment" and "Recommendation: ship" in f["text"] for f in ds):
        actions.append({"link": "One-tap remittance flow", "verdict": "A/B test passed",
                        "action": "Ship the new flow and focus onboarding messages on younger customers, where the "
                                  "lift is largest."})

    caveats = [
        "All links are statistical associations in governed, synthetic data, not proof of causation.",
        "Time-series links use month-over-month growth to remove shared trends; p-values are permutation-based and "
        "Bonferroni-adjusted for the number of lags or pairs tried.",
        f"The discovery scan checked {n_pairs} metric pairs; its candidates need analyst review before use.",
    ]
    if any(l["verdict"] == "control FAILED" for l in control):
        caveats.insert(0, "The control test FAILED: treat every confirmed link with extra caution.")
    elif control:
        caveats.append("The control (placebo) test passed, so the method is not finding links everywhere.")
    if discoveries:
        caveats.append("Discovered co-movement may reflect a shared driver such as the seasonal calendar.")

    return {
        "title": "Wadi Bank: cross-business review",
        "as_of": as_of,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "scope": scope,
        "executive_summary": summary,
        "alerts": [f["text"] for f in alerts],
        "watch": [f["text"] for f in watch],
        "domains": {d: {"label": DOMAINS[d]["label"], "findings": by_domain.get(d, [])} for d in scope["domains"]},
        "links": links,
        "discoveries": discoveries,
        "actions": actions,
        "data_science": ds,
        "caveats": caveats,
    }


def briefing_text(b: dict) -> str:
    """Every sentence the verifier must check (summary, alerts, actions are free text)."""
    parts = list(b["executive_summary"]) + b["alerts"] + b["watch"]
    parts += [a["action"] for a in b["actions"]]
    return "\n".join(parts)


def to_markdown(b: dict) -> str:
    L = [f"# {b['title']}", "", f"*Data as of {b['as_of']} · generated {b['generated_at']} · "
         f"domains: {', '.join(DOMAINS[d]['label'] for d in b['scope']['domains'])} · synthetic data*", ""]
    L += ["## Executive summary", ""] + [f"- {s}" for s in b["executive_summary"]] + [""]
    if b["alerts"] or b["watch"]:
        L += ["## Risks and watch-list", ""]
        L += [f"- 🔴 {a}" for a in b["alerts"]] + [f"- 🟠 {w}" for w in b["watch"]] + [""]
    L += ["## Cross-business impact map", "", "| Link | From → To | Verdict | Evidence |", "|---|---|---|---|"]
    for l in b["links"]:
        L.append(f"| {l['title']} | {DOMAINS[l['from_domain']]['label']} → {DOMAINS[l['to_domain']]['label']} | "
                 f"{VERDICT_ICON.get(l['verdict'], '')} {l['verdict']} | {l['summary']} |")
    L.append("")
    if b["discoveries"]:
        L += ["## Candidate links found by the discovery scan (need review)", ""]
        L += [f"- {d['text']}" for d in b["discoveries"]] + [""]
    L += ["## Recommended actions", ""]
    L += [f"{i}. **{a['link']}** ({a['verdict']}): {a['action']}" for i, a in enumerate(b["actions"], 1)] + [""]
    if b.get("data_science"):
        L += ["## Data science lab (causal, prediction, forecast, experiment, segments)", ""]
        L += [f"- {f['text']}" for f in b["data_science"]] + [""]
    L += ["## Domain scorecards", ""]
    icon = {"alert": "🔴", "watch": "🟠", "positive": "🟢", "info": "⚪"}
    for d, v in b["domains"].items():
        L += [f"### {v['label']}", ""] + [f"- {icon.get(f['severity'], '')} {f['text']}" for f in v["findings"]] + [""]
    L += ["## Method and caveats", ""] + [f"- {c}" for c in b["caveats"]] + [""]
    if b.get("verification"):
        v = b["verification"]
        L += [f"*Verifier: {'passed' if v['passed'] else 'FAILED'} · {v['numbers_checked']} numbers checked "
              f"against computed facts · {len(v['issues'])} issue(s).*", ""]
    return "\n".join(L)
