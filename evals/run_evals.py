"""Run the golden-question evaluation and write evals/report.md.

    python evals/run_evals.py                  # offline rule-based planner (CI default)
    python evals/run_evals.py --planner llm    # evaluate the Claude planner (needs ANTHROPIC_API_KEY)
    python evals/run_evals.py --min-score 0.9  # exit code 1 if overall accuracy is below 90%
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bi_agent import db  # noqa: E402
from bi_agent.graph import BIAgent  # noqa: E402
from bi_agent.plan import Filter, QueryPlan  # noqa: E402


def _norm_filters(fs) -> set:
    out = set()
    for f in fs or []:
        f = f if isinstance(f, Filter) else Filter(**f)
        out.add((f.dimension, f.op, tuple(sorted(f.values))))
    return out


def plan_matches(plan: QueryPlan, exp: dict) -> tuple[bool, list[str]]:
    diffs = []
    checks = {
        "metrics": (sorted(plan.metrics), sorted(exp.get("metrics", []))),
        "dimensions": (sorted(plan.dimensions), sorted(exp.get("dimensions", []))),
        "time_grain": (plan.time_grain, exp.get("time_grain")),
    }
    for key in ("start_date", "end_date", "order_by", "order_dir", "limit"):
        if key in exp:
            checks[key] = (getattr(plan, key), exp[key])
    if "filters" in exp:
        checks["filters"] = (_norm_filters(plan.filters), _norm_filters(exp["filters"]))
    elif plan.filters and set(f.dimension for f in plan.filters) - set(plan.dimensions):
        checks["filters"] = (_norm_filters(plan.filters), set())
    for k, (got, want) in checks.items():
        if got != want:
            diffs.append(f"{k}: got {got!r}, expected {want!r}")
    return not diffs, diffs


def results_match(got: pd.DataFrame, ref: pd.DataFrame, metrics: list[str],
                  allow_suppressed: bool = False) -> tuple[bool, str]:
    keys = [c for c in ref.columns if c not in metrics]
    if len(got) != len(ref):
        return False, f"row count {len(got)} vs reference {len(ref)}"
    g, r = got.copy(), ref.copy()
    for k in keys:
        if k not in g:
            return False, f"missing column {k}"
        g[k], r[k] = g[k].astype(str), r[k].astype(str)
    if keys:
        m = g.merge(r, on=keys, how="outer", suffixes=("_got", "_ref"), indicator=True)
        if (m["_merge"] != "both").any():
            return False, "group keys differ"
    else:
        m = pd.concat([g.add_suffix("_got"), r.add_suffix("_ref")], axis=1)
    for met in metrics:
        a, b = m[f"{met}_got"].astype(float), m[f"{met}_ref"].astype(float)
        for x, y in zip(a, b):
            if allow_suppressed and math.isnan(x) and not math.isnan(y):
                continue  # small-cell suppression hid this value on purpose
            if not (math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6) or (math.isnan(x) and math.isnan(y))):
                return False, f"{met}: {x} vs {y}"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["rules", "llm"], default="rules")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--report", default=str(ROOT / "evals" / "report.md"))
    args = ap.parse_args()

    cases = yaml.safe_load((ROOT / "evals" / "golden_questions.yaml").read_text())["cases"]
    agent = BIAgent(planner=args.planner)
    rows, t_all = [], time.perf_counter()

    for c in cases:
        exp = c["expect"]
        t0 = time.perf_counter()
        prev = None
        if c.get("previous"):
            prev = agent.ask(c["previous"]).get("plan")
        out = agent.ask(c["question"], prev)
        ms = (time.perf_counter() - t0) * 1000
        want_status = exp.get("status", "ok")
        status_ok = out.get("status") == want_status
        plan_ok, ans_ok, detail = None, None, ""
        if want_status == "ok" and status_ok:
            plan = QueryPlan.from_dict(out["plan"])
            plan_ok, diffs = plan_matches(plan, exp)
            if not c.get("reference_sql"):
                detail = "; ".join(diffs)
                passed = status_ok and plan_ok is not False
                rows.append({"id": c["id"], "question": c["question"], "status_ok": status_ok, "plan_ok": plan_ok,
                             "answer_ok": None, "passed": passed, "ms": round(ms), "detail": detail or "plan-only case"})
                print(f"{'PASS' if passed else 'FAIL'}  {c['id']:<32} {round(ms):>5} ms  {detail or '(plan only)'}")
                continue
            ref = db.run(c["reference_sql"])
            ans_ok, why = results_match(out["result"], ref, list(ref.columns.intersection(plan.metrics)) or plan.metrics,
                                        allow_suppressed=bool(c.get("allow_suppressed")))
            if ans_ok and c.get("allow_suppressed") and not out.get("suppressed_cells"):
                ans_ok, why = False, "expected small-cell suppression but nothing was masked"
            detail = "; ".join(diffs) if diffs else ("" if ans_ok else why)
        elif not status_ok:
            detail = f"status {out.get('status')!r}, expected {want_status!r}: {out.get('message', '')[:80]}"
        passed = status_ok and plan_ok is not False and ans_ok is not False
        rows.append({"id": c["id"], "question": c["question"], "status_ok": status_ok, "plan_ok": plan_ok,
                     "answer_ok": ans_ok, "passed": passed, "ms": round(ms), "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'}  {c['id']:<32} {round(ms):>5} ms  {detail}")

    df = pd.DataFrame(rows)
    answered = df[df["plan_ok"].notna()]
    score = df["passed"].mean()
    summary = {
        "cases": len(df),
        "overall pass rate": f"{score:.0%}",
        "plan accuracy (answerable)": f"{answered['plan_ok'].astype(bool).mean():.0%}",
        "answer accuracy vs reference SQL": f"{answered['answer_ok'].dropna().astype(bool).mean():.0%}",
        "guardrail / clarify cases": f"{df[df['plan_ok'].isna()]['passed'].mean():.0%}",
        "median latency": f"{df['ms'].median():.0f} ms",
        "planner": args.planner,
    }
    lines = ["# Evaluation report", "", f"Generated by `evals/run_evals.py` in {time.perf_counter() - t_all:.1f}s.", "",
             "| Measure | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summary.items()]
    lines += ["", "| Case | Question | Plan | Answer | Result | ms | Notes |", "|---|---|---|---|---|---|---|"]
    mark = lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else ("✅" if v else "❌")  # noqa: E731
    for r in rows:
        lines.append(f"| {r['id']} | {r['question']} | {mark(r['plan_ok'])} | {mark(r['answer_ok'])} | "
                     f"{'PASS' if r['passed'] else 'FAIL'} | {r['ms']} | {r['detail']} |")
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(f"{k:<36} {v}" for k, v in summary.items()))
    print(f"\nReport written to {args.report}")
    return 0 if score >= args.min_score else 1


if __name__ == "__main__":
    raise SystemExit(main())
