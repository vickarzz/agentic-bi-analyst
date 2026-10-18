"""Run the multi-agent business review from the command line.

    python -m bi_agent.orchestrator
    python -m bi_agent.orchestrator "focus on credit risk and customer churn" --out reports/review.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph import BusinessReviewOrchestrator


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-agent cross-business review (governed metrics only).")
    ap.add_argument("request", nargs="*", default=["Review the whole business and how each line affects the others"])
    ap.add_argument("--out", default="reports/business_review.md")
    ap.add_argument("--writer", choices=["deterministic", "llm"], default=None)
    args = ap.parse_args()

    out = BusinessReviewOrchestrator(writer=args.writer).run(" ".join(args.request))
    print("Agent trace")
    for t in out["trace"]:
        tag = " ∥" if t["group"] == "parallel" else "  "
        print(f"{tag} {t['agent']:<18} {t['ms']:>8.1f} ms  {t['detail']}")
    print(f"   governed tool calls: {len(out['tool_calls'])}")
    print("\n" + out["markdown"])
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out["markdown"])
    b = dict(out["briefing"])
    path.with_suffix(".json").write_text(json.dumps(b, indent=2, default=str))
    print(f"\nSaved {path} and {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
