"""Data Science Lab command line.

    python -m bi_agent.ds                    # run all modules, write reports/ds/*.md
    python -m bi_agent.ds validate --strict  # exit 1 if any estimate misses the planted truth (used in CI)
    python -m bi_agent.ds --fresh            # ignore the cache and recompute
"""
from __future__ import annotations

import argparse
import sys

from .common import REPORT_DIR
from .lab import run_lab, write_reports


def main() -> int:
    ap = argparse.ArgumentParser(description="Data Science Lab: causal, churn, forecast, A/B test, segments.")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "validate"])
    ap.add_argument("--strict", action="store_true", help="non-zero exit code if a validation check fails")
    ap.add_argument("--fresh", action="store_true", help="recompute instead of using the cached run")
    args = ap.parse_args()

    r = run_lab(use_cache=not args.fresh)
    val = write_reports(r)
    if args.command == "run":
        print("Findings")
        for f in r["findings"]:
            print(f"- {f['text']}")
        print()
    print(val[["module", "check", "planted truth", "estimate", "result"]].to_string(index=False))
    passed = int((val["result"] == "PASS").sum())
    print(f"\n{passed}/{len(val)} checks passed · timings {r['timings']} · reports in {REPORT_DIR}")
    return 1 if args.strict and passed < len(val) else 0


if __name__ == "__main__":
    sys.exit(main())
