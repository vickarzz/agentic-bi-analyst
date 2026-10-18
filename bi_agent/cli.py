"""Command-line interface.

    python -m bi_agent.cli "NPF ratio by segment"
    python -m bi_agent.cli --show-sql "top 3 corridors by remittance volume in 2026"
    python -m bi_agent.cli            # interactive mode with follow-up questions
"""
from __future__ import annotations

import argparse

from .graph import BIAgent


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask Wadi Bank's governed data a question.")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--show-sql", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--planner", choices=["rules", "llm"], default=None)
    args = ap.parse_args()

    agent = BIAgent(planner=args.planner)
    prev = None

    def answer(q: str) -> None:
        nonlocal prev
        out = agent.ask(q, prev)
        print("\n" + agent.format_answer(out))
        if args.show_sql and out.get("display_sql"):
            print("\n-- SQL --\n" + out["display_sql"])
        if args.trace:
            print("\n-- trace --")
            for t in out["trace"]:
                print(f"[{t['step']:<11}] {t['ms']:>7} ms  {t['detail']}")
        if out.get("status") == "ok":
            prev = out["plan"]

    if args.question:
        answer(" ".join(args.question))
        return
    print("Agentic BI Analyst — Wadi Bank (synthetic data). Type a question, or 'exit'.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            answer(q)


if __name__ == "__main__":
    main()
