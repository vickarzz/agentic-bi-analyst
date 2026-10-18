"""Teach the agents new business vocabulary or a new test question, safely.

    python scripts/teach.py synonym npf_ratio "kredit macet"
    python scripts/teach.py synonym region "wilayah"
    python scripts/teach.py alias "murabaha card" product_category "Cards"
    python scripts/teach.py golden "NPF ratio by region last year"        # adds a plan-only golden question
    python scripts/teach.py synonym npf_ratio "kredit bermasalah" --check  # also re-run the evals

Every change is validated (the semantic layer must still load) and rolled back if invalid.
Changes are plain YAML edits, so they show up in git diffs and code review like any other change.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEM = ROOT / "semantic" / "semantic_layer.yaml"
GOLDEN = ROOT / "evals" / "golden_questions.yaml"


def _validate_or_rollback(path: Path, backup: Path) -> None:
    from bi_agent import semantic
    try:
        semantic.load(SEM)
    except Exception as e:  # noqa: BLE001
        shutil.copy(backup, path)
        sys.exit(f"Rejected: the semantic layer would become invalid ({e}). Nothing changed.")
    finally:
        if backup.exists():
            backup.unlink()


def add_synonym(name: str, phrase: str) -> str:
    from bi_agent import semantic
    sem = semantic.load(SEM)
    phrase = phrase.strip().lower()
    if name not in sem.metrics and name not in sem.dimensions:
        sys.exit(f"Unknown metric or dimension '{name}'. Metrics: {sorted(sem.metrics)}")
    for kind, items in (("metric", sem.metrics), ("dimension", sem.dimensions)):
        for other, obj in items.items():
            if other != name and phrase in [x.lower() for x in obj.synonyms]:
                sys.exit(f"'{phrase}' already means {kind} '{other}'. Remove it there first to avoid ambiguity.")
    lines = SEM.read_text().splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if re.fullmatch(rf"  {re.escape(name)}:\s*\n?", l)), None)
    if start is None:
        sys.exit(f"Could not find '{name}' block in {SEM.name}")
    for i in range(start + 1, len(lines)):
        if re.match(r"  \S", lines[i]) or re.match(r"\S", lines[i]):
            break
        m = re.match(r"(\s+synonyms:\s*\[)(.*)(\]\s*)$", lines[i])
        if m:
            existing = [x.strip().lower() for x in m.group(2).split(",") if x.strip()]
            if phrase in existing:
                return f"'{phrase}' is already a synonym of {name}."
            backup = SEM.with_suffix(".bak")
            shutil.copy(SEM, backup)
            lines[i] = f"{m.group(1)}{m.group(2)}, {phrase}{m.group(3)}"
            SEM.write_text("".join(lines))
            _validate_or_rollback(SEM, backup)
            return f"Added '{phrase}' as a synonym of {name}."
    sys.exit(f"'{name}' has no synonyms line to extend.")


def add_alias(phrase: str, dimension: str, value: str) -> str:
    from bi_agent import db, semantic
    sem = semantic.load(SEM)
    if dimension not in sem.dimensions:
        sys.exit(f"Unknown dimension '{dimension}'. Dimensions: {sorted(sem.dimensions)}")
    known = db.dimension_values(sem)
    if value not in known.get(dimension, []):
        sys.exit(f"'{value}' is not a value of {dimension}. Known values: {known.get(dimension)}")
    text = SEM.read_text()
    if "\nvalue_aliases:\n" not in text:
        sys.exit("No value_aliases section in the semantic layer.")
    phrase = phrase.strip().lower()
    if re.search(rf"\n  {re.escape(json.dumps(phrase))}:|\n  {re.escape(phrase)}:", text):
        return f"'{phrase}' is already an alias."
    backup = SEM.with_suffix(".bak")
    shutil.copy(SEM, backup)
    entry = f"  {json.dumps(phrase)}: {{dimension: {dimension}, value: {json.dumps(value)}}}\n"
    SEM.write_text(text.replace("\nvalue_aliases:\n", "\nvalue_aliases:\n" + entry, 1))
    _validate_or_rollback(SEM, backup)
    return f"'{phrase}' now means {dimension} = {value}."


def add_golden(question: str, yes: bool) -> str:
    from bi_agent.graph import BIAgent
    out = BIAgent(planner="rules").ask(question)
    if out.get("status") != "ok":
        sys.exit(f"The agent cannot answer this yet ({out.get('status')}: {out.get('message')}). "
                 "Teach a synonym or alias first.")
    p = out["plan"]
    expect = {"metrics": p["metrics"], "dimensions": p["dimensions"], "time_grain": p["time_grain"]}
    for k in ("start_date", "end_date"):
        if p.get(k):
            expect[k] = p[k]
    if p.get("filters"):
        expect["filters"] = p["filters"]
    print("The agent understood the question as:\n" + json.dumps(expect, indent=2))
    if not yes and input("Save as a golden question? [y/N] ").strip().lower() != "y":
        return "Not saved."
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:40]
    block = (f"\n  - id: taught_{slug}\n    question: {json.dumps(question)}\n"
             f"    expect: {json.dumps(expect)}\n"
             f"    # plan-only case; add reference_sql to also check the numbers\n")
    GOLDEN.write_text(GOLDEN.read_text().rstrip() + "\n" + block)
    return f"Saved golden question taught_{slug}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("synonym"); a.add_argument("name"); a.add_argument("phrase")
    b = sub.add_parser("alias"); b.add_argument("phrase"); b.add_argument("dimension"); b.add_argument("value")
    c = sub.add_parser("golden"); c.add_argument("question"); c.add_argument("--yes", action="store_true")
    for p in (a, b, c):
        p.add_argument("--check", action="store_true", help="re-run the golden-question evals afterwards")
    args = ap.parse_args()
    if args.cmd == "synonym":
        print(add_synonym(args.name, args.phrase))
    elif args.cmd == "alias":
        print(add_alias(args.phrase, args.dimension, args.value))
    else:
        print(add_golden(args.question, args.yes))
    if args.check:
        sys.exit(subprocess.call([sys.executable, str(ROOT / "evals" / "run_evals.py"), "--min-score", "0.9"]))


if __name__ == "__main__":
    main()
