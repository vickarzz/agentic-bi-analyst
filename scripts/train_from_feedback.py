"""Turn the feedback log into a training report: what to teach the agents next.

    python scripts/train_from_feedback.py                     # reads data/feedback/log.jsonl
    python scripts/train_from_feedback.py --log my_log.jsonl  # e.g. a log downloaded from the cloud app

Writes reports/training_report.md with:
  1. Health: answered / clarified / blocked rates and thumbs-up rate
  2. Questions the agent could not map, with ready-to-run `teach.py` commands
  3. Thumbs-down answers to review (wrong metric, dimension or period?)
  4. Thumbs-up answers that can become golden questions (regression tests)
  5. Blocked questions to double-check for false positives

Nothing is changed automatically: a person reviews the report and runs the suggested commands.
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bi_agent import feedback, semantic  # noqa: E402

STOP = set("""a an the of for by in on at to and or vs versus what which who how many much is are was were
show me give tell list our my we last this next year month quarter week day top bottom per each all total
please can you i want see compare between from since with about trend over time rate ratio""".split())


def phrases(q: str) -> list[str]:
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]+", q.lower())]
    out = []
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            chunk = words[i:i + n]
            if chunk[0] in STOP or chunk[-1] in STOP:
                continue
            out.append(" ".join(chunk))
    return out


def vocab(sem) -> tuple[dict, dict]:
    metric_v, dim_v = {}, {}
    for name, m in sem.metrics.items():
        for x in [name.replace("_", " ")] + m.synonyms:
            metric_v[x.lower()] = name
    for name, d in sem.dimensions.items():
        for x in [name.replace("_", " ")] + d.synonyms:
            dim_v[x.lower()] = name
    for x, v in sem.value_aliases.items():
        dim_v[x] = v["dimension"]
    return metric_v, dim_v


def analyse(q: str, sem) -> dict:
    """Split an unmapped question into known words, unknown words and close metric matches."""
    metric_v, dim_v = vocab(sem)
    text = " " + q.lower() + " "
    known = sorted({f"{w} ({dim_v[w]})" for w in dim_v if re.search(rf"\b{re.escape(w)}\b", text)})
    words = [w for w in re.findall(r"[a-z][a-z\-]+", q.lower()) if w not in STOP]
    known_words = {k.split(" (")[0] for k in known} | {t for k in known for t in k.split(" (")[0].split()}
    unknown = [w for w in words if w not in known_words]
    guesses = []
    for ph in phrases(q):
        if ph in known_words:
            continue
        for match in difflib.get_close_matches(ph, metric_v.keys(), n=1, cutoff=0.72):
            guesses.append((ph, metric_v[match], difflib.SequenceMatcher(None, ph, match).ratio()))
    guesses = sorted(set(guesses), key=lambda g: -g[2])[:2]
    return {"known": known, "unknown": unknown, "guesses": guesses}


def build_report(events: list[dict], sem) -> str:
    n = len(events)
    if not n:
        return "# Training report\n\nNo feedback yet. Ask the app some questions and rate the answers first.\n"
    status = Counter(e.get("status") for e in events)
    rated = [e for e in events if "rating" in e]
    ups = sum(e["rating"] == 1 for e in rated)
    L = ["# Training report", "", f"Questions logged: **{n}** · rated: **{len(rated)}**", "",
         "| Measure | Value |", "|---|---|",
         f"| Answered | {status.get('ok', 0) / n:.0%} |",
         f"| Needed clarification (not understood) | {status.get('clarify', 0) / n:.0%} |",
         f"| Blocked by guardrails | {status.get('blocked', 0) / n:.0%} |",
         f"| Thumbs up (of rated) | {ups / len(rated):.0%} |" if rated else "| Thumbs up (of rated) | n/a |", ""]

    unmapped = [e for e in events if e.get("status") == "clarify"]
    L += ["## 1. Questions the agent could not map", ""]
    if not unmapped:
        L.append("None.")
    words = Counter(w for e in unmapped for w in analyse(e["question"], sem)["unknown"])
    for e in unmapped:
        a = analyse(e["question"], sem)
        L.append(f"- **{e['question']}**")
        if a["known"]:
            L.append(f"  - understood: {', '.join(a['known'])}")
        if a["unknown"]:
            phrase = " ".join(a["unknown"])
            L.append(f"  - not understood: *{phrase}*. Decide which metric it means, then run "
                     f"`python scripts/teach.py synonym <metric> \"{phrase}\" --check`")
        for ph, target, score in a["guesses"]:
            L.append(f"  - spelling-similar to `{target}` ({score:.2f}): "
                     f"`python scripts/teach.py synonym {target} \"{ph}\" --check`")
    if words:
        L += ["", "Most frequent unknown words (teach these first): " + ", ".join(f"*{w}* ({c})" for w, c in words.most_common(8))]
    L.append("")

    downs = [e for e in rated if e["rating"] == 0]
    L += ["## 2. Answers rated 👎 (review what the agent understood)", ""]
    if not downs:
        L.append("None.")
    for e in downs:
        p = e.get("plan") or {}
        L.append(f"- **{e['question']}** → metrics `{p.get('metrics')}`, dimensions `{p.get('dimensions')}`, "
                 f"period `{p.get('start_date')}` to `{p.get('end_date')}`"
                 + (f" · comment: {e['comment']}" if e.get("comment") else ""))
    if downs:
        L += ["", "Fix: teach the missing synonym or alias, then lock the correct behaviour with "
                  "`python scripts/teach.py golden \"<question>\"`."]
    L.append("")

    ups_ok = [e for e in rated if e["rating"] == 1 and e.get("status") == "ok"]
    L += ["## 3. Answers rated 👍 → golden-question candidates", ""]
    if not ups_ok:
        L.append("None yet.")
    for e in ups_ok[:20]:
        L.append(f"- `python scripts/teach.py golden \"{e['question']}\"`")
    L.append("")

    blocked = [e for e in events if e.get("status") == "blocked"]
    L += ["## 4. Blocked questions (check for false positives)", ""]
    if not blocked:
        L.append("None.")
    for e in blocked:
        L.append(f"- {e['question']}")
    if blocked:
        L += ["", "If a legitimate aggregate question was blocked, adjust the patterns in "
                  "`bi_agent/guardrails.py` and add a test in `tests/test_guardrails.py`."]
    L += ["", "---", "After teaching, run `make test` and `make evals`. Commit the YAML changes so the "
          "vocabulary history is reviewable."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "training_report.md"))
    args = ap.parse_args()
    events = feedback.load(Path(args.log) if args.log else None)
    report = build_report(events, semantic.load())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
