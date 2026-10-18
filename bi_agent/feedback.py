"""Feedback log: every question the agent sees, what it did, and how the user rated it.

This is the raw material for "training" the agents. Nothing changes automatically:
`scripts/train_from_feedback.py` turns the log into suggestions that a human approves
(new synonyms, new golden questions), which keeps the semantic layer governed.

The log is a local JSON Lines file (data/feedback/log.jsonl). It stores the question text,
status and plan only, never query results. On a free cloud host the file resets on restart,
so run the training loop on your own machine or download the log first.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import config

LOG_PATH = Path(os.getenv("BI_AGENT_FEEDBACK", config.ROOT / "data" / "feedback" / "log.jsonl"))


def log_event(question: str, out: dict, path: Path | None = None) -> str:
    """Record one answered (or refused) question. Returns its id for later rating."""
    p = Path(path or LOG_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    event_id = uuid.uuid4().hex[:12]
    rec = {
        "type": "question", "id": event_id, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question, "status": out.get("status"), "planner": out.get("planner"),
        "plan": out.get("plan"), "message": out.get("message"),
        "rows": int(len(out["result"])) if out.get("status") == "ok" and out.get("result") is not None else None,
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return event_id


def log_rating(event_id: str, rating: int, comment: str = "", path: Path | None = None) -> None:
    """rating: 1 = thumbs up, 0 = thumbs down."""
    p = Path(path or LOG_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"type": "rating", "id": event_id, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "rating": int(rating), "comment": comment}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load(path: Path | None = None) -> list[dict]:
    """Questions merged with their latest rating."""
    p = Path(path or LOG_PATH)
    if not p.exists():
        return []
    questions: dict[str, dict] = {}
    ratings: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("type") == "question":
            questions[rec["id"]] = rec
        elif rec.get("type") == "rating":
            ratings[rec["id"]] = rec
    for qid, r in ratings.items():
        if qid in questions:
            questions[qid]["rating"] = r["rating"]
            questions[qid]["comment"] = r.get("comment", "")
    return list(questions.values())
