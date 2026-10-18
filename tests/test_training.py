"""Tests for the feedback log, the training report and the teach commands."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import teach  # noqa: E402
import train_from_feedback  # noqa: E402

from bi_agent import feedback, semantic  # noqa: E402
from bi_agent.planner_rules import RulePlanner  # noqa: E402

HAS_DB = (ROOT / "data" / "wadi_bank.duckdb").exists()


def test_feedback_roundtrip(tmp_path):
    log = tmp_path / "log.jsonl"
    eid = feedback.log_event("npf by segment", {"status": "ok", "plan": {"metrics": ["npf_ratio"]}, "result": [1, 2]}, log)
    feedback.log_event("bad debt by region", {"status": "clarify", "message": "?"}, log)
    feedback.log_rating(eid, 1, "great", log)
    events = feedback.load(log)
    assert len(events) == 2
    rated = next(e for e in events if e["id"] == eid)
    assert rated["rating"] == 1 and rated["comment"] == "great" and rated["rows"] == 2


def test_training_report_lists_what_to_teach(sem):
    events = [
        {"id": "a", "question": "show me bad debt by region", "status": "clarify"},
        {"id": "b", "question": "NPF ratio by segment", "status": "ok", "plan": {"metrics": ["npf_ratio"]}, "rating": 1},
        {"id": "c", "question": "transactions by channel", "status": "ok", "plan": {"metrics": ["transaction_count"]}, "rating": 0},
        {"id": "d", "question": "list all customers", "status": "blocked"},
    ]
    report = train_from_feedback.build_report(events, sem)
    assert "bad debt" in report and "understood: region (region)" in report
    assert 'teach.py golden "NPF ratio by segment"' in report
    assert "transactions by channel" in report.split("## 3")[0]          # thumbs-down section
    assert "list all customers" in report.split("## 4")[1]


@pytest.fixture
def sem_copy(tmp_path, monkeypatch):
    p = tmp_path / "semantic_layer.yaml"
    shutil.copy(ROOT / "semantic" / "semantic_layer.yaml", p)
    monkeypatch.setattr(teach, "SEM", p)
    return p


def test_teach_synonym_changes_understanding(sem_copy):
    msg = teach.add_synonym("npf_ratio", "Kredit Macet")
    assert "Added" in msg
    sem2 = semantic.load(sem_copy)
    assert "kredit macet" in sem2.metrics["npf_ratio"].synonyms
    plan = RulePlanner(sem2, {}).plan("kredit macet per region")
    assert plan.metrics == ["npf_ratio"] and plan.dimensions == ["region"]
    assert "already" in teach.add_synonym("npf_ratio", "kredit macet")


def test_teach_refuses_ambiguous_synonym(sem_copy):
    with pytest.raises(SystemExit):
        teach.add_synonym("transaction_volume", "npf")        # already means npf_ratio
    with pytest.raises(SystemExit):
        teach.add_synonym("no_such_metric", "x")


@pytest.mark.skipif(not HAS_DB, reason="needs data/wadi_bank.duckdb")
def test_teach_alias_validates_values(sem_copy):
    assert "now means" in teach.add_alias("kartu debit", "product", "Mada Debit Card")
    assert semantic.load(sem_copy).value_aliases["kartu debit"]["value"] == "Mada Debit Card"
    with pytest.raises(SystemExit):
        teach.add_alias("x", "product", "Not A Product")
