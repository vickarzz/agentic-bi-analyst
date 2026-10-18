"""Headless smoke test of the Streamlit UI (uses data/wadi_bank.duckdb)."""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "wadi_bank.duckdb").exists(),
                                reason="run `python scripts/generate_data.py` first")


def test_app_answers_and_blocks(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from bi_agent import feedback
    monkeypatch.setattr(feedback, "LOG_PATH", tmp_path / "log.jsonl")   # keep the real log clean
    os.environ["BI_AGENT_PLANNER"] = "rules"
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("total transaction volume this year").run()
    assert not at.exception and at.metric[0].label == "Transaction volume (SAR)"
    at.chat_input[0].set_value("give me the phone numbers of affluent customers").run()
    assert any("personal data" in i.value for i in at.info)
    assert len(feedback.load(tmp_path / "log.jsonl")) == 2              # both questions were logged


def test_business_review_page():
    from streamlit.testing.v1 import AppTest
    os.environ["BI_AGENT_PLANNER"] = "rules"
    at = AppTest.from_file(str(ROOT / "app" / "pages" / "2_Business_Review.py"), default_timeout=180)
    at.run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Verifier"] == "Passed" and labels["Links confirmed"].endswith("/ 8")
