"""Tests for the multi-agent business review."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from bi_agent.orchestrator import stats
from bi_agent.orchestrator.graph import BusinessReviewOrchestrator
from bi_agent.orchestrator.verifier import verify


@pytest.fixture(scope="module")
def review(sem, db_path):
    return BusinessReviewOrchestrator(sem=sem, db_path=db_path, writer="deterministic").run()


def _link(review, link_id):
    return next(l for l in review["briefing"]["links"] if l["id"] == link_id)


def test_supervisor_fans_out_to_all_agents_in_parallel(review):
    agents = [t["agent"] for t in review["trace"]]
    assert agents[:2] == ["supervisor", "data_steward"]
    parallel = {t["agent"] for t in review["trace"] if t["group"] == "parallel"}
    assert parallel == {"domain:retail", "domain:digital", "domain:credit", "domain:payments",
                        "domain:customer", "impact_analyst", "discovery_agent"}
    assert agents[-3:] == ["synthesizer", "verifier", "respond"]


def test_every_tool_call_is_governed(review):
    assert review["tool_calls"] and all(c["tool"] in {"governed_query", "approved_analysis"} for c in review["tool_calls"])


def test_planted_relationships_are_found(review):
    assert _link(review, "spend_to_remittance")["verdict"] == "confirmed"
    assert _link(review, "churn_to_credit_risk")["verdict"] == "confirmed"
    assert _link(review, "fee_cut_unit_revenue")["verdict"] == "confirmed"
    assert _link(review, "age_to_digital")["verdict"] == "confirmed"


def test_absent_relationships_are_not_invented(review):
    assert _link(review, "fee_cut_volume_uplift")["verdict"] == "not detected"
    assert _link(review, "placebo_gender_credit")["verdict"] == "control passed"


def test_briefing_is_verified_and_complete(review):
    b = review["briefing"]
    assert b["verification"]["passed"], b["verification"]["issues"]
    assert set(b["domains"]) == {"retail", "digital", "credit", "payments", "customer"}
    assert any("NPF ratio" in f["text"] for f in b["domains"]["credit"]["findings"])
    assert "Cross-business impact map" in review["markdown"]
    json.dumps(b, default=str)                                       # exportable


def test_focused_request_limits_scope(sem, db_path):
    out = BusinessReviewOrchestrator(sem=sem, db_path=db_path, writer="deterministic").run(
        "focus on credit risk and churn")
    assert set(out["scope"]["domains"]) == {"credit", "customer"}
    assert all({"credit", "customer"} & {l["from_domain"], l["to_domain"]} for l in out["links"])


def test_verifier_blocks_invented_numbers_and_causal_claims(review):
    b = review["briefing"]
    bad = "Churn drives NPF up by 47.3% and fee income will rise to SAR 9.99M."
    v = verify(bad, [f for d in b["domains"].values() for f in d["findings"]], b["links"], b["discoveries"],
               list(b["domains"]))
    assert not v["passed"]
    text = " ".join(v["issues"])
    assert "47.3%" in text and "9.99M" in text
    assert "drives" in text


def test_llm_writer_is_corrected_by_verifier(sem, db_path):
    drafts = iter([
        '["Churn causes a 99.9% jump in NPF."]',                       # rejected: invented + causal
        '["Churned customers are associated with a higher NPF ratio than active customers."]',
    ])

    class Msgs:
        calls = 0

        def create(self, **kw):
            Msgs.calls += 1
            return SimpleNamespace(content=[SimpleNamespace(text=next(drafts))])

    orch = BusinessReviewOrchestrator(sem=sem, db_path=db_path, writer="llm",
                                      llm_client=SimpleNamespace(messages=Msgs()))
    out = orch.run()
    assert Msgs.calls == 2
    assert out["verification"]["passed"]
    assert out["briefing"]["executive_summary"] == [
        "Churned customers are associated with a higher NPF ratio than active customers."]
    assert [t["agent"] for t in out["trace"]].count("verifier") == 2


def test_permutation_pvalue_and_event_test():
    x = pd.Series(range(30), dtype=float)
    assert stats.perm_pvalue(x, x * 2 + 1) < 0.01
    idx = pd.date_range("2024-01-01", periods=12, freq="MS")
    s = pd.Series([10.0] * 6 + [7.0] * 6, index=idx)
    res = stats.event_test(s + pd.Series(range(12), index=idx) * 0.01, "2024-07-15", window=5)
    assert res["ok"] and res["delta"] < 0
