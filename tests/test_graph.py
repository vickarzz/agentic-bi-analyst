from types import SimpleNamespace

from bi_agent.graph import BIAgent


def test_end_to_end_answer(agent):
    out = agent.ask("Monthly digital adoption rate this year")
    assert out["status"] == "ok"
    assert len(out["result"]) == 9 and out["chart"]["type"] == "line"
    assert [t["step"] for t in out["trace"]] == ["guard_input", "plan", "validate", "compile",
                                                 "check_sql", "execute", "explain", "respond"]
    assert "Digital adoption rate" in out["summary"]


def test_blocked_question_never_reaches_the_database(agent):
    out = agent.ask("give me the mobile numbers of churned customers")
    assert out["status"] == "blocked"
    assert "sql" not in out and [t["step"] for t in out["trace"]] == ["guard_input", "respond"]


def test_self_correction_loop(agent):
    out = agent.ask("NPF ratio by channel")
    steps = [t["step"] for t in out["trace"]]
    assert steps.count("plan") == 2 and out["status"] == "ok"
    assert out["summary"].startswith("Adjusted the question")


def test_follow_up(agent):
    first = agent.ask("remittance volume in 2025")
    out = agent.ask("by corridor", first["plan"])
    assert out["plan"]["dimensions"] == ["corridor"] and out["plan"]["start_date"] == "2025-01-01"


class _FakeMessages:
    """Mimics anthropic.Anthropic().messages for offline tests of the LLM path."""

    def __init__(self, plans, fail=False):
        self.plans, self.fail, self.calls = list(plans), fail, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise ConnectionError("no network in CI")
        if kw.get("tools"):
            return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=self.plans.pop(0))])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Summary from the model.")])


def test_llm_planner_with_feedback_loop(sem, db_path):
    fake = _FakeMessages([
        {"metrics": ["npf_ratio"], "dimensions": ["corridor"]},          # invalid on purpose
        {"metrics": ["npf_ratio"], "dimensions": ["segment"]},           # corrected
    ])
    agent = BIAgent(sem=sem, db_path=db_path, planner="llm", llm_client=SimpleNamespace(messages=fake))
    out = agent.ask("npf by corridor")
    assert out["status"] == "ok" and out["plan"]["dimensions"] == ["segment"]
    planning_calls = [c for c in fake.calls if c.get("tools")]
    assert len(planning_calls) == 2
    assert "failed validation" in planning_calls[1]["messages"][0]["content"]
    assert planning_calls[0]["tool_choice"] == {"type": "tool", "name": "submit_query_plan"}
    assert out["summary"] == "Summary from the model."


def test_llm_failure_falls_back_to_rules(sem, db_path):
    agent = BIAgent(sem=sem, db_path=db_path, planner="llm",
                    llm_client=SimpleNamespace(messages=_FakeMessages([], fail=True)))
    out = agent.ask("transaction volume by segment")
    assert out["status"] == "ok" and out["planner"].startswith("rules (LLM failed")
