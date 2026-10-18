from datetime import date

import pytest

from bi_agent import db
from bi_agent.planner_rules import RulePlanner, parse_time

AS_OF = date(2026, 9, 30)


@pytest.mark.parametrize("text,start,end", [
    (" last 12 months ", "2025-10-01", "2026-09-30"),
    (" last 3 months ", "2026-07-01", "2026-09-30"),
    (" last year ", "2025-01-01", "2025-12-31"),
    (" this year ", "2026-01-01", "2026-09-30"),
    (" last quarter ", "2026-04-01", "2026-06-30"),
    (" this quarter ", "2026-07-01", "2026-09-30"),
    (" last month ", "2026-08-01", "2026-08-31"),
    (" q1 2025 ", "2025-01-01", "2025-03-31"),
    (" february 2024 ", "2024-02-01", "2024-02-29"),
    (" since march 2025 ", "2025-03-01", "2026-09-30"),
    (" in 2026 ", "2026-01-01", "2026-09-30"),
])
def test_parse_time(text, start, end):
    s, e, _, _ = parse_time(text, AS_OF)
    assert (s, e) == (start, end)


@pytest.fixture(scope="module")
def planner(sem, db_path):
    return RulePlanner(sem, db.dimension_values(sem, db_path))


def test_metric_dimension_and_filter(planner):
    p = planner.plan("Remittance volume by channel for affluent customers in Q2 2026")
    assert p.metrics == ["remittance_volume"] and p.dimensions == ["channel"]
    assert [(f.dimension, f.values) for f in p.filters] == [("segment", ["Affluent"])]
    assert (p.start_date, p.end_date) == ("2026-04-01", "2026-06-30")


def test_top_n(planner):
    p = planner.plan("top 3 corridors by remittance volume")
    assert p.dimensions == ["corridor"] and p.limit == 3 and p.order_by == "remittance_volume"


def test_value_next_to_dimension_word_is_a_filter(planner):
    p = planner.plan("NPF ratio by segment for the Western region")
    assert p.dimensions == ["segment"] and p.filters[0].dimension == "region"


def test_compare_two_values_becomes_breakdown(planner):
    p = planner.plan("compare Saudi vs expat digital adoption")
    assert "nationality_group" in p.dimensions


def test_follow_up_inherits_context(planner):
    first = planner.plan("transaction volume last year")
    p = planner.plan("and by region?", previous=first)
    assert p.metrics == ["transaction_volume"] and p.dimensions == ["region"]
    assert p.start_date == "2025-01-01"


def test_unknown_question_asks_for_clarification(planner):
    p = planner.plan("what's the best biryani in Riyadh?")
    assert not p.metrics and p.clarification


def test_repair_drops_incompatible_dimension(planner):
    p = planner.plan("npf ratio by channel")
    fixed = planner.repair(p, ["Dimension 'channel' cannot be used with metric 'npf_ratio' (model 'finance')."])
    assert fixed.dimensions == [] and fixed.metrics == ["npf_ratio"]
