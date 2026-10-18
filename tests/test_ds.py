"""Tests for the Data Science Lab.

These assert properties that should hold for any reasonable sample (on an 8,000-customer
database), not exact numbers: no leakage, effects recovered within tolerance, honest
uncertainty, governed data access.
"""
import shutil

import duckdb
import numpy as np
import pandas as pd
import pytest

from bi_agent.ds import causal, churn, experiment, features, forecast, segments
from bi_agent.ds.common import GovernanceError, ground_truth, psi, template
from bi_agent.ds.lab import lab_markdown, validate
from bi_agent.guardrails import check_sql


# ------------------------------------------------------------------------------- governance
@pytest.mark.parametrize("sql", [features.FEATURE_SQL, causal.CARD_SQL, causal.BOOK_SQL, causal.REMIT_SQL,
                                 forecast.DAILY_SQL, experiment.EXPERIMENT_SQL, segments.SEG_SQL])
def test_every_template_passes_the_sql_guard(sem, sql):
    assert check_sql(sql, sem) == []


def test_template_with_pii_is_rejected(ds_db_path):
    with pytest.raises(GovernanceError):
        template("SELECT full_name FROM dim_customer LIMIT 5", db_path=ds_db_path)


def test_protected_attributes_are_not_model_inputs():
    assert not set(features.PROTECTED) & set(features.FEATURES)


# ------------------------------------------------------------------------------- features
def test_features_are_point_in_time(ds_db_path, tmp_path):
    """Deleting everything after the cutoff must not change any feature (only the label may differ)."""
    cutoff = "2025-06-30"
    before = features.customer_features(cutoff, db_path=ds_db_path)
    copy = tmp_path / "truncated.duckdb"
    shutil.copy(ds_db_path, copy)
    con = duckdb.connect(str(copy))
    con.execute(f"DELETE FROM fact_transaction WHERE txn_date > DATE '{cutoff}'")
    con.execute(f"DELETE FROM fact_remittance WHERE remittance_date > DATE '{cutoff}'")
    con.execute(f"DELETE FROM fact_finance WHERE disbursement_date > DATE '{cutoff}'")
    con.close()
    after = features.customer_features(cutoff, db_path=str(copy))
    cols = [c for c in features.NUMERIC + features.CATEGORICAL]
    pd.testing.assert_frame_equal(before.sort_values("customer_id")[cols].reset_index(drop=True),
                                  after.sort_values("customer_id")[cols].reset_index(drop=True))


def test_label_looks_only_forward(ds_db_path):
    df = features.customer_features("2025-06-30", db_path=ds_db_path)
    churn_dates = template("SELECT customer_id, churn_date FROM dim_customer LIMIT 100000", db_path=ds_db_path)
    m = df.merge(churn_dates, on="customer_id")
    pos = m[m["label"] == 1]
    assert (pd.to_datetime(pos["churn_date"]) > pd.Timestamp("2025-06-30")).all()
    assert (pd.to_datetime(pos["churn_date"]) <= pd.Timestamp("2025-09-28")).all()


# ------------------------------------------------------------------------------- churn
def test_churn_model_is_out_of_time_and_useful(ds_lab):
    from bi_agent.ds.common import ds_config
    cfg = ds_config()["churn"]
    assert max(cfg["train_cutoffs"]) < min(cfg["validation_cutoffs"]) < min(cfg["test_cutoffs"])
    m = ds_lab["churn"]["metrics"]
    assert m["roc_auc"] >= 0.75
    assert m["top_decile_lift"] > 2
    assert m["brier_calibrated"] <= m["brier_raw"]          # calibration helps


def test_retention_decision_responds_to_costs(ds_lab):
    scored = ds_lab["churn"]["_scored"]
    cheap, dear = churn.recommend(scored, 10, 0.25), churn.recommend(scored, 400, 0.25)
    assert cheap["contacts"] >= dear["contacts"]


def test_model_card_documents_use_and_fairness(ds_lab):
    card = churn.model_card(ds_lab["churn"], ds_lab["fingerprint"])
    for section in ("Intended use", "Fairness audit", "Limitations", "Monitoring"):
        assert section in card


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(1, 1, 5000)) > 0.2


# ------------------------------------------------------------------------------- causal
def test_campaign_effect_recovers_planted_truth(ds_lab, ds_db_path):
    truth = ground_truth(ds_db_path)["campaign"]["uplift"]
    c = ds_lab["causal"]["campaign"]
    assert abs(c["did"] - truth) < 0.06
    assert c["ri_p"] < 0.05
    es = c["event_study"].set_index("rel_month")["effect"]
    assert es.loc[-12:-2].abs().mean() < 0.06                 # no effect before
    assert es.loc[0:2].mean() > 0.10                         # effect during
    assert es.loc[4:8].abs().mean() < 0.06                   # gone after


def test_fee_cut_naive_is_wrong_and_its_is_not(ds_lab):
    f = ds_lab["causal"]["fee_cut"]
    ch = f["fits"][f["chosen"]]
    assert f["naive"] > 0.03                                 # a dashboard would claim a lift
    assert ch["lo_pts"] <= 0 <= ch["hi_pts"]                 # the trend-aware model does not


# ------------------------------------------------------------------------------- forecast
def test_forecast_beats_baselines_and_learns_ramadan(ds_lab):
    fo = ds_lab["forecast"]
    s = fo["backtest_summary"]
    assert s["Calendar model (Hijri-aware)"] < s["Seasonal naive (same week last year)"]
    assert 0.2 < fo["effects"]["Ramadan week"] < 0.5
    assert (fo["forecast"]["lo"] <= fo["forecast"]["forecast"]).all()
    assert (fo["forecast"]["forecast"] <= fo["forecast"]["hi"]).all()


# ------------------------------------------------------------------------------- experiment
def test_experiment_readout_is_honest(ds_lab):
    e = ds_lab["experiment"]
    assert e["srm_p"] > 0.001
    assert 0.02 <= e["aa"]["false_positive_rate"] <= 0.08
    assert e["variance_reduction"] > 0.15
    assert (e["cuped"]["hi"] - e["cuped"]["lo"]) < (e["raw"]["hi"] - e["raw"]["lo"])


def test_sample_size_formula():
    n = experiment.sample_size(sd=1.0, base=1.0, rel_effect=0.1)
    assert 1500 < n < 1700                                   # 2 * (2.80 / 0.1)^2 ≈ 1,570


# ------------------------------------------------------------------------------- segments
def test_segments_are_stable_and_suppressed(ds_lab):
    s = ds_lab["segments"]
    assert s["stability_ari"] >= 0.7
    assert (s["profiles"]["customers"] >= 10).all()


# ------------------------------------------------------------------------------- lab
def test_lab_runs_without_errors_and_reports(ds_lab, ds_db_path):
    assert ds_lab["errors"] == {}
    val = validate(ds_lab, ds_db_path)
    assert len(val) >= 15
    md = lab_markdown(ds_lab, val)
    assert "Validation against planted truth" in md and "Synthetic control" in md


def test_data_scientist_agent_joins_the_review(sem, ds_db_path, ds_lab):
    from bi_agent.orchestrator.graph import BusinessReviewOrchestrator
    out = BusinessReviewOrchestrator(sem=sem, db_path=ds_db_path, writer="deterministic").run()
    agents = {t["agent"] for t in out["trace"] if t["group"] == "parallel"}
    assert "data_scientist" in agents
    assert out["briefing"]["data_science"]
    assert out["verification"]["passed"], out["verification"]["issues"]
