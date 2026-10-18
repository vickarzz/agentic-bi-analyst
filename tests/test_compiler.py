from bi_agent import db
from bi_agent.compiler import compile_plan, render_sql
from bi_agent.plan import Filter, QueryPlan, validate_plan


def test_single_model_sql_is_parameterised(sem):
    plan = QueryPlan(metrics=["transaction_volume"], dimensions=["segment"],
                     filters=[Filter("city", ["Riyadh"])], start_date="2025-01-01", end_date="2025-12-31")
    cq = compile_plan(plan, sem)
    assert cq.params == ["2025-01-01", "2025-12-31", "Riyadh"]
    assert "'Riyadh'" not in cq.sql                       # values are bound, not concatenated
    assert "LEFT JOIN dim_branch b" in cq.sql and "LEFT JOIN dim_customer c" in cq.sql
    assert cq.sql.index("dim_customer") < cq.sql.index("dim_branch")   # dependency order
    assert "'Riyadh'" in render_sql(cq)


def test_only_needed_joins(sem):
    cq = compile_plan(QueryPlan(metrics=["transaction_count"], dimensions=["channel"]), sem)
    assert "JOIN" not in cq.sql


def test_multi_model_uses_ctes_not_fact_to_fact_join(sem, db_path):
    plan = QueryPlan(metrics=["transaction_volume", "remittance_volume"], dimensions=["region"])
    cq = compile_plan(plan, sem)
    assert cq.sql.startswith("WITH m0 AS") and "FULL OUTER JOIN m1 USING (region)" in cq.sql
    df = db.run(cq.sql, cq.params, db_path)
    single = db.run(*_sql(sem, QueryPlan(metrics=["transaction_volume"], dimensions=["region"])), db_path)
    merged = df.merge(single, on="region", suffixes=("", "_single"))
    # no fan-out: totals equal the single-model query
    assert (merged["transaction_volume"].round(2) == merged["transaction_volume_single"].round(2)).all()


def _sql(sem, plan):
    cq = compile_plan(plan, sem)
    return cq.sql, cq.params


def test_limit_is_capped(sem):
    cq = compile_plan(QueryPlan(metrics=["transaction_count"], dimensions=["branch"], limit=10_000), sem)
    assert cq.sql.rstrip().endswith(f"LIMIT {sem.governance['max_rows']}")


def test_time_grain_and_order(sem):
    cq = compile_plan(QueryPlan(metrics=["new_customers"], time_grain="quarter"), sem)
    assert "date_trunc('quarter', c.join_date)" in cq.sql and "ORDER BY period ASC" in cq.sql


def test_validation_catches_incompatible_dimension(sem):
    errs = validate_plan(QueryPlan(metrics=["npf_ratio"], dimensions=["channel"]), sem)
    assert errs and "cannot be used" in errs[0]


def test_validation_catches_unknown_names_and_bad_dates(sem):
    errs = validate_plan(QueryPlan(metrics=["profit"], start_date="2025-13-01"), sem)
    assert any("Unknown metric" in e for e in errs) and any("ISO" in e for e in errs)


def test_every_metric_compiles_and_runs(sem, db_path):
    for name in sem.metrics:
        cq = compile_plan(QueryPlan(metrics=[name], dimensions=["segment"]), sem)
        df = db.run(cq.sql, cq.params, db_path)
        assert len(df) == 3, name
