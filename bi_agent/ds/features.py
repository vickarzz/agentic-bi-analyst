"""Point-in-time customer features (a tiny feature store).

For a cutoff date T, every feature uses ONLY data dated on or before T, and the label looks
strictly after T. This is the most common source of leakage in churn models, so it is tested
explicitly (tests/test_ds.py::test_features_are_point_in_time).

Note: fact_finance.status is the status at the data's as-of date, so it would leak the future.
It is deliberately NOT used; only contracts disbursed before T are counted.
"""
from __future__ import annotations

import pandas as pd

from .common import template

FEATURE_SQL = """
WITH base AS (
    SELECT c.customer_id, c.segment, c.age_band, c.gender, c.nationality_group, b.region,
           DATE_DIFF('day', CAST(c.join_date AS DATE), CAST($1 AS DATE)) AS tenure_days,
           c.churn_date
    FROM dim_customer c
    LEFT JOIN dim_branch b ON b.branch_id = c.branch_id
    WHERE c.join_date <= CAST($1 AS DATE) - to_days(CAST($2 AS INTEGER))
      AND (c.churn_date IS NULL OR c.churn_date > CAST($1 AS DATE))
),
txn AS (
    SELECT t.customer_id,
           COUNT(*) FILTER (WHERE t.txn_date > CAST($1 AS DATE) - INTERVAL 30 DAY)  AS txn_30d,
           COUNT(*) FILTER (WHERE t.txn_date > CAST($1 AS DATE) - INTERVAL 90 DAY)  AS txn_90d,
           COUNT(*)                                                                  AS txn_180d,
           SUM(t.amount_sar) FILTER (WHERE t.txn_date > CAST($1 AS DATE) - INTERVAL 90 DAY) AS amount_90d,
           SUM(t.amount_sar)                                                         AS amount_180d,
           AVG(CASE WHEN t.is_digital THEN 1.0 ELSE 0.0 END)                         AS digital_share_180d,
           AVG(CASE WHEN t.txn_type IN ('Card Purchase', 'E-commerce') THEN 1.0 ELSE 0.0 END) AS card_share_180d,
           COUNT(DISTINCT t.channel)                                                 AS channels_180d,
           DATE_DIFF('day', MAX(t.txn_date), CAST($1 AS DATE))                       AS days_since_txn
    FROM fact_transaction t
    WHERE t.txn_date <= CAST($1 AS DATE) AND t.txn_date > CAST($1 AS DATE) - INTERVAL 180 DAY
    GROUP BY t.customer_id
),
rem AS (
    SELECT r.customer_id,
           COUNT(*) FILTER (WHERE r.remittance_date > CAST($1 AS DATE) - INTERVAL 90 DAY) AS remit_90d,
           COUNT(*)                                                                       AS remit_180d,
           DATE_DIFF('day', MAX(r.remittance_date), CAST($1 AS DATE))                     AS days_since_remit
    FROM fact_remittance r
    WHERE r.remittance_date <= CAST($1 AS DATE) AND r.remittance_date > CAST($1 AS DATE) - INTERVAL 180 DAY
    GROUP BY r.customer_id
),
fin AS (
    SELECT f.customer_id, COUNT(*) AS finance_contracts, SUM(f.principal_sar) AS finance_principal
    FROM fact_finance f
    WHERE f.disbursement_date <= CAST($1 AS DATE)
    GROUP BY f.customer_id
)
SELECT base.customer_id, base.segment, base.age_band, base.gender, base.nationality_group, base.region,
       base.tenure_days,
       COALESCE(txn.txn_30d, 0)            AS txn_30d,
       COALESCE(txn.txn_90d, 0)            AS txn_90d,
       COALESCE(txn.txn_180d, 0)           AS txn_180d,
       COALESCE(txn.amount_90d, 0)         AS amount_90d,
       COALESCE(txn.amount_180d, 0)        AS amount_180d,
       COALESCE(txn.digital_share_180d, 0) AS digital_share_180d,
       COALESCE(txn.card_share_180d, 0)    AS card_share_180d,
       COALESCE(txn.channels_180d, 0)      AS channels_180d,
       COALESCE(txn.days_since_txn, 181)   AS days_since_txn,
       COALESCE(rem.remit_90d, 0)          AS remit_90d,
       COALESCE(rem.remit_180d, 0)         AS remit_180d,
       COALESCE(rem.days_since_remit, 181) AS days_since_remit,
       COALESCE(fin.finance_contracts, 0)  AS finance_contracts,
       COALESCE(fin.finance_principal, 0)  AS finance_principal,
       CASE WHEN base.churn_date IS NOT NULL
             AND base.churn_date <= CAST($1 AS DATE) + to_days(CAST($3 AS INTEGER)) THEN 1 ELSE 0 END AS label
FROM base
LEFT JOIN txn ON txn.customer_id = base.customer_id
LEFT JOIN rem ON rem.customer_id = base.customer_id
LEFT JOIN fin ON fin.customer_id = base.customer_id
LIMIT 1000000
"""

NUMERIC = ["tenure_days", "txn_30d", "txn_90d", "txn_180d", "amount_90d", "amount_180d", "digital_share_180d",
           "card_share_180d", "channels_180d", "days_since_txn", "remit_90d", "remit_180d", "days_since_remit",
           "finance_principal", "activity_trend"]
CATEGORICAL = ["segment", "age_band", "region"]
PROTECTED = ["gender", "nationality_group"]          # audited, never used as inputs
FEATURES = NUMERIC + CATEGORICAL


def customer_features(cutoff: str, horizon_days: int = 90, min_tenure_days: int = 90,
                      db_path: str | None = None) -> pd.DataFrame:
    df = template(FEATURE_SQL, [cutoff, min_tenure_days, horizon_days], db_path, name="customer_features")
    # recent 30 days vs the monthly average of the previous 150 days: < 1 means slowing down
    df["activity_trend"] = (df["txn_30d"] + 0.5) / ((df["txn_180d"] - df["txn_30d"]) / 5 + 0.5)
    df["cutoff"] = pd.Timestamp(cutoff)
    return df


def build_dataset(cutoffs: list[str], horizon_days: int = 90, min_tenure_days: int = 90,
                  db_path: str | None = None) -> pd.DataFrame:
    return pd.concat([customer_features(c, horizon_days, min_tenure_days, db_path) for c in cutoffs],
                     ignore_index=True)
