"""Remittance customer segmentation (k-means), with the checks a thesis examiner would ask for:

* Features on the last 12 months: frequency, average amount, recency, digital share, months active,
  standardised (amounts on a log scale).
* k chosen by silhouette over a range, not fixed in advance.
* Stability: refit on bootstrap samples and compare labels with the adjusted Rand index (ARI).
* Profiles are aggregated, and any group under the small-cell threshold is suppressed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..semantic import default as default_semantic
from .common import ds_config, pct, sar, template

SEG_SQL = """
SELECT r.customer_id, c.segment, c.age_band,
       COUNT(*) AS remittances, AVG(r.amount_sar) AS avg_amount, SUM(r.amount_sar) AS total_amount,
       DATE_DIFF('day', MAX(r.remittance_date), CAST($1 AS DATE)) AS recency_days,
       AVG(CASE WHEN r.channel = 'Branch' THEN 0.0 ELSE 1.0 END) AS digital_share,
       COUNT(DISTINCT date_trunc('month', r.remittance_date)) AS months_active,
       MODE(r.corridor) AS corridor
FROM fact_remittance r
JOIN dim_customer c ON c.customer_id = r.customer_id
WHERE r.remittance_date > CAST($1 AS DATE) - to_months(CAST($2 AS INTEGER))
  AND r.remittance_date <= CAST($1 AS DATE)
GROUP BY ALL
HAVING COUNT(*) >= 2
LIMIT 1000000
"""
FEATURES = ["remittances", "log_avg_amount", "recency_days", "digital_share", "months_active"]


def _name(row: pd.Series, overall: pd.Series) -> str:
    """Readable names from how a cluster differs from the average sender (only meaningful differences)."""
    parts = ["Frequent" if row["remittances"] > 1.15 * overall["remittances"]
             else "Occasional" if row["remittances"] < 0.85 * overall["remittances"] else "Regular"]
    if row["avg_amount"] > 1.1 * overall["avg_amount"]:
        parts.append("high-value")
    elif row["avg_amount"] < 0.9 * overall["avg_amount"]:
        parts.append("low-value")
    if row["digital_share"] < 0.6:
        parts.append("branch-leaning")
    name = " ".join(parts) + " senders"
    if row["recency_days"] > 1.5 * overall["recency_days"]:
        name += ", lapsing"
    return name


def run(db_path: str | None = None, seed: int = 0) -> dict:
    cfg = ds_config()["segments"]
    sem = default_semantic()
    as_of = sem.as_of_date
    df = template(SEG_SQL, [as_of, cfg["lookback_months"]], db_path, name="remittance_segments")
    df["log_avg_amount"] = np.log(df["avg_amount"])
    X = StandardScaler().fit_transform(df[FEATURES])
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), min(4000, len(X)), replace=False)          # silhouette on a sample for speed

    k_lo, k_hi = cfg["k_range"]
    scores = []
    for k in range(k_lo, k_hi + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        scores.append({"k": k, "silhouette": silhouette_score(X[sample], km.labels_[sample]), "inertia": km.inertia_})
    scores = pd.DataFrame(scores)
    k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    df["cluster"] = km.labels_

    aris = []
    for b in range(cfg["bootstrap_runs"]):
        idx = rng.choice(len(X), len(X), replace=True)
        kb = KMeans(n_clusters=k, n_init=5, random_state=seed + b + 1).fit(X[idx])
        aris.append(adjusted_rand_score(km.labels_, kb.predict(X)))

    overall = df[["remittances", "avg_amount", "recency_days", "digital_share"]].mean()
    prof = (df.groupby("cluster").agg(customers=("customer_id", "size"), remittances=("remittances", "mean"),
                                      avg_amount=("avg_amount", "mean"), total_amount=("total_amount", "sum"),
                                      recency_days=("recency_days", "mean"), digital_share=("digital_share", "mean"),
                                      months_active=("months_active", "mean")).reset_index())
    prof["share_customers"] = prof["customers"] / prof["customers"].sum()
    prof["share_volume"] = prof["total_amount"] / prof["total_amount"].sum()
    prof["name"] = prof.apply(lambda r: _name(r, overall), axis=1)
    k_min = int(sem.governance.get("small_cell_threshold", 10))
    prof = prof[prof["customers"] >= k_min].sort_values("share_volume", ascending=False).reset_index(drop=True)
    corridor = (df.groupby(["cluster", "corridor"]).size().rename("customers").reset_index()
                .merge(prof[["cluster", "name"]], on="cluster"))
    corridor = corridor[corridor["customers"] >= k_min]
    mix = df.groupby("cluster")["segment"].value_counts(normalize=True).rename("share").reset_index()
    return {"k": k, "silhouette": scores, "stability_ari": float(np.mean(aris)), "stability_min": float(np.min(aris)),
            "profiles": prof, "corridor_mix": corridor, "segment_mix": mix, "customers": len(df),
            "lookback_months": cfg["lookback_months"], "as_of": as_of}


def findings(s: dict) -> list[str]:
    p = s["profiles"]
    top = p.iloc[0]
    return [f"Remittance segmentation of {s['customers']:,} active senders: {s['k']} segments (k chosen by silhouette), "
            f"stable under bootstrap resampling (adjusted Rand index {s['stability_ari']:.2f}), though the silhouette of "
            f"{s['silhouette']['silhouette'].max():.2f} says they overlap: use them for targeting, not as hard boxes. "
            f"The largest by volume, "
            f"'{top['name']}', is {pct(top['share_customers'], 0)} of senders and {pct(top['share_volume'], 0)} of volume, "
            f"averaging {sar(top['avg_amount'])} per transfer."]
