"""Churn prediction: who is likely to leave in the next 90 days, and is it worth calling them?

Design (what a model-risk reviewer will ask about):
  * Point-in-time features at quarterly cutoffs (features.py); label = churn in (T, T+90 days].
  * Out-of-time split: train 2024-06..2025-09, validate 2025-12, test 2026-03 and 2026-06.
  * Two candidate models: logistic regression (interpretable baseline) and gradient boosting.
    The winner is chosen on validation PR-AUC (churn is rare, so ROC-AUC alone flatters).
  * Isotonic calibration on the validation set, so a "12% risk" means about 12 in 100.
  * Gender and nationality are NOT inputs; they are used to audit fairness.
  * Reason codes: how much each feature moves a customer's risk vs. a typical customer.
  * Decision layer: contact a customer when expected saved value exceeds the contact cost.
  * Output is aggregated for display; scored lists would go to the CRM with access controls.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .common import ds_config, pct, psi, sar
from .features import CATEGORICAL, FEATURES, NUMERIC, PROTECTED, build_dataset, customer_features

FEATURE_LABELS = {
    "tenure_days": "Tenure", "txn_30d": "Transactions, last 30 days", "txn_90d": "Transactions, last 90 days",
    "txn_180d": "Transactions, last 180 days", "amount_90d": "Spend, last 90 days", "amount_180d": "Spend, last 180 days",
    "digital_share_180d": "Digital share", "card_share_180d": "Card share", "channels_180d": "Channels used",
    "days_since_txn": "Days since last transaction", "remit_90d": "Remittances, last 90 days",
    "remit_180d": "Remittances, last 180 days", "days_since_remit": "Days since last remittance",
    "finance_contracts": "Finance contracts", "finance_principal": "Finance principal",
    "activity_trend": "Activity trend (30d vs prior)", "segment": "Segment", "age_band": "Age band", "region": "Region",
}


def _models(seed: int = 0) -> dict[str, Pipeline]:
    log_num = Pipeline([("scale", StandardScaler())])
    lr = Pipeline([
        ("prep", ColumnTransformer([("num", log_num, NUMERIC),
                                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])),
        ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
    ])
    gb = Pipeline([
        ("prep", ColumnTransformer([("num", "passthrough", NUMERIC),
                                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)])),
        ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80,
                                               l2_regularization=1.0, categorical_features=list(range(len(NUMERIC), len(FEATURES))),
                                               random_state=seed)),
    ])
    return {"Logistic regression": lr, "Gradient boosting": gb}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for c in ("amount_90d", "amount_180d", "finance_principal"):
        x[c] = np.log1p(x[c])
    return x


def _lift_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    order = np.argsort(-p)
    y, p = y[order], p[order]
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(len(y)), bins)):
        rows.append({"decile": i + 1, "customers": len(idx), "predicted": p[idx].mean(), "observed": y[idx].mean(),
                     "captured": y[: idx[-1] + 1].sum() / max(y.sum(), 1)})
    t = pd.DataFrame(rows)
    t["lift"] = t["observed"] / max(y.mean(), 1e-9)
    return t


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return (pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin")
            .agg(predicted=("p", "mean"), observed=("y", "mean"), customers=("y", "size")).reset_index(drop=True))


def _reason_codes(model, x: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Model-agnostic local attribution: risk change when one feature is set to a typical value."""
    base = model.predict_proba(x)[:, 1]
    typical = {c: (x[c].median() if c in NUMERIC else x[c].mode().iloc[0]) for c in FEATURES}
    contrib = {}
    for c in FEATURES:
        xc = x.copy()
        xc[c] = typical[c]
        contrib[c] = base - model.predict_proba(xc)[:, 1]
    cm = pd.DataFrame(contrib, index=x.index)
    out = []
    for i in range(len(cm)):
        row = cm.iloc[i].sort_values(ascending=False)
        out.append([FEATURE_LABELS[k] for k, v in row.head(top).items() if v > 0])
    return pd.Series(out, index=x.index)


def train_and_evaluate(db_path: str | None = None, seed: int = 0) -> dict:
    cfg = ds_config()["churn"]
    t0 = time.perf_counter()
    h, mt = cfg["horizon_days"], cfg["min_tenure_days"]
    train = build_dataset(cfg["train_cutoffs"], h, mt, db_path)
    valid = build_dataset(cfg["validation_cutoffs"], h, mt, db_path)
    test = build_dataset(cfg["test_cutoffs"], h, mt, db_path)
    now = customer_features(cfg["score_cutoff"], h, mt, db_path)

    xtr, ytr = _prep(train), train["label"].to_numpy()
    xva, yva = _prep(valid), valid["label"].to_numpy()
    xte, yte = _prep(test), test["label"].to_numpy()

    candidates = {}
    for name, m in _models(seed).items():
        m.fit(xtr, ytr)
        pv = m.predict_proba(xva)[:, 1]
        candidates[name] = {"model": m, "val_pr_auc": average_precision_score(yva, pv), "val_roc_auc": roc_auc_score(yva, pv)}
    best_name = max(candidates, key=lambda k: candidates[k]["val_pr_auc"])
    model = candidates[best_name]["model"]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(model.predict_proba(xva)[:, 1], yva)
    calibrate = lambda raw: iso.predict(raw)  # noqa: E731

    raw_te = model.predict_proba(xte)[:, 1]
    p_te = calibrate(raw_te)
    metrics = {
        "roc_auc": roc_auc_score(yte, p_te), "pr_auc": average_precision_score(yte, p_te),
        "base_rate": yte.mean(), "brier_raw": brier_score_loss(yte, raw_te), "brier_calibrated": brier_score_loss(yte, p_te),
        "brier_base_rate": brier_score_loss(yte, np.full_like(p_te, ytr.mean())),
        "test_rows": len(yte), "test_positives": int(yte.sum()), "train_rows": len(ytr), "train_positives": int(ytr.sum()),
    }
    for name, c in candidates.items():
        pt = c["model"].predict_proba(xte)[:, 1]
        c["test_roc_auc"], c["test_pr_auc"] = roc_auc_score(yte, pt), average_precision_score(yte, pt)
    lift = _lift_table(yte, p_te)
    metrics["top_decile_lift"] = float(lift["lift"].iloc[0])
    metrics["top_decile_capture"] = float(lift["captured"].iloc[0])

    imp = permutation_importance(model, xte, yte, scoring="roc_auc", n_repeats=5, random_state=seed)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .assign(label=lambda d: d["feature"].map(FEATURE_LABELS))
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    # ---- fairness audit on the out-of-time test set
    econ = cfg["retention_economics"]
    thr = np.quantile(p_te, 0.9)                           # contact the top 10% (audit threshold)
    fair_rows = []
    for attr in PROTECTED:
        for g, idx in test.groupby(attr).groups.items():
            yy, pp = yte[test.index.get_indexer(idx)], p_te[test.index.get_indexer(idx)]
            pos = yy == 1
            fair_rows.append({"attribute": attr, "group": g, "customers": len(yy), "observed": yy.mean(),
                              "predicted": pp.mean(), "calibration_gap_pts": (pp.mean() - yy.mean()) * 100,
                              "roc_auc": roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan,
                              "contact_rate": (pp >= thr).mean(), "recall": (pp[pos] >= thr).mean() if pos.any() else np.nan})
    fairness = pd.DataFrame(fair_rows)
    tol = cfg["fairness_tolerance"]
    flags = []
    for attr, g in fairness.groupby("attribute"):
        cg = g["calibration_gap_pts"].abs().max()
        rg = (g["recall"].max() - g["recall"].min()) * 100
        if cg > tol["calibration_gap_pts"]:
            flags.append(f"{attr}: calibration gap {cg:.1f} pts exceeds {tol['calibration_gap_pts']} pts")
        if rg > tol["recall_gap_pts"]:
            flags.append(f"{attr}: recall gap {rg:.1f} pts exceeds {tol['recall_gap_pts']} pts")

    # ---- score today's customers and decide who to contact
    xnow = _prep(now)
    p_now = calibrate(model.predict_proba(xnow)[:, 1])
    value = now["segment"].map(econ["annual_value_sar"]).to_numpy(float)
    ev = p_now * econ["save_rate"] * value - econ["contact_cost_sar"]
    contact = ev > 0
    now_scored = now[["segment", "region", "age_band", "gender", "nationality_group"]].copy()
    now_scored["p"], now_scored["value"], now_scored["ev"], now_scored["contact"] = p_now, value, ev, contact
    by_segment = (now_scored.groupby("segment")
                  .agg(customers=("p", "size"), expected_churners=("p", "sum"), contact=("contact", "sum"),
                       expected_net_value=("ev", lambda s: s[s > 0].sum())).reset_index())
    by_region = (now_scored.groupby("region").agg(customers=("p", "size"), expected_churners=("p", "sum"),
                                                  avg_risk=("p", "mean")).reset_index())
    top = now_scored[contact]
    reasons = _reason_codes(model, xnow[contact].head(400)) if contact.any() else pd.Series(dtype=object)
    reason_freq = (pd.Series([r for rs in reasons for r in rs]).value_counts().head(6)
                   .rename_axis("reason").reset_index(name="customers")) if len(reasons) else pd.DataFrame()

    # ---- backtest the decision rule on the test set (would the contacts have paid off?)
    v_te = test["segment"].map(econ["annual_value_sar"]).to_numpy(float)
    c_te = (p_te * econ["save_rate"] * v_te - econ["contact_cost_sar"]) > 0
    realised = (yte[c_te] * econ["save_rate"] * v_te[c_te]).sum() - c_te.sum() * econ["contact_cost_sar"]

    drift = {"score_psi": psi(calibrate(model.predict_proba(xtr)[:, 1]), p_now),
             "feature_psi": {f: psi(xtr[f].to_numpy(float), xnow[f].to_numpy(float)) for f in NUMERIC}}

    return {
        "model_name": best_name, "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in candidates.items()},
        "metrics": metrics, "lift": lift, "calibration": _calibration(yte, p_te), "importance": importance,
        "fairness": fairness, "fairness_flags": flags,
        "scoring": {"cutoff": cfg["score_cutoff"], "customers": len(now), "expected_churners": float(p_now.sum()),
                    "contacts": int(contact.sum()), "expected_saves": float((p_now[contact] * econ["save_rate"]).sum()),
                    "expected_net_value": float(ev[contact].sum()), "by_segment": by_segment, "by_region": by_region,
                    "reason_codes": reason_freq, "contact_threshold_risk": float(top["p"].min()) if len(top) else None},
        "decision_backtest": {"contacts": int(c_te.sum()), "churners_reached": int(yte[c_te].sum()),
                              "realised_net_value": float(realised), "cutoffs": cfg["test_cutoffs"]},
        "drift": drift, "economics": econ,
        "_scored": now_scored,       # kept in memory for the interactive economics; never written to reports
        "seconds": time.perf_counter() - t0,
    }


def recommend(scored: pd.DataFrame, cost: float, save_rate: float) -> dict:
    """Re-run the contact decision with different business assumptions (used by the UI slider)."""
    ev = scored["p"] * save_rate * scored["value"] - cost
    c = ev > 0
    return {"contacts": int(c.sum()), "expected_saves": float((scored.loc[c, "p"] * save_rate).sum()),
            "expected_net_value": float(ev[c].sum())}


def findings(r: dict) -> list[str]:
    m, s = r["metrics"], r["scoring"]
    out = [
        f"Churn model ({r['model_name'].lower()}) on an out-of-time test set: ROC-AUC {m['roc_auc']:.2f}, "
        f"PR-AUC {m['pr_auc']:.2f} vs a {pct(m['base_rate'])} base rate; the riskiest 10% of customers hold "
        f"{pct(m['top_decile_capture'], 0)} of the churners ({m['top_decile_lift']:.1f}x lift).",
        f"Next 90 days: about {s['expected_churners']:.0f} expected churners among {s['customers']:,} customers. "
        f"Contacting {s['contacts']:,} customers has an expected net value of {sar(s['expected_net_value'])} "
        f"under the agreed retention assumptions.",
    ]
    if r["fairness_flags"]:
        out.append("Fairness audit flags: " + "; ".join(r["fairness_flags"]) + ".")
    else:
        out.append("Fairness audit passed: calibration and recall are within tolerance across gender and nationality, "
                   "which are not model inputs.")
    if r["drift"]["score_psi"] > ds_config()["churn"]["psi_alert"]:
        out.append(f"Score drift alert: PSI {r['drift']['score_psi']:.2f} between training and today.")
    return out


def model_card(r: dict, data_fingerprint: str) -> str:
    m, cfg = r["metrics"], ds_config()["churn"]
    fair = r["fairness"].copy()
    lines = [
        "# Model card: 90-day churn model", "",
        f"*Data snapshot `{data_fingerprint}` · synthetic data (Wadi Bank is fictional) · owner: Customer Analytics*", "",
        "## Intended use",
        "Rank active customers by risk of closing their relationship in the next 90 days, so the retention team "
        "can prioritise calls. Not for credit decisions, pricing or any decision about an individual's eligibility.", "",
        "## Data and features",
        f"- Training cutoffs: {', '.join(cfg['train_cutoffs'])}; validation: {', '.join(cfg['validation_cutoffs'])}; "
        f"out-of-time test: {', '.join(cfg['test_cutoffs'])}.",
        f"- {m['train_rows']:,} training rows ({m['train_positives']:,} churners); {m['test_rows']:,} test rows "
        f"({m['test_positives']:,} churners).",
        "- Inputs: " + ", ".join(FEATURE_LABELS[f] for f in FEATURES) + ".",
        "- Excluded on purpose: gender and nationality (used only to audit fairness), and finance status "
        "(it reflects the as-of date and would leak the future).", "",
        "## Performance (out-of-time test)",
        "| Model | Validation PR-AUC | Test ROC-AUC | Test PR-AUC |", "|---|---|---|---|",
    ]
    for name, c in r["candidates"].items():
        lines.append(f"| {name}{' (selected)' if name == r['model_name'] else ''} | {c['val_pr_auc']:.3f} | "
                     f"{c['test_roc_auc']:.3f} | {c['test_pr_auc']:.3f} |")
    lines += ["", f"- Base rate {pct(m['base_rate'], 2)}; top-decile lift {m['top_decile_lift']:.1f}x, "
                  f"capturing {pct(m['top_decile_capture'], 0)} of churners.",
              f"- Brier score: base rate {m['brier_base_rate']:.4f} · raw model {m['brier_raw']:.4f} · "
              f"after isotonic calibration {m['brier_calibrated']:.4f}.", "",
              "## Fairness audit", "| Attribute | Group | Customers | Observed | Predicted | Recall at top 10% |",
              "|---|---|---|---|---|---|"]
    for _, f in fair.iterrows():
        lines.append(f"| {f['attribute']} | {f['group']} | {f['customers']:,} | {pct(f['observed'], 2)} | "
                     f"{pct(f['predicted'], 2)} | {pct(f['recall'], 0)} |")
    lines += ["", ("Flags: " + "; ".join(r["fairness_flags"])) if r["fairness_flags"] else "No fairness flags.", "",
              "## Decision rule",
              f"Contact a customer when risk × save rate ({cfg['retention_economics']['save_rate']}) × annual value "
              f"exceeds the contact cost (SAR {cfg['retention_economics']['contact_cost_sar']}). On the test period this "
              f"rule would have contacted {r['decision_backtest']['contacts']:,} customers, reached "
              f"{r['decision_backtest']['churners_reached']:,} churners and returned {sar(r['decision_backtest']['realised_net_value'])} "
              "net, under the same assumptions.", "",
              "## Monitoring",
              f"- Score PSI training vs today: {r['drift']['score_psi']:.3f} (alert above {cfg['psi_alert']}).",
              "- Retrain quarterly; re-run the fairness audit at every retrain.", "",
              "## Limitations",
              "- Synthetic data with a planted disengagement pattern; real churn has more causes (pricing, service, life events).",
              "- Churn is rare (about 1% per quarter), so precision at any threshold is modest; use for prioritisation, "
              "not automation.",
              "- The save rate and customer values are business assumptions; the decision changes when they change."]
    return "\n".join(lines) + "\n"
