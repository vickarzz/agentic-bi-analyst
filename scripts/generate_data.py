"""
Synthetic data generator for *Wadi Bank* — a FICTIONAL Saudi retail bank.

Every row is randomly generated. No real customer, bank or employer data is used.

Creates a small star schema in DuckDB:

    dim_date, dim_branch, dim_product, dim_customer           (dimensions)
    fact_transaction, fact_finance, fact_remittance           (facts)
    fact_experiment                                           (A/B test readout)

Built-in business patterns (so the agent has real "stories" to find):
  * Digital adoption (Mobile App + Internet Banking share) rises from ~50% to ~75%.
  * Spending spikes during Ramadan and before Eid al-Adha.
  * Affluent / Private customers have larger tickets.
  * Expat customers send remittances to six corridors (Indonesia, India, ...).
  * Non-performing finance (90+ DPD) is higher in the Mass segment.
  * ~9% of customers churn, more often low-activity Mass customers.

Usage:
    python scripts/generate_data.py                 # default size (~20k customers)
    python scripts/generate_data.py --customers 5000 --out data/small.duckdb
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

import duckdb
import numpy as np
import pandas as pd

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-09-30")  # the dataset's "as-of" date

REGIONS = {
    "Central": ["Riyadh", "Riyadh", "Riyadh", "Riyadh", "Al Kharj", "Buraydah"],
    "Western": ["Jeddah", "Jeddah", "Jeddah", "Makkah", "Madinah", "Taif"],
    "Eastern": ["Dammam", "Al Khobar", "Al Khobar", "Jubail", "Al Ahsa"],
    "Southern": ["Abha", "Jazan", "Najran"],
    "Northern": ["Tabuk", "Hail", "Al Jouf"],
}
REGION_WEIGHT = {"Central": 0.38, "Western": 0.28, "Eastern": 0.20, "Southern": 0.08, "Northern": 0.06}

PRODUCTS = [
    # id, name, category, is_islamic
    ("P01", "Mada Debit Card", "Cards", True),
    ("P02", "Classic Credit Card", "Cards", True),
    ("P03", "Signature Credit Card", "Cards", True),
    ("P04", "Current Account", "Accounts", True),
    ("P05", "Mudaraba Savings Account", "Accounts", True),
    ("P06", "Murabaha Personal Finance", "Personal Finance", True),
    ("P07", "Auto Ijarah Finance", "Auto Finance", True),
    ("P08", "Instant Remittance", "Remittance", True),
]

# Approximate Gregorian windows for Ramadan and the run-up to Eid al-Adha.
SEASONAL_WINDOWS = [
    ("2024-03-11", "2024-04-10", 1.35), ("2024-06-06", "2024-06-16", 1.25),
    ("2025-03-01", "2025-03-31", 1.35), ("2025-05-27", "2025-06-06", 1.25),
    ("2026-02-18", "2026-03-20", 1.35), ("2026-05-17", "2026-05-27", 1.25),
]

FIRST = ["Abdullah", "Mohammed", "Fahad", "Khalid", "Saad", "Nora", "Sara", "Reem", "Lama",
         "Faisal", "Omar", "Ahmed", "Huda", "Maha", "Budi", "Siti", "Ravi", "Priya", "Jose",
         "Maria", "Ali", "Fatima", "Yusuf", "Aisha", "Hassan", "Rahul", "Dewi", "Imran"]
LAST = ["Al-Harbi", "Al-Qahtani", "Al-Otaibi", "Al-Shehri", "Al-Ghamdi", "Al-Zahrani",
        "Al-Dosari", "Santoso", "Sharma", "Khan", "Reyes", "Hossain", "Mahmoud", "Rahman"]

CORRIDORS = {  # destination country: (weight, mean amount SAR)
    "Indonesia": (0.18, 1400), "India": (0.27, 1800), "Pakistan": (0.20, 1500),
    "Philippines": (0.12, 1300), "Egypt": (0.15, 2200), "Bangladesh": (0.08, 1100),
}


def _date_weights() -> pd.Series:
    days = pd.date_range(START, END, freq="D")
    w = pd.Series(1.0, index=days)
    w[days.dayofweek.isin([3, 4])] *= 1.12          # Thu/Fri: weekend shopping
    w[days.day >= 25] *= 1.10                        # salary week (25th onwards)
    for s, e, f in SEASONAL_WINDOWS:
        w[(days >= s) & (days <= e)] *= f
    return w / w.max()


def _sample_dates(rng, lo: np.ndarray, hi: np.ndarray, weights: pd.Series) -> np.ndarray:
    """Sample one date per row uniformly in [lo, hi] (day ints) with seasonal rejection."""
    base = weights.index[0]
    w = weights.to_numpy()
    out = np.empty(len(lo), dtype=np.int64)
    todo = np.arange(len(lo))
    for _ in range(30):
        if len(todo) == 0:
            break
        cand = rng.integers(lo[todo], hi[todo] + 1)
        keep = rng.random(len(todo)) < w[cand]
        out[todo[keep]] = cand[keep]
        todo = todo[~keep]
    out[todo] = rng.integers(lo[todo], hi[todo] + 1)
    return (base + pd.to_timedelta(out, unit="D")).to_numpy()


def build(n_customers: int = 20_000, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    total_days = (END - START).days
    weights = _date_weights()

    # ---------------- dim_branch ----------------
    rows, bid = [], 1
    for region, cities in REGIONS.items():
        for city in cities:
            n_in_city = sum(1 for r in rows if r["city"] == city) + 1
            rows.append({"branch_id": f"B{bid:03d}", "branch_name": f"{city} Branch {n_in_city}",
                         "city": city, "region": region})
            bid += 1
    dim_branch = pd.DataFrame(rows)
    b_weight = dim_branch["region"].map(REGION_WEIGHT) / dim_branch.groupby("region")["region"].transform("count")
    b_weight = (b_weight / b_weight.sum()).to_numpy()

    # ---------------- dim_product ----------------
    dim_product = pd.DataFrame(PRODUCTS, columns=["product_id", "product_name", "product_category", "is_islamic"])

    # ---------------- dim_customer ----------------
    n = n_customers
    segment = rng.choice(["Mass", "Affluent", "Private"], n, p=[0.72, 0.24, 0.04])
    nationality = rng.choice(["Saudi", "Expat"], n, p=[0.64, 0.36])
    age_band = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n, p=[0.13, 0.31, 0.27, 0.18, 0.11])
    gender = rng.choice(["Male", "Female"], n, p=[0.61, 0.39])
    branch_idx = rng.choice(len(dim_branch), n, p=b_weight)
    join_off = rng.integers(-6 * 365, total_days - 30, n)  # many customers joined before 2024
    join_date = START + pd.to_timedelta(join_off, unit="D")

    # Churn: Mass and older-tenure-free customers churn more.
    churn_p = np.where(segment == "Mass", 0.11, np.where(segment == "Affluent", 0.05, 0.02))
    churned = rng.random(n) < churn_p
    churn_lo = np.maximum((join_date - START).days + 60, 120)
    churn_off = np.where(churned & (churn_lo < total_days),
                         rng.integers(np.minimum(churn_lo, total_days - 1), total_days, n), -1)
    churn_date = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    has_churn = churn_off >= 0
    churn_date[has_churn] = START + pd.to_timedelta(churn_off[has_churn], unit="D")

    first = rng.choice(FIRST, n)
    last = rng.choice(LAST, n)
    dim_customer = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(1, n + 1)],
        "full_name": [f"{f} {l}" for f, l in zip(first, last)],                 # PII (synthetic)
        "mobile_number": [f"+9665{x:08d}" for x in rng.integers(0, 10**8, n)],  # PII (synthetic)
        "email": [f"{f.lower()}.{i}@example.com" for i, f in enumerate(first)],  # PII (synthetic)
        "segment": segment,
        "nationality_group": nationality,
        "age_band": age_band,
        "gender": gender,
        "branch_id": dim_branch["branch_id"].to_numpy()[branch_idx],
        "join_date": join_date.normalize(),
        "churn_date": churn_date,
    })
    dim_customer["is_active"] = dim_customer["churn_date"].isna()

    # Active window for activity, as day offsets from START
    win_lo = np.maximum((join_date - START).days.to_numpy(), 0)
    win_hi = np.where(has_churn, churn_off, total_days)
    win_hi = np.maximum(win_hi, win_lo)
    win_len = win_hi - win_lo + 1

    # ---------------- fact_transaction ----------------
    seg_rate = {"Mass": 0.55, "Affluent": 0.9, "Private": 1.2}  # txns per day-ish /  scaled
    rate = np.vectorize(seg_rate.get)(segment) * np.where(churned, 0.45, 1.0)
    n_txn = rng.poisson(rate * win_len / 30.0 * 1.6)
    cust_rep = np.repeat(np.arange(n), n_txn)
    tdates = _sample_dates(rng, win_lo[cust_rep], win_hi[cust_rep], weights)
    t_frac = (pd.to_datetime(tdates) - START).days.to_numpy() / total_days

    # Digital share rises over time and is higher for younger customers.
    age_boost = pd.Series(age_band).map({"18-24": 0.12, "25-34": 0.08, "35-44": 0.0,
                                         "45-54": -0.08, "55+": -0.18}).to_numpy()[cust_rep]
    p_digital = np.clip(0.48 + 0.27 * t_frac + age_boost, 0.1, 0.95)
    is_digital = rng.random(len(cust_rep)) < p_digital
    digital_ch = rng.choice(["Mobile App", "Internet Banking"], len(cust_rep), p=[0.82, 0.18])
    phys_ch = rng.choice(["POS", "ATM", "Branch"], len(cust_rep), p=[0.66, 0.26, 0.08])
    channel = np.where(is_digital, digital_ch, phys_ch)

    type_map = {
        "Mobile App": (["Transfer", "Bill Payment", "E-commerce"], [0.45, 0.25, 0.30]),
        "Internet Banking": (["Transfer", "Bill Payment"], [0.6, 0.4]),
        "POS": (["Card Purchase"], [1.0]),
        "ATM": (["Cash Withdrawal", "Cash Deposit"], [0.85, 0.15]),
        "Branch": (["Transfer", "Cash Deposit"], [0.5, 0.5]),
    }
    txn_type = np.empty(len(channel), dtype=object)
    for ch, (types, p) in type_map.items():
        m = channel == ch
        txn_type[m] = rng.choice(types, m.sum(), p=p)

    seg_mult = pd.Series(segment).map({"Mass": 1.0, "Affluent": 2.4, "Private": 6.0}).to_numpy()[cust_rep]
    type_mean = pd.Series(txn_type).map({"Transfer": 1500, "Bill Payment": 420, "E-commerce": 260,
                                         "Card Purchase": 190, "Cash Withdrawal": 600,
                                         "Cash Deposit": 1800}).to_numpy()
    amount = np.round(rng.lognormal(np.log(type_mean * seg_mult), 0.65), 2)

    card_prod = np.where(pd.Series(segment).to_numpy()[cust_rep] == "Mass",
                         rng.choice(["P01", "P02"], len(cust_rep), p=[0.75, 0.25]),
                         rng.choice(["P01", "P02", "P03"], len(cust_rep), p=[0.45, 0.2, 0.35]))
    product = np.where(np.isin(txn_type, ["Card Purchase", "E-commerce"]), card_prod,
                       np.where(txn_type == "Cash Withdrawal", "P01", "P04"))

    fact_transaction = pd.DataFrame({
        "txn_id": np.arange(1, len(cust_rep) + 1),
        "customer_id": dim_customer["customer_id"].to_numpy()[cust_rep],
        "product_id": product,
        "txn_date": pd.to_datetime(tdates).normalize(),
        "channel": channel,
        "txn_type": txn_type,
        "amount_sar": amount,
        "is_digital": is_digital,
    }).sort_values(["txn_date", "txn_id"]).reset_index(drop=True)
    fact_transaction["txn_id"] = np.arange(1, len(fact_transaction) + 1)

    # ---------------- fact_finance (Murabaha / Ijarah) ----------------
    fin_p = np.where(nationality == "Saudi", 0.34, 0.14)
    has_fin = rng.random(n) < fin_p
    fc = np.where(has_fin)[0]
    m = len(fc)
    disb_lo = np.maximum((join_date[fc] - START).days.to_numpy(), -365)
    disb_off = rng.integers(np.minimum(disb_lo, total_days - 31), total_days - 30)
    disb_date = START + pd.to_timedelta(disb_off, unit="D")
    prod = rng.choice(["P06", "P07"], m, p=[0.72, 0.28])
    seg_f = segment[fc]
    principal = np.round(rng.lognormal(np.log(np.where(prod == "P06", 85_000, 120_000)
                                              * pd.Series(seg_f).map({"Mass": 1, "Affluent": 1.9, "Private": 3.5}).to_numpy()), 0.35), -2)
    tenor = rng.choice([12, 24, 36, 48, 60], m, p=[0.08, 0.17, 0.3, 0.2, 0.25])
    profit_rate = np.round(rng.uniform(0.035, 0.08, m), 4)
    months_elapsed = np.clip(((END - disb_date).days.to_numpy() / 30.44).astype(int), 0, None)
    paid_frac = np.clip(months_elapsed / tenor, 0, 1)
    outstanding = np.round(principal * (1 + profit_rate * tenor / 12) * (1 - paid_frac), 2)

    npf_p = pd.Series(seg_f).map({"Mass": 0.055, "Affluent": 0.022, "Private": 0.01}).to_numpy()
    npf_p = npf_p * np.where(churned[fc], 2.5, 1.0)
    r = rng.random(m)
    status = np.select([paid_frac >= 1, r < npf_p, r < npf_p * 1.8, r < npf_p * 3.0],
                       ["Closed", "90+ DPD", "60 DPD", "30 DPD"], default="Current")
    outstanding = np.where(status == "Closed", 0.0, outstanding)

    fact_finance = pd.DataFrame({
        "finance_id": [f"F{i:06d}" for i in range(1, m + 1)],
        "customer_id": dim_customer["customer_id"].to_numpy()[fc],
        "product_id": prod,
        "disbursement_date": disb_date.normalize(),
        "principal_sar": principal,
        "profit_rate": profit_rate,
        "tenor_months": tenor,
        "status": status,
        "outstanding_sar": outstanding,
    })

    # ---------------- fact_remittance ----------------
    exp = np.where(nationality == "Expat")[0]
    corr_names = list(CORRIDORS)
    corr_w = np.array([CORRIDORS[c][0] for c in corr_names])
    cust_corr = rng.choice(corr_names, len(exp), p=corr_w / corr_w.sum())
    n_rem = rng.poisson(np.where(churned[exp], 0.6, 1.0) * win_len[exp] / 30.0 * 0.9)
    rep = np.repeat(np.arange(len(exp)), n_rem)
    ci = exp[rep]
    rdates = _sample_dates(rng, win_lo[ci], win_hi[ci], weights)
    r_frac = (pd.to_datetime(rdates) - START).days.to_numpy() / total_days
    corr = cust_corr[rep]
    mean_amt = pd.Series(corr).map({c: v[1] for c, v in CORRIDORS.items()}).to_numpy()
    r_amount = np.round(rng.lognormal(np.log(mean_amt), 0.5), 2)
    r_digital = rng.random(len(rep)) < np.clip(0.55 + 0.3 * r_frac, 0, 0.95)
    r_channel = np.where(r_digital, rng.choice(["Mobile App", "Internet Banking"], len(rep), p=[0.85, 0.15]), "Branch")
    fee = np.where(r_channel == "Branch", 25.0, np.where(r_frac > 0.6, 10.0, 15.0))

    fact_remittance = pd.DataFrame({
        "remittance_id": np.arange(1, len(rep) + 1),
        "customer_id": dim_customer["customer_id"].to_numpy()[ci],
        "product_id": "P08",
        "remittance_date": pd.to_datetime(rdates).normalize(),
        "corridor": corr,
        "channel": r_channel,
        "amount_sar": r_amount,
        "fee_sar": fee,
    }).sort_values(["remittance_date", "remittance_id"]).reset_index(drop=True)
    fact_remittance["remittance_id"] = np.arange(1, len(fact_remittance) + 1)

    # ---------------- dim_date ----------------
    d = pd.date_range("2018-01-01", END, freq="D")
    dim_date = pd.DataFrame({"date": d, "year": d.year, "quarter": d.quarter, "month": d.month,
                             "month_name": d.strftime("%b"), "day_of_week": d.strftime("%a")})

    tables = {"dim_date": dim_date, "dim_branch": dim_branch, "dim_product": dim_product,
              "dim_customer": dim_customer, "fact_transaction": fact_transaction,
              "fact_finance": fact_finance, "fact_remittance": fact_remittance}
    return plant_ds_signals(tables, seed)


# ---------------------------------------------------------------------------
# Data-science ground truth. These signals are planted with a SEPARATE random
# stream after the base tables are built, so the BI numbers stay stable. The
# Data Science Lab (bi_agent/ds) must recover them; `python -m bi_agent.ds validate`
# compares every estimate with these planted values.
# ---------------------------------------------------------------------------
GROUND_TRUTH = {
    "pre_churn_disengagement": {
        "window_days": 120, "max_drop": 0.7,
        "description": "Transactions thin out in the 120 days before a customer churns (up to 70% fewer).",
    },
    "campaign": {
        "id": "western_cashback", "name": "Western region card cashback campaign",
        "region": "Western", "start": "2025-11-01", "end": "2026-01-31", "uplift": 0.18,
        "description": "Card purchase and e-commerce transactions of Western-region customers rise 18% "
                       "during the campaign; no effect before or after.",
    },
    "experiment": {
        "id": "remit_one_tap_v2", "name": "One-tap remittance flow in the mobile app",
        "start": "2026-06-01", "days": 28,
        "lift_by_age": {"18-24": 0.10, "25-34": 0.09, "35-44": 0.05, "45-54": 0.02, "55+": 0.0},
        "description": "Randomised 50/50 test on active expat remitters; the new flow lifts remittances "
                       "most for younger customers and not at all for 55+.",
    },
    "fee_cut": {
        "date": "2025-08-25", "volume_effect": 0.0,
        "description": "The digital fee fell from SAR 15 to SAR 10; customer behaviour did not change.",
    },
}


def plant_ds_signals(t: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed + 1_000)
    ft, cust, br = t["fact_transaction"], t["dim_customer"], t["dim_branch"]

    # 1) disengagement before churn: drop a growing share of transactions as churn approaches
    g = GROUND_TRUTH["pre_churn_disengagement"]
    churn_date = ft["customer_id"].map(cust.set_index("customer_id")["churn_date"])
    days_before = (churn_date - ft["txn_date"]).dt.days
    p_drop = (g["max_drop"] * (1 - days_before / g["window_days"])).where(days_before.between(0, g["window_days"] - 1), 0)
    ft = ft[~(rng.random(len(ft)) < p_drop.fillna(0).to_numpy())]

    # 2) regional campaign: extra card transactions for Western customers during the campaign
    c = GROUND_TRUTH["campaign"]
    region = cust["branch_id"].map(br.set_index("branch_id")["region"])
    treated = set(cust.loc[region == c["region"], "customer_id"])
    in_campaign = (ft["customer_id"].isin(treated) & ft["txn_type"].isin(["Card Purchase", "E-commerce"])
                   & ft["txn_date"].between(pd.Timestamp(c["start"]), pd.Timestamp(c["end"])))
    base = ft[in_campaign]
    extra = base.iloc[rng.choice(len(base), int(round(len(base) * c["uplift"])), replace=False)].copy()
    extra["amount_sar"] = rng.permutation(base["amount_sar"].to_numpy())[: len(extra)]  # same amount distribution
    ft = pd.concat([ft, extra]).sort_values(["txn_date", "txn_id"]).reset_index(drop=True)
    ft["txn_id"] = np.arange(1, len(ft) + 1)
    t["fact_transaction"] = ft

    # 3) a randomised A/B test, as an experimentation platform would export it
    e = GROUND_TRUTH["experiment"]
    start = pd.Timestamp(e["start"])
    fr = t["fact_remittance"]
    pre_window = fr[(fr["remittance_date"] >= start - pd.Timedelta(days=e["days"])) & (fr["remittance_date"] < start)]
    recent = set(fr.loc[(fr["remittance_date"] >= start - pd.Timedelta(days=90)) & (fr["remittance_date"] < start), "customer_id"])
    alive = cust["churn_date"].isna() | (cust["churn_date"] >= start + pd.Timedelta(days=e["days"]))
    elig = cust[(cust["nationality_group"] == "Expat") & alive & cust["customer_id"].isin(recent)].copy()
    elig["pre"] = elig["customer_id"].map(pre_window.groupby("customer_id").size()).fillna(0).astype(int)
    treat = rng.random(len(elig)) < 0.5
    lift = elig["age_band"].map(e["lift_by_age"]).to_numpy() * treat
    lam = (0.45 + 0.8 * elig["pre"].to_numpy()) * (1 + lift)
    y = rng.poisson(lam)
    t["fact_experiment"] = pd.DataFrame({
        "experiment_id": e["id"], "customer_id": elig["customer_id"].to_numpy(),
        "variant": np.where(treat, "treatment", "control"), "assigned_date": start,
        "pre_remittances_28d": elig["pre"].to_numpy(), "remittances_28d": y,
        "amount_28d_sar": np.round(y * rng.lognormal(np.log(1500), 0.35, len(y)), 2),
        "failed_transfers_28d": rng.binomial(y, 0.03),          # guardrail: 3% of transfers fail in both arms
    })
    return t


def write_duckdb(tables: dict[str, pd.DataFrame], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    con = duckdb.connect(path)
    for name, df in tables.items():
        con.register("tmp_df", df)
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        replace = f" REPLACE ({', '.join(f'CAST({c} AS DATE) AS {c}' for c in date_cols)})" if date_cols else ""
        con.execute(f"CREATE TABLE {name} AS SELECT *{replace} FROM tmp_df")
        con.unregister("tmp_df")
    con.execute("CREATE TABLE _metadata AS SELECT ? AS as_of_date, ? AS generated_by",
                [date(END.year, END.month, END.day), "scripts/generate_data.py (synthetic)"])
    # planted truth, used ONLY by `python -m bi_agent.ds validate` (never by the agents)
    con.execute("CREATE TABLE _ground_truth AS SELECT ? AS truth_json", [json.dumps(GROUND_TRUTH)])
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--customers", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/wadi_bank.duckdb")
    args = ap.parse_args()
    tables = build(args.customers, args.seed)
    write_duckdb(tables, args.out)
    for k, v in tables.items():
        print(f"{k:<18} {len(v):>9,} rows")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
