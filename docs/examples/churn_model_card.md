# Model card: 90-day churn model

*Data snapshot `ddff5e4e54d4` · synthetic data (Wadi Bank is fictional) · owner: Customer Analytics*

## Intended use
Rank active customers by risk of closing their relationship in the next 90 days, so the retention team can prioritise calls. Not for credit decisions, pricing or any decision about an individual's eligibility.

## Data and features
- Training cutoffs: 2024-06-30, 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30; validation: 2025-12-31; out-of-time test: 2026-03-31, 2026-06-30.
- 92,236 training rows (961 churners); 34,617 test rows (450 churners).
- Inputs: Tenure, Transactions, last 30 days, Transactions, last 90 days, Transactions, last 180 days, Spend, last 90 days, Spend, last 180 days, Digital share, Card share, Channels used, Days since last transaction, Remittances, last 90 days, Remittances, last 180 days, Days since last remittance, Finance principal, Activity trend (30d vs prior), Segment, Age band, Region.
- Excluded on purpose: gender and nationality (used only to audit fairness), and finance status (it reflects the as-of date and would leak the future).

## Performance (out-of-time test)
| Model | Validation PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic regression (selected) | 0.096 | 0.922 | 0.166 |
| Gradient boosting | 0.088 | 0.914 | 0.139 |

- Base rate 1.30%; top-decile lift 7.3x, capturing 73% of churners.
- Brier score: base rate 0.0128 · raw model 0.1131 · after isotonic calibration 0.0119.

## Fairness audit
| Attribute | Group | Customers | Observed | Predicted | Recall at top 10% |
|---|---|---|---|---|---|
| gender | Female | 13,591 | 1.31% | 0.66% | 78% |
| gender | Male | 21,026 | 1.29% | 0.69% | 80% |
| nationality_group | Expat | 12,538 | 1.27% | 0.62% | 74% |
| nationality_group | Saudi | 22,079 | 1.32% | 0.72% | 81% |

No fairness flags.

## Decision rule
Contact a customer when risk × save rate (0.25) × annual value exceeds the contact cost (SAR 40). On the test period this rule would have contacted 205 customers, reached 33 churners and returned SAR 15.65K net, under the same assumptions.

## Monitoring
- Score PSI training vs today: 0.009 (alert above 0.2).
- Retrain quarterly; re-run the fairness audit at every retrain.

## Limitations
- Synthetic data with a planted disengagement pattern; real churn has more causes (pricing, service, life events).
- Churn is rare (about 1% per quarter), so precision at any threshold is modest; use for prioritisation, not automation.
- The save rate and customer values are business assumptions; the decision changes when they change.
