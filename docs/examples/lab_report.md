# Data Science Lab report

*Data snapshot `ddff5e4e54d4` · synthetic data (Wadi Bank is fictional) · 17/17 validation checks passed*

## Headlines

- Cashback campaign (Western, Nov 2025 to Jan 2026): difference-in-differences estimates +17.0% card transactions per customer (95% CI +12.9% to +21.2%; randomisation p < 0.001 over 2,380 city assignments). A naive before/after says +13.5%; synthetic control says +16.8%.
- Fee cut: digital remittances rose +11.2% in the 6 months after the cut, but digital's share was already climbing +0.9 pts a month. An interrupted time series (level shift + linear trend) finds a shift of +0.4 pts (95% CI -0.2 to +0.9), within the range of 9 placebo dates (-0.3 to +0.8 pts): no measurable effect.
- Churn model (logistic regression) on an out-of-time test set: ROC-AUC 0.92, PR-AUC 0.15 vs a 1.3% base rate; the riskiest 10% of customers hold 73% of the churners (7.3x lift).
- Next 90 days: about 129 expected churners among 17,785 customers. Contacting 131 customers has an expected net value of SAR 5.74K under the agreed retention assumptions.
- Fairness audit passed: calibration and recall are within tolerance across gender and nationality, which are not model inputs.
- Remittance forecast for the next 13 weeks: SAR 32.79M (80% interval SAR 32.05M to SAR 33.05M).
- Across 20 rolling backtests the Hijri-aware model's error (WAPE) is 3.0% vs 11.8% for 'same week last year'; around Ramadan 2026 it is 3.4% vs 14.7%, because Ramadan moved 11 days earlier. A full Ramadan week carries about 33% more remittance volume.
- A/B test of the one-tap remittance flow (6,046 customers; sample-ratio check p = 0.18): +5.2% remittances per customer with CUPED (95% CI +0.5% to +10.0%, p = 0.027). Without CUPED the same data gives +3.7% (CI -2.1% to +9.4%, p = 0.20): using pre-period behaviour cut variance by 33% and turned an inconclusive test into a clear one.
- The lift falls with age (18-24: +11.0%, 55+: +2.5%), but no single age band is significant after Bonferroni: the test was powered for the overall effect (minimum detectable effect 6.6%). Failure rate per transfer stayed within the 1.0-point guardrail (3.0% vs 3.0%). Recommendation: ship to all expat remitters.
- Remittance segmentation of 6,727 active senders: 2 segments (k chosen by silhouette), stable under bootstrap resampling (adjusted Rand index 0.95), though the silhouette of 0.26 says they overlap: use them for targeting, not as hard boxes. The largest by volume, 'Frequent senders', is 61% of senders and 74% of volume, averaging SAR 1.84K per transfer.

## Causal inference: Western cashback campaign

| Method | Estimate | Uncertainty |
|---|---|---|
| Naive before/after (campaign months vs the 3 months before) | +13.5% | none |
| Difference-in-differences (city and month fixed effects) | +17.0% | 95% CI +12.9% to +21.2%; randomisation p = 0.0004 |
| DiD + treated-group trend (robustness) | +18.3% | |
| Synthetic control (Western region) | +16.8% | rank 1 of 14 placebos (p = 0.071) |

Pre-trend check: treated vs control slope difference -0.22% a month (p = 0.046). Spend per customer (noisier): +11.8%.

## Causal inference: digital fee cut

| Interrupted time-series model (digital share) | Shift at the cut | 95% CI | p | AIC |
|---|---|---|---|---|
| Level shift + linear trend (chosen) | +0.37 pts | -0.17 to +0.91 | 0.179 | -243.9 |
| + slope change | +0.28 pts | -0.33 to +0.89 | 0.364 | -242.5 |
| + quadratic trend | +0.29 pts | -0.34 to +0.93 | 0.364 | -242.1 |

Naive: digital remittances +11.2%, digital share +6.6 pts (6 months after vs before). Pre-period trend: +0.89 pts a month (p = 0). Placebo dates: -0.34 to +0.83 pts. Fee change, digital vs branch: SAR -5.00.

## Churn model

Selected: Logistic regression. See `churn_model_card.md` for the full model card.

| Decile | Customers | Predicted | Observed | Lift | Cumulative capture |
|---|---|---|---|---|---|
| 1 | 3,462 | 4.99% | 9.53% | 7.3x | 73% |
| 2 | 3,462 | 1.04% | 1.94% | 1.5x | 88% |
| 3 | 3,462 | 0.52% | 0.81% | 0.6x | 94% |
| 4 | 3,462 | 0.13% | 0.32% | 0.2x | 97% |
| 5 | 3,462 | 0.07% | 0.17% | 0.1x | 98% |
| 6 | 3,462 | 0.05% | 0.17% | 0.1x | 100% |
| 7 | 3,462 | 0.00% | 0.00% | 0.0x | 100% |
| 8 | 3,461 | 0.00% | 0.00% | 0.0x | 100% |
| 9 | 3,461 | 0.00% | 0.03% | 0.0x | 100% |
| 10 | 3,461 | 0.00% | 0.03% | 0.0x | 100% |

Next-quarter retention plan by segment:

| Segment | Customers | Expected churners | Contact | Expected net value |
|---|---|---|---|---|
| Affluent | 4,471 | 18 | 104 | SAR 3.91K |
| Mass | 12,553 | 110 | 6 | SAR 130 |
| Private | 761 | 1 | 21 | SAR 1.70K |

## Forecast

| Model | Mean WAPE (rolling backtests) | WAPE around Ramadan 2026 |
|---|---|---|
| Calendar model (Hijri-aware) | 3.0% | 3.4% |
| Moving average (last 4 weeks) | 9.5% | 14.0% |
| Seasonal naive (same week last year) | 11.8% | 14.7% |

Calendar effects learned by the model: Ramadan week +32.8%, Pre-Eid week +22.1%, Payday week +9.5%, Growth per year +12.4%.

## A/B test

- Power: minimum detectable effect 8.1% without CUPED, 6.6% with CUPED, at 2,971 customers per arm. A 3% effect needs 21,769 per arm without CUPED or 14,506 with it.
- A/A check: 4.8% of 500 random splits were 'significant'.

| Age band | Customers | Lift | Bonferroni CI |
|---|---|---|---|
| 18-24 | 788 | +11.0% | -7.6% to +29.7% |
| 25-34 | 1,867 | +6.0% | -5.1% to +17.1% |
| 35-44 | 1,634 | +4.1% | -7.8% to +15.9% |
| 45-54 | 1,069 | +2.9% | -11.8% to +17.6% |
| 55+ | 688 | +2.5% | -16.1% to +21.2% |

## Segmentation

| Segment | Senders | Share of volume | Transfers / year | Avg transfer | Days since last |
|---|---|---|---|---|---|
| Frequent senders | 4,104 | 74% | 12.5 | SAR 1.84K | 23 |
| Occasional senders, lapsing | 2,623 | 26% | 6.8 | SAR 1.80K | 60 |

## Validation against planted truth

| Module | Check | Planted truth | Estimate | Result |
|---|---|---|---|---|
| Causal | Campaign effect (DiD 95% CI covers truth) | +18.0% | +17.0% (+12.9% to +21.2%) | PASS |
| Causal | Campaign effect is significant (randomisation inference) | non-zero | p = 0.0004 | PASS |
| Causal | Synthetic control close to truth (±6 pts) | +18.0% | +16.8% | PASS |
| Causal | Fee cut: interrupted time series CI covers zero | 0 pts | +0.37 pts (-0.17 to +0.91) | PASS |
| Causal | Fee cut: naive before/after is biased (shows why the method matters) | 0% | +11.2% | PASS |
| Churn | Out-of-time ROC-AUC ≥ 0.75 | disengagement before churn | 0.92 | PASS |
| Churn | Top decile captures ≥ 40% of churners |  | 73% | PASS |
| Churn | Calibration beats the base rate (Brier) |  | 0.0119 vs 0.0128 | PASS |
| Churn | No fairness flags | gender/nationality not used | none | PASS |
| Forecast | Ramadan effect recovered (±10 pts) | +35% | +32.8% | PASS |
| Forecast | Pre-Eid effect recovered (±10 pts) | +25% | +22.1% | PASS |
| Forecast | Beats 'same week last year' in backtests |  | WAPE 3.0% vs 11.8% | PASS |
| A/B test | CUPED 95% CI covers the planted lift | +5.8% | +5.2% (+0.5% to +10.0%) | PASS |
| A/B test | A/A false-positive rate near 5% | 5% | 4.8% | PASS |
| A/B test | No sample ratio mismatch | 50/50 | p = 0.18 | PASS |
| A/B test | Guardrail unchanged | no effect on failures | -0.03 pts | PASS |
| Segments | Stable under bootstrap (ARI ≥ 0.8) | no planted segments | 0.95 | PASS |

Timings (seconds): churn 10.83, causal 2.9, forecast 0.94, experiment 0.87, segments 3.76.
