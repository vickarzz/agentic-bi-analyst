# Validation against planted truth

| module   | check                                                                | planted truth               | estimate                   | result   | note                          |
|:---------|:---------------------------------------------------------------------|:----------------------------|:---------------------------|:---------|:------------------------------|
| Causal   | Campaign effect (DiD 95% CI covers truth)                            | +18.0%                      | +17.0% (+12.9% to +21.2%)  | PASS     |                               |
| Causal   | Campaign effect is significant (randomisation inference)             | non-zero                    | p = 0.0004                 | PASS     |                               |
| Causal   | Synthetic control close to truth (±6 pts)                            | +18.0%                      | +16.8%                     | PASS     |                               |
| Causal   | Fee cut: interrupted time series CI covers zero                      | 0 pts                       | +0.37 pts (-0.17 to +0.91) | PASS     |                               |
| Causal   | Fee cut: naive before/after is biased (shows why the method matters) | 0%                          | +11.2%                     | PASS     | expected to differ from truth |
| Churn    | Out-of-time ROC-AUC ≥ 0.75                                           | disengagement before churn  | 0.92                       | PASS     |                               |
| Churn    | Top decile captures ≥ 40% of churners                                |                             | 73%                        | PASS     |                               |
| Churn    | Calibration beats the base rate (Brier)                              |                             | 0.0119 vs 0.0128           | PASS     |                               |
| Churn    | No fairness flags                                                    | gender/nationality not used | none                       | PASS     |                               |
| Forecast | Ramadan effect recovered (±10 pts)                                   | +35%                        | +32.8%                     | PASS     |                               |
| Forecast | Pre-Eid effect recovered (±10 pts)                                   | +25%                        | +22.1%                     | PASS     |                               |
| Forecast | Beats 'same week last year' in backtests                             |                             | WAPE 3.0% vs 11.8%         | PASS     |                               |
| A/B test | CUPED 95% CI covers the planted lift                                 | +5.8%                       | +5.2% (+0.5% to +10.0%)    | PASS     |                               |
| A/B test | A/A false-positive rate near 5%                                      | 5%                          | 4.8%                       | PASS     |                               |
| A/B test | No sample ratio mismatch                                             | 50/50                       | p = 0.18                   | PASS     |                               |
| A/B test | Guardrail unchanged                                                  | no effect on failures       | -0.03 pts                  | PASS     |                               |
| Segments | Stable under bootstrap (ARI ≥ 0.8)                                   | no planted segments         | 0.95                       | PASS     |                               |
