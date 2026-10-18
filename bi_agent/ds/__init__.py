"""Data Science Lab: causal inference, churn prediction, forecasting, A/B testing and
segmentation on the same governed synthetic bank.

Every module reads data only through reviewed SQL templates that pass the SQL guard
(no PII columns, allow-listed tables), returns aggregates for display, and can be checked
against the truth planted by scripts/generate_data.py (`python -m bi_agent.ds validate`).
"""
