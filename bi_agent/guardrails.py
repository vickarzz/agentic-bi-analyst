"""Guardrails: defence in depth around the agent.

1. Input guard   — blocks requests for personal data, record-level exports and prompt injection.
2. SQL guard     — the compiled SQL must be a single read-only SELECT over allow-listed tables,
                   with no PII columns and a LIMIT. (The compiler already guarantees this; the
                   guard proves it independently, which is what auditors want to see.)
3. Output guard  — small-cell suppression: groups with fewer than N customers are masked so
                   no one can be re-identified from an aggregate (k-anonymity style rule).
"""
from __future__ import annotations

import math
import re

import pandas as pd

from .semantic import SemanticLayer

_PII_PATTERNS = [
    r"\b(full\s+)?names?\s+of\s+(the\s+)?customers?\b", r"\bcustomer\s+names?\b", r"\bwho\s+(are|is)\s+the\s+customers?\b",
    r"\bphone\b", r"\bmobile\s+(number|no\.?|numbers)\b", r"\bcontact\s+(details|info)\b",
    r"\be-?mail(s|\s+address(es)?)?\b", r"\bnational\s+id\b", r"\biqama\b", r"\bpassport\b",
    r"\biban\b", r"\baccount\s+numbers?\b", r"\bcard\s+numbers?\b", r"\baddress(es)?\s+of\b",
    r"\bc\d{6}\b",                                     # a specific customer id
    r"\b(list|export|download|dump)\b.{0,30}\b(customers|clients|records|rows|transactions)\b(?!.{0,25}\b(by|per|count|number)\b)",
    r"\bshow\s+(me\s+)?(all|the|every)\s+(individual\s+)?(customers|clients|records|rows)\b(?!.{0,25}\b(by|per|count|number)\b)",
]
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|the\s+)?(previous|prior|above)\s+instructions", r"system\s+prompt",
    r"\b(drop|delete|truncate|alter|insert|update)\s+(table|from|into|the\s+table)\b",
    r"\bdeveloper\s+mode\b", r"\bjailbreak\b",
]

_FORBIDDEN_SQL = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|COPY|PRAGMA|INSTALL|LOAD|EXPORT|IMPORT|CALL|SET|GRANT|REVOKE|TRUNCATE|REPLACE)\b"


def check_question(question: str) -> tuple[bool, str | None]:
    q = question.lower()
    for p in _INJECTION_PATTERNS:
        if re.search(p, q):
            return False, ("That looks like an instruction to change my rules or the data. "
                           "I can only answer read-only analytical questions.")
    for p in _PII_PATTERNS:
        if re.search(p, q):
            return False, ("I only return governed, aggregated metrics — never personal data or "
                           "record-level customer details. Try asking for a count, total or rate "
                           "broken down by segment, region, channel, etc.")
    return True, None


def check_sql(sql: str, sem: SemanticLayer) -> list[str]:
    """Return violations (empty = safe)."""
    problems: list[str] = []
    body = re.sub(r"'(?:[^']|'')*'", "''", sql)          # ignore string literals
    stripped = body.strip().rstrip(";")
    if ";" in stripped:
        problems.append("Multiple statements are not allowed")
    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.I):
        problems.append("Only SELECT/WITH queries are allowed")
    bad = re.findall(_FORBIDDEN_SQL, stripped, re.I)
    if bad:
        problems.append(f"Forbidden keyword(s): {sorted(set(b.upper() for b in bad))}")
    for col in sem.governance.get("pii_columns", []):
        name = col.split(".")[-1]
        if re.search(rf"\b{re.escape(name)}\b", stripped, re.I):
            problems.append(f"PII column referenced: {col}")
    ctes = set(re.findall(r"\b(\w+)\s+AS\s*\(", stripped, re.I))
    tables = set(re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w\.]*)", stripped, re.I))
    allowed = set(sem.governance.get("allowed_tables", []))
    illegal = sorted(t for t in tables - ctes if t not in allowed)
    if illegal:
        problems.append(f"Table(s) not allow-listed: {illegal}")
    if not re.search(r"\bLIMIT\s+\d+\s*$", stripped, re.I):
        problems.append("Query must end with a LIMIT")
    return problems


def suppress_small_cells(df: pd.DataFrame, helpers: dict[str, list[str]], threshold: int) -> tuple[pd.DataFrame, int]:
    """Mask metric values computed from fewer than `threshold` customers. Returns (df, n_masked_cells)."""
    df = df.copy()
    masked = 0
    for helper, metrics in helpers.items():
        if helper not in df:
            continue
        small = df[helper].fillna(0) < threshold
        # a missing helper (row came only from another model in an outer join) is not "small"
        small &= df[helper].notna()
        for m in metrics:
            if m in df:
                masked += int((small & df[m].notna()).sum())
                df.loc[small, m] = math.nan
    df = df.drop(columns=[h for h in helpers if h in df])
    return df, masked
