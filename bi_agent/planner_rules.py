"""Offline, deterministic planner: question -> QueryPlan using the semantic layer's synonyms.

It needs no API key, so the whole project runs (and is tested in CI) for free.
When ANTHROPIC_API_KEY is set, the Claude planner (planner_llm.py) is used instead and
this module becomes the fallback.
"""
from __future__ import annotations

import calendar
import difflib
import re
from datetime import date, timedelta

from .plan import Filter, QueryPlan
from .semantic import SemanticLayer

# Extra business vocabulary for dimension values (user word -> (dimension, canonical value)).
VALUE_ALIASES: dict[str, tuple[str, str]] = {
    "murabaha": ("product", "Murabaha Personal Finance"),
    "personal finance": ("product_category", "Personal Finance"),
    "ijarah": ("product", "Auto Ijarah Finance"),
    "auto finance": ("product_category", "Auto Finance"),
    "car finance": ("product_category", "Auto Finance"),
    "credit cards": ("product_category", "Cards"),
    "credit card": ("product_category", "Cards"),
    "cards": ("product_category", "Cards"),
    "mada": ("product", "Mada Debit Card"),
    "mobile app": ("channel", "Mobile App"),
    "mobile banking": ("channel", "Mobile App"),
    "app": ("channel", "Mobile App"),
    "online banking": ("channel", "Internet Banking"),
    "internet banking": ("channel", "Internet Banking"),
    "atm": ("channel", "ATM"),
    "pos": ("channel", "POS"),
    "e-commerce": ("transaction_type", "E-commerce"),
    "ecommerce": ("transaction_type", "E-commerce"),
    "online shopping": ("transaction_type", "E-commerce"),
    "bill payments": ("transaction_type", "Bill Payment"),
    "cash withdrawals": ("transaction_type", "Cash Withdrawal"),
    "expats": ("nationality_group", "Expat"),
    "expatriates": ("nationality_group", "Expat"),
    "expatriate": ("nationality_group", "Expat"),
    "saudis": ("nationality_group", "Saudi"),
    "nationals": ("nationality_group", "Saudi"),
    "women": ("gender", "Female"),
    "men": ("gender", "Male"),
    "90+": ("finance_status", "90+ DPD"),
    "90 dpd": ("finance_status", "90+ DPD"),
    "delinquent": ("finance_status", "30 DPD"),   # expanded to all DPD buckets below
}
# Values too generic to treat as filters when seen in free text.
AMBIGUOUS_VALUES = {"current", "closed", "transfer", "branch", "male", "female"}
DELINQUENT = ["30 DPD", "60 DPD", "90+ DPD"]

DIM_TRIGGERS = r"(by|per|across|each|every|split by|broken down by|breakdown by|grouped by|vs\.?|versus|compare|comparing|between)"

GRAINS = [
    ("day", r"\b(daily|(by|per|each|and) days?|day by day)\b"),
    ("week", r"\b(weekly|(by|per|each|and) weeks?|week over week|wow)\b"),
    ("quarter", r"\b(quarterly|(by|per|each|and) quarters?|quarter over quarter|qoq)\b"),
    ("year", r"\b(yearly|annual|annually|(by|per|each|and) years?|year over year|yoy)\b"),
    ("month", r"\b(monthly|(by|per|each|and) months?|month over month|mom|over time|trend|trends|trending|evolution)\b"),
]
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "twelve": 12, "eighteen": 18, "twenty-four": 24}


def _norm(text: str) -> str:
    t = text.lower().replace("’", "'")
    for w, n in NUM_WORDS.items():
        t = re.sub(rf"\b{w}\b", str(n), t)
    return " " + re.sub(r"\s+", " ", t) + " "


def _find(phrases: list[tuple[str, object]], text: str) -> tuple[list[tuple[int, object, str]], str]:
    """Longest-first phrase matching; matched spans are blanked so shorter phrases can't re-match."""
    hits = []
    for phrase, target in sorted(phrases, key=lambda x: -len(x[0])):
        pat = rf"(?<![\w-]){re.escape(phrase)}s?(?![\w-])"
        for m in re.finditer(pat, text):
            hits.append((m.start(), target, m.group(0)))
            text = text[:m.start()] + "#" * (m.end() - m.start()) + text[m.end():]
    hits.sort(key=lambda h: h[0])
    return hits, text


def _month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def _shift_months(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return date(y, m + 1, 1)


def parse_time(text: str, as_of: date) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (start, end, label, implied_grain)."""
    t = text
    cur_q = (as_of.month - 1) // 3 + 1

    if m := re.search(r"\b(last|past|previous|trailing)\s+(\d+)\s+(day|week|month|quarter|year)s?\b", t):
        n, unit = int(m.group(2)), m.group(3)
        if unit == "day":
            start = as_of - timedelta(days=n - 1)
        elif unit == "week":
            start = as_of - timedelta(weeks=n) + timedelta(days=1)
        else:
            months = n * {"month": 1, "quarter": 3, "year": 12}[unit]
            start = _shift_months(date(as_of.year, as_of.month, 1), -(months - 1))
        return start.isoformat(), as_of.isoformat(), f"last {n} {unit}s", None
    if re.search(r"\b(ytd|year to date|this year|current year)\b", t):
        return date(as_of.year, 1, 1).isoformat(), as_of.isoformat(), f"{as_of.year} year-to-date", None
    if re.search(r"\b(last|previous|prior) year\b", t):
        y = as_of.year - 1
        return f"{y}-01-01", f"{y}-12-31", str(y), None
    if re.search(r"\b(this|current) quarter\b|\bqtd\b", t):
        s = date(as_of.year, (cur_q - 1) * 3 + 1, 1)
        return s.isoformat(), as_of.isoformat(), f"Q{cur_q} {as_of.year}", None
    if re.search(r"\b(last|previous|prior) quarter\b", t):
        s = _shift_months(date(as_of.year, (cur_q - 1) * 3 + 1, 1), -3)
        e = _month_end(s.year, s.month + 2)
        return s.isoformat(), e.isoformat(), f"Q{(s.month - 1) // 3 + 1} {s.year}", None
    if re.search(r"\b(this|current) month\b|\bmtd\b", t):
        return date(as_of.year, as_of.month, 1).isoformat(), as_of.isoformat(), as_of.strftime("%B %Y"), None
    if re.search(r"\b(last|previous|prior) month\b", t):
        s = _shift_months(date(as_of.year, as_of.month, 1), -1)
        return s.isoformat(), _month_end(s.year, s.month).isoformat(), s.strftime("%B %Y"), None

    # Q1 2026 / 2026 Q1
    if m := re.search(r"\bq([1-4])[\s-]*(20\d\d)\b|\b(20\d\d)[\s-]*q([1-4])\b", t):
        q = int(m.group(1) or m.group(4)); y = int(m.group(2) or m.group(3))
        return date(y, 3 * q - 2, 1).isoformat(), _month_end(y, 3 * q).isoformat(), f"Q{q} {y}", None
    month_pat = "|".join(sorted(MONTHS, key=len, reverse=True))
    # since 2025 / since march 2025
    if m := re.search(rf"\b(since|from)\s+(?:({month_pat})\s+)?(20\d\d)\b", t):
        y = int(m.group(3)); mo = MONTHS.get(m.group(2) or "", 1)
        return date(y, mo, 1).isoformat(), as_of.isoformat(), f"since {date(y, mo, 1):%b %Y}", None
    # March 2026
    if m := re.search(rf"\b({month_pat})\.?\s+(20\d\d)\b", t):
        mo, y = MONTHS[m.group(1)], int(m.group(2))
        return date(y, mo, 1).isoformat(), _month_end(y, mo).isoformat(), date(y, mo, 1).strftime("%B %Y"), None
    years = sorted({int(y) for y in re.findall(r"\b(20\d\d)\b", t)})
    if len(years) == 1:
        y = years[0]
        end = min(date(y, 12, 31), as_of)
        return f"{y}-01-01", end.isoformat(), str(y), None
    if len(years) >= 2:
        end = min(date(years[-1], 12, 31), as_of)
        return f"{years[0]}-01-01", end.isoformat(), f"{years[0]}–{years[-1]}", "year"
    return None, None, None, None


class RulePlanner:
    name = "rules"

    def __init__(self, sem: SemanticLayer, dim_values: dict[str, list[str]]):
        self.sem = sem
        self.as_of = date.fromisoformat(sem.as_of_date)
        self.metric_phrases = []
        for name, m in sem.metrics.items():
            self.metric_phrases.append((name.replace("_", " "), name))
            self.metric_phrases += [(s.lower(), name) for s in m.synonyms]
        self.dim_phrases = []
        for name, d in sem.dimensions.items():
            self.dim_phrases.append((name.replace("_", " "), name))
            self.dim_phrases += [(s.lower(), name) for s in d.synonyms]
        self.value_phrases: list[tuple[str, tuple[str, str]]] = []
        for dim, vals in dim_values.items():
            for v in vals:
                if v.lower() not in AMBIGUOUS_VALUES:
                    self.value_phrases.append((v.lower(), (dim, v)))
        self.value_phrases += list(VALUE_ALIASES.items())
        # aliases taught through semantic_layer.yaml (scripts/teach.py) override the built-in ones
        self.value_phrases += [(k, (v["dimension"], v["value"])) for k, v in sem.value_aliases.items()]

    # ------------------------------------------------------------------
    def plan(self, question: str, previous: QueryPlan | None = None,
             feedback: list[str] | None = None) -> QueryPlan:
        text = _norm(question)

        grain = None
        for g, pat in GRAINS:
            if re.search(pat, text):
                grain = g
                text = re.sub(pat, " ", text)
                break
        start, end, tlabel, implied_grain = parse_time(text, self.as_of)
        grain = grain or implied_grain

        m_hits, text = _find(self.metric_phrases, text)
        # bare entity nouns ("private customers", "mobile transactions") qualify another metric
        bare = {"customers", "customer", "transactions", "remittances", "volume"}
        if len({h[1] for h in m_hits}) > 1:
            m_hits = [h for h in m_hits if h[2].strip() not in bare] or m_hits
        metrics = list(dict.fromkeys(h[1] for h in m_hits))

        v_hits, text_no_vals = _find(self.value_phrases, text)
        d_hits, _ = _find(self.dim_phrases, text_no_vals)

        # --- dimensions: a dimension word preceded (closely) by a trigger like "by"/"per"/"vs"
        dims: list[str] = []
        value_ends = {(dim, pos + len(raw)) for pos, (dim, _), raw in v_hits}
        for pos, dim, _ in d_hits:
            before = text_no_vals[max(0, pos - 60):pos]
            if any(d == dim and 0 <= pos - e <= 2 for d, e in value_ends):
                continue            # "affluent segment", "western region" -> a filter, not a breakdown
            if re.search(rf"\b{DIM_TRIGGERS}\b(\s+[\w#&'-]+,?){{0,3}}\s+$", before):
                dims.append(dim)
            elif re.search(r"\b(which|what|top|bottom|best|worst|each|every|all)\s+(\d+\s+)?([a-z]+\s+)?$", before):
                dims.append(dim)
        dims = list(dict.fromkeys(dims))

        # --- filters from recognised values
        values_by_dim: dict[str, list[str]] = {}
        for _, (dim, val), _ in v_hits:
            vals = DELINQUENT if val == "30 DPD" and dim == "finance_status" else [val]
            values_by_dim.setdefault(dim, [])
            values_by_dim[dim] += [v for v in vals if v not in values_by_dim[dim]]
        filters: list[Filter] = []
        for dim, vals in values_by_dim.items():
            negate = bool(re.search(rf"\b(excluding|except|without|other than|not)\b[^#]*{re.escape(vals[0].lower())}", text))
            filters.append(Filter(dimension=dim, values=vals, op="not_in" if negate else "in"))
            # "Saudi vs Expat" -> compare = group by the dimension
            if len(vals) >= 2 and dim not in dims and re.search(r"\b(vs\.?|versus|compare|comparing|and)\b", text):
                dims.append(dim)

        # --- ranking
        order_by, order_dir, limit = None, "desc", None
        if m := re.search(r"\b(top|best|highest|largest|biggest|most)\b\s*(\d+)?", text):
            order_dir = "desc"; limit = int(m.group(2)) if m.group(2) else (5 if m.group(1) == "top" else None)
            order_by = "rank"
        if m := re.search(r"\b(bottom|worst|lowest|smallest|least|fewest)\b\s*(\d+)?", text):
            order_dir = "asc"; limit = int(m.group(2)) if m.group(2) else (5 if m.group(1) == "bottom" else None)
            order_by = "rank"

        # --- follow-up questions inherit context from the previous plan
        if not metrics and previous and previous.metrics:
            metrics = list(previous.metrics)
            if not dims and not grain:
                dims, grain = list(previous.dimensions), previous.time_grain
            if not start and not end:
                start, end, tlabel = previous.start_date, previous.end_date, previous.time_label
            merged = {f.dimension: f for f in previous.filters}
            merged.update({f.dimension: f for f in filters})
            filters = list(merged.values())

        if not metrics:
            return QueryPlan(metrics=[], clarification=self._suggest(question))

        plan = QueryPlan(metrics=metrics, dimensions=dims[:3], time_grain=grain, start_date=start,
                         end_date=end, filters=filters, time_label=tlabel,
                         order_by=metrics[0] if order_by and (dims or not grain) else None,
                         order_dir=order_dir, limit=limit if dims else None)
        if feedback:
            plan = self.repair(plan, feedback)
        return plan

    # ------------------------------------------------------------------
    def repair(self, plan: QueryPlan, errors: list[str]) -> QueryPlan:
        """Self-correction: drop whatever the validator rejected and keep the rest of the question."""
        bad_dims = set(re.findall(r"Dimension '(\w+)' cannot be used", " ".join(errors)))
        bad_dims |= set(re.findall(r"Unknown dimension '(\w+)'", " ".join(errors)))
        bad_metrics = set(re.findall(r"'(\w+)'", " ".join(e for e in errors if e.startswith("Unknown metric"))))
        plan.dimensions = [d for d in plan.dimensions if d not in bad_dims]
        plan.filters = [f for f in plan.filters if f.dimension not in bad_dims]
        plan.metrics = [m for m in plan.metrics if m not in bad_metrics]
        if plan.order_by and plan.order_by not in plan.metrics:
            plan.order_by = plan.metrics[0] if plan.metrics else None
        if not plan.dimensions:
            plan.limit = None
        if len(plan.dimensions) > 3:
            plan.dimensions = plan.dimensions[:3]
        return plan

    def _suggest(self, question: str) -> str:
        words = re.findall(r"[a-z][a-z\-]+", question.lower())
        vocab = {p: t for p, t in self.metric_phrases}
        close = []
        for w in words:
            for c in difflib.get_close_matches(w, vocab.keys(), n=1, cutoff=0.8):
                close.append(vocab[c])
        msg = "I couldn't map that question to a governed metric."
        if close:
            msg += f" Did you mean: {', '.join(dict.fromkeys(close))}?"
        msg += " Available metrics: " + ", ".join(self.sem.metrics) + "."
        return msg
