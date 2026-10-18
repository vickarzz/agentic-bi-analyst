# Appendix: How the Agentic BI Analyst Was Built

This appendix explains, from start to finish, **what was built, how, and why**. You can use it to:

1. rebuild the project yourself step by step,
2. explain every design decision in an interview, and
3. extend it toward a real enterprise stack (Power BI / Fabric, dbt, Denodo, Databricks).

All data is synthetic. *Wadi Bank* is fictional.

---

## Contents

- [A. Goal and what the project proves](#a-goal-and-what-the-project-proves)
- [B. Setup](#b-setup)
- [C. Step 1: Synthetic data](#c-step-1-synthetic-data)
- [D. Step 2: The semantic layer](#d-step-2-the-semantic-layer)
- [E. Step 3: QueryPlan, the agent's only output](#e-step-3-queryplan-the-agents-only-output)
- [F. Step 4: The SQL compiler](#f-step-4-the-sql-compiler)
- [G. Step 5: Guardrails](#g-step-5-guardrails)
- [H. Step 6: Planners (offline rules and Claude)](#h-step-6-planners-offline-rules-and-claude)
- [I. Step 7: The LangGraph agent](#i-step-7-the-langgraph-agent)
- [J. Step 8: Verified insights and charts](#j-step-8-verified-insights-and-charts)
- [K. Step 9: Streamlit UI and CLI](#k-step-9-streamlit-ui-and-cli)
- [L. Step 10: Tests and the evaluation harness](#l-step-10-tests-and-the-evaluation-harness)
- [M. Step 11: Continuous integration](#m-step-11-continuous-integration)
- [N. Step 12: Publish, deploy and record a demo](#n-step-12-publish-deploy-and-record-a-demo)
- [O. Mapping to enterprise tools](#o-mapping-to-enterprise-tools)
- [P. Interview guide](#p-interview-guide)
- [Q. LinkedIn post and CV bullet](#q-linkedin-post-and-cv-bullet)
- [R. Glossary](#r-glossary)
- [S. Troubleshooting](#s-troubleshooting)
- [T. Multi-agent orchestration: the cross-business review](#t-multi-agent-orchestration-the-cross-business-review)
- [U. Training the agents](#u-training-the-agents)

---

## A. Goal and what the project proves

**Business problem.** Managers wait days for analysts to answer simple questions ("What was our NPF ratio by segment last quarter?"). Self-service BI helps, but people still need to know where to click. LLM chatbots answer instantly, but they write their own SQL, so numbers drift, joins double-count and personal data can leak. In a regulated bank (SAMA, NDMO/PDPL), that is not acceptable.

**Solution.** An agent that understands natural language but is **only allowed to use governed metric definitions**. The LLM decides *what* to ask. A deterministic compiler decides *how* to compute it.

**Skills this demonstrates (mapped to 2026 job postings):**

| Hiring signal | Where it shows in this repo |
|---|---|
| Agentic AI / multi-step orchestration (LangGraph) | `bi_agent/graph.py`: state machine with conditional edges and a retry loop |
| LLM tool use / structured output | `bi_agent/planner_llm.py`: forced tool call with an enum-constrained JSON schema |
| Semantic layer and data modeling | `semantic/semantic_layer.yaml`, `bi_agent/compiler.py` |
| Data governance (CDMP / NDMO mindset) | PII blocking, allow-listed tables, small-cell suppression, metric owners |
| Evaluation / LLMOps | `evals/`: golden questions checked against independent SQL, gated in CI |
| Production engineering | 70 tests, CI pipeline, read-only connections, parameterised SQL, fallbacks |
| Domain knowledge | Islamic finance (Murabaha, Ijarah, NPF), remittance corridors, digital adoption |

---

## B. Setup

Requirements: Python 3.11+ and Git. No API key is needed.

```bash
git clone https://github.com/vickarzz/agentic-bi-analyst.git
cd agentic-bi-analyst
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/generate_data.py      # ~5 seconds
python -m pytest -q                  # 70 tests
python evals/run_evals.py            # writes evals/report.md
streamlit run app/streamlit_app.py
```

**Optional Claude mode:** copy `.env.example` to `.env` (or export the variables in your shell):

| Variable | Meaning | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Enables the Claude planner and summaries | empty (offline) |
| `BI_AGENT_PLANNER` | `auto`, `llm` or `rules` | `auto` (Claude if a key exists) |
| `BI_AGENT_MODEL` | Claude model ID | `claude-sonnet-5-5` |
| `BI_AGENT_DB` | Path to the DuckDB file | `data/wadi_bank.duckdb` |

> The app does not load `.env` automatically. Either `export` the variables or add `python-dotenv` if you prefer.

**Why these tools?**

- **DuckDB**: an in-process analytical database. No server, columnar and fast, and it uses standard SQL, so the patterns transfer directly to Snowflake, Oracle or Impala.
- **LangGraph**: makes the agent an explicit, testable state machine instead of a black-box loop.
- **Streamlit**: fastest way to ship a usable UI in pure Python, with free hosting.
- **Claude via the Anthropic SDK**: strong tool use, and forced tool choice gives reliably structured plans.

---

## C. Step 1: Synthetic data

**File:** `scripts/generate_data.py`

### Why synthetic?
You must **never** use real customer data, or any code or data from an employer, in a public portfolio. Synthetic data also lets you **plant business patterns on purpose**, so the demo has real stories to discover.

### Schema (star schema)

```
                 dim_date
                    │
dim_branch ─── dim_customer ─── fact_transaction ─── dim_product
                    │       └── fact_finance ─────── dim_product
                    │       └── fact_remittance
```

| Table | Rows (default) | Grain | Key columns |
|---|---|---|---|
| `dim_customer` | 20,000 | one customer | segment, nationality_group, age_band, gender, branch_id, join_date, churn_date, **PII: full_name, mobile_number, email** |
| `dim_branch` | 23 | one branch | city, region (Central, Western, Eastern, Southern, Northern) |
| `dim_product` | 8 | one product | product_category, is_islamic |
| `fact_transaction` | ~560k | one transaction | txn_date, channel, txn_type, amount_sar, is_digital |
| `fact_finance` | ~5.4k | one Murabaha/Ijarah contract | disbursement_date, principal_sar, profit_rate, tenor_months, status, outstanding_sar |
| `fact_remittance` | ~174k | one outbound remittance | remittance_date, corridor, channel, amount_sar, fee_sar |

The PII columns exist **deliberately**, so the guardrails have something real to protect.

### Planted patterns (the "stories")

| Pattern | How it is generated |
|---|---|
| Digital adoption rises from ~50% (Q1 2024) to ~75% (Q3 2026) | Probability of a digital channel = 0.48 + 0.27 × (time fraction) + age adjustment |
| Younger customers are more digital | Age adjustments from +12 pts (18–24) to −18 pts (55+) |
| Ramadan and pre-Eid al-Adha spending spikes | Daily weights ×1.35 / ×1.25 in those windows; dates are sampled by rejection sampling against the weights |
| Payday and weekend effects | ×1.10 from the 25th of the month, ×1.12 on Thursday and Friday |
| Affluent and Private customers have bigger tickets | Log-normal amounts with segment multipliers 1× / 2.4× / 6× |
| NPF is higher in Mass than Affluent | Delinquency probabilities by segment, ×2.5 for churned customers |
| Expats send remittances to 6 corridors | Each expat has a home corridor (India 27%, Pakistan 20%, Indonesia 18%, ...) |
| Mass customers churn most (~11% vs ~2% Private) | Segment-based churn probability. Churned customers stop transacting at churn_date |

### Key techniques
- **Vectorised NumPy** generates 560k rows in about 3 seconds (no Python loops per row).
- **Activity windows:** each customer only transacts between max(join_date, 2024-01-01) and churn_date (or the as-of date).
- **Outstanding balance** = principal × (1 + profit_rate × tenor/12) × (1 − months_elapsed/tenor). Contracts past their tenor are `Closed`.
- **Reproducible:** a fixed `--seed` produces identical data every time, which makes tests and evals stable.

```bash
python scripts/generate_data.py --customers 5000 --seed 1 --out data/small.duckdb
```

---

## D. Step 2: The semantic layer

**File:** `semantic/semantic_layer.yaml`, loaded and validated by `bi_agent/semantic.py`.

A semantic layer is a **contract between the business and the data**. "NPF ratio" is defined once, with an owner, and every consumer (dashboard, analyst, AI agent) gets the same number.

### Building blocks

```yaml
models:                       # a fact table + the joins it may use
  finance:
    table: fact_finance
    alias: f
    time_column: f.disbursement_date
    customer_key: f.customer_id        # used for small-cell suppression
    joins:
      c: {sql: "LEFT JOIN dim_customer c ON c.customer_id = f.customer_id"}
      b: {sql: "LEFT JOIN dim_branch b ON b.branch_id = c.branch_id", requires: [c]}

dimensions:                   # how metrics can be sliced
  region:
    expr: b.region
    requires: [c, b]                   # needs both joins, in dependency order
    synonyms: [region, area, province]

metrics:                      # governed calculations
  npf_ratio:
    model: finance
    additive: false                    # a ratio cannot be summed across groups
    expr: >-
      SUM(CASE WHEN f.status = '90+ DPD' THEN f.outstanding_sar ELSE 0 END)
      / NULLIF(SUM(CASE WHEN f.status <> 'Closed' THEN f.outstanding_sar ELSE 0 END), 0)
    owner: Retail Credit Risk
    description: Non-performing finance ratio ...
    synonyms: [npf, npl, non-performing, default rate]
```

### Design decisions

- **Compatibility is computed, not hard-coded.** A dimension can be used with a model only if every alias it `requires` is the model's own alias or one of its joins. So `channel` works for transactions and remittances but not for finance. The validator uses this rule to reject impossible questions.
- **`{alias}` placeholders** let one dimension (such as `channel`) map to different tables.
- **A model can carry a `where`** (the `churn` model only includes churned customers) and its own time column, so "churned customers this year" filters on `churn_date`, not `join_date`.
- **The `additive` flag** stops the insight engine from claiming "Mass is 60% of the total NPF ratio", which would be meaningless.
- **Validation at load time** catches broken references (an unknown model, an unreachable dimension, a missing join dependency) before the app starts.

### How to add a metric (example: credit card spend)

```yaml
  card_spend:
    model: transactions
    label: Card spend (SAR)
    expr: SUM(CASE WHEN t.txn_type IN ('Card Purchase', 'E-commerce') THEN t.amount_sar ELSE 0 END)
    format: currency
    owner: Cards Business
    description: Value of POS and e-commerce card transactions.
    synonyms: [card spend, card spending, card sales]
```

Then add a golden question for it in `evals/golden_questions.yaml` and run `make evals`. No Python change is needed.

---

## E. Step 3: QueryPlan, the agent's only output

**File:** `bi_agent/plan.py`

```python
QueryPlan(
    metrics=["remittance_volume"], dimensions=["corridor"],
    time_grain=None, start_date="2026-01-01", end_date="2026-09-30",
    filters=[Filter("segment", ["Affluent"], op="in")],
    order_by="remittance_volume", order_dir="desc", limit=3,
    time_label="2026", clarification=None,
)
```

`validate_plan()` returns plain-English errors, for example:

> Dimension 'channel' cannot be used with metric 'npf_ratio' (model 'finance'). Compatible dimensions: [...]

These messages are written **for the planner to read**, because they are fed back into the next planning attempt (section I).

Why a plan instead of SQL? The plan is small, inspectable, easy to validate, easy to test and **identical** whether it came from rules or from an LLM. This is the key architectural boundary of the project.

---

## F. Step 4: The SQL compiler

**File:** `bi_agent/compiler.py`

Algorithm for each question:

1. **Group metrics by model.** `transaction_volume` and `remittance_volume` live in different fact tables.
2. **For each model, build one `SELECT`:**
   - time bucket: `CAST(date_trunc('<grain>', <time_column>) AS DATE) AS period`
   - dimension expressions and metric expressions from the YAML
   - a helper column `COUNT(DISTINCT customer_key) AS __n_<model>` for privacy suppression
   - joins: only those required, resolved recursively in dependency order (`c` before `b`)
   - `WHERE`: model filter + date range + dimension filters, all as `?` **bound parameters**
   - `GROUP BY ALL` (DuckDB) when there are dimensions or a time grain
3. **One model:** use that query. **Several models:** wrap each in a CTE and `FULL OUTER JOIN ... USING (shared dimensions)`, or `CROSS JOIN` for single-row KPIs.
4. **Order and limit:** explicit ranking, else chronological for trends, else by the first metric. The `LIMIT` is capped at `governance.max_rows`.

**Why CTEs for multiple facts?** Joining `fact_transaction` to `fact_remittance` directly on customer multiplies rows (the classic fan-out bug) and inflates totals. Aggregating each fact separately and then joining the aggregates is correct. `tests/test_compiler.py::test_multi_model_uses_ctes_not_fact_to_fact_join` proves the totals match a single-model query.

**Why bound parameters?** Filter values come from user text. Binding them (never string-concatenating) removes SQL injection by construction. `render_sql()` inlines the values for **display only**.

Example for *"Transaction and remittance volume by region for Affluent customers, quarterly, 2026"*:

```sql
WITH m0 AS (
  SELECT CAST(date_trunc('quarter', t.txn_date) AS DATE) AS period, b.region AS region,
         SUM(t.amount_sar) AS transaction_volume, COUNT(DISTINCT t.customer_id) AS __n_transactions
  FROM fact_transaction t
  LEFT JOIN dim_customer c ON c.customer_id = t.customer_id
  LEFT JOIN dim_branch b ON b.branch_id = c.branch_id
  WHERE CAST(t.txn_date AS DATE) >= ? AND CAST(t.txn_date AS DATE) <= ? AND c.segment IN (?)
  GROUP BY ALL),
m1 AS ( ... same shape over fact_remittance ... )
SELECT period, region, transaction_volume, remittance_volume, __n_transactions, __n_remittances
FROM m0 FULL OUTER JOIN m1 USING (period, region)
ORDER BY period ASC, region
LIMIT 500
```

---

## G. Step 5: Guardrails

**File:** `bi_agent/guardrails.py`. This is defence in depth: three independent layers.

| Layer | When | What it does | Example |
|---|---|---|---|
| **1. Input guard** | before planning | Regex rules block requests for personal data (names, phone, email, IDs, IBAN), record-level exports, specific customer IDs, and prompt injection ("ignore previous instructions", "drop table") | "Give me the phone numbers of affluent customers" → refused, and the database is never touched |
| **2. SQL guard** | after compiling | Independently checks: single statement, starts with SELECT/WITH, no DDL/DML keywords (string literals are ignored), no PII column names, only allow-listed tables, ends with LIMIT | `SELECT * FROM read_csv('/etc/passwd')` → rejected (not allow-listed) |
| **3. Output guard** | after executing | **Small-cell suppression:** any metric computed from fewer than 10 distinct customers is masked | Private churners YTD = 7 → shown as "n/a (suppressed)" |

On top of that, the DuckDB connection is opened `read_only=True`, so even a missed bug cannot write.

**Why guard SQL that our own compiler produced?** Auditors ask "how do you *know*?" A separate check that does not trust the compiler is evidence. It also protects a future change, such as letting an LLM write SQL for an "expert mode".

**Why small-cell suppression?** Aggregates can re-identify people ("customers in Najran, Private segment, age 55+, female: 1"). Masking groups below *k* is standard practice in statistical disclosure control. The threshold lives in the YAML (`governance.small_cell_threshold`).

**Known limits (state them openly):** regex input guards can be evaded by creative phrasing. That is why layers 2 and 3 exist and why the semantic layer simply *cannot express* PII. In production, add an LLM-based classifier and database-level row/column security (Denodo, Unity Catalog, Power BI RLS).

---

## H. Step 6: Planners (offline rules and Claude)

Both planners implement the same interface:

```python
plan(question, previous_plan=None, feedback=None) -> QueryPlan
```

### H1. Rule-based planner (`bi_agent/planner_rules.py`)

It runs without an API key and is deterministic, which makes it ideal for CI and demos.

1. **Normalise** the text (lowercase, number words to digits).
2. **Time grain:** patterns such as `monthly`, `by quarter`, `trend`, `yoy`.
3. **Time range:** resolved against the dataset's as-of date (2026-09-30): `last N months/quarters/years`, `this/last year|quarter|month`, `ytd`, `Q2 2026`, `March 2026`, `since March 2025`, `in 2025`, `2024 vs 2025` (which also implies a yearly grain).
4. **Metrics:** longest-first phrase matching over metric names and synonyms. Matched spans are blanked so "number of transactions" does not also match "transactions". Bare nouns such as "customers" in "private customers" are dropped when a real metric is also present.
5. **Filter values:** distinct dimension values are loaded from the database (Riyadh, Affluent, Mobile App, India, ...) plus a business alias table (murabaha, expats, e-commerce, delinquent). Words such as "excluding/except/not" turn a filter into `not_in`.
6. **Dimensions:** a dimension word counts as a breakdown when it follows a trigger (`by`, `per`, `across`, `vs`, `which`, `top`) within three words. "Western **region**" right after a value is treated as a filter, not a breakdown. Two values of one dimension with "vs" ("Saudi vs expat") become a breakdown.
7. **Ranking:** `top N`/`highest` → descending with a limit. `bottom`/`lowest` → ascending.
8. **Follow-ups:** with no metric in the text, the previous plan's metrics, dates and filters carry over. New dimensions or filters replace the old ones.
9. **Repair:** on validator feedback, it drops the offending dimensions, filters or metrics and keeps the rest of the question.
10. **Clarify:** if no metric is found, it suggests close matches (difflib) and lists the available metrics.

### H2. Claude planner (`bi_agent/planner_llm.py`)

- **System prompt** = rules + the catalog (metric and dimension definitions, which dimensions work with which model) + known filter values + the as-of date. **No data rows are ever sent.**
- **Forced tool use:** `tool_choice={"type": "tool", "name": "submit_query_plan"}`. The tool's JSON schema uses **enums** of metric and dimension names, so the model cannot invent a field.
- **Feedback loop:** on a retry, the validator's errors are appended ("Your previous plan failed validation. Fix these problems: ..."). This is the self-correction pattern.
- **Follow-ups:** the previous plan is included as JSON.
- **Fallback:** any exception (network, quota, malformed output) silently falls back to the rule planner and is recorded in the trace.
- **Summaries:** optionally, Claude rewrites the *verified facts* (section J) into 2–3 sentences, with an instruction to use only numbers that appear in those facts.

Tested offline with a fake client (`tests/test_graph.py`): first plan invalid → second plan corrected → answer, plus a test of the failure fallback.

---

## I. Step 7: The LangGraph agent

**File:** `bi_agent/graph.py`

```
guard_input ──blocked──────────────────────────────► respond
     │ ok
     ▼
   plan ◄──────── errors (attempts < 3) ──────┐
     │                                        │
     ▼                                        │
 validate ────────────────────────────────────┘
     │ ok             └── clarify / attempts exhausted ──► respond
     ▼
  compile ─► check_sql ─violation─► respond
                 │ ok
                 ▼
             execute ─► explain ─► respond ─► END
```

- **State** (`AgentState`, a TypedDict): question, previous plan, plan, errors, attempts, SQL, params, result DataFrame, facts, summary, chart spec, status (`ok | blocked | clarify | error`), message, and `trace`.
- **`trace` uses a reducer** (`Annotated[list, operator.add]`), so every node *appends* a step with its timing. The UI shows it as an audit log.
- **Conditional edges** implement the control flow. The retry limit comes from `BI_AGENT_MAX_ATTEMPTS` (default 3).
- **Why a graph and not a free-running agent loop?** Banking needs predictable paths. Every possible route is visible in the code, testable (`test_graph.py` asserts the exact node sequence) and bounded (no infinite loops).

---

## J. Step 8: Verified insights and charts

**File:** `bi_agent/insights.py`

Narratives are computed with pandas from the result, so every number is real:

| Question shape | Facts produced |
|---|---|
| KPI (no breakdown) | value per metric, with the time label |
| Trend | latest value, change vs previous period, change since the first period, peak |
| Breakdown | highest, lowest, share of total (additive metrics only) or "X is N× Y" (ratios) |
| Breakdown over time | strongest and weakest change between the first and last period, largest in the latest period |

Charts are chosen by rule: trend → line, breakdown → bar (with a second dimension as colour), KPI → metric tiles.

---

## K. Step 9: Streamlit UI and CLI

**`app/streamlit_app.py`**
- Chat interface with conversation memory (the last successful plan is passed to follow-ups).
- Each answer shows a summary, a Plotly chart and tabs for **Table** (with verified facts), **SQL**, **Plan** (JSON) and **Trace** (node-by-node timings).
- Sidebar: sample questions, a metric glossary (definition and owner), governance rules and a "new conversation" button.
- The agent is cached with `st.cache_resource`. On first run in the cloud, the app generates the database if it is missing.

**`bi_agent/cli.py`**: one-shot or interactive mode, with `--show-sql`, `--trace` and `--planner rules|llm`.

---

## L. Step 10: Tests and the evaluation harness

### Unit and integration tests (`tests/`, 70 tests, including 10 for the orchestrator and 5 for training)

| File | Covers |
|---|---|
| `test_compiler.py` | parameter binding, join ordering, minimal joins, no fan-out, LIMIT cap, every metric compiles and runs |
| `test_guardrails.py` | blocked and allowed questions, SQL guard rejections, string-literal handling, suppression |
| `test_planner.py` | 11 date expressions, filters vs dimensions, top-N, compare, follow-ups, clarification, repair |
| `test_graph.py` | full node sequence, blocked path skips the DB, self-correction, follow-ups, mocked Claude path, LLM-failure fallback |
| `test_app.py` | headless Streamlit run via `streamlit.testing.v1.AppTest` |

A session-scoped fixture builds a small 3,000-customer database in a temp folder, so the tests are fast (~2 s) and isolated.

### Golden-question evals (`evals/`)

Each case states the **expected plan** and an **independently hand-written reference SQL** that deliberately does *not* use the semantic layer:

```yaml
- id: murabaha_region_q2
  question: Murabaha disbursed by region in Q2 2026
  expect: {metrics: [finance_disbursed], dimensions: [region], start_date: "2026-04-01", end_date: "2026-06-30",
           filters: [{dimension: product, values: [Murabaha Personal Finance], op: in}]}
  reference_sql: |
    SELECT b.region, SUM(f.principal_sar) AS finance_disbursed
    FROM fact_finance f JOIN dim_customer c USING (customer_id) JOIN dim_branch b USING (branch_id)
    WHERE f.product_id = 'P06' AND f.disbursement_date BETWEEN '2026-04-01' AND '2026-06-30' GROUP BY 1
```

The runner reports **plan accuracy**, **answer accuracy** (values match within 1e-6), guardrail/clarify accuracy and latency, writes `evals/report.md`, and returns a non-zero exit code below `--min-score`. Cases cover KPIs, trends, breakdowns, top-N, multi-fact questions, follow-ups, self-correction, small-cell suppression (`allow_suppressed: true`), PII requests, injection and out-of-scope questions.

**Honest framing for interviews:** the 28 golden questions were written while building the rule planner, so 100% means "no regressions on known questions", not "understands everything". Next steps: a held-out set written by someone else, and running the same harness with `--planner llm` to compare the planners.

---

## M. Step 11: Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:
1. set up Python 3.12 with a pip cache
2. install requirements
3. generate the synthetic data
4. run pytest
5. run the evals with `--min-score 0.9` (the build fails if quality drops)
6. publish `evals/report.md` to the GitHub job summary

CI uses `BI_AGENT_PLANNER=rules`, so it needs no secrets and is fully deterministic.

---

## N. Step 12: Publish, deploy and record a demo

### Push to GitHub
```bash
git init && git add . && git commit -m "Agentic BI Analyst: governed NL-to-metrics agent"
git branch -M main
git remote add origin https://github.com/vickarzz/agentic-bi-analyst.git
git push -u origin main
```
On GitHub:
- **Description:** "Governed agentic BI: natural-language questions → semantic layer → safe SQL. LangGraph + Claude + DuckDB."
- **Topics:** `agentic-ai`, `langgraph`, `semantic-layer`, `business-intelligence`, `text-to-sql`, `data-governance`, `llm`, `duckdb`, `streamlit`, `banking`
- **Pin** the repo on your profile, and link it from your profile README.

### Deploy for free (a live link matters to recruiters)
- **Streamlit Community Cloud:** connect the repo, set the main file to `app/streamlit_app.py`, and optionally add `ANTHROPIC_API_KEY` under *Secrets*. The data is generated on first start.
- **Hugging Face Spaces:** create a Streamlit Space and push the repo. This works the same way.

> If you add an API key to a public demo, set a spending limit on the key. Without a key, the demo still works in offline mode.

### Record a 20–40 second demo GIF
Use ScreenToGif (Windows) or Kap (macOS). Show the following in order: a trend question → the SQL tab → a follow-up ("now by region") → a blocked PII request → a suppressed cell. Save it as `docs/demo.gif` and uncomment the line in the README.

---

## O. Mapping to enterprise tools

| This repo | Power BI / Fabric | dbt Semantic Layer | Denodo | Databricks |
|---|---|---|---|---|
| `semantic_layer.yaml` metrics | DAX measures in a semantic model | MetricFlow metrics | Derived / interface views | Unity Catalog metric views |
| `compiler.py` | DAX query via XMLA / REST `executeQueries` | Semantic Layer API (JDBC/GraphQL) | VQL / REST data services | SQL against metric views, or Genie |
| Guardrails (PII, allow-list) | RLS / OLS, sensitivity labels | Warehouse grants | Denodo role-based security, column masking | Unity Catalog grants, column masks |
| Small-cell suppression | Measure logic (`IF(DISTINCTCOUNT(...) < 10, BLANK())`) | Metric filter / post-processing | Derived view with a HAVING rule | Masking functions |
| Planner | Fabric data agent / Copilot | any LLM + API | any LLM + REST | Genie |

**Swapping DuckDB for your warehouse:** only `bi_agent/db.py` changes (for example, use `oracledb`, `snowflake-connector-python` or `impyla`). SQL functions such as `date_trunc` and `GROUP BY ALL` may need small dialect tweaks in `compiler.py`.

**Next level: an MCP server.** Wrap `BIAgent.ask` as an MCP tool (`ask_governed_metrics(question)`) so any MCP-capable assistant can query governed numbers. This is on the roadmap and is a strong follow-up post.

---

## P. Interview guide

### 60-second pitch (STAR)
- **Situation:** Business users wait for analysts, and LLM "chat with data" tools produce inconsistent, ungoverned numbers, which is unacceptable under banking regulation.
- **Task:** Build an assistant that answers natural-language questions with the *same* numbers as official dashboards, without exposing personal data.
- **Action:** I separated *understanding* from *computation*. A LangGraph agent turns the question into a structured plan restricted to a governed semantic layer. A deterministic compiler generates parameterised SQL. Three guardrail layers protect PII. A golden-question eval runs in CI against independent SQL.
- **Result:** 15 governed metrics, 100% plan and answer accuracy on 28 golden questions, around 15 ms median latency offline, every answer auditable (plan, SQL, trace), and zero PII exposure by design.

### Likely questions and strong answers

**"Why not let the LLM write SQL directly?"**
Consistency, safety and testability. With a semantic layer, the LLM picks *what* (metric names), and governed code decides *how*. The same question always yields the same SQL, so evals are meaningful.

**"What happens when the question doesn't fit the catalog?"**
The validator explains why. The planner retries with that feedback up to three times. Then the agent asks for clarification instead of guessing. See the "NPF ratio by channel" example.

**"How would you scale this to 500 metrics?"**
Retrieve only the relevant part of the catalog (embedding search over metric descriptions) before planning. Group metrics by domain, and have owners maintain synonyms. The compiler and guards don't change.

**"How do you measure quality in production?"**
The golden set runs in CI, plus logging of plans, clarifications and user thumbs-up/down. Questions that fail become new golden cases. Track plan accuracy by planner and by model version.

**"What are the limitations?"**
The rule planner handles known phrasings only. The regex input guard can be evaded (that is why the semantic layer can't express PII at all). There is no Arabic support yet. The golden set is in-distribution.

**"How does this relate to your experience?"**
It is the same layered pattern I used with Denodo virtual views and Power BI/Tableau semantic models at banks: one governed definition, many consumers, now with an AI consumer added.

---

## Q. LinkedIn post and CV bullet

### LinkedIn post (draft; edit it into your own voice)

> **"Chat with your data" is easy. Getting the *same* number twice is the hard part.**
>
> I built an open-source **Agentic BI Analyst** for a fictional Saudi bank (100% synthetic data):
>
> 🔹 Ask in plain English: "Which region has the highest NPF ratio?" or "Top 3 remittance corridors in 2026"
> 🔹 A LangGraph agent turns the question into a plan restricted to a **governed semantic layer** (15 metrics with owners and definitions)
> 🔹 A deterministic compiler writes parameterised SQL. The LLM never writes SQL
> 🔹 3 guardrail layers: PII requests blocked, SQL allow-list, and groups under 10 customers suppressed
> 🔹 Self-correction: invalid plans get validator feedback and retry
> 🔹 28 golden questions checked against independent SQL on every commit
>
> Stack: Python · LangGraph · Claude (tool use) · DuckDB · Streamlit · GitHub Actions
>
> After 15 years in BI across banking and telecom, my takeaway: **AI doesn't replace the semantic layer; it makes it more important.**
>
> 🔗 Code + live demo: [link]
> What would you add next: an MCP server, or Arabic questions?
>
> #AgenticAI #BusinessIntelligence #DataGovernance #LangGraph #SemanticLayer #Banking

Tips: attach the demo GIF or a short video (native video gets more reach than a link), post Tuesday to Thursday in the morning, reply to comments in the first hour, and put the repo in the *Featured* section.

### CV / LinkedIn Projects bullet
> **Agentic BI Analyst (open source):** Built a LangGraph + Claude agent that answers natural-language banking questions through a governed semantic layer (15 metrics), with a deterministic SQL compiler, three-layer PII guardrails, small-cell suppression and a CI-gated golden-question eval (100% plan and answer accuracy on 28 cases; 70 automated tests). Added a multi-agent orchestrator (supervisor, 5 parallel domain agents, impact analyst, discovery agent, verifier) that tests 8 cross-business hypotheses with permutation statistics and a placebo control..

---

## R. Glossary

| Term | Meaning |
|---|---|
| **Semantic layer** | A business-friendly layer that defines metrics, dimensions and joins once, for all tools |
| **Murabaha** | Islamic cost-plus sale financing, commonly used for personal finance |
| **Ijarah** | Islamic lease financing, used here for auto finance |
| **DPD** | Days past due. 30/60/90+ DPD are delinquency buckets |
| **NPF ratio** | Non-performing finance: 90+ DPD outstanding / open outstanding (the Islamic-banking equivalent of NPL) |
| **Corridor** | Destination country of a remittance |
| **Digital adoption rate** | Share of transactions via the mobile app or internet banking |
| **Fan-out** | Row multiplication when joining two fact tables, which inflates sums |
| **Small-cell suppression** | Hiding aggregates built from too few individuals to prevent re-identification |
| **Golden set** | Curated questions with known correct answers, used for regression evaluation |
| **Self-correction loop** | Feeding validation errors back to the planner so it can fix its plan |

---

## S. Troubleshooting

| Symptom | Fix |
|---|---|
| `data/wadi_bank.duckdb not found` | Run `python scripts/generate_data.py` |
| `ModuleNotFoundError: bi_agent` | Run commands from the repo root (`python -m bi_agent.cli ...`) |
| App always says "Offline rule-based planner" | Export `ANTHROPIC_API_KEY` in the same shell before `streamlit run` |
| Trace shows `rules (LLM failed: ...)` | Check the key, network and model name (`BI_AGENT_MODEL`). The app keeps working offline |
| A question gets "couldn't map" | Add a synonym to the metric in the YAML, then add a golden case |
| DuckDB "database is locked" | Close other processes that opened the file in write mode. The app opens it read-only |
| Different numbers after regenerating data | Use the same `--seed`. Evals compare against live reference SQL, so they stay valid |

---

## T. Multi-agent orchestration: the cross-business review

### T1. What it adds

The chat agent answers one question at a time. The **Business Review Orchestrator** answers a bigger one: *"How is the whole bank doing, and how does each business line affect the others?"* It runs a team of agents, all of them limited to governed metrics, and produces an executive briefing that a verifier has checked.

```bash
python -m bi_agent.orchestrator                                   # whole business
python -m bi_agent.orchestrator "focus on credit risk and churn"  # narrower scope
make review                                                       # same, writes reports/business_review.md + .json
```

In the Streamlit app, open **Business Review** from the sidebar.

### T2. The agents

| Agent | Job | Tools it may use |
|---|---|---|
| **Supervisor** | Reads the request, chooses the domains in scope (keywords such as "credit", "churn"; "whole/all/seluruh" means everything), dispatches the team | none |
| **Data steward** | Builds one governed monthly KPI panel (12 metrics × 33 months) shared by every specialist | `governed_query` |
| **Domain agents ×5** (Retail, Digital, Credit, Payments, Customer) | Latest-quarter KPIs with QoQ and YoY, 12-month breakdowns, alert thresholds (NPF > 5%), unusual months (robust z ≥ 3) explained with the business calendar | `governed_query` |
| **Impact analyst** | Tests every hypothesis in `semantic/impact_links.yaml`; returns verdict, effect, p-value and the owner's action | `governed_query`, `approved_analysis` |
| **Discovery agent** | Scans all 66 KPI pairs for co-movement nobody has hypothesised; proposes candidates only | panel only |
| **Synthesizer** | Writes the briefing: summary, risks, impact map, actions, scorecards, caveats | none |
| **Verifier** | Checks the briefing before release; sends it back if it fails | none |

All seven specialists run **in parallel** using LangGraph's `Send` API, and a join edge waits for all of them before the synthesizer starts:

```python
g.add_conditional_edges("build_panel", self.fan_out, ["domain_agent", "impact_analyst", "discovery_agent"])
g.add_edge(["domain_agent", "impact_analyst", "discovery_agent"], "synthesizer")
```

`findings`, `trace` and `tool_calls` use `operator.add` reducers, so parallel agents append to them without overwriting each other.

### T3. Governance carries over

Agents have exactly two data tools (`bi_agent/orchestrator/tools.py`):

- **`governed_query(plan)`**: the same QueryPlan → validator → compiler → SQL guard → small-cell suppression path as the chat agent.
- **`approved_analysis(name, sql)`**: runs a reviewed SQL template stored in `impact_links.yaml` (for cohort comparisons that need joins across fact tables). Templates return no identifiers and still pass the SQL guard. Agents cannot write new SQL.

Every call is logged; the test suite asserts that no other tool was used.

### T4. Cross-business hypotheses (`semantic/impact_links.yaml`)

Business owners write the links they believe in; the agent tests them. Each entry has a title, the from/to business line, a hypothesis in plain words, the expected direction, an owner, a test design and two recommended actions (`if_confirmed`, `if_not_detected`).

| Test type | Question it answers | Method |
|---|---|---|
| `timeseries` | Do two KPIs move together, possibly with a delay? | Correlate **month-over-month growth** of X with growth of Y at lags 0..N; permutation p-value; Bonferroni across lags; elasticity with a bootstrap interval when confirmed |
| `cross_section` | Where X is high, is Y high too? | Correlation across groups (e.g. segment × region); permutation p-value; slope |
| `cohort` | Is the outcome different for one group of customers? | Approved SQL returns contract-level rows; NPF ratio per cohort; permutation test on the cohort labels |
| `event` | Did something change after a dated business event? | Mean of 6 months before vs 6 after (the event month is skipped); permutation test |

Verdicts: **confirmed** (expected sign, p < 0.05 and a meaningful effect), **contradicted** (significant, wrong sign), **not detected**, and for the placebo, **control passed / control FAILED**.

To add a hypothesis, append a YAML block and run `make review`. For example:

```yaml
  - id: digital_to_fee_income
    title: Digital adoption and remittance fee income
    from: {domain: digital, metric: digital_adoption_rate}
    to: {domain: payments, metric: remittance_fee_income}
    hypothesis: Months with faster digital adoption see slower fee income growth.
    expected: negative
    owner: Payments
    if_confirmed: "Model fee income under the digital adoption target when budgeting."
    if_not_detected: "No adjustment needed."
    test: {type: timeseries, transform: growth, max_lag: 2}
```

### T5. Why these statistical choices

- **Growth, not levels.** Almost every business line grew over 2024–2026, so raw levels correlate at ~0.9 for unrelated KPIs. Month-over-month growth removes the shared trend; what remains is real co-movement.
- **Permutation p-values.** "How often would shuffled months show a link this strong?" is easy to explain to executives and makes no distribution assumptions.
- **Bonferroni adjustment.** Trying 3 lags or 66 pairs inflates false positives, so p-values are multiplied by the number of tries.
- **A placebo control.** Gender has no planted effect on credit risk. If the method ever "confirms" it, the verifier fails the briefing.
- **Associations, not causes.** The verifier rejects causal verbs. Event and cohort designs are stronger evidence than correlations, but still not experiments.

### T6. What the agents find in the synthetic data

| Link | Verdict | Evidence |
|---|---|---|
| Spending and remittances move together | confirmed | growth r = +0.98; elasticity +0.94 (90% interval +0.86 to +0.98) |
| Churned customers and credit risk | confirmed | NPF 12.1% vs 3.6% (3.3x, p < 0.001) |
| Churn rate and NPF across segments and regions | confirmed | r = +0.74 across 10 groups |
| Fee cut and revenue per remittance | confirmed | SAR 18.05 → SAR 13.59 (-24.7%) |
| Fee cut and remittance activity | not detected | YoY growth 12.8% → 14.1%, p = 0.82 |
| Customer age and digital adoption | confirmed | r = -0.99 across age bands |
| New customers and next-month activity | not detected | r = +0.09 |
| Control: gender vs credit risk | control passed | 1.3x difference, p = 0.17 (not significant) |

The two fee-cut results combined give the most "executive" insight in the briefing: **the price cut gave away about a quarter of revenue per transfer without a measurable lift in activity**, so a targeted (corridor or segment) price is worth testing instead of a blanket cut.

These results match how the data was generated (section C): churned customers have 2.5x delinquency, spending and remittances share the same seasonal weights, the digital fee changed on 2025-08-25 without changing behaviour, and gender was never used. **The agents recover planted truths and do not invent the absent ones,** which is the strongest evidence that the method works.

### T7. The verifier loop

`verifier.py` checks four things:

1. **Grounding**: every number in the summary, alerts and actions must appear in a computed fact (findings, link evidence or discoveries).
2. **Language**: no "drives / causes / leads to / due to / proves".
3. **Evidence**: anything labelled confirmed must have p < 0.05; a failed control blocks sign-off.
4. **Coverage**: every domain in scope has findings.

With the default deterministic writer, the briefing is assembled from verified sentences and passes first time. With the Claude writer (`--writer llm`, or automatically when `ANTHROPIC_API_KEY` is set), Claude rewrites only the executive summary. If the verifier rejects it, the issues are sent back for another draft (up to 2 rewrites); if it still fails, the deterministic summary is shipped instead. `tests/test_orchestrator.py::test_llm_writer_is_corrected_by_verifier` proves the loop with a fake model whose first draft invents "99.9%" and says "causes".

### T8. The Business Review page

- KPI tiles: links confirmed, risk alerts, watch items, verifier status
- Executive summary and an **impact map** (business lines as nodes; teal solid lines = confirmed, grey dotted = not detected; hover for evidence)
- Tabs: impact tests with a chart per test type (lag bars, cross-section scatter, cohort bars, before/after bars), recommended actions, risks, domain scorecards, discovery heatmap (growth correlations), agent trace (parallel agents highlighted) and the full briefing with Markdown/JSON download

### T9. Tests

`tests/test_orchestrator.py` (9 tests) and one page test check:

- the supervisor fans out to all seven agents in parallel, and the run ends synthesizer → verifier → respond
- every tool call is governed
- planted relationships are confirmed and absent ones are not
- a focused request limits both domains and links
- the verifier catches invented numbers and causal claims
- the LLM rewrite loop converges

These tests run on the small 3,000-customer test database, so they also show the findings are robust to sample size.

### T10. Interview talking points

- *"I separated analysis from narration. Specialist agents compute facts with governed tools; one agent writes; another agent audits the writing against the facts."*
- *"Cross-business links are owned hypotheses, not LLM speculation. The system tests them with the right design for each question and includes a placebo control."*
- *"The most useful output was a negative result: the fee cut did not lift volume. Good analytics tells you what didn't work."*
- *"The same pattern scales: add a domain agent and a few YAML hypotheses, and the supervisor, verifier and UI pick them up."*

### T11. Limitations and next steps

- Hypotheses are tested one at a time; a multivariate model (e.g. regression with controls) would separate overlapping effects such as segment vs churn.
- 33 months is a short history for lagged effects; production would use weekly data or a longer history.
- The supervisor uses keywords for scope; with Claude it could plan custom investigations ("why did fee income fall?").
- Next: schedule the review monthly and post the briefing to Teams or Slack, and expose `run_review` as an MCP tool.

---

## U. Training the agents

**Principle.** Nothing here fine-tunes a model. In a regulated bank that is a feature: behaviour changes only through reviewed, versioned knowledge files, so every answer remains explainable. "Training" means growing that knowledge with a human in the loop.

**Levers, from most to least frequent:**

| Lever | File | Tool |
|---|---|---|
| Synonyms (any language: "kredit macet", "wilayah") | `semantic/semantic_layer.yaml` | `scripts/teach.py synonym` |
| Value aliases ("kartu debit" → product = Mada Debit Card) | `semantic/semantic_layer.yaml` (`value_aliases`) | `scripts/teach.py alias` |
| Golden questions (lock correct behaviour) | `evals/golden_questions.yaml` | `scripts/teach.py golden` (plan-only; add `reference_sql` to check numbers) |
| New metrics | `semantic/semantic_layer.yaml` | edit YAML + golden question |
| Cross-business hypotheses | `semantic/impact_links.yaml` | edit YAML, `make review` |
| Business events | `semantic/business_calendar.yaml` | edit YAML |
| Alert thresholds | `bi_agent/orchestrator/domains.py` | code change + test |

**Feedback loop.**
1. `bi_agent/feedback.py` appends every chat question (text, status, plan) and every 👍/👎 to `data/feedback/log.jsonl`. Results are never logged.
2. `scripts/train_from_feedback.py` groups the log into four lists: unmapped questions (split into understood vs unknown words, with a ready-to-run `teach.py` command), thumbs-down answers, thumbs-up answers (golden-question candidates) and blocked questions (possible false positives).
3. `scripts/teach.py` makes the change, validates that the semantic layer still loads, refuses a synonym that already means something else, rolls back on error, and optionally re-runs the evals (`--check`).
4. `make test`, `make evals`, then commit. The git history becomes the audit trail of what the agents learned and who approved it.

**Worked example (from the test run).** Before training, "kredit macet per wilayah" returned *clarify*. The report listed the unknown words *kredit macet* and *wilayah*. Two commands (`synonym npf_ratio "kredit macet"`, `synonym region "wilayah"`) fixed it, and `golden "kredit macet per wilayah"` added a regression test. Evals stayed at 100%.

**With Claude.** New synonyms and metrics flow into the Claude planner's catalog automatically. Compare planners on the same golden set with `python evals/run_evals.py --planner rules` vs `--planner llm`.

**Cloud note.** On the free Streamlit tier the log resets when the app restarts. Run the loop locally, or (next step) send feedback to a database or a sheet.

**Tests.** `tests/test_training.py` covers the log round-trip, the report sections, synonym teaching (including refusing ambiguous words) and alias validation against real dimension values.
