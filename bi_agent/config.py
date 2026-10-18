"""Central configuration. Everything can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DB_PATH = Path(os.getenv("BI_AGENT_DB", ROOT / "data" / "wadi_bank.duckdb"))
SEMANTIC_PATH = Path(os.getenv("BI_AGENT_SEMANTIC", ROOT / "semantic" / "semantic_layer.yaml"))

# "auto"  -> use Claude when ANTHROPIC_API_KEY is set, otherwise the offline rule-based planner
# "llm"   -> always use Claude (fails if no key)
# "rules" -> always use the offline planner (deterministic; used in CI and evals)
PLANNER = os.getenv("BI_AGENT_PLANNER", "auto").lower()
LLM_MODEL = os.getenv("BI_AGENT_MODEL", "claude-sonnet-5-5")
MAX_PLAN_ATTEMPTS = int(os.getenv("BI_AGENT_MAX_ATTEMPTS", "3"))


def use_llm() -> bool:
    if PLANNER == "rules":
        return False
    if PLANNER == "llm":
        return True
    return bool(os.getenv("ANTHROPIC_API_KEY"))
