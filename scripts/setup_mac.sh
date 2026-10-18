#!/usr/bin/env bash
# One-command setup for macOS (also works on Linux).
#
#   bash scripts/setup_mac.sh          # install, build data, run all checks
#   bash scripts/setup_mac.sh --run    # ... and open the app
#
# Needs Python 3.11+ (python.org installer) and an internet connection the first time only.
set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.12 from https://www.python.org/downloads/ and run this again."; exit 1
fi
"$PY" - <<'EOF' || { echo "Python 3.11 or newer is required (you have $($PY --version))."; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)
EOF

say "1/6 Virtual environment (.venv)"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip -q

say "2/6 Libraries (needs internet the first time)"
pip install -r requirements.txt -q

say "3/6 Synthetic bank data"
if [ ! -f data/wadi_bank.duckdb ] || [ "${1:-}" = "--fresh" ]; then
  python scripts/generate_data.py
else
  echo "data/wadi_bank.duckdb already exists (use --fresh to rebuild)"
fi

export BI_AGENT_PLANNER=${BI_AGENT_PLANNER:-rules}

say "4/6 Tests"
python -m pytest -q

say "5/6 Golden-question evals"
python evals/run_evals.py --min-score 0.9 | tail -8

say "6/6 Multi-agent business review"
python -m bi_agent.orchestrator > /dev/null
echo "Briefing written to reports/business_review.md"

cat <<'EOF'

All set. Everything below works offline from now on:

  source .venv/bin/activate
  streamlit run app/streamlit_app.py            # chat app + Business Review page
  python -m bi_agent.cli                         # chat in the terminal
  python -m bi_agent.orchestrator                # whole-business review
  python scripts/train_from_feedback.py          # what to teach the agents next

EOF

if [ "${1:-}" = "--run" ] || [ "${2:-}" = "--run" ]; then
  exec streamlit run app/streamlit_app.py --browser.gatherUsageStats false
fi
