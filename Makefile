.PHONY: setup setup-mac data test evals app ask review ds train publish all

setup:
	pip install -r requirements.txt

data:
	python scripts/generate_data.py

test:
	BI_AGENT_PLANNER=rules python -m pytest -q

evals:
	BI_AGENT_PLANNER=rules python evals/run_evals.py --min-score 0.9

app:
	streamlit run app/streamlit_app.py

ask:
	python -m bi_agent.cli --show-sql "$(q)"

setup-mac:
	bash scripts/setup_mac.sh

train:
	python scripts/train_from_feedback.py

publish:
	bash scripts/publish_github.sh $(user)

ds:
	python -m bi_agent.ds validate --strict

review:
	python -m bi_agent.orchestrator --out reports/business_review.md

all: data test evals ds review
