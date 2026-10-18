.PHONY: install test lint typecheck run worker eval-smoke eval-full eval-compare ablations format dataset-mine dataset-label dataset-verify reproduce

install:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run python -m mypy src tests evals

test:
	uv run pytest tests/

run:
	uv run uvicorn prism.api:app --reload --host 0.0.0.0 --port 8000

worker:
	uv run arq prism.queue.worker.WorkerSettings

eval-smoke:
	uv run pytest tests/test_eval_smoke.py

eval-full:
	uv run python -m evals.harness.runner --config evals/configs/baseline.yaml

eval-compare:
	@if [ -z "$(RUN_A)" ] || [ -z "$(RUN_B)" ]; then echo "Usage: make eval-compare RUN_A=runs/<a> RUN_B=runs/<b>"; exit 1; fi
	uv run python -m evals.harness.compare --run-a $(RUN_A) --run-b $(RUN_B)

ablations:
	uv run python -m evals.harness.ablations --configs-dir evals/configs/ablations --split dev --output docs/ablations.md

dataset-mine:
	uv run python -m evals.dataset.miner \
		--config evals/configs/repos.yml \
		--out data/raw/prs.jsonl \
	uv run pytest tests/test_dataset.py -v

dataset-label:
	uv run python -m evals.dataset.label \
		--input data/raw/prs.jsonl \
		--output-dir data/final \
		--stats-out docs/dataset-stats.md

dataset-verify:
	uv run streamlit run evals/dataset/verify_app.py -- --file data/final/test.jsonl

update-baseline:
	@CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"); \
	if [ "$$CURRENT_BRANCH" != "main" ] && [ "$$CURRENT_BRANCH" != "master" ]; then \
		echo "Error: update-baseline can only be executed on the 'main' branch (current: $$CURRENT_BRANCH)"; \
		exit 1; \
	fi
	uv run python -m evals.harness.regression_gate --update-baseline

eval-regression:
	uv run python -m evals.harness.regression_gate --output-comment comment.md

train-prep:
	uv run python -m training.data_prep --distill

train-run:
	uv run python -m training.run_experiments

serve-mock:
	uv run python -m training.serve --mock

serve-cost:
	uv run python -m training.serve --cost-comparison

eval-final:
	uv run python -m evals.harness.final_comparison

reproduce:
	uv run python -m evals.harness.final_comparison

