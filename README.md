# prism-review

Prism Review is a GitHub App that reviews pull requests against custom engineering standards and posts inline comments, coupled with a rigorous evaluation harness for measuring review quality and calibration.

## Project Structure

```text
prism-review/
  CLAUDE.md                 (Operational rules and engineering principles)
  pyproject.toml            (uv, ruff, mypy strict, pytest)
  Makefile                  (Build & development targets)
  docker-compose.yml        (PostgreSQL+pgvector, Redis, optional Langfuse)
  .env.example              (Environment variable template)
  src/prism/
    config.py               (pydantic-settings configuration)
    models.py               (Pydantic domain models)
    github/                 (App auth, webhook routing, diff parsing, comment posting)
    review/                 (Pipeline, prompts, standards, LLM clients, postprocessing)
    retrieval/              (Vector indexer, search, reranker)
    queue/                  (Arq worker and background review jobs)
    observability/          (Distributed tracing and metrics)
    api.py                  (FastAPI service entrypoint)
  evals/
    dataset/                (PR mining, labeling, dataset schemas)
    harness/                (Evaluation runner, metrics, reporting)
    configs/                (Experiment YAML configurations)
  training/                 (Data prep, fine-tuning scripts, weight merging, serving)
  tests/                    (Unit tests, import tests, eval smoke tests)
  docs/                     (Architecture and getting started documentation)
```

## Quick Start

1. Install dependencies:
   ```bash
   make install
   ```

2. Run code quality checks:
   ```bash
   make lint
   make typecheck
   make test
   make eval-smoke
   ```

3. Start services via Docker Compose:
   ```bash
   docker compose up -d
   ```
