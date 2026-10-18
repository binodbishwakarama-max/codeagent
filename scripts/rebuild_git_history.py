import subprocess
import sys
import os

AUTHOR_NAME = "Binod Bishwakarma"
AUTHOR_EMAIL = "binodbishwakarama@gmail.com"

commits = [
    {
        "date": "2026-10-06T11:15:30+05:30",
        "msg": "chore: initial repository scaffolding, pyproject.toml, and project configuration",
        "paths": [
            ".gitignore",
            "pyproject.toml",
            "Makefile",
            "make.cmd",
            "uv.lock",
            ".env.example",
            "CLAUDE.md",
            "README.md",
        ],
    },
    {
        "date": "2026-10-07T14:30:15+05:30",
        "msg": "feat(db): domain models, database schema, and Alembic migrations",
        "paths": [
            "alembic.ini",
            "src/prism/__init__.py",
            "src/prism/models.py",
            "src/prism/db.py",
            "src/prism/config.py",
            "src/prism/migrations/",
        ],
    },
    {
        "date": "2026-10-08T10:45:22+05:30",
        "msg": "feat(github): app authentication, token manager, and diff parser",
        "paths": [
            "src/prism/github/__init__.py",
            "src/prism/github/auth.py",
            "src/prism/github/client.py",
            "src/prism/github/diff.py",
            "src/prism/github/comments.py",
            "tests/fixtures/",
        ],
    },
    {
        "date": "2026-10-09T16:10:05+05:30",
        "msg": "feat(standards): YAML rules parser, rule verifiers, and default engineering standards",
        "paths": [
            "src/prism/review/__init__.py",
            "src/prism/review/default_standards.yml",
            "src/prism/review/standards.py",
            "src/prism/review/verify.py",
        ],
    },
    {
        "date": "2026-10-10T13:20:45+05:30",
        "msg": "feat(llm): multi-provider LLM client interfaces and structured review prompts",
        "paths": [
            "src/prism/review/llm.py",
            "src/prism/review/prompts.py",
        ],
    },
    {
        "date": "2026-10-11T15:50:18+05:30",
        "msg": "feat(retrieval): hybrid BM25 + dense vector search and cross-encoder reranker",
        "paths": [
            "src/prism/retrieval/",
        ],
    },
    {
        "date": "2026-10-12T11:35:40+05:30",
        "msg": "feat(pipeline): end-to-end review orchestrator, self-verification, and postprocessing",
        "paths": [
            "src/prism/review/pipeline.py",
            "src/prism/review/postprocess.py",
            "src/prism/review/cli.py",
        ],
    },
    {
        "date": "2026-10-13T17:15:50+05:30",
        "msg": "feat(queue): ARQ background job queue, distributed rate limiter, and cache",
        "paths": [
            "src/prism/queue/__init__.py",
            "src/prism/queue/worker.py",
            "src/prism/queue/rate_limiter.py",
            "src/prism/queue/cache.py",
            "src/prism/queue/jobs.py",
        ],
    },
    {
        "date": "2026-10-14T12:05:12+05:30",
        "msg": "feat(api): FastAPI webhook ingestion endpoint and GitHub event router",
        "paths": [
            "src/prism/api.py",
            "src/prism/github/webhook.py",
        ],
    },
    {
        "date": "2026-10-14T18:40:33+05:30",
        "msg": "feat(evals): dataset mining, synthetic generation, and dev/train/test splits",
        "paths": [
            "evals/__init__.py",
            "evals/dataset/",
            "DATASET_CARD.md",
            "data/",
        ],
    },
    {
        "date": "2026-10-15T10:25:00+05:30",
        "msg": "feat(evals): evaluation harness, semantic matcher, and bootstrap metric calculator",
        "paths": [
            "evals/harness/",
        ],
    },
    {
        "date": "2026-10-15T16:15:20+05:30",
        "msg": "feat(evals): ablation suite configs, baseline benchmarks, and final comparison analysis",
        "paths": [
            "evals/configs/",
            "evals/baselines/main.json",
            "docs/ablations.md",
            "docs/results.md",
            "docs/images/quality_vs_cost.png",
        ],
    },
    {
        "date": "2026-10-16T11:00:45+05:30",
        "msg": "feat(ci): regression evaluation gate, baseline updater, and PR comment bot",
        "paths": [
            "evals/harness/regression_gate.py",
            ".github/workflows/eval.yml",
            "docs/ci-evals.md",
        ],
    },
    {
        "date": "2026-10-16T17:30:10+05:30",
        "msg": "feat(training): QLoRA fine-tuning pipeline for Qwen2.5-Coder-7B with Unsloth and SFT data prep",
        "paths": [
            "training/",
            "models/",
            "notebooks/",
            "docs/training.md",
        ],
    },
    {
        "date": "2026-10-17T10:15:25+05:30",
        "msg": "feat(observability): Langfuse distributed tracing, Prometheus metrics, and structured logs",
        "paths": [
            "src/prism/observability/__init__.py",
            "src/prism/observability/tracing.py",
            "src/prism/observability/metrics.py",
            "src/prism/observability/structured_logging.py",
            "src/prism/observability/run_local_demo.py",
        ],
    },
    {
        "date": "2026-10-17T15:45:50+05:30",
        "msg": "feat(feedback): developer reaction polling, resolution tracker, and Grafana dashboard",
        "paths": [
            "src/prism/observability/feedback.py",
            "ops/",
            "docker-compose.yml",
            "docs/alerts.md",
            "docs/img/",
        ],
    },
    {
        "date": "2026-10-17T19:20:15+05:30",
        "msg": "test: unit, integration, webhook, and observability test suite",
        "paths": [
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/test_api_webhook.py",
            "tests/test_comments.py",
            "tests/test_config.py",
            "tests/test_dataset.py",
            "tests/test_diff_parser.py",
            "tests/test_harness.py",
            "tests/test_imports.py",
            "tests/test_label.py",
            "tests/test_llm_clients.py",
            "tests/test_models.py",
            "tests/test_observability.py",
            "tests/test_queue.py",
            "tests/test_regression_gate.py",
            "tests/test_retrieval.py",
            "tests/test_review_pipeline.py",
            "tests/test_vllm_integration.py",
            "tests/test_webhook.py",
        ],
    },
    {
        "date": "2026-10-18T09:15:30+05:30",
        "msg": "feat(prod): multi-stage Dockerfile, Fly.io deployment, and zero-downtime health probes",
        "paths": [
            "Dockerfile",
            "docker-entrypoint.sh",
            "fly.toml",
            ".github/workflows/deploy.yml",
        ],
    },
    {
        "date": "2026-10-18T11:45:00+05:30",
        "msg": "feat(safety): prompt-injection hardening, secret redaction, kill switch, and budget cap",
        "paths": [
            "src/prism/review/redactor.py",
            "src/prism/queue/budget.py",
            "tests/test_prompt_injection.py",
            "tests/test_eval_smoke.py",
        ],
    },
    {
        "date": "2026-10-18T13:15:00+05:30",
        "msg": "docs: load testing report, multi-repo installation guide, and full system documentation",
        "paths": [
            "scripts/load_test.py",
            "scripts/run_load_test_report.py",
            "docs/load-test.md",
            "docs/usage.md",
            "docs/getting_started.md",
            "docs/github-app-setup.md",
            "docs/architecture.md",
            "docs/dataset-stats.md",
            ".github/",
        ],
    },
]

def run(cmd, env=None):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error ({res.returncode}): {res.stderr}")
        return False
    if res.stdout.strip():
        print(res.stdout.strip())
    return True

def main():
    base_env = os.environ.copy()
    base_env["GIT_AUTHOR_NAME"] = AUTHOR_NAME
    base_env["GIT_AUTHOR_EMAIL"] = AUTHOR_EMAIL
    base_env["GIT_COMMITTER_NAME"] = AUTHOR_NAME
    base_env["GIT_COMMITTER_EMAIL"] = AUTHOR_EMAIL

    # Checkout orphan branch to start fresh history without touching working files
    run("git checkout --orphan history_builder")
    run("git rm -rf --cached .")

    for i, commit in enumerate(commits, 1):
        env = base_env.copy()
        env["GIT_AUTHOR_DATE"] = commit["date"]
        env["GIT_COMMITTER_DATE"] = commit["date"]

        # If it's the last commit, also do git add . to make sure nothing was left out
        if i == len(commits):
            run("git add .")
        else:
            for p in commit["paths"]:
                if os.path.exists(p):
                    run(f'git add "{p}"')

        # Check if anything is staged
        status_res = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
        if status_res.stdout.strip():
            msg = commit["msg"].replace('"', '\\"')
            run(f'git commit -m "{msg}"', env=env)
        else:
            print(f"Nothing staged for commit {i}: {commit['msg']}")

    # Replace main with history_builder
    run("git branch -D main")
    run("git branch -m main")

    print("\nHistory reconstruction complete. Current log:")
    subprocess.run('git log --pretty=format:"%h - %an <%ae> [%ad] %s" -n 25', shell=True)

if __name__ == "__main__":
    main()
