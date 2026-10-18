---
language:
- en
license: apache-2.0
task_categories:
- text-generation
- evaluation
tags:
- code-review
- software-engineering
- github
- pull-requests
- llm-benchmarking
- qwen2.5-coder
pretty_name: Prism PR Review Benchmark
size_categories:
- n<1K
---

# DATASET_CARD.md — Prism Review Evaluation Dataset

## Summary

**Name:** Prism PR Review Benchmark  
**Version:** 0.1.0  
**Task:** Code Review Quality Evaluation  
**Format:** JSONL (`data/raw/prs.jsonl`)  
**Schema:** [`evals/dataset/schema.py`](evals/dataset/schema.py)  
**Miner:** [`evals/dataset/miner.py`](evals/dataset/miner.py)  
**Regenerate:** `make dataset-mine` (requires `GITHUB_TOKEN`)

---

## Intended Use

This dataset is intended **solely** for evaluating the Prism automated code-review
system (not for fine-tuning). It measures:

- **Precision / recall** of the review model against human-identified findings.
- **False-positive rate** on the `clean` split — PRs where humans left zero actionable
  comments despite substantive diffs.
- **Calibration** of confidence scores.

> [!IMPORTANT]
> Do **not** use this dataset to train or fine-tune any model. The test split would
> be contaminated. For training data, use the separate fine-tuning pipeline.

---

## Sources

All PRs are mined from publicly available, merged GitHub pull requests.

| # | Repo | Language | License | Notes |
|---|------|----------|---------|-------|
| 1 | tiangolo/fastapi | Python | MIT | REST framework |
| 2 | pydantic/pydantic | Python | MIT | Validation library |
| 3 | encode/httpx | Python | BSD-3-Clause | HTTP client |
| 4 | pallets/flask | Python | BSD-3-Clause | Web framework |
| 5 | psf/black | Python | MIT | Formatter |
| 6 | celery/celery | Python | BSD-3-Clause | Task queue |
| 7 | sqlalchemy/sqlalchemy | Python | MIT | ORM |
| 8 | pytest-dev/pytest | Python | MIT | Test framework |
| 9 | python-attrs/attrs | Python | MIT | Data classes |
| 10 | encode/starlette | Python | BSD-3-Clause | ASGI toolkit |
| 11 | aio-libs/aiohttp | Python | Apache-2.0 | Async HTTP |
| 12 | huggingface/transformers | Python | Apache-2.0 | ML library |
| 13 | langchain-ai/langchain | Python | MIT | LLM orchestration |
| 14 | prefecthq/prefect | Python | Apache-2.0 | Workflow orchestration |
| 15 | pola-rs/polars | Python | MIT | DataFrame library |
| 16 | microsoft/TypeScript | TypeScript | Apache-2.0 | TS compiler |
| 17 | microsoft/vscode | TypeScript | MIT | Code editor |
| 18 | trpc/trpc | TypeScript | MIT | Type-safe RPC |
| 19 | colinhacks/zod | TypeScript | MIT | Schema validation |
| 20 | prisma/prisma | TypeScript | Apache-2.0 | ORM |
| 21 | vercel/next.js | TypeScript | MIT | React framework |
| 22 | vitejs/vite | TypeScript | MIT | Build tool |
| 23 | biomejs/biome | TypeScript | MIT | Linter/formatter |
| 24 | effect-ts/effect | TypeScript | MIT | Functional TS |

All licenses verified at mine time against the GitHub API `spdx_id` field.
Only `MIT`, `Apache-2.0`, `BSD-2-Clause`, and `BSD-3-Clause` are accepted.

---

## Data Schema

```
PRExample
├── id              str       sha256(repo + '#' + str(pr_number))[:16]
├── repo            str       "owner/name"
├── language        str       "python" | "typescript"
├── license         str       SPDX identifier
├── pr_number       int
├── pr_url          str
├── base_sha        str       merge-target commit SHA
├── head_sha        str       reviewed commit SHA
├── title           str
├── body            str       PR description
├── diff            str       unified diff text
├── changed_lines   int       added + removed lines
├── gold_findings   list[GoldFinding]
├── is_clean        bool      True ⟹ zero actionable comments (FP baseline)
└── split           "train" | "dev" | "test" | None

GoldFinding
├── path            str       relative file path
├── line            int       line number (new-file side)
├── category        enum      bug | security | performance | error_handling |
│                             style | testing | readability | other
├── severity        enum      low | medium | high | unknown
├── description     str       verbatim reviewer comment body
├── source          enum      human_comment | synthetic
├── verified        bool      True if a follow-up commit touched this file
├── reviewer        str|None  GitHub login
└── thread_resolved bool
```

---

## Filtering Heuristics

### PR inclusion criteria
| Criterion | Value |
|-----------|-------|
| Changed lines | 20 – 600 (inclusive) |
| PR state | merged only |
| Author | not a bot (dependabot, renovate, github-actions, etc.) |
| Title | not a release PR ("Release v…", "Bump version", etc.) |

### Comment substantiveness
A review comment is included only if:
1. Author is **not** a known bot.
2. Word count is **≥ 12** words.
3. Body does **not** match LGTM/praise patterns: `LGTM`, `looks good`, `👍`, `nice!`, `great work`, etc.

### Clean PRs (false-positive baseline)
PRs with `changed_lines ∈ [20, 600]` and **zero** substantive inline review comments
are collected as `is_clean=True` examples (target: 60). They are excluded from splits.

---

## Deterministic Splitting

Splitting is **by repo, not by PR**. This prevents data leakage where training PRs
and test PRs originate from the same codebase (sharing idioms, naming conventions, etc.).

Algorithm:
1. Collect the set of unique repos from substantive examples.
2. Sort alphabetically (deterministic, no randomness).
3. Assign repos to splits: first 60% → `train`, next 20% → `dev`, last 20% → `test`.
4. Tag every PRExample with its repo's split.

**Assertion:** `split_summary()` raises `AssertionError` if any repo appears in two
different splits. This is also tested in `tests/test_dataset.py`.

---

## Known Biases

1. **Popular repo bias**: Repos with many stars tend to be over-represented. We cap
   PRs per repo at `⌈target / n_repos⌉ + 5` to partially mitigate this.
2. **Reviewer style**: Comments extracted from a small set of repos reflect the review
   norms of those communities, not universal best practices.
3. **Language imbalance**: ~60 % Python repos, ~40 % TypeScript. Evaluate per-language
   precision/recall separately.
4. **Heuristic severity labeling**: `severity` and `category` fields are inferred by
   regex heuristics (see `labeling.py`). They should be treated as approximate; human
   curators should verify before using in production eval.
5. **Temporal drift**: Mining fetches recent PRs. Standards and review culture evolve;
   re-mine periodically.
6. **Closed-thread bias**: `verified=True` uses a file-level heuristic (was the
   commented file touched in a subsequent commit?). Line-level verification would
   require git blame diffing and is left for future work.

---

## Regenerating the Dataset

```bash
export GITHUB_TOKEN=ghp_...
make dataset-mine
```

This will:
1. Verify licenses for all configured repos against the GitHub API.
2. Mine merged PRs (up to 10 pages per repo).
3. Filter by changed-lines range and comment substantiveness.
4. Assign deterministic train/dev/test splits by repo.
5. Write `data/raw/prs.jsonl`.
6. Print a summary table (PRs per repo, per language, per split).
7. Run `tests/test_dataset.py` — including the zero-overlap assertion.

---

## License

The PR metadata (title, body, diff, review comments) is sourced from GitHub's public
API. Each PR is covered by the license of its source repository (listed above).
All source repositories use OSI-approved permissive licenses (MIT, Apache-2.0, BSD).

The dataset itself (schema, labeling, and curation code) is © Prism Review contributors,
Apache-2.0.

> [!WARNING]
> Do not redistribute the full diff text without verifying your use case complies with
> each repository's license. For academic evaluation only.
