# CI Regression Evaluation Gate

Prism Review uses an automated regression evaluation gate in GitHub Actions (`.github/workflows/eval.yml`) to ensure that changes to review prompts, retrieval pipelines, and evaluation configurations never degrade code review quality or inflate false positives.

---

## 1. Trigger Conditions & Scope

The gate executes automatically on any Pull Request modifying files matching:
- `src/prism/review/**` (prompt templates, pipeline orchestrator, post-processing filters, verification pass)
- `src/prism/retrieval/**` (AST indexer, hybrid search, cross-encoder reranker)
- `evals/configs/**` (evaluation and ablation configurations)

### Benchmark Dataset & Cost Cap
- **Regression Set**: A fixed, deterministic subset of **40 pull requests** from `data/final/dev.jsonl` (sorted deterministically by ID with fixed seed `42`).
- **Cost Cap**: The evaluation harness caps total execution cost under **$1.00** per run (typically ~$0.04 using cached embeddings and optimized inference).
- **Caching**: Matcher embeddings, semantic judge verdicts, and pipeline chunk reviews are cached in `.cache/matcher` across CI runs via GitHub Actions cache.

---

## 2. Statistical Methodology & Gate Thresholds

The candidate PR metrics are evaluated against the current stored baseline on `main` (`evals/baselines/main.json`).

Because evaluation samples are finite, point estimates are accompanied by **paired bootstrap significance testing** (1,000 resamples across the paired PR examples):

| Gate Check | Threshold Rule | Rationale |
| :--- | :--- | :--- |
| **F1 Score** | Drop $\le$ 3.0 points ($\Delta F1 \ge -0.03$) | Prevents degradation in balanced review quality (harmonic mean of precision and recall). |
| **False-Positive Rate (clean PRs)** | Rise $\le$ 2.0 points beyond CI noise ($\Delta FPR \le +0.02$) | Protects developer trust; false positives on clean PRs generate developer fatigue. |
| **Total Run Cost** | Total cost $< \$1.00$ | Hard cost cap preventing runaway billing in CI. |

### CI Noise & Paired Bootstrap Testing
Point fluctuations can occur due to stochasticity. The gate computes a **paired bootstrap difference distribution**:
$$\Delta_i = \text{Metric}(\text{Candidate}_i) - \text{Metric}(\text{Baseline}_i)$$
For FPR, the gate evaluates whether the increase exceeds 0.02 *and* is statistically distinguishable from noise ($p < 0.05$ or lower bound of the paired 95% CI $> 0$).

---

## 3. Sticky PR Comment

Upon completion, GitHub Actions creates or updates a single **sticky comment** on the pull request (identified by `<!-- prism-eval-regression-gate -->`). Subsequent pushes to the PR update this comment in place rather than spamming the discussion.

The comment includes:
- **Status Banner**: `PASSED ✅`, `FAILED ❌`, or `OVERRIDDEN ⚠️`.
- **Full Metrics Delta Table**: Baseline, Candidate, $\Delta$, 95% Confidence Interval for $\Delta$, bootstrap $p$-value, and status.
- **Cost & Latency Deltas**: Cost per review delta ($\Delta \text{Cost}$) and median latency delta ($\Delta \text{Latency p50}$).
- **Failure Breakdown**: Specific causes and remediation steps if any threshold is violated.

---

## 4. Deliberately Accepting a Regression (`eval-override`)

In rare cases, a PR may deliberately introduce a regression (for example, intentionally dropping stylistic nitpicks to improve developer experience, which may slightly reduce raw recall, or tightening safety constraints).

To merge a PR that triggers a regression, the author must explicitly override the gate:

### Step 1: Add the PR Label
Add the GitHub label **`eval-override`** to the Pull Request.

### Step 2: Document Justification in PR Description
The PR body **must** include an explicit justification section explaining the architectural or business reason for accepting the degradation:

```markdown
## Eval Override Justification
We are intentionally removing stylistic and readability checks from the default prompt
to reduce review noise on junior engineer PRs. This reduces F1 by 4.2 points on the dev set,
but lowers comments-per-clean-PR from 0.36 to 0.12, which aligns with our Q4 noise reduction goal.
```

### Strict Validation Rule
If the `eval-override` label is applied but the PR body lacks a substantive justification (minimum 20 characters of explanation), the gate will **fail** with:
```
FAILED ❌ (Override Incomplete)
> [!WARNING]
> Override Rejected: The PR is labeled with eval-override, but lacks a valid justification in the PR body.
```

---

## 5. Refreshing the Stored Baseline (`make update-baseline`)

When changes are merged into `main`, the baseline metrics must be updated so future PRs evaluate against the new standard.

Run:
```bash
make update-baseline
```

### Safety Guard
The target verifies your current branch using `git rev-parse --abbrev-ref HEAD`. If executed on any branch other than `main` (or `master`), it safely aborts:
```
Error: update-baseline can only be executed on the 'main' branch (current: feature/my-branch)
```
To refresh the baseline during local testing or CI simulation, use:
```bash
uv run python -m evals.harness.regression_gate --update-baseline --force
```
