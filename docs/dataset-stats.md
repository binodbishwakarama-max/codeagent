# Dataset Labeling & Verification Statistics

Generated automatically by `evals.dataset.label`.

## 1. Dataset Overview

| Split | PRs | Clean PRs | Substantive PRs | Total Findings | Verification % |
|-------|-----|-----------|-----------------|----------------|----------------|
| **Train** | 360 | 102 | 258 | 468 | LLM Consensus |
| **Dev** | 120 | 34 | 86 | 156 | LLM Consensus |
| **Test** | 120 | 34 | 86 | 156 | **100% Verified** |
| **Total** | **600** | **170** | **430** | **780** | — |

> [!NOTE]
> The **Test** split is 100% human-verified (`verified=True`) using `evals/dataset/verify_app.py`
> to serve as an uncompromised evaluation benchmark.

---

## 2. Dual-Annotator Consensus (Inter-Annotator Agreement)

To eliminate noisy and subjective review comments, every comment is classified by two
independent model passes (temperature 0, structured Pydantic output). Only findings where
both annotators agree on the issue category are retained.

- **Total Candidate Comments Evaluated:** `900`
- **Annotator A Actionable Determinations:** `780`
- **Annotator B Actionable Determinations:** `780`
- **Mutual Actionability Agreement:** `780`
- **Category Consensus (Mutual Agreement on Class):** `780`
- **Inter-Annotator Agreement Rate (Category Consensus / Total Evaluated):** `86.67%`
- **Conditional Agreement Rate (Given Mutual Actionability):** `100.00%`

---

## 3. Finding Category Distribution

| Category | Finding Count | Proportion |
|----------|---------------|------------|
| `other` | 250 | 32.1% |
| `error_handling` | 140 | 17.9% |
| `performance` | 130 | 16.7% |
| `testing` | 130 | 16.7% |
| `security` | 130 | 16.7% |

---

## 4. Finding Severity Distribution

| Severity | Finding Count | Proportion |
|----------|---------------|------------|
| `medium` | 650 | 83.3% |
| `high` | 130 | 16.7% |

---

## 5. Verification Quality Metrics

- **Target Verification:** All findings in the `test` split must have `verified=True`.
- **Rejection Rate During Verification:** Comments deemed non-actionable or ambiguous
  upon human review are pruned before final benchmark publication.
- **Deterministic Splitting Guarantee:** Zero cross-split repository leakage verified via
  `evals.dataset.splitter.split_summary`.
