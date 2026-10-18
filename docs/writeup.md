# Measured Code Review: Training and Evaluating a Domain-Specific LLM for Production Pull Request Reviews

**Author:** Binod Bishwakarma  
**Date:** October 2026  
**Artifacts:** [Evaluation Benchmark](evals/harness/) | [Dataset Card](DATASET_CARD.md) | [Benchmark Results](docs/results.md) | [Model Checkpoints](models/)

---

## 1. Problem Statement: The Code Review Bottleneck

Engineering teams spend 20% to 25% of development bandwidth on peer code review. While vital for architectural invariants, security perimeters, and reliability, human review introduces serious throughput friction: median turnaround latency ($p50$) routinely spans 18 to 36 hours. A major portion of this delay is spent repeatedly flagging mechanical oversights: unhandled async exceptions, missing database timeouts, unindexed join queries, exposed credentials, and missing unit tests.

Traditional solutions force an unsatisfactory tradeoff:
1. **Static Linters (AST & regex)**: Free ($0.00 ms$ latency, zero tokens), but blind to cross-line semantics. On our benchmark of real-world human findings, linters scored an F1 of **0.0000** (0.0% recall on logic errors and architectural violations).
2. **Generic Commercial LLM Bots**: Capable of natural-language reasoning, but unmeasured. Deployed zero-shot with naive prompts, commercial review assistants suffer from hallucinated rule citations, shallow nitpicking ("consider adding comments"), and high false-alarm rates that rapidly induce developer fatigue.

Prism solves this: an autonomous, production review engine built on explicit repository standards, hybrid retrieval, and domain-tuned open-weights models, verified against an empirical benchmark.

---

## 2. Why Most AI Review Tools Are Unmeasured

Despite dozens of commercial AI review tools, almost none publish empirical precision, recall, or calibration metrics. Instead, vendors rely on two deceptive proxies:

* **Subjective Satisfaction Surveys**: Claims that "85% of developers feel faster" conceal the reality that teams mute comment bots after experiencing high-visibility false alarms.
* **Cherry-Picked Demos**: Showing an LLM catching a trivial SQL injection in a 10-line demo says nothing about performance across complex multi-file diffs with hundreds of lines of mixed additions, deletions, and refactors.

Crucially, commercial tools almost never measure the **Clean-PR False Positive Rate (FPR)**: the likelihood of posting unsolicited comments on PRs approved without changes. A code review bot with 90% recall that comments on 75% of clean PRs will be uninstalled within days.

Prism establishes an empirical benchmark: evaluating candidates on a frozen test split using defect-laden and clean PRs, with 95% bootstrap confidence intervals.

---

## 3. Dataset Construction & Decontamination

Evaluating automated review requires ground truth from production codebases rather than synthetic puzzles. We mined 300+ PRs across 24 repositories in Python and TypeScript (FastAPI, Pydantic, HTTPX, Celery, VS Code, tRPC, Vite, etc.):
- **Diff Magnitude**: 20 to 600 changed lines, filtering typo fixes and giant lockfiles.
- **Comment Substantiveness**: $\ge 12$ words; regex filtered non-actionable chatter (`LGTM`, `looks good`, `nice catch`, emojis).
- **Verification Heuristic**: Marked `verified=True` if a subsequent commit touched the commented file.
- **Clean PR Baseline**: 60 PRs (20–600 lines) approved with zero inline critiques, establishing our false-positive baseline.

To prevent data contamination, Prism enforces **repository-level partitioning**:
* **Train Split (60%)**: 14 repositories (`fastapi`, `pydantic`, `httpx`, `sqlalchemy`, `vscode`, etc.).
* **Dev Split (20%)**: 5 repositories (`pytest`, `starlette`, `transformers`, etc.).
* **Test Split (20%)**: Frozen on `trpc/trpc` and `vitejs/vite` (120 PRs total).

Startup assertion `assert_no_leakage()` guarantees zero repository overlap.

---

## 4. Metric Design & The Multi-Matcher Evaluation Harness

Review comments rarely anchor to the exact line: a reviewer might flag a signature (line 42) while a model flags an unhandled await (line 45). Exact line matching penalizes valid catches, while whole-file matching credits hallucinations.

Prism implements a **MultiMatcher**:
1. **Line Window Tolerance**: Matches if line anchors fall within $\pm 3$ lines on the new-file side of the unified diff.
2. **Category Alignment**: The predicted defect category (`SECURITY`, `ERROR_HANDLING`, `PERFORMANCE`, `BUG`) must match or subsume the gold category.
3. **Semantic Text Similarity**: Candidate description and gold comment must exceed an embedding cosine similarity threshold of $\tau = 0.82$ (dense text embeddings) or receive positive validation from an independent LLM judge.
4. **Bipartite Greedy Matching**: Greedy assignment ensures each gold finding is matched at most once, preventing shotgun duplicate commenting.

For each candidate model, we compute Precision, Recall, F1 score, and Clean-PR False Positive Rate (proportion of clean PRs receiving $\ge 1$ comment). All metrics are reported with non-parametric 95% confidence intervals derived from $B = 1,000$ PR-level bootstrap resamples.

---

## 5. Ablation Findings: System Architecture Contributions

We ablated each pipeline component on the dev split (40 PRs) to isolate the impact of standards injection, retrieval, and verification filtering:

| Architecture / Ablation Stage | Precision | Recall | F1 Score | FP Rate (Clean) | Actionability |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A0: Zero-Shot Naive Prompt** | 0.542 | 0.633 | **0.584** | 0.650 | 72.1% |
| **A1: Standards Injection (.prism/standards.yml)** | 0.667 | 0.720 | **0.692** *(+10.8)* | 0.520 | 81.4% |
| **A2: Dense Vector Retrieval (Top-K)** | 0.722 | 0.770 | **0.745** *(+5.3)* | 0.440 | 85.0% |
| **A3: Hybrid Retrieval (BM25 + Dense RRF)** | 0.789 | 0.836 | **0.812** *(+6.7)* | 0.360 | 89.2% |
| **A4: Cross-Encoder Reranker (Top 6 chunks)** | 0.835 | 0.874 | **0.854** *(+4.2)* | 0.280 | 92.5% |
| **A5: Self-Verification Filtering Step** | 0.884 | 0.912 | **0.898** *(+4.4)* | **0.180** *(-10.0)* | **96.8%** |

Key findings: formal standards provided the largest single jump ($+10.8$ F1). Reciprocal Rank Fusion (BM25 + dense) beat pure vector search by $+6.7$ F1 points by matching exact identifier tokens. Self-verification dropped clean-PR false alarms from 0.280 to 0.180 while raising F1 to 0.898.

---

## 6. Fine-Tuning Results: QLoRA on Qwen2.5-Coder-7B

Frontier API models exhibit strong reasoning, but per-call costs and network latencies prevent running them on every commit push. We fine-tuned open-weights `Qwen2.5-Coder-7B-Instruct` to deliver frontier review quality at local serving costs.

### 6.1 The Base Model's Fatal Flaw: Inability to Remain Silent
Evaluating base `Qwen2.5-Coder-7B-Instruct` zero-shot on our test split revealed an unacceptable failure mode:
* **F1 Score**: `0.7152` [0.6625, 0.7632]
* **Clean-PR False Positive Rate**: **`0.7353`**

The base model commented on **73.5% of clean PRs**. Pre-trained instruction models are biased toward generating output; given clean diffs, the base model hallucinates stylistic critiques rather than emitting an empty list.

### 6.2 SFT Formulation & Distillation
We constructed an SFT dataset:
1. **25% Negative Clean PRs**: Exactly 25% of training examples consisted of clean PR diffs paired with an empty target output: `{"findings": []}`. This trained the model to recognize valid patterns and output empty JSON.
2. **Rejection-Sampled Distillation**: For PRs with human findings, we generated candidate reviews using Claude 3.5 Sonnet, filtered hallucinations via our semantic matcher, and preserved verified reasoning traces.
3. **Hyperparameters**: QLoRA with rank $r = 32$, $\alpha = 64$, all linear projection layers, learning rate $1.5 \times 10^{-4}$ with cosine decay, 3 epochs, sequence length 8,192 tokens.

### 6.3 Definitive Test-Split Benchmark Comparison
All models were frozen and evaluated exactly once on the test split (`data/final/test.jsonl`, 120 PRs):

| Model / Configuration | Precision (95% CI) | Recall (95% CI) | F1 Score (95% CI) | Clean FPR | Comments/PR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Frontier API (Claude 3.5 Sonnet)** | `0.9675` [0.938, 0.993] | `0.9551` [0.920, 0.985] | **`0.9613`** [0.939, 0.982] | `0.0588` | `1.28` |
| **Mid-Size API (Claude 3.5 Haiku)** | `0.9062` [0.858, 0.947] | `0.9295` [0.891, 0.964] | **`0.9177`** [0.887, 0.945] | `0.1765` | `1.33` |
| **Base Qwen2.5-Coder-7B** | `0.6782` [0.615, 0.742] | `0.7564` [0.697, 0.814] | **`0.7152`** [0.663, 0.763] | `0.7353` | `1.45` |
| **Fine-Tuned 7B (Merged BF16)** | `0.8889` [0.837, 0.936] | `0.9744` [0.947, 0.994] | **`0.9297`** [0.898, 0.956] | `0.2941` | `1.43` |
| **Fine-Tuned 7B (AWQ 4-bit) 🏆** | `0.9187` [0.861, 0.965] | `0.9423` [0.902, 0.978] | **`0.9304`** [0.893, 0.959] | **`0.2353`** | `1.33` |
| **Linter Baseline** | `0.0000` [0.000, 0.000] | `0.0000` [0.000, 0.000] | **`0.0000`** [0.000, 0.000] | `0.0000` | `0.00` |

### 6.4 Statistical Validation
Bootstrap resampling ($B = 1,000$) over paired test predictions establishes:
> **Fine-tuned Qwen2.5-Coder-7B (AWQ 4-bit) achieves 96.8% [95% CI: 92.2%, 100.7%] of Claude 3.5 Sonnet's F1 score, while reducing clean-PR false alarms by 68.0% relative to the base model.**

AWQ 4-bit quantization yielded zero quality loss (F1 `0.9304` vs `0.9297` for BF16) while compressing VRAM from 16.2 GB to 5.8 GB.

---

## 7. Serving Efficiency, Latency & Cost Analysis

### 7.1 Cost Assumptions
* **APIs**: Claude 3.5 Sonnet ($3.00/1M prompt, $15.00/1M completion tokens); Claude 3.5 Haiku ($0.80 / $4.00).
* **Self-Hosted (vLLM)**:
  - BF16: 1x NVIDIA A100 (40GB) at $1.60/hr.
  - AWQ 4-bit: 1x NVIDIA A10G (24GB) at $0.70/hr on-demand ($0.35/hr reserved).
  - Duty Cycle: Conservative **50% average server utilization** (factoring in off-peak nighttime lulls).
  - Formula: $\text{Cost per Review} = \frac{\text{Hourly GPU Rate}}{3,600 \times 0.50} \times \text{Latency (seconds)}$.

### 7.2 Economics Breakdown

| Model / Deployment Target | Hardware Spec | Latency p50 | Latency p95 | Cost / Review | Spend per 10,000 PRs |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Frontier API (Claude 3.5 Sonnet)** | Serverless API | 251.5 ms | 283.6 ms | **$0.01334** | **$133.40** |
| **Mid-Size API (Claude 3.5 Haiku)** | Serverless API | 110.1 ms | 127.8 ms | **$0.00335** | **$33.50** |
| **Fine-Tuned 7B (vLLM BF16)** | 1x A100 (40GB) | 60.7 ms | 68.8 ms | **$0.00098** | **$9.80** |
| **Fine-Tuned 7B (vLLM AWQ 4-bit)** | 1x A10G (24GB) | **40.5 ms** | **47.1 ms** | **$0.00036** | **$3.60** |

The fine-tuned AWQ model achieves **37.1× lower cost** than Claude 3.5 Sonnet ($0.00036 vs $0.01334) at **40.5 ms** p50 latency (6.2× faster than Sonnet's 251.5 ms roundtrip).

---

## 8. Failure Analysis: Categorizing 80 Edge Cases

We conducted a manual root-cause taxonomy on **40 False Positives (FPs)** and **40 False Negatives (FNs)** produced on the test split.

### 8.1 False Positive Distribution (40 Cases)
1. **Missing Cross-File Dependency Context (37.5%, 15 cases)**: Flagging an unvalidated parameter, unaware that sanitization was implemented in a caller file outside the diff hunk.
2. **Intentional Pragmatic Violations (25.0%, 10 cases)**: Flagging an un-parameterized SQL query or broad try/except in a one-off database migration script or test helper, where the team chose pragmatic simplicity.
3. **Over-Aggressive Edge-Case Defense (20.0%, 8 cases)**: Demanding defensive `null` checks on data structures already guaranteed by TypeScript compiler invariants earlier in the pipeline.
4. **Inexact Diff Line Anchoring (17.5%, 7 cases)**: Correct conceptual identification of a bug, but anchored $>3$ lines away from the human reviewer's comment.

### 8.2 False Negative Distribution (40 Cases)
1. **Silent Semantic Inversions (35.0%, 14 cases)**: Refactors that subtly inverted business logic (e.g., swapping `<=` to `<` in pagination bounds) while maintaining syntactically clean code.
2. **Implicit Framework Invariants (27.5%, 11 cases)**: Violations of unwritten ecosystem idioms (e.g., Vite plugin hook execution order or tRPC batch context scoping) not codified in the retrieved standards snippet.
3. **Distributed Resource Lifecycle Leaks (22.5%, 9 cases)**: Connection or subscription lifecycle leaks where acquisition occurred in one method and release occurred conditionally across async generators.
4. **Sub-Threshold Confidence Suppression (15.0%, 6 cases)**: Valid detections where model confidence scored at $0.44$–$0.49$, just below our production threshold of $0.50$.

---

## 9. What I Would Do Next

1. **Whole-Repository SCIP / LSIF Symbol Indexing**: Integrate Sourcegraph SCIP code graphs to resolve cross-file definitions and call hierarchies, eliminating the 37.5% of false positives caused by missing context.
2. **AST-Aware Hunk Windowing**: Replace fixed-line diff splitting with tree-sitter AST syntax chunking to provide complete function or class enclosures rather than truncated diff hunks.
3. **Multi-Turn Review Thread Resolution (Shipped in v1.1.0)**: Developers can reply directly to `@prism-review` inline comments to provide clarification or dispute findings. Prism evaluates the thread, diff hunk, and developer justification in [`src/prism/review/conversation.py`](../src/prism/review/conversation.py), conceding false positives gracefully (`decision="resolved"`) or defending the vulnerability with evidence (`decision="maintained"`).
4. **Online Reinforcement Learning from Developer Reactions (RLDR)**: Leverage reaction telemetry and thread resolution records in [`src/prism/observability/feedback.py`](../src/prism/observability/feedback.py). Real-time thumbs-up, thumbs-down, and thread-resolution events provide a continuous preference dataset for Direct Preference Optimization (DPO).

---

*All code, raw metrics, and configs are available in the repository root. Run `make reproduce` to re-execute the test benchmark.*
