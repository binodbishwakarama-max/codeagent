# Prism Review: Public Launch & Social Proof Playbook

This document contains publication-ready launch copy, empirical tables, and discussion hooks for Hacker News, LinkedIn, Twitter/X, and longform technical blogging.

All links and metrics are verified against:
* **Interactive Live Space**: [https://huggingface.co/spaces/Vinod8848/prism-review](https://huggingface.co/spaces/Vinod8848/prism-review)
* **Hugging Face Benchmark Dataset**: [https://huggingface.co/datasets/Vinod8848/prism-review-benchmark](https://huggingface.co/datasets/Vinod8848/prism-review-benchmark)
* **GitHub Repository**: [https://github.com/binodbishwakarama-max/codeagent](https://github.com/binodbishwakarama-max/codeagent)
* **GitHub Release v1.0.0**: [https://github.com/binodbishwakarama-max/codeagent/releases/tag/v1.0.0](https://github.com/binodbishwakarama-max/codeagent/releases/tag/v1.0.0)

---

## 1. Show HN (Hacker News) Launch Post

**Target Subreddit / Platform**: `news.ycombinator.com`  
**Suggested Titles**:
* `Show HN: Prism – Why base LLMs comment on 73.5% of clean PRs, and how we fixed it`
* `Show HN: Prism – An open-source AI code reviewer with a measured clean-PR false positive rate`

### Post Body (Copy & Paste Ready)

```text
Hi HN,

I built Prism (https://github.com/binodbishwakarama-max/codeagent), an open-source automated code review engine designed around a metric that commercial AI review tools almost never measure: the Clean-PR False Positive Rate (Clean FPR).

### The Core Problem: The Silent Failure Mode of AI Reviewers

Over the past two years, dozens of AI code review assistants have launched. Most claim high satisfaction or demonstrate catching an injection bug in a 10-line snippet. 

However, when you benchmark base instruction models (like Qwen2.5-Coder-7B-Instruct or Llama-3.1-8B) across real pull requests, you uncover a fatal flaw:
* They score an F1 of ~0.71 on defect-laden PRs.
* But they post unsolicited critiques on 73.5% of completely clean, human-approved PRs.

Pre-trained instruction models have an innate bias toward generating tokens. When presented with clean code, they hallucinate nitpicks, stylistic bike-shedding, and phantom edge cases rather than returning an empty list. In a real engineering team, a bot that comments on 3 out of 4 clean PRs will be uninstalled within a week.

### What We Did

1. **Negative SFT Training on Clean PRs**:
We fine-tuned Qwen2.5-Coder-7B with QLoRA where exactly 25% of the training dataset consisted of clean pull requests mapped to an empty output (`{"findings": []}`). This trained the model to recognize idiomatic implementations and maintain silence when no defect exists.
Clean FPR dropped from 73.5% down to 23.5% (a 68% reduction).

2. **Self-Verification Guardrail**:
Every candidate finding passes through an independent verification step before posting. This drops clean PR false alarms further (down to 18.0% in ablation tests) and filters out hallucinations.

3. **Hybrid Code Retrieval (BM25 + Dense RRF)**:
Pure vector search often misses exact variable names or method signatures. We use Reciprocal Rank Fusion combining BM25 keyword matching with dense embeddings, followed by a cross-encoder reranker, to supply relevant repository standards and definitions.

4. **Multi-Turn Review Thread Resolution**:
Developers can reply directly to @prism-review on inline comments. The bot evaluates the developer's justification against the diff hunk. If the developer notes that the input was sanitized in upstream middleware, Prism concedes gracefully (marking it resolved) rather than arguing defensively.

5. **AWQ 4-Bit Serving on Cheap Hardware**:
We quantized the fine-tuned model to AWQ 4-bit. It runs in vLLM on a single NVIDIA A10G (24GB) with a median p50 latency of 40.5 ms and a serving cost of $0.00036 per review ($3.60 per 10,000 PRs), compared to $133.40 for Claude 3.5 Sonnet.

### Empirical Benchmark on Frozen Test Split (120 PRs)

We mined 300+ real pull requests across 24 repositories (FastAPI, Vite, tRPC, Pydantic, HTTPX, etc.) with human review comments verified by follow-up commits. The test split is frozen with zero repository overlap.

| Model / Architecture | F1 Score (95% CI) | Clean PR FPR | p50 Latency | Cost / 10k PRs |
| :--- | :---: | :---: | :---: | :---: |
| Frontier (Claude 3.5 Sonnet) | 0.9613 [0.939, 0.982] | 5.88% | 251.5 ms | $133.40 |
| Mid-Size (Claude 3.5 Haiku) | 0.9177 [0.887, 0.945] | 17.65% | 110.1 ms | $33.50 |
| Base Qwen2.5-Coder-7B | 0.7152 [0.663, 0.763] | 73.53% | 60.7 ms | $9.80 |
| Fine-Tuned 7B (BF16) | 0.9297 [0.898, 0.956] | 29.41% | 60.7 ms | $9.80 |
| Fine-Tuned 7B (AWQ 4-bit) 🏆 | 0.9304 [0.893, 0.959] | 23.53% | 40.5 ms | $3.60 |
| AST Linters Baseline | 0.0000 [0.000, 0.000] | 0.00% | 0.0 ms | $0.00 |

*Fine-tuned 7B reaches 96.8% of Claude 3.5 Sonnet's F1 score while serving 6.2x faster at 37x lower cost.*

### Open Source & Public Artifacts

All code, evaluation harnesses, and datasets are open:
* Interactive Web Demo: https://huggingface.co/spaces/Vinod8848/prism-review
* GitHub: https://github.com/binodbishwakarama-max/codeagent
* Hugging Face Benchmark Dataset: https://huggingface.co/datasets/Vinod8848/prism-review-benchmark
* Release v1.0.0: https://github.com/binodbishwakarama-max/codeagent/releases/tag/v1.0.0
* Local Simulator: You can run the entire webhook & review pipeline locally in terminal without external dependencies using `make simulate-webhook`.

Would love to hear your thoughts on automated review heuristics, false positive mitigation, and how teams evaluate LLM code assistants!
```

---

## 2. LinkedIn Launch Post

**Audience**: Staff/Principal Engineers, VP/Directors of Engineering, Applied AI & ML Engineers.  
**Tone**: Empirical, technical, architectural, problem-focused.

```text
Why do most engineering teams eventually mute or uninstall AI code review bots?

It’s almost never because the model missed a bug.
It’s because of the Clean-PR False Positive Rate.

When we benchmarked base open-weights coding models (like Qwen2.5-Coder-7B-Instruct) across real production PRs from FastAPI, Vite, and Pydantic:
👉 They scored a reasonable 0.71 F1 on pull requests containing defects.
👉 But they posted unsolicited nitpicks on 73.5% of completely clean, human-approved PRs.

Pre-trained LLMs have an innate generation bias: given code, they want to output comments. If there are no bugs, they hallucinate stylistic gripes, phantom null-pointer concerns, or redundant refactoring ideas.

To fix this, we built Prism Review (https://github.com/binodbishwakarama-max/codeagent):

1️⃣ Negative SFT Distillation: We injected 25% clean PRs with an explicit target of {"findings": []} into our fine-tuning mixture. Teaching the model to remain silent cut clean-PR false alarms by 68%.
2️⃣ Hybrid RRF Code Retrieval: BM25 keyword matching + dense embeddings + cross-encoder reranking to ground findings in explicit repo standards (.prism/standards.yml).
3️⃣ Self-Verification Guardrail: Every candidate finding is independently audited before posting, reducing false alarms to 18%.
4️⃣ Multi-Turn Thread Resolution: Developers can reply directly to @prism-review. If you explain that an input is already validated in upstream middleware, Prism concedes gracefully and marks the finding resolved.
5️⃣ AWQ 4-Bit Inference: Quantized serving via vLLM on a single NVIDIA A10G runs at 40.5ms p50 latency and costs $0.00036 per review ($3.60 per 10,000 PRs vs $133.40 for Claude 3.5 Sonnet).

All benchmarks, evaluation harnesses, and datasets are public:
🌐 Interactive Live Playground: https://huggingface.co/spaces/Vinod8848/prism-review
📦 Hugging Face Benchmark Dataset: https://huggingface.co/datasets/Vinod8848/prism-review-benchmark
🚀 GitHub Release v1.0.0: https://github.com/binodbishwakarama-max/codeagent/releases/tag/v1.0.0
💻 Codebase & Local Simulator: https://github.com/binodbishwakarama-max/codeagent

If your team is experimenting with automated code review or local LLM serving, I’d love to connect and hear how you approach developer trust and noise reduction.

#MachineLearning #LLM #SoftwareEngineering #CodeReview #vLLM #OpenSource #DevOps #AI
```

---

## 3. Twitter / X Deep-Dive Thread (6 Tweets)

### Tweet 1 (Hook)
> Most AI code review tools fail because of a metric they never publish:
>
> The Clean-PR False Positive Rate.
>
> Base coding LLMs comment on 73.5% of completely clean PRs.
>
> Here is how we built Prism to drop false alarms by 68% and achieve 96.8% of Claude 3.5 Sonnet’s F1 at 37x lower cost 🧵👇
> https://github.com/binodbishwakarama-max/codeagent

### Tweet 2 (The Flaw)
> Why do base LLMs spam clean PRs?
>
> Pre-trained instruction models are biased toward generating tokens. When given a diff with no bugs, they hallucinate stylistic nitpicks and redundant refactors rather than returning an empty list.
>
> In production, developer fatigue sets in within days.

### Tweet 3 (The Fix: Negative Training)
> The solution: 25% negative training.
>
> During QLoRA fine-tuning on Qwen2.5-Coder-7B, we fed exactly 25% clean PRs with target output `{"findings": []}`.
>
> Teaching the model *when to remain silent* dropped the clean-PR false positive rate from 73.5% down to 23.5%.

### Tweet 4 (Architecture & Guardrails)
> We wrapped the model in a 4-layer production pipeline:
> 1. Automated secret scrubber (pre-LLM)
> 2. Hybrid BM25 + dense RRF standards retrieval
> 3. Self-verification filter (drops hallucinations)
> 4. Multi-turn thread replies (@prism-review concedes false alarms when context is provided)

### Tweet 5 (Economics)
> Serving efficiency with AWQ 4-bit in vLLM:
> 
> ⚡ Latency: 40.5 ms p50 (vs 251.5 ms for Sonnet)
> 💰 Cost per 10k reviews: $3.60 (vs $133.40 for Sonnet)
> 🎯 F1 Score: 0.9304 vs 0.9613 (reaches 96.8% of Sonnet quality)
> 
> Runs comfortably on a single 24GB GPU.

### Tweet 6 (CTA & Links)
> Everything is open source, reproducible, and benchmarked on 300+ real PRs:
>
> 🌐 Live Interactive Playground: https://huggingface.co/spaces/Vinod8848/prism-review
> 📂 Hugging Face Dataset: https://huggingface.co/datasets/Vinod8848/prism-review-benchmark
> 🏷️ Release v1.0.0: https://github.com/binodbishwakarama-max/codeagent/releases/tag/v1.0.0
> 💻 GitHub: https://github.com/binodbishwakarama-max/codeagent
>
> Run `make simulate-webhook` to test the entire pipeline locally!

---

## 4. Full Engineering Blog Post (Substack / Dev.to / Medium)

```markdown
# Measured Code Review: Why Base LLMs Spam Clean PRs and How to Train Domain-Specific AI Reviewers

**By Binod Bishwakarma**  
*Try the live playground on [Hugging Face Spaces](https://huggingface.co/spaces/Vinod8848/prism-review), read the open-source code on [GitHub](https://github.com/binodbishwakarama-max/codeagent), or download the benchmark dataset on [Hugging Face](https://huggingface.co/datasets/Vinod8848/prism-review-benchmark).*

---

Every engineering team wants faster pull request turnaround times. Today, peer review takes 20% to 25% of developer bandwidth, and PR turnaround latencies routinely stretch between 18 and 36 hours. 

When teams turn to generic commercial AI review bots, they often face a frustrating cycle: initial excitement followed by developer irritation and eventual uninstallation.

Why does this happen?

The answer lies in an unmeasured variable: the **Clean-PR False Positive Rate**.

```
+-----------------------------------------------------------------------------+
|                           THE AI CODE REVIEW DILEMMA                        |
|                                                                             |
|  AST Linters (flake8, eslint):      0% Recall on Architectural Logic Bugs   |
|  Base Coding LLMs (Zero-Shot):     73.5% False Positive Rate on Clean PRs   |
|  Prism (Negative SFT + Guardrails): 93.0% F1 Score with 23.5% Clean FPR      |
+-----------------------------------------------------------------------------+
```

### The Benchmark: Measuring What Matters

To establish an honest baseline, we mined 300+ real-world pull requests across 24 leading open-source repositories in Python and TypeScript (FastAPI, Pydantic, HTTPX, Celery, VS Code, tRPC, Vite, etc.).

Crucially, our benchmark includes:
1. **Defect-laden PRs**: Verified by subsequent commits that modified the flagged code.
2. **Clean PRs (Negative controls)**: PRs approved with zero inline critiques, establishing our false-positive baseline.
3. **Repository-Level Split**: 60% Train, 20% Dev, and 20% frozen Test (120 PRs on `trpc/trpc` and `vitejs/vite`) with zero repository overlap.

### The Problem with Zero-Shot LLMs

When we evaluated base `Qwen2.5-Coder-7B-Instruct` zero-shot on our test split, it achieved an F1 of `0.7152`. On surface level, that sounds acceptable.

Then we checked the negative controls:
* **Clean-PR False Positive Rate: 73.53%**

Base models have been instruction-tuned to *always* provide helpful answers. When handed a clean diff with no bugs, they do not output empty text; instead, they invent stylistic suggestions, debate variable naming, or suggest defensive `null` checks on types guaranteed by the compiler.

### The 4 Architectural Pillars of Prism

To build an automated reviewer that developers actually trust, we designed a multi-layer pipeline:

```
[Incoming PR Webhook]
          │
          ▼
1. HMAC-SHA256 & Deduplication (Idempotent replay protection)
          │
          ▼
2. Automated Secret Redaction (Pre-LLM scrubbing of API keys/tokens)
          │
          ▼
3. Hybrid Standards Retrieval (BM25 + Dense RRF + Cross-Encoder)
          │
          ▼
4. Domain-Tuned Model Inference (QLoRA with 25% negative clean PRs)
          │
          ▼
5. Self-Verification Filter (Rejects low-confidence hallucinations)
          │
          ▼
6. GitHub Check Run & Inline Reviews (with one-click commit suggestions)
          │
          ▼
7. Multi-Turn Thread Resolver (Answers developer disputes & concessions)
```

#### 1. Negative Training for Silence
In our QLoRA fine-tuning mixture, exactly 25% of all training examples were clean pull requests paired with a target output of `{"findings": []}`. This trained the model to recognize idiomatic code and taught it that returning an empty list is a valid, high-reward answer.
* Result: Clean FPR plummeted from **73.5% to 23.5%** (a 68% reduction).

#### 2. Hybrid Retrieval (BM25 + Dense RRF)
Dense embeddings alone often miss exact symbol names (`get_user_by_id`, `StripeClient`). We implemented Reciprocal Rank Fusion combining BM25 keyword matching with dense vectors, followed by a cross-encoder reranker. In ablation testing, this boosted F1 by **+6.7 points** over vector-only search.

#### 3. Self-Verification Step
Every finding generated by the review model is independently audited against the diff hunk before posting. If the verification check cannot substantiate the rule violation, the finding is discarded. This dropped clean-PR false alarms by another **10.0%**.

#### 4. Multi-Turn Conversational Thread Resolution
In real engineering workflows, review is an interactive discussion. When a developer replies:
> *"@prism-review false positive: customer_id is verified upstream in customer_router.py before billing lookup"*

Prism evaluates the developer's justification against the diff hunk. If the objection is valid, Prism concedes gracefully:
```markdown
<!-- prism-reply -->
### 💬 Prism Review Response
✅ **Status**: Conceded (False Positive Accepted)

Conceded: Verified against the request validation pipeline. The customer UUID format is enforced prior to dispatch, so injection is safely mitigated.
```
Every concession is logged to a Postgres telemetry table to feed continuous Reinforcement Learning from Developer Reactions (RLDR).

---

### Empirical Results

All models were evaluated on the frozen test split (120 PRs):

| Model / Configuration | Precision | Recall | F1 Score | Clean FPR | Cost / 10k PRs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Frontier API (Claude 3.5 Sonnet)** | 0.9675 | 0.9551 | **0.9613** | 5.88% | $133.40 |
| **Mid-Size API (Claude 3.5 Haiku)** | 0.9062 | 0.9295 | **0.9177** | 17.65% | $33.50 |
| **Base Qwen2.5-Coder-7B** | 0.6782 | 0.7564 | **0.7152** | 73.53% | $9.80 |
| **Fine-Tuned 7B (BF16)** | 0.8889 | 0.9744 | **0.9297** | 29.41% | $9.80 |
| **Fine-Tuned 7B (AWQ 4-bit) 🏆** | 0.9187 | 0.9423 | **0.9304** | **23.53%** | **$3.60** |
| **AST Linter Baseline** | 0.0000 | 0.0000 | **0.0000** | 0.00% | $0.00 |

### Serving Economics

Quantizing the fine-tuned model with AWQ 4-bit allowed us to serve it in vLLM on a single NVIDIA A10G (24GB VRAM):
* **Median Latency (p50)**: **40.5 ms** (6.2× faster than Claude 3.5 Sonnet's 251.5 ms).
* **Cost per Review**: **$0.00036** (37.1× cheaper than Sonnet's $0.01334).
* **Throughput**: $3.60 per 10,000 pull requests under a conservative 50% duty cycle.

---

### Try It Yourself

1. **Interactive Web Playground on Hugging Face Spaces**:
Explore and review realistic code diffs in real time without any local setup at [Vinod8848/prism-review](https://huggingface.co/spaces/Vinod8848/prism-review).

2. **Benchmark Dataset on Hugging Face**:
Explore the splits, manifests, and PR metadata at [Vinod8848/prism-review-benchmark](https://huggingface.co/datasets/Vinod8848/prism-review-benchmark).

3. **Simulate Webhook & Reviews Locally**:
You can run the full GitHub App pipeline locally with zero external dependencies:
```bash
git clone https://github.com/binodbishwakarama-max/codeagent.git
cd codeagent
uv sync
python scripts/simulate_webhook.py
```

4. **Check the v1.0.0 Release**:
Release notes, migration scripts, and architecture docs are published on [GitHub Releases](https://github.com/binodbishwakarama-max/codeagent/releases/tag/v1.0.0).

---

*Comments, feedback, and pull requests are warmly welcomed on [GitHub](https://github.com/binodbishwakarama-max/codeagent).*
```
