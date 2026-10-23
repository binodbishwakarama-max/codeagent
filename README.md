<div align="center">

![Prism Review Hero Banner](docs/images/prism_hero_banner.png)

# 💎 PRISM REVIEW

### Autonomous, Standards-Driven AI Code Reviewer with Empirical Benchmark Verification

[![CI / CD](https://github.com/binodbishwakarama-max/codeagent/actions/workflows/deploy.yml/badge.svg)](https://github.com/binodbishwakarama-max/codeagent/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-FFD21E.svg)](https://huggingface.co/datasets/Vinod8848/prism-review-benchmark)
[![Live Demo Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Live%20Demo-blue.svg)](https://huggingface.co/spaces/Vinod8848/prism-review)
[![vLLM Powered](https://img.shields.io/badge/Served%20with-vLLM%20AWQ-purple.svg)](https://github.com/vllm-project/vllm)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Strict Types: mypy](https://img.shields.io/badge/types-mypy%20strict-1f425f.svg)](https://mypy-lang.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![Deploy on Fly.io](https://img.shields.io/badge/Deploy-Fly.io-24185B.svg?logo=flydotio&logoColor=white)](fly.toml)

<p align="center">
  <a href="#-headline-metrics">Headline Metrics</a> •
  <a href="#-live-pull-request-demo">Live Demo</a> •
  <a href="#-2-minute-video-walkthrough">Video Walkthrough</a> •
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-empirical-benchmark-results">Benchmark</a> •
  <a href="#-quickstart-in-5-commands">Quickstart</a> •
  <a href="#-observability--telemetry">Observability</a> •
  <a href="#-documentation-index">Docs</a>
</p>

</div>

---

> **Prism Review** is a production-grade GitHub App that autonomously reviews pull requests against explicit engineering standards (`.prism/standards.yml`), verified by an empirical evaluation benchmark on frozen open-source repositories.
>
> By fine-tuning **Qwen2.5-Coder-7B-Instruct** with QLoRA on multi-file diffs and explicitly training the model to remain silent on clean code, Prism captures **96.8% of Claude 3.5 Sonnet's F1 score** (`0.9304` vs `0.9613`) at **37.1× lower cost** ($0.00036 vs $0.01334 per PR) with **40.5 ms** $p50$ latency, posting high-signal inline comments with interactive one-click commit suggestions.

---

## ⚡ Headline Metrics

<div align="center">

| Metric | Result | Benchmark Context |
|:---|:---:|:---|
| **Review F1 Score** | **`0.9304`** | `96.8%` of Claude 3.5 Sonnet (`0.9613`) on frozen test split |
| **Serving Cost per Review** | **`$0.00036`** | **`37.1×` cheaper** than Claude 3.5 Sonnet ($0.01334/PR) |
| **Monthly Spend (10k PRs)** | **`$3.60`** | Compared to `$133.40` for Frontier API at 50% GPU utilization |
| **Clean-PR False Positive Rate** | **`0.2353`** | **`68.0%` noise reduction** vs. Base Qwen-7B (`0.7353`) |
| **Serving Latency (vLLM)** | **`40.5 ms`** | `6.2×` faster than Frontier API roundtrip (251.5 ms) |
| **Memory Footprint (AWQ 4-bit)** | **`5.8 GB`** | Fits on commodity 24GB GPUs (NVIDIA A10G / RTX 4090) |

</div>

---

## 🎯 The Core Problem & The Prism Solution

| The Status Quo | Why It Breaks | The Prism Solution |
|---|---|---|
| **Human Peer Review** | Consumes 20–25% of developer bandwidth; median PR turnaround ($p50$) ranges between **18–36 hours**. | **Instant Turnaround**: Webhook ingestion in **11.8 ms**; full review posted in **< 1 second**. |
| **Static Linters** | Fast and free, but blind to multi-line semantics; scored **0.0000 F1** on our real-world logic bug benchmark. | **Semantic Understanding**: Catches unhandled async exceptions, SQL injections, and race conditions. |
| **Commercial AI Reviewers** | Zero-shot generic prompts comment on **73.5% of clean PRs**, spamming shallow nitpicks and causing alert fatigue. | **Trained Silence**: SFT dataset with 25% negative clean PRs reduces clean-PR false alarms by **68%**. |

---

## 🎬 Live Pull Request Demo

> 🚀 **Interactive Live Playground**: You can test Prism Review directly in your browser without any setup at **[huggingface.co/spaces/Vinod8848/prism-review](https://huggingface.co/spaces/Vinod8848/prism-review)** with pre-loaded realistic diff scenarios (clean PRs, SQL injection, token leakage, unhandled async exceptions).

![Prism Review Demo](docs/images/pr_review_demo.png)

*Figure 1: Prism detects a hardcoded API token in `src/auth.ts`, cites repository standard `STD-SEC-01`, explains security impact, and formats an interactive GitHub "Commit suggestion" markdown diff block.*

---

## 🎥 2-Minute Video Walkthrough

[![Watch the 2-Minute Walkthrough](docs/images/video_demo_thumbnail.png)](docs/video_script.md)

*Figure 2: Shot-by-shot 2-minute walkthrough covering the code review bottleneck, live GitHub review demo, evaluation harness results, and Pareto serving economics. Read the complete script in [docs/video_script.md](docs/video_script.md).*

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph GitHub ["GitHub Platform"]
        PR[Developer Pull Request Event]
        WH[Webhook Delivery]
        CR[GitHub Checks API & Inline Review Comments]
        RX[Developer Reactions 👍/👎 & Thread Resolution]
    end

    subgraph Ingestion ["Ingestion & Async Queue Layer"]
        API[FastAPI Gateway /webhook/github<br/>p95: 11.8 ms]
        HMAC[HMAC-SHA256 Signature Verification]
        KILL{Per-Installation Kill Switch?}
        REDIS[(Redis Job Queue)]
        WORKER[ARQ Async Worker Pool]
    end

    subgraph Intelligence ["Review Intelligence Pipeline"]
        REDACT[Secret Redactor<br/>Scrub API Keys & Credentials]
        UNTRUST[Untrusted XML Encapsulation<br/>Neutralize Prompt Injection]
        RETRIB[Hybrid Retrieval<br/>BM25 + Dense Embeddings via RRF]
        STANDARDS[(.prism/standards.yml)]
        LLM[vLLM Inference Engine<br/>Fine-Tuned Qwen2.5-Coder-7B AWQ]
        VERIFY[Self-Verification Filter<br/>Discard Hallucinations]
    end

    subgraph Observability ["Observability & Feedback Telemetry"]
        TRC[Langfuse Distributed Tracing]
        MET[Prometheus Metrics /metrics]
        DASH[Grafana Live Dashboard<br/>Throughput, Latency, Cost, Acceptance Rate]
    end

    PR --> WH --> API
    API --> HMAC --> KILL
    KILL -- Blocked --> DROP[Drop & Return 200 OK]
    KILL -- Allowed --> REDIS --> WORKER
    WORKER --> REDACT --> UNTRUST
    STANDARDS --> RETRIB --> UNTRUST
    UNTRUST --> LLM --> VERIFY --> CR
    RX --> API --> MET --> DASH
    WORKER --> TRC
```

### Architectural Highlights:
1. **Zero-Downtime Ingestion**: Webhook payloads are authenticated via HMAC-SHA256 and queued in Redis in **9.8 ms** ($p50$), completely shielding GitHub from worker processing.
2. **Hybrid Standards Retrieval**: Combines BM25 token matching with dense vector embeddings via **Reciprocal Rank Fusion (RRF)** to retrieve exact rules (`STD-SEC-01`) alongside conceptual architectural standards.
3. **Enterprise Prompt-Injection Hardening**: Diff hunks and PR titles are strictly encapsulated inside `<untrusted_diff>` and `<untrusted_pr_body>` tags with delimiter neutralization ([sanitize_untrusted_text](src/prism/review/prompts.py)).
4. **Automated Secret Redactor**: Scrubs OpenAI/Anthropic keys, GitHub PATs, AWS credentials, private keys, bearer tokens, and database passwords before prompting ([redact_file_diffs](src/prism/review/redactor.py)).
5. **Self-Verification Filter**: An autonomous second-pass filter checks every candidate finding against the exact diff lines, discarding ungrounded claims and slashing false alarms.

---

## 📊 Empirical Benchmark Results

We evaluated all candidate models under **identical conditions** across a leak-free frozen test split of **120 PRs** (`trpc/trpc` and `vitejs/vite`) using the `MultiMatcher` ($\pm 3$ line tolerance, semantic cosine similarity $\ge 0.82$, $B=1,000$ paired bootstrap iterations):

| Model / Candidate | Precision (95% CI) | Recall (95% CI) | F1 Score (95% CI) | Clean-PR FPR | Latency p50 | Cost / Review | Spend (10k PRs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frontier API (Claude 3.5 Sonnet)** | `0.9675` [0.938, 0.993] | `0.9551` [0.920, 0.985] | **`0.9613`** [0.939, 0.982] | `0.0588` | 251.5 ms | **$0.01334** | $133.40 |
| **Mid-Size API (Claude 3.5 Haiku)** | `0.9062` [0.858, 0.947] | `0.9295` [0.891, 0.964] | **`0.9177`** [0.887, 0.945] | `0.1765` | 110.1 ms | **$0.00335** | $33.50 |
| **Base Qwen2.5-Coder-7B (Zero-Shot)** | `0.6782` [0.615, 0.742] | `0.7564` [0.697, 0.814] | **`0.7152`** [0.663, 0.763] | `0.7353` | 67.9 ms | **$0.00003** | $3.88 |
| **Fine-Tuned 7B (Merged BF16)** | `0.8889` [0.837, 0.936] | `0.9744` [0.947, 0.994] | **`0.9297`** [0.898, 0.956] | `0.2941` | 60.7 ms | **$0.00098** | $9.80 |
| **Fine-Tuned 7B (AWQ 4-bit) 🏆** | `0.9187` [0.861, 0.965] | `0.9423` [0.902, 0.978] | **`0.9304`** [0.893, 0.959] | **`0.2353`** | **40.5 ms** | **$0.00036** | **$3.60** |
| **Linter Baseline** | `0.0000` [0.000, 0.000] | `0.0000` [0.000, 0.000] | **`0.0000`** [0.000, 0.000] | `0.0000` | 0.0 ms | **$0.00000** | $0.00 |

### 📈 Quality vs. Cost Pareto Frontier

<div align="center">

![Quality vs Cost Pareto Frontier](docs/images/quality_vs_cost.png)

*Figure 3: Review Quality (F1 score with 95% bootstrap CIs) vs. Cost per review ($ USD, log-scale). The green dashed line traces the empirical Pareto frontier, proving Fine-Tuned 7B AWQ occupies the optimal efficiency sweet spot.*

</div>

---

## 🔭 Observability & Telemetry

Prism features a first-class production observability stack:

<div align="center">

| Grafana Live Dashboard | Langfuse Distributed Tracing |
|:---:|:---:|
| ![Grafana Dashboard](docs/img/grafana_dashboard.png) | ![Langfuse Tracing](docs/img/langfuse_trace_tree.png) |
| *Real-time metrics: reviews/day, latency histograms, daily budget, and developer acceptance rate.* | *End-to-end trace tree: diff parsing, hybrid retrieval, model generation, and GitHub comment latency.* |

</div>

### Developer Feedback Loop
When developers react with 👍 / 👎 or resolve comment threads on GitHub, Prism captures the event via webhook and updates its live acceptance rate metric:
$$\text{Acceptance Rate} = \frac{\text{Resolved Threads} + \text{Positive Reactions}}{\text{Total Review Comments Posted}}$$

---

## 🚀 GitHub Action Quickstart (Zero-Server Setup)

You can run Prism autonomously in any GitHub repository with zero infrastructure. It posts inline PR review comments, generates rich Markdown job summaries, and emits GitHub workflow annotations.

Create `.github/workflows/prism.yml`:

```yaml
name: Prism Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write
  checks: write

jobs:
  review:
    name: Autonomous Standards Review
    runs-on: ubuntu-latest
    steps:
      - name: Checkout PR Branch
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run Prism Review
        uses: binodbishwakarama-max/codeagent@main
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          standards-file: ".prism/standards.yml"
          fail-on-severity: "high" # none | medium | high
```

---

## 💻 CLI Usage: Review Any PR or Local Branch

Review live GitHub pull requests, local working trees, or patch files from your terminal:

```bash
# 1. Install Prism CLI
pip install git+https://github.com/binodbishwakarama-max/codeagent.git

# 2. Review any public GitHub pull request by URL
prism --url https://github.com/fastapi/fastapi/pull/11020

# 3. Review uncommitted or staged local changes
prism --git
prism --staged

# 4. Review a git patch or piped diff
git diff origin/main | prism --diff -
```

---

## ⚡ Quickstart in 5 Commands

Reproduce the full benchmark or run Prism locally in under 3 minutes:

```bash
# 1. Clone repository
git clone https://github.com/binodbishwakarama-max/codeagent.git
cd codeagent

# 2. Install dependencies with uv
uv sync --extra dev

# 3. Reproduce the test-split benchmark in seconds
make reproduce

# 4. Start backing services (PostgreSQL + pgvector, Redis, Prometheus, Grafana)
docker compose up -d

# 5. Launch the FastAPI review webhook gateway
make run
```

---

## 🐳 Pinned Reproducibility Container

For 100% reproducible benchmark execution across any Linux, macOS, or Windows environment, execute the evaluation inside our pinned Docker container:

```bash
# Build reproducibility container
docker build -f Dockerfile.eval -t prism-eval .

# Run full test benchmark
docker run --rm prism-eval
```

---

## 📜 Custom Standards as Code (`.prism/standards.yml`)

Add a `.prism/standards.yml` file to the root of your repository to enforce your team's engineering standards:

```yaml
version: "1.0"
description: "Core engineering standards for production services."

standards:
  - id: "STD-SEC-01"
    title: "No Hardcoded Secrets or Tokens"
    category: "SECURITY"
    severity: "HIGH"
    description: "Never commit API keys, private keys, database passwords, or auth tokens. Load all secrets strictly from environment variables."
    good_example: |
      api_key = os.environ.get("OPENAI_API_KEY")
    bad_example: |
      api_key = "sk-live-51P2uJSH4zK7L9M1oB3n6c5D7e8T9"

  - id: "STD-ERR-01"
    title: "Explicit Async Exception & Timeout Handling"
    category: "ERROR_HANDLING"
    severity: "HIGH"
    description: "All outbound network, HTTP, and database requests must declare explicit timeouts and handle exceptions gracefully."
    good_example: |
      response = await client.get(url, timeout=10.0)
    bad_example: |
      response = await client.get(url)  # Unbounded timeout

  - id: "STD-SQL-01"
    title: "Parameterized Database Queries"
    category: "SECURITY"
    severity: "HIGH"
    description: "Always use parameterized query placeholders or ORM bind parameters. Never interpolate user strings into SQL."
    good_example: |
      await db.execute(text("SELECT * FROM users WHERE email = :email"), {"email": email})
    bad_example: |
      await db.execute(f"SELECT * FROM users WHERE email = '{email}'")
```

---

## 🛡️ Production Safety Controls

* **Per-Installation Kill Switch**: Set `BLOCKED_INSTALLATIONS="123456,789012"` in environment variables to immediately acknowledge and drop incoming webhooks for specific repositories with zero compute or LLM tokens consumed.
* **Daily Cost Budget Cap**: Configured via `DAILY_COST_BUDGET_USD="15.0"` and `BUDGET_ENFORCED="true"`. Cumulative daily USD spend is tracked in Redis; if exceeded, reviews and comment posting are automatically disabled.
* **Prompt Injection Containment**: Diffs and PR descriptions are encapsulated inside `<untrusted_diff>` XML blocks with tag neutralization, preventing prompt injection attacks from overriding system instructions.
* **Automated Secret Redactor**: Strips API keys (`sk-`, `sk-ant-`), GitHub PATs (`ghp_`), AWS credentials (`AKIA`), private keys, and database passwords from diffs before sending to model inference.

---

## 📚 Documentation Index

| Document | Description |
|---|---|
| 📖 **[Technical Write-Up](docs/writeup.md)** | 1,995-word technical report covering metrics, dataset construction, fine-tuning, and failure analysis. |
| 📊 **[Benchmark Results](docs/results.md)** | Complete evaluation report with bootstrap CIs, per-category breakdown, and serving economics. |
| 📦 **[Dataset Card](DATASET_CARD.md)** | PR mining methodology, schemas, license compliance, and Hugging Face export steps. |
| 🎬 **[Video Walkthrough Script](docs/video_script.md)** | Shot-by-shot 2-minute video script and pinned 5-minute repository orientation guide. |
| 💼 **[Resume & Interview Guide](docs/resume_and_interview.md)** | 3 targeted resume bullet variants and 60-second interview architectural pitch. |
| 🚀 **[Installation & Usage](docs/usage.md)** | Multi-repo GitHub App setup, Fly.io deployment runbook, and kill-switch guide. |
| 📈 **[Load Test Report](docs/load-test.md)** | 50 PR event burst simulation with 0.0% error rate and 11.8ms p95 ingestion latency. |
| 🐳 **[Evaluation Dockerfile](Dockerfile.eval)** | Pinned, reproducible container specification for benchmark replication. |

---

## 🗺️ Project Roadmap

- [x] Multi-stage Dockerfile and rolling zero-downtime Fly.io deployment.
- [x] MultiMatcher evaluation harness with $\pm 3$ line tolerance and semantic similarity.
- [x] Fine-tuned Qwen2.5-Coder-7B with QLoRA ($r=32$) and 25% negative silence training.
- [x] 4-bit AWQ quantization with vLLM serving engine (40.5ms p50 latency).
- [x] Secret redaction, prompt injection hardening, and daily budget controls.
- [x] Distributed tracing (Langfuse), metrics (Prometheus), and dashboards (Grafana).
- [ ] **Whole-Repository SCIP Symbol Indexing**: Integrate Sourcegraph SCIP code graphs to eliminate cross-file context false positives.
- [ ] **AST-Aware Hunk Windowing**: Replace line-based hunk truncation with tree-sitter syntax enclosures.
- [ ] **Direct Preference Optimization (DPO)**: Online reinforcement learning from developer 👍/👎 reaction telemetry.

---

## ⚖️ License

The Prism Review engine, evaluation harness, and dataset curation tooling are released under the [Apache-2.0 License](LICENSE). Source pull request metadata is covered by original repository permissive open-source licenses (MIT, Apache-2.0, BSD).

---

<div align="center">

**Built with precision by [Binod Bishwakarma](https://github.com/binodbishwakarama-max)**  
*Run `make reproduce` to re-execute the test split benchmark.*

</div>
