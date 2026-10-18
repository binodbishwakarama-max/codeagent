# Prism Review: Resume Bullets & 60-Second Interview Walkthrough

All numbers cited below are strictly derived from and verified in [docs/results.md](results.md).

---

## 📄 Resume Bullets (3 Targeted Variants)

### Variant 1: Machine Learning & Applied AI Engineer (Focus: Fine-Tuning, Quantization, Evaluation)
> * **Fine-Tuned Open-Weights LLM with QLoRA ($r=32$) on Multi-File Diffs**: Fine-tuned `Qwen2.5-Coder-7B-Instruct` on curated GitHub PR diffs to reach **96.8%** of Claude 3.5 Sonnet's F1 score (**0.9304** vs **0.9613**, 95% bootstrap CI: [92.2%, 100.7%]) at **37.1× lower cost** ($0.00036 vs $0.01334 per PR), reducing clean-PR false positive alarms by **68.0%** (0.7353 down to 0.2353).
> * **Built Rigorous Multi-Matcher Evaluation Harness**: Engineered an empirical benchmark evaluating candidate models across a frozen 120-PR test split (`trpc/trpc` and `vitejs/vite`) with $\pm 3$ line window tolerance, semantic cosine similarity ($\ge 0.82$), and $B=1,000$ paired bootstrap confidence intervals.
> * **Quantization & High-Throughput Serving**: Compressed model memory footprint from 16.2 GB to **5.8 GB VRAM** via 4-bit AWQ quantization with zero degradation in review F1 (`0.9304`), achieving **40.5 ms** $p50$ latency and **47.1 ms** $p95$ latency on an NVIDIA A10G using vLLM.

---

### Variant 2: Senior Systems & Backend Infrastructure Engineer (Focus: Distributed Architecture, Latency, Economics)
> * **Architected High-Throughput Autonomous Code Review System**: Built distributed GitHub App processing webhook events in **11.8 ms** ($p95$) via FastAPI, ARQ async worker pool, Redis job queue, and PostgreSQL with pgvector.
> * **Slashed Production LLM Review Spend by 37.1×**: Transitioned code review pipeline from frontier APIs ($0.01334/PR) to self-hosted vLLM AWQ container ($0.00036/PR), reducing monthly operational spend for 10,000 PRs from $133.40 to **$3.60** at 50% GPU utilization.
> * **Engineered Production Safety & Observability Infrastructure**: Implemented automated secret redaction for API keys/tokens, per-installation kill switches, daily cost budget caps, Langfuse distributed tracing, and Prometheus metrics tracking live review acceptance rates.

---

### Variant 3: Full-Stack AI & Product Engineer (Focus: End-to-End Delivery, DX, Benchmarking)
> * **Engineered End-to-End AI Code Review Platform**: Designed and deployed Prism Review (FastAPI, Redis, PostgreSQL, vLLM) on Fly.io with zero-downtime rolling deploys, reviewing PRs against custom repository standards (`.prism/standards.yml`).
> * **Delivered Frontier-Grade Accuracy at Low Cost**: Achieved **0.9187** Precision, **0.9423** Recall, and **0.9304** F1 score on a frozen test split, capturing **96.8%** of Claude 3.5 Sonnet's review quality at **37.1× lower cost** ($0.00036 per review).
> * **Designed Closed-Loop Telemetry & Developer Feedback**: Captured developer thumbs-up reactions and resolved comment threads via GitHub webhooks, streaming live acceptance metrics to Prometheus and Grafana.

---

## 🎙️ 60-Second Interview Walkthrough Script

**Prompt:** *"Walk me through the architecture of Prism Review and how you approached building it."*

**Time:** 60 seconds (approx. 150 words spoken at natural conversational pace)

---

### Spoken Walkthrough

> *"I built Prism because commercial AI review bots suffer from high false-alarm rates and almost never publish empirical evaluation metrics.
> 
> **Architectural Decisions:**  
> Rather than relying on generic prompts, I anchored reviews to explicit repository standards in `.prism/standards.yml`. I paired hybrid BM25 and vector search with Reciprocal Rank Fusion to retrieve relevant rules, and added a multi-stage self-verification filter that eliminated hallucinations before posting.
> 
> **One Thing That Failed:**  
> Out of the box, base `Qwen2.5-Coder-7B` had a disastrous **73.5% false positive rate on clean PRs**—it felt compelled to critique perfectly fine code. I fixed this by fine-tuning with 25% negative clean examples paired with empty finding outputs, which curbed clean-PR false alarms by **68%** down to 0.2353 and lifted F1 to **0.9304**, reaching **96.8% of Claude 3.5 Sonnet** at **37.1× lower cost**.
> 
> **One Tradeoff I'd Revisit:**  
> Currently, the diff parser evaluates $\pm 3$ line windows within local hunks. While fast, 37.5% of remaining false positives stem from missing cross-file caller context. I'd upgrade from line hunks to whole-repo SCIP symbol graph indexing."*

---

## 💡 Quick Interview Q&A Cheatsheet

| Question | Concise Answer |
|---|---|
| **Why not just use Claude 3.5 Sonnet in production?** | Sonnet costs **$0.01334** per review and has a 251.5 ms latency roundtrip. Our fine-tuned 7B AWQ model costs **$0.00036** (37.1× cheaper) and runs in **40.5 ms** ($p50$), making it economically feasible to run on every commit push. |
| **Why is the Clean-PR FPR metric so important?** | Linters have 0% FPR because they don't hallucinate, but have 0% recall on logic errors. Pretrained LLMs have high recall but spam comments on 73.5% of clean PRs. Developers uninstall noisy bots; keeping FPR low is what makes an AI reviewer production-viable. |
| **How did you prevent data leakage during training?** | We partitioned strictly by repository: 60% train repos, 20% dev repos, and 20% test repos (`trpc` and `vite`). We asserted zero repository overlap in code (`assert_no_leakage()`). |
| **How did 4-bit AWQ compare to 16-bit BF16?** | AWQ quantization achieved identical F1 (`0.9304` vs `0.9297`), but reduced VRAM from 16.2 GB to **5.8 GB**, allowing deployment on a single commodity NVIDIA A10G ($0.70/hr). |
