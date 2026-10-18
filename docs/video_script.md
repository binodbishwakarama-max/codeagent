# Prism Review: 2-Minute Video Walkthrough Script & 5-Minute Repo Guide

---

## 📌 Pinned: How to Read This Repo in 5 Minutes

If you only have 5 minutes to understand how Prism Review works, read these 6 files in order:

1. **[docs/results.md](results.md)** *(2 mins)*: The final benchmark comparison table. Understand the headline numbers: fine-tuned 7B reaches **96.8% of Claude 3.5 Sonnet's F1** (`0.9304` vs `0.9613`) at **37.1× lower cost** ($0.00036 vs $0.01334 per PR) with **0.2353 clean-PR FPR**.
2. **[docs/writeup.md](writeup.md)** *(1.5 mins)*: The technical deep dive covering dataset construction, metric design, failure analysis (40 FPs / 40 FNs), and serving economics.
3. **[src/prism/review/pipeline.py](../src/prism/review/pipeline.py)** *(45 secs)*: The core review engine coordinating diff parsing, standards retrieval, LLM inference, and self-verification.
4. **[evals/harness/matcher.py](../evals/harness/matcher.py)** *(30 secs)*: The `MultiMatcher` evaluation logic matching line ranges ($\pm 3$ tolerance), semantic embeddings ($\ge 0.82$), and AST categories.
5. **[training/data_prep.py](../training/data_prep.py)** *(30 secs)*: The SFT data preparation showing how **25% clean negative PRs** were synthesized to teach the 7B model to stay silent on defect-free code.
6. **[Makefile](../Makefile)** *(15 secs)*: Run `make reproduce` to re-execute the test-split benchmark in seconds from cached evaluation artifacts.

---

## 🎬 2-Minute Video Script (Shot-by-Shot)

**Target Duration:** 120 seconds (2:00)  
**Tone:** Confident, concise, technical, metrics-driven. No generic AI hype.

---

### [0:00 – 0:15] Scene 1: The Problem
* **Visual:** Split screen. On the left: a long, pending GitHub PR waiting 36 hours for human review. On the right: a generic AI review bot spamming 12 shallow comments ("consider adding comments", "variable name could be clearer"). Cut to terminal showing Base Model evaluation stats: Clean-PR False Positive Rate: **73.5%**.
* **Audio / Voiceover:**
  > *"Every engineering team loses 20 to 25% of their time to peer code review, with PRs sitting idle for 24 to 36 hours. But traditional linters catch 0% of logic bugs, while naive LLM bots comment on 73% of completely clean PRs. Developers end up muting them. I built Prism to solve this: an autonomous code review engine with measured, production-grade precision."*
* **On-screen Text Overlay:**
  * Linters: 0.0000 F1 on logic defects
  * Base 7B Model: 73.5% False Positive Rate on clean PRs

---

### [0:15 – 0:50] Scene 2: Live PR Review Demo
* **Visual:** Screen capture of GitHub web UI. A developer pushes a commit adding an unescaped SQL query and an unhandled async HTTP timeout to `src/auth.ts`.
* Fast cut to Fly.io live log showing webhook ingestion in **9.8 ms**, Redis job enqueue, and worker processing.
* Switch back to the GitHub PR: `@prism-review[bot]` posts an inline comment on line 37 citing `STD-SEC-01: Parameterized Database Queries`, providing the exact vulnerability explanation, and rendering an interactive "Commit suggestion" markdown diff block.
* Developer clicks the 👍 reaction button; cut to Prometheus dashboard showing `feedback_reactions_total` incrementing and acceptance rate updating in real-time.
* **Audio / Voiceover:**
  > *"Here's Prism live in production. A developer opens a pull request. The webhook hits our FastAPI gateway—acknowledged in under 12 milliseconds. Our worker retrieves the repository's explicit standards from `.prism/standards.yml`, uses hybrid BM25 and vector search to find relevant rules, scrubs secrets, and runs review inference.*
  > *Prism posts an inline comment directly on the violating line with rule citation and a one-click commit suggestion. When the developer reacts with thumbs up or resolves the thread, our telemetry records it live to track real acceptance."*
* **On-screen Text Overlay:**
  * Ingestion p95: 11.8 ms
  * Zero Secrets Leaked (Regex Scrubbing)
  * Live Acceptance Rate Telemetry

---

### [0:50 – 1:20] Scene 3: The Evaluation Harness & Results Table
* **Visual:** Switch to code editor showing `evals/harness/matcher.py` with the `MultiMatcher` architecture diagram. Then screen transitions to the benchmark table in `docs/results.md`. Highlighting the test split numbers with camera zoom on the `Fine-Tuned 7B (AWQ 4-bit)` row.
* **Audio / Voiceover:**
  > *"Most AI tools are unmeasured because fuzzy matching on multi-file diffs is hard. I built an evaluation harness with a MultiMatcher that accounts for plus-or-minus 3 line tolerances and embedding cosine similarity.*
  > *We tested across a frozen, decontaminated 120-PR test split. The results: Base Qwen-7B was unusable with 73.5% false positives on clean code. By fine-tuning with 25% negative clean examples and rejection-sampled distillation, we curbed false positives by 68%, reaching an F1 score of 0.9304."*
* **On-screen Text Overlay:**
  * Test Benchmark: 120 PRs (Frozen `trpc` & `vite` repos)
  * Frontier Sonnet F1: 0.9613
  * Fine-Tuned 7B AWQ F1: 0.9304 (95% CI: [0.893, 0.959])
  * Clean-PR FPR: 0.2353

---

### [1:20 – 1:45] Scene 4: The Quality-vs-Cost Chart & Economics
* **Visual:** Full-screen render of `docs/images/quality_vs_cost.png` (Pareto Frontier plot). Visual cursor highlights Claude 3.5 Sonnet at $0.01334 and traces down the Pareto curve to Fine-Tuned 7B AWQ at $0.00036.
* An animated bar chart compares monthly spend on 10,000 PRs: $133.40 (Sonnet) vs $3.60 (Prism 7B AWQ).
* **Audio / Voiceover:**
  > *"Here is the Pareto frontier. Running Claude 3.5 Sonnet costs over 1.3 cents per review and takes 250 milliseconds. Our fine-tuned 7B model, quantized to 4-bit AWQ and served on an NVIDIA A10G via vLLM, delivers 40.5 millisecond p50 latency and costs 0.036 cents per review.*
  > *That is 37.1 times cheaper and 6 times faster, while achieving 96.8% of frontier F1 quality."*
* **On-screen Text Overlay:**
  * Cost per Review: $0.00036 (37.1× lower than Frontier API)
  * Latency p50: 40.5 ms (vLLM AWQ)
  * 10,000 PRs: $3.60 total GPU cost

---

### [1:45 – 2:00] Scene 5: What It Taught Me & Conclusion
* **Visual:** Terminal executing `make reproduce`. The evaluation script runs, verifies all 120 PRs, and prints the exact matching confidence intervals. Concludes with GitHub repository URL `github.com/binodbishwakarama-max/codeagent`.
* **Audio / Voiceover:**
  > *"What this project taught me is that LLMs don't fail at finding bugs—they fail at knowing when to stay silent. The breakthrough was teaching silence through negative dataset balance and multi-stage self-verification.*
  > *The entire benchmark is reproducible with a single command: `make reproduce`. Clone the repo, check out the write-up, and inspect the numbers yourself."*
* **On-screen Text Overlay:**
  * Key Insight: Teaching Silence via Negative Examples
  * Reproduce: `make reproduce`
  * Repo: `github.com/binodbishwakarama-max/codeagent`

---
