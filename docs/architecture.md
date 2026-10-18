# Architecture Overview

Prism Review is a GitHub App providing automated, context-aware pull request reviews aligned with an organization's custom engineering standards.

```
+-----------------------------------------------------------------------------------------+
|                                    GitHub Ecosystem                                     |
+-----------------------------------------------------------------------------------------+
        | Webhook Events (PR opened/synchronize)                      ^ Inline Comments
        v                                                             | (Reviews API)
+------------------+         +--------------------+         +--------------------+
|  FastAPI App     |  enqueue|     Redis Queue    | dequeue |     Arq Worker     |
| (prism.api)      | ------->|   (prism.queue)    | ------->| (prism.queue.jobs) |
+------------------+         +--------------------+         +--------------------+
                                                                      |
                                                                      v
                                                            +--------------------+
                                                            |  Review Pipeline   |
                                                            |   (prism.review)   |
                                                            +--------------------+
                                                              |                |
                                     Fetch Standards & Style  |                | Call Traced LLM
                                                              v                v
                                                    +------------------+ +---------------+
                                                    | pgvector Indexer | |  LLM Client   |
                                                    | (prism.retrieval)| | (Langfuse/OTel|
                                                    +------------------+ +---------------+
```

## System Components

1. **GitHub Ingestion (`prism.github`)**:
   - Authenticates as a GitHub App using RS256 JWTs and installation access tokens.
   - Verifies HMAC webhook signatures.
   - Parses unified diffs, line numbers, and pull request changes.

2. **Async Task Execution (`prism.queue`)**:
   - Webhook requests acknowledge immediately to GitHub with 202 Accepted.
   - Reviews are scheduled as asynchronous jobs through Redis and `arq`.

3. **Knowledge Retrieval (`prism.retrieval`)**:
   - Encodes internal style guides, architectural decisions (ADRs), and rules into pgvector.
   - Retrieves top-k relevant rules based on changed files, diff hunks, and commit messages.

4. **Review & LLM Execution (`prism.review`)**:
   - Formulates prompts incorporating diff context and retrieved standards.
   - Enforces execution timeouts, typed output validation, and OpenTelemetry / Langfuse tracing.
   - Filters false positives and posts inline comments at exact hunk line coordinates.

5. **Evaluation Harness (`evals`)**:
   - Mines real-world PRs and benchmarks review precision, recall, and hallucination rates.
   - Runs deterministic smoke tests (`make eval-smoke`) and full experiment evaluations (`make eval-full`).

6. **Model Adaptation (`training`)**:
   - Pipelines for formatting review instructions and fine-tuning lightweight models (e.g., LoRA).
