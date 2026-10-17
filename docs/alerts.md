# Operational Alerting Rules & Runbooks

This document defines the production alerting policies, PromQL triggers, severity classifications, and operational runbook steps for Prism Review.

---

## Alerting Policies Overview

Alert rules are defined declaratively in [`ops/prometheus/alerts.yml`](../ops/prometheus/alerts.yml) and evaluated continuously by Prometheus.

| Alert Name | Condition / Expression | Window | Severity | Target Tier |
| :--- | :--- | :---: | :---: | :--- |
| **`HighReviewErrorRate`** | Review failure rate $> 5\%$ | $10\text{ min}$ | **Critical** | Worker / GitHub API |
| **`HighReviewLatencyP95`** | $p95\text{ latency} > 120\text{ s}$ | $5\text{ min}$ | **Warning** | Worker / LLM Model |
| **`HighQueueDepth`** | Pending jobs $> 50$ | $5\text{ min}$ | **Warning** | Queue / Redis |

---

## 1. HighReviewErrorRate

### Definition
Fires when the proportion of failed PR review jobs exceeds 5% over a continuous 10-minute sliding window.

- **PromQL Expression**:
  ```promql
  (sum(rate(reviews_total{status="failed"}[10m])) / (sum(rate(reviews_total[10m])) > 0 or 1)) > 0.05
  ```
- **Severity**: Critical (P1)
- **Impact**: Developer pull requests are failing to receive automated review findings, blocking CI check runs.

### Operational Runbook:
1. **Inspect Worker Error Logs**:
   Filter logs by `level="ERROR"` and correlation ID:
   ```bash
   docker-compose logs --tail=100 -f | jq 'select(.level=="ERROR")'
   ```
2. **Examine Dead Letter Queue**:
   Check permanently failed records in Postgres `dead_letter_jobs` table:
   ```sql
   SELECT id, repo, pr, error_message, attempts, created_at
   FROM dead_letter_jobs
   ORDER BY created_at DESC LIMIT 10;
   ```
3. **Check GitHub API Outages & Rate Limits**:
   Inspect the `github_api_errors_total` metric:
   ```promql
   sum(rate(github_api_errors_total[5m])) by (operation)
   ```
   If GitHub returned `403 SecondaryRateLimit` or `5xx`, verify installation access token rotation and token-bucket limiter state.
4. **Verify Database Connectivity**:
   Confirm Postgres and pgvector connection pool health:
   ```bash
   docker-compose exec postgres pg_isready -U prism -d prism
   ```

---

## 2. HighReviewLatencyP95

### Definition
Fires when the 95th percentile ($p95$) of end-to-end pull request review processing latency exceeds 120 seconds over a 5-minute evaluation window.

- **PromQL Expression**:
  ```promql
  histogram_quantile(0.95, sum(rate(review_latency_seconds_bucket[5m])) by (le)) > 120
  ```
- **Severity**: Warning (P2)
- **Impact**: Delayed review comments on developer PRs; feedback loops degraded.

### Operational Runbook:
1. **Trace Breakdown in Langfuse**:
   Open the Langfuse dashboard (`http://localhost:3000`) and locate the slowest traces. Check the breakdown across spans:
   - `diff_fetch`: slow GitHub API response or massive diff (>10,000 lines).
   - `retrieval`: slow vector embedding or AST chunk scan across repository.
   - `llm_call`: LLM model endpoint slow response, throttling, or repair prompts triggered.
2. **Evaluate Cache Hit Rate**:
   Check whether the Redis result cache is active:
   ```promql
   cache_hit_ratio
   ```
   If hit ratio drops to 0%, check Redis connection health.
3. **Review Concurrency Limits**:
   Review concurrency is bounded by `max_concurrency` (default 5 concurrent chunks). Adjust `LLM_TIMEOUT_SECONDS` or scaling worker instances.

---

## 3. HighQueueDepth

### Definition
Fires when the number of queued review jobs waiting in Redis exceeds 50 for more than 5 minutes.

- **PromQL Expression**:
  ```promql
  queue_depth > 50
  ```
- **Severity**: Warning (P2)
- **Impact**: Backlog of pull requests waiting for review workers.

### Operational Runbook:
1. **Check Worker Process Health**:
   Verify arq worker processes are actively consuming jobs:
   ```bash
   ps aux | grep "prism.queue.worker"
   ```
2. **Examine Redis Memory & Queue Length**:
   Inspect Redis key lengths:
   ```bash
   docker-compose exec redis redis-cli zcard arq:queue
   ```
3. **Scale Up Worker Instances**:
   Scale background review workers horizontally:
   ```bash
   docker-compose up -d --scale worker=4
   ```
