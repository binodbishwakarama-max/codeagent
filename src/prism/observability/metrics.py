"""Operational metrics collection for latency, throughput, token usage, and errors.

Exposes Prometheus metrics for Grafana dashboards and Prometheus scrapers.
"""

from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ---------------------------------------------------------------------------
# Prometheus Metric Instruments
# ---------------------------------------------------------------------------

# reviews_total{status}
reviews_total = Counter(
    "reviews_total",
    "Total number of pull request reviews processed",
    ["status"],
)

# review_latency_seconds (histogram)
review_latency_seconds = Histogram(
    "review_latency_seconds",
    "End-to-end pull request review latency in seconds",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

# llm_cost_usd_total
llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Cumulative dollar cost of all LLM invocations in USD",
)

# cache_hit_ratio
cache_hit_ratio = Gauge(
    "cache_hit_ratio",
    "Current cache hit ratio for LLM review results (0.0 to 1.0)",
)

# queue_depth
queue_depth = Gauge(
    "queue_depth",
    "Current number of pending review jobs in the job queue",
)

# findings_per_review
findings_per_review = Histogram(
    "findings_per_review",
    "Distribution of actionable findings posted per review",
    buckets=(0, 1, 2, 5, 10, 20, 50),
)

# github_api_errors_total
github_api_errors_total = Counter(
    "github_api_errors_total",
    "Total number of GitHub API errors encountered",
    ["operation"],
)

# acceptance_rate
acceptance_rate = Gauge(
    "acceptance_rate",
    "Live developer acceptance rate of review findings (0.0 to 1.0)",
)


# ---------------------------------------------------------------------------
# Helper instrumentation functions
# ---------------------------------------------------------------------------


def record_review_completed(
    status: str,
    latency_seconds: float,
    cost_usd: float = 0.0,
    findings_count: int = 0,
) -> None:
    """Record metrics for a finished review job."""
    reviews_total.labels(status=status).inc()
    if latency_seconds >= 0.0:
        review_latency_seconds.observe(latency_seconds)
    if cost_usd > 0.0:
        llm_cost_usd_total.inc(cost_usd)
    findings_per_review.observe(findings_count)


def record_cache_stats(hits: int, misses: int) -> None:
    """Update cache hit ratio gauge based on current hit/miss counters."""
    total = hits + misses
    ratio = (hits / total) if total > 0 else 0.0
    cache_hit_ratio.set(round(ratio, 4))


def record_queue_depth(depth: int) -> None:
    """Update the current queue depth gauge."""
    queue_depth.set(max(0, depth))


def record_github_api_error(operation: str = "general") -> None:
    """Increment GitHub API error counter."""
    github_api_errors_total.labels(operation=operation).inc()


def record_acceptance_rate(rate: float) -> None:
    """Update live acceptance rate gauge."""
    acceptance_rate.set(max(0.0, min(1.0, rate)))


def get_metrics_payload(registry: Any = None) -> tuple[bytes, str]:
    """Generate Prometheus exposition text format bytes and matching content-type header."""
    reg = registry if registry is not None else REGISTRY
    data = generate_latest(reg)
    return data, CONTENT_TYPE_LATEST
