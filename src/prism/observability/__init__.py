"""Observability instrumentation: telemetry tracing, Prometheus metrics, structured logging, and developer feedback."""

from prism.observability.feedback import (
    detect_touched_comment_lines,
    poll_comment_reactions,
    record_comment_reaction,
    record_line_touched,
    record_thread_resolved,
    refresh_acceptance_rate_metric,
)
from prism.observability.metrics import (
    acceptance_rate,
    cache_hit_ratio,
    findings_per_review,
    get_metrics_payload,
    github_api_errors_total,
    llm_cost_usd_total,
    queue_depth,
    record_acceptance_rate,
    record_cache_stats,
    record_github_api_error,
    record_queue_depth,
    record_review_completed,
    review_latency_seconds,
    reviews_total,
)
from prism.observability.structured_logging import (
    StructuredJSONFormatter,
    bind_log_context,
    clear_log_context,
    get_correlation_id,
    set_correlation_id,
    setup_structured_logging,
)
from prism.observability.tracing import (
    LLMSpan,
    ReviewTrace,
    TraceSpanNode,
    compute_config_hash,
    get_current_trace,
    set_current_trace,
    trace_llm_call,
)

__all__ = [
    # Metrics
    "reviews_total",
    "review_latency_seconds",
    "llm_cost_usd_total",
    "cache_hit_ratio",
    "queue_depth",
    "findings_per_review",
    "github_api_errors_total",
    "acceptance_rate",
    "record_review_completed",
    "record_cache_stats",
    "record_queue_depth",
    "record_github_api_error",
    "record_acceptance_rate",
    "get_metrics_payload",
    # Tracing
    "ReviewTrace",
    "TraceSpanNode",
    "compute_config_hash",
    "get_current_trace",
    "set_current_trace",
    "trace_llm_call",
    "LLMSpan",
    # Logging
    "StructuredJSONFormatter",
    "setup_structured_logging",
    "set_correlation_id",
    "get_correlation_id",
    "bind_log_context",
    "clear_log_context",
    # Feedback
    "record_comment_reaction",
    "record_thread_resolved",
    "record_line_touched",
    "detect_touched_comment_lines",
    "poll_comment_reactions",
    "refresh_acceptance_rate_metric",
]
