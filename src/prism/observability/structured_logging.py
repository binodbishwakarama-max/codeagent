"""Structured JSON logging with correlation ID tracking across API and background worker."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

# Context variables storing per-request or per-job correlation metadata
_correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_context_metadata_ctx: ContextVar[dict[str, Any] | None] = ContextVar(
    "context_metadata", default=None
)


def set_correlation_id(delivery_id: str | None) -> None:
    """Set the delivery_id / correlation_id for the current asynchronous execution context."""
    _correlation_id_ctx.set(delivery_id)


def get_correlation_id() -> str | None:
    """Retrieve the active delivery_id / correlation_id, if any."""
    return _correlation_id_ctx.get()


def bind_log_context(**kwargs: Any) -> None:
    """Attach structured contextual fields (e.g., repo, pr, head_sha) to subsequent log records."""
    current = dict(_context_metadata_ctx.get() or {})
    current.update({k: v for k, v in kwargs.items() if v is not None})
    _context_metadata_ctx.set(current)


def clear_log_context() -> None:
    """Clear correlation ID and contextual metadata for the current context."""
    _correlation_id_ctx.set(None)
    _context_metadata_ctx.set(None)


class StructuredJSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects with correlation IDs and context."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize LogRecord into a standardized JSON string."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include correlation ID if set
        correlation_id = _correlation_id_ctx.get()
        if correlation_id:
            log_data["correlation_id"] = correlation_id
            log_data["delivery_id"] = correlation_id

        # Include bound context metadata (repo, pr, head_sha, etc.)
        metadata = _context_metadata_ctx.get() or {}
        for k, v in metadata.items():
            if k not in log_data:
                log_data[k] = v

        # Include exception information if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include extra attributes attached to record
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            for k, v in record.extra_fields.items():
                if k not in log_data:
                    log_data[k] = v

        return json.dumps(log_data, default=str)


def setup_structured_logging(level: str = "INFO", handler: logging.Handler | None = None) -> None:
    """Configure the root logger or custom handler with StructuredJSONFormatter."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    formatter = StructuredJSONFormatter()

    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
