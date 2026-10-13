"""Arq worker settings, process lifecycle, and graceful shutdown configuration."""

import logging
import signal
from typing import Any

from prism.config import Settings
from prism.queue.jobs import process_pull_request_review, run_review

logger = logging.getLogger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    """Initialize worker-level shared resources on startup."""
    settings = Settings()

    # Initialize Redis client for shared use across jobs
    try:
        from redis.asyncio import Redis

        ctx["redis"] = Redis.from_url(settings.redis_url, decode_responses=False)
        logger.info("Worker Redis connected: %s", settings.redis_url)
    except Exception as exc:
        logger.warning("Worker Redis unavailable: %s", exc)
        ctx["redis"] = None

    ctx["settings"] = settings
    logger.info("Worker started")


async def shutdown(ctx: dict[str, Any]) -> None:
    """Gracefully close shared resources on worker shutdown."""
    redis = ctx.get("redis")
    if redis is not None:
        await redis.aclose()
        logger.info("Worker Redis connection closed")

    logger.info("Worker shutdown complete")


def _handle_sigterm(signum: int, frame: Any) -> None:
    """Handle SIGTERM for graceful shutdown."""
    logger.info("Received SIGTERM, initiating graceful shutdown...")
    raise SystemExit(0)


# Register SIGTERM handler for graceful shutdown in containers
try:
    signal.signal(signal.SIGTERM, _handle_sigterm)
except (OSError, ValueError):
    # signal.signal can fail in some environments (e.g., non-main thread)
    pass


class WorkerSettings:
    """Arq worker configuration with retry policy and lifecycle hooks.

    - max_tries=3: retries with arq's built-in exponential backoff
    - job_timeout=300s: kill jobs that hang beyond 5 minutes
    - health_check_interval=30s: arq health monitoring
    """

    functions = [run_review, process_pull_request_review]
    on_startup = startup
    on_shutdown = shutdown
    max_tries = 3
    job_timeout = 300
    health_check_interval = 30
    retry_jobs = True
