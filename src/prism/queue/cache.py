"""Redis-based LLM result cache keyed by deterministic content hashes."""

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cache TTL: 7 days
CACHE_TTL_SECONDS = 7 * 24 * 3600

# Metrics counters (in-process; emitted as structured logs)
_cache_hits = 0
_cache_misses = 0


def _build_cache_key(
    model: str,
    prompt_version: str,
    standards_hash: str,
    chunk_text: str,
) -> str:
    """Build a deterministic cache key from review inputs.

    Returns sha256 hex digest of concatenated inputs.
    """
    content = f"{model}:{prompt_version}:{standards_hash}:{chunk_text}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"prism:llm_cache:{digest}"


def compute_standards_hash(standards_text: str) -> str:
    """Compute sha256 digest of rendered standards for cache key construction."""
    return hashlib.sha256(standards_text.encode("utf-8")).hexdigest()


async def get_cached_result(
    redis_client: Any,
    model: str,
    prompt_version: str,
    standards_hash: str,
    chunk_text: str,
) -> dict[str, Any] | None:
    """Look up a cached LLM result in Redis.

    Returns the deserialized JSON dict on cache hit, or None on miss.
    """
    global _cache_hits, _cache_misses

    if redis_client is None:
        _cache_misses += 1
        return None

    key = _build_cache_key(model, prompt_version, standards_hash, chunk_text)

    try:
        raw = await redis_client.get(key)
        if raw is not None:
            _cache_hits += 1
            logger.debug("LLM cache HIT: %s", key[:24])
            data = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            if isinstance(data, dict):
                return data
            return None
        _cache_misses += 1
        logger.debug("LLM cache MISS: %s", key[:24])
        return None
    except Exception as exc:
        _cache_misses += 1
        logger.warning("LLM cache read error: %s", exc)
        return None


async def set_cached_result(
    redis_client: Any,
    model: str,
    prompt_version: str,
    standards_hash: str,
    chunk_text: str,
    result: dict[str, Any],
    ttl: int = CACHE_TTL_SECONDS,
) -> None:
    """Store an LLM result in Redis with the configured TTL."""
    if redis_client is None:
        return

    key = _build_cache_key(model, prompt_version, standards_hash, chunk_text)

    try:
        serialized = json.dumps(result)
        await redis_client.set(key, serialized, ex=ttl)
        logger.debug("LLM cache SET: %s (TTL=%ds)", key[:24], ttl)
    except Exception as exc:
        logger.warning("LLM cache write error: %s", exc)


def get_cache_stats() -> dict[str, int]:
    """Return current in-process cache hit/miss counters."""
    return {
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "total_lookups": _cache_hits + _cache_misses,
        "hit_rate_pct": (
            round(100 * _cache_hits / (_cache_hits + _cache_misses))
            if (_cache_hits + _cache_misses) > 0
            else 0
        ),
    }


def log_cache_stats() -> None:
    """Emit cache hit/miss stats as a structured log line."""
    stats = get_cache_stats()
    logger.info(
        "LLM cache stats: hits=%d misses=%d total=%d hit_rate=%d%%",
        stats["cache_hits"],
        stats["cache_misses"],
        stats["total_lookups"],
        stats["hit_rate_pct"],
    )
