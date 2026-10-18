"""FastAPI application entrypoint for webhook ingestion and administrative APIs."""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from prism.config import Settings
from prism.github.webhook import (
    check_and_set_idempotency,
    should_process_pull_request,
    verify_signature,
)
from prism.observability.feedback import (
    record_comment_reaction,
    record_thread_resolved,
)
from prism.observability.metrics import (
    get_metrics_payload,
)
from prism.observability.structured_logging import (
    bind_log_context,
    clear_log_context,
    set_correlation_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

_redis_client: Any = None
_arq_pool: Any = None
_settings: Settings | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and teardown shared application resources."""
    global _redis_client, _arq_pool, _settings

    _settings = Settings()

    try:
        from redis.asyncio import Redis

        _redis_client = Redis.from_url(_settings.redis_url, decode_responses=False)
        logger.info("Redis connected: %s", _settings.redis_url)
    except Exception as exc:
        logger.warning("Redis unavailable at startup (will run without cache): %s", exc)
        _redis_client = None

    try:
        import arq

        _arq_pool = await arq.create_pool(
            arq.connections.RedisSettings.from_dsn(_settings.redis_url)
        )
        logger.info("Arq job pool initialised")
    except Exception as exc:
        logger.warning("Arq pool unavailable at startup (jobs will not be enqueued): %s", exc)
        _arq_pool = None

    yield

    # Teardown
    if _redis_client is not None:
        await _redis_client.aclose()
    if _arq_pool is not None:
        await _arq_pool.aclose()


# ---------------------------------------------------------------------------
# FastAPI app & Middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="prism-review",
    description="Automated PR reviews against custom engineering standards",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
    """Propagate correlation ID (delivery_id) across execution context and response headers."""
    delivery_id = (
        request.headers.get("x-github-delivery")
        or request.headers.get("x-correlation-id")
        or str(uuid.uuid4())
    )
    set_correlation_id(delivery_id)
    try:
        response: Response = await call_next(request)
        response.headers["x-correlation-id"] = delivery_id
        return response
    finally:
        clear_log_context()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_settings() -> Settings:
    try:
        return Settings()
    except Exception:
        if _settings is not None:
            return _settings
        return Settings.model_construct(
            github_app_id="default-app",
            github_private_key="default-key",
            github_webhook_secret="default-secret",
            database_url="",
            redis_url="",
            blocked_installations=[],
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/readyz", tags=["Health"])
async def readiness_check() -> JSONResponse:
    """Readiness probe checking database and Redis connectivity."""
    settings = _get_settings()
    db_ok = True
    db_err: str | None = None
    redis_ok = True
    redis_err: str | None = None

    # 1. Check Database connection
    if settings.database_url:
        try:
            from sqlalchemy import text

            from prism.db import get_async_engine

            engine = get_async_engine(settings.database_url)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as exc:
            db_ok = False
            db_err = str(exc)
    else:
        db_ok = False
        db_err = "database_url not configured"

    # 2. Check Redis connection
    if _redis_client is not None:
        try:
            await _redis_client.ping()
        except Exception as exc:
            redis_ok = False
            redis_err = str(exc)
    elif settings.redis_url:
        try:
            from redis.asyncio import Redis

            r = Redis.from_url(settings.redis_url)
            await r.ping()
            await r.close()
        except Exception as exc:
            redis_ok = False
            redis_err = str(exc)
    else:
        redis_ok = False
        redis_err = "redis_url not configured"

    if db_ok and redis_ok:
        return JSONResponse(
            status_code=200,
            content={"status": "ready", "database": "ok", "redis": "ok"},
        )
    return JSONResponse(
        status_code=503,
        content={
            "status": "not_ready",
            "database": "ok" if db_ok else f"unhealthy: {db_err}",
            "redis": "ok" if redis_ok else f"unhealthy: {redis_err}",
        },
    )


@app.get("/metrics", tags=["Observability"])
async def metrics_endpoint() -> Response:
    """Expose Prometheus metrics for scrapers."""
    payload, content_type = get_metrics_payload()
    return Response(content=payload, media_type=content_type)


@app.post("/webhook/github", tags=["Webhooks"])
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
    x_github_delivery: str | None = Header(default=None),
) -> Response:
    """Receive and process incoming GitHub App webhook events.

    Verifies the HMAC-SHA256 signature, routes events (pull_request, review comment feedback),
    enforces idempotency via Redis, and enqueues background review jobs.
    Always returns 202 Accepted for new PR reviews — never blocks on review work.
    """
    settings = _get_settings()
    raw_body = await request.body()

    # Set correlation ID and bind contextual logging
    effective_delivery = x_github_delivery or str(uuid.uuid4())
    set_correlation_id(effective_delivery)
    bind_log_context(delivery_id=effective_delivery, github_event=x_github_event)

    # 1. Verify HMAC-SHA256 signature
    if not verify_signature(raw_body, settings.github_webhook_secret, x_hub_signature_256):
        logger.warning("Rejected webhook with invalid signature (delivery=%s)", effective_delivery)
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # 2. Parse payload
    try:
        payload: dict[str, Any] = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc

    # Check per-installation kill switch early
    installation_id = (payload.get("installation") or {}).get("id")
    if installation_id and int(installation_id) in settings.blocked_installations:
        logger.warning(
            "Webhook rejected: installation %d is blocked by per-installation kill switch",
            int(installation_id),
        )
        return Response(content="Installation blocked by kill switch", status_code=200)

    # 3. Handle developer feedback webhook events: thread resolution & comment reactions
    if x_github_event == "pull_request_review_thread":
        action = payload.get("action")
        if action == "resolved":
            thread = payload.get("thread", {})
            comments = thread.get("comments", [])
            comment_id = str(comments[0].get("id")) if comments else str(thread.get("id", "0"))
            repo_name = payload.get("repository", {}).get("full_name", "")
            pr_num = payload.get("pull_request", {}).get("number", 0)
            author = payload.get("sender", {}).get("login")
            if settings.database_url:
                await record_thread_resolved(
                    database_url=settings.database_url,
                    repo=repo_name,
                    pr=int(pr_num),
                    comment_id=comment_id,
                    author=author,
                )
            return JSONResponse(
                status_code=200,
                content={"status": "recorded", "feedback": "resolved"},
            )
        return JSONResponse(
            status_code=200,
            content={"status": "ignored", "reason": f"thread action '{action}' not tracked"},
        )

    if x_github_event == "reaction":
        action = payload.get("action")
        if action == "created":
            reaction = payload.get("reaction", {})
            content = reaction.get("content", "")
            comment = payload.get("comment", {})
            comment_id = str(comment.get("id", ""))
            repo_name = payload.get("repository", {}).get("full_name", "")
            pr_num = payload.get("pull_request", {}).get("number") or payload.get("issue", {}).get(
                "number", 0
            )
            author = reaction.get("user", {}).get("login")
            if settings.database_url and comment_id:
                await record_comment_reaction(
                    database_url=settings.database_url,
                    repo=repo_name,
                    pr=int(pr_num),
                    comment_id=comment_id,
                    reaction_content=content,
                    author=author,
                )
            return JSONResponse(
                status_code=200,
                content={"status": "recorded", "feedback": "reaction"},
            )
        return JSONResponse(
            status_code=200,
            content={"status": "ignored", "reason": f"reaction action '{action}' not tracked"},
        )

    if x_github_event == "pull_request_review_comment":
        action = payload.get("action")
        comment = payload.get("comment", {})
        body = comment.get("body", "").strip()
        repo_name = payload.get("repository", {}).get("full_name", "")
        pr_num = payload.get("pull_request", {}).get("number", 0)
        comment_id = str(comment.get("id", ""))
        author = comment.get("user", {}).get("login")

        if action == "created" and body in ("+1", "-1", "👍", "👎", "thumbs_up", "thumbs_down"):
            if settings.database_url and comment_id:
                await record_comment_reaction(
                    database_url=settings.database_url,
                    repo=repo_name,
                    pr=int(pr_num),
                    comment_id=comment_id,
                    reaction_content=body,
                    author=author,
                )
            return JSONResponse(
                status_code=200,
                content={"status": "recorded", "feedback": "review_comment_reaction"},
            )

        if action == "created":
            sender = payload.get("sender", {})
            sender_login = sender.get("login") or comment.get("user", {}).get("login")
            sender_type = sender.get("type") or comment.get("user", {}).get("type")
            in_reply_to_id = comment.get("in_reply_to_id")
            parent_body = (
                comment.get("in_reply_to", {}).get("body")
                if isinstance(comment.get("in_reply_to"), dict)
                else None
            )

            from prism.review.conversation import should_respond_to_comment

            should_reply, reply_reason = should_respond_to_comment(
                comment_body=body,
                parent_body=parent_body,
                in_reply_to_id=in_reply_to_id,
                sender_login=sender_login,
                sender_type=sender_type,
            )

            if should_reply:
                installation_id = payload.get("installation", {}).get("id")
                reply_job_payload: dict[str, Any] = {
                    "installation_id": installation_id,
                    "repo_full_name": repo_name,
                    "pr_number": int(pr_num),
                    "comment_id": int(comment_id) if comment_id else 0,
                    "in_reply_to_id": int(in_reply_to_id) if in_reply_to_id else None,
                    "body": body,
                    "author": author or sender_login or "developer",
                    "path": comment.get("path", ""),
                    "diff_hunk": comment.get("diff_hunk"),
                    "line": comment.get("line"),
                    "parent_body": parent_body,
                    "delivery_id": x_github_delivery,
                }

                if _arq_pool is not None:
                    await _arq_pool.enqueue_job("run_thread_reply", reply_job_payload)
                    logger.info(
                        "Enqueued thread reply for %s#%s comment %s",
                        repo_name,
                        pr_num,
                        comment_id,
                    )
                else:
                    from prism.queue.jobs import run_thread_reply

                    ctx = {"settings": settings, "redis": _redis_client}
                    asyncio.create_task(run_thread_reply(ctx, reply_job_payload))
                    logger.info(
                        "Scheduled direct thread reply task for %s#%s comment %s",
                        repo_name,
                        pr_num,
                        comment_id,
                    )

                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "accepted",
                        "event": "thread_reply",
                        "repo": repo_name,
                        "pr": pr_num,
                        "comment_id": comment_id,
                    },
                )

            return JSONResponse(
                status_code=200,
                content={
                    "status": "ignored",
                    "reason": f"comment not addressed to prism ({reply_reason})",
                },
            )

        return JSONResponse(
            status_code=200,
            content={"status": "ignored", "reason": f"comment action '{action}' not tracked"},
        )

    # 4. Process pull_request events
    if x_github_event != "pull_request":
        return JSONResponse(
            status_code=200,
            content={"status": "ignored", "reason": f"event '{x_github_event}' not handled"},
        )

    # Validate the pull request should be processed (action, draft, bot, fork)
    should_process, reason = should_process_pull_request(payload)
    if not should_process:
        logger.info("Ignoring pull_request event: %s", reason)
        return JSONResponse(
            status_code=200,
            content={"status": "ignored", "reason": reason},
        )

    # 5. Extract key fields for the review job
    pr = payload["pull_request"]
    installation_id = payload.get("installation", {}).get("id")
    repo_full_name = payload.get("repository", {}).get("full_name")
    pr_number = pr.get("number")
    head_sha = pr.get("head", {}).get("sha")

    if not all([installation_id, repo_full_name, pr_number, head_sha]):
        raise HTTPException(
            status_code=422,
            detail=(
                "Payload missing required fields: installation.id, "
                "repository.full_name, pull_request.number, head.sha"
            ),
        )

    # 6. Idempotency check: deduplicate by delivery_id and (repo, pr, head_sha)
    is_new = await check_and_set_idempotency(
        redis_client=_redis_client,
        delivery_id=x_github_delivery,
        repo_full_name=str(repo_full_name),
        pr_number=int(pr_number),
        head_sha=str(head_sha),
    )
    if not is_new:
        return JSONResponse(
            status_code=202,
            content={"status": "duplicate", "reason": "Already enqueued this delivery"},
        )

    # 7. Enqueue background review job
    job_payload: dict[str, Any] = {
        "installation_id": installation_id,
        "repo_full_name": repo_full_name,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "delivery_id": x_github_delivery,
    }

    if _arq_pool is not None:
        await _arq_pool.enqueue_job("run_review", job_payload)
        logger.info(
            "Enqueued review job for %s#%s at %s (delivery=%s)",
            repo_full_name,
            pr_number,
            head_sha,
            x_github_delivery,
        )
    else:
        # Log warning if pool is not initialized (e.g., in tests without Redis)
        logger.warning(
            "Arq pool unavailable — review job NOT enqueued for %s#%s",
            repo_full_name,
            pr_number,
        )

    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "repo": repo_full_name,
            "pr": pr_number,
            "sha": head_sha,
        },
    )
