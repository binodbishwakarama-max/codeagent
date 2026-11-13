"""FastAPI application entrypoint for webhook ingestion and administrative APIs."""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

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


class DirectDiffReviewRequest(BaseModel):
    diff: str = Field(description="Unified git diff text to review")
    title: str = Field(default="Direct Review", description="Pull request title")
    standards_yaml: str | None = Field(
        default=None, description="Optional custom .prism/standards.yml content"
    )


FALLBACK_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Prism Review | Automated Code Review Platform</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; background: #090d16; color: #f1f5f9; padding: 2.5rem; text-align: center; }
    h1 { font-size: 2.5rem; margin-bottom: 1rem; color: #38bdf8; }
    p { color: #94a3b8; font-size: 1.1rem; margin-bottom: 1.5rem; }
    a { color: #38bdf8; text-decoration: none; border: 1px solid #38bdf8; padding: 0.5rem 1rem; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>💎 Prism Review Platform</h1>
  <p>Autonomous, standards-driven AI code review for pull requests with empirical benchmark verification.</p>
  <a href="/docs">View Interactive API Documentation</a>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def landing_page() -> str:
    """Serve interactive web dashboard and code review playground."""
    from pathlib import Path

    web_index = Path(__file__).parent / "web" / "index.html"
    if web_index.exists():
        return web_index.read_text(encoding="utf-8")
    return FALLBACK_LANDING_HTML


@app.post("/api/v1/review/diff", tags=["Review"])
async def review_diff_direct(req: DirectDiffReviewRequest) -> dict[str, Any]:
    """Analyze a unified diff directly and return structured review findings."""
    from prism.github.diff import parse_unified_diff
    from prism.review.redactor import redact_file_diffs
    from prism.review.standards import load_default_standards, parse_repo_standards

    file_diffs = parse_unified_diff(req.diff)
    redacted_diffs = redact_file_diffs(file_diffs)
    standards = (
        parse_repo_standards(req.standards_yaml) if req.standards_yaml else load_default_standards()
    )

    settings = _get_settings()
    findings: list[dict[str, Any]] = []

    if settings.anthropic_api_key or settings.llm_base_url:
        from prism.review.llm import create_llm_client
        from prism.review.pipeline import PRContext, review_pr

        pr_ctx = PRContext(
            repo_full_name="playground/review",
            pr_number=1,
            title=req.title,
            file_diffs=redacted_diffs,
            standards=standards.standards,
        )
        res = await review_pr(pr_ctx, settings, llm_client=create_llm_client(settings))
        for f in res.findings:
            findings.append(
                {
                    "path": f.path,
                    "line": f.line,
                    "category": f.category.value,
                    "severity": f.severity.value,
                    "message": f.message,
                    "suggestion": f.suggestion,
                    "confidence": f.confidence,
                    "standard_id": f.standard_id,
                }
            )
    else:
        # Deterministic standards rules engine (ideal for zero-credential cloud demos)
        from prism.review.rules_engine import evaluate_rules_as_dicts

        findings = evaluate_rules_as_dicts(redacted_diffs)

    return {
        "status": "completed",
        "files_analyzed": len(redacted_diffs),
        "findings_count": len(findings),
        "findings": findings,
    }


class PRUrlReviewRequest(BaseModel):
    url: str = Field(
        description="GitHub Pull Request URL or shorthand (e.g. https://github.com/owner/repo/pull/123)"
    )
    token: str | None = Field(
        default=None, description="Optional GitHub token for private repositories"
    )
    standards_yaml: str | None = Field(default=None, description="Optional custom standards YAML")


@app.post("/api/v1/review/pr-url", tags=["Review"])
async def review_pr_url(req: PRUrlReviewRequest) -> dict[str, Any]:
    """Fetch a public or private GitHub PR and execute automated review analysis."""
    from prism.review.cli import fetch_pr_details_and_diff, parse_pr_url

    try:
        repo, pr_number = parse_pr_url(req.url)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err

    try:
        meta, raw_diff = await fetch_pr_details_and_diff(repo, pr_number, req.token)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to fetch PR from GitHub: {exc}"
        ) from exc

    diff_req = DirectDiffReviewRequest(
        diff=raw_diff,
        title=meta.get("title", f"PR #{pr_number}"),
        standards_yaml=req.standards_yaml,
    )
    result = await review_diff_direct(diff_req)
    result["repo"] = repo
    result["pr_number"] = pr_number
    result["title"] = meta.get("title", f"PR #{pr_number}")
    result["raw_diff"] = raw_diff
    return result


@app.get("/healthz", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/readyz", tags=["Health"])
async def readiness_check() -> JSONResponse:
    """Readiness probe checking database, Redis, and external LLM connectivity."""
    settings = _get_settings()
    db_ok = True
    db_err: str | None = None
    redis_ok = True
    redis_err: str | None = None
    llm_ok = True
    llm_err: str | None = None

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

    # 3. Check External vLLM / LLM connection (when custom endpoint configured)
    check_llm_url = settings.llm_base_url
    if not check_llm_url and settings.llm_provider == "openai_compat":
        check_llm_url = "http://localhost:8000/v1"

    if check_llm_url:
        try:
            import httpx

            base = check_llm_url.rstrip("/")
            health_url = f"{base[:-3]}/health" if base.endswith("/v1") else f"{base}/health"

            async with httpx.AsyncClient(timeout=2.0) as client:
                try:
                    resp = await client.get(health_url)
                except httpx.HTTPError:
                    # Fallback to models endpoint probe
                    models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
                    resp = await client.get(models_url)

                if resp.status_code >= 400:
                    llm_ok = False
                    llm_err = f"Inference server returned HTTP {resp.status_code}"
        except Exception as exc:
            llm_ok = False
            llm_err = str(exc)

    all_ready = db_ok and redis_ok and llm_ok
    status_code = 200 if all_ready else 503

    content: dict[str, Any] = {
        "status": "ready" if all_ready else "not_ready",
        "database": "ok" if db_ok else f"unhealthy: {db_err}",
        "redis": "ok" if redis_ok else f"unhealthy: {redis_err}",
    }
    if check_llm_url:
        content["llm"] = "ok" if llm_ok else f"unhealthy: {llm_err}"

    return JSONResponse(status_code=status_code, content=content)


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
                    try:
                        await _arq_pool.enqueue_job("run_thread_reply", reply_job_payload)
                        logger.info(
                            "Enqueued thread reply for %s#%s comment %s",
                            repo_name,
                            pr_num,
                            comment_id,
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to enqueue thread reply into Arq pool for %s#%s: %s",
                            repo_name,
                            pr_num,
                            exc,
                        )
                        raise HTTPException(
                            status_code=503,
                            detail="Background worker queue is temporarily unavailable",
                        ) from exc
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
        try:
            await _arq_pool.enqueue_job("run_review", job_payload)
            logger.info(
                "Enqueued review job for %s#%s at %s (delivery=%s)",
                repo_full_name,
                pr_number,
                head_sha,
                x_github_delivery,
            )
        except Exception as exc:
            logger.error(
                "Failed to enqueue review job into Arq pool for %s#%s: %s",
                repo_full_name,
                pr_number,
                exc,
            )
            raise HTTPException(
                status_code=503,
                detail="Background review queue is temporarily unavailable",
            ) from exc
    else:
        logger.error(
            "Arq pool unavailable — cannot enqueue review job for %s#%s (delivery=%s)",
            repo_full_name,
            pr_number,
            x_github_delivery,
        )
        raise HTTPException(
            status_code=503,
            detail="Background review queue is currently unavailable",
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
