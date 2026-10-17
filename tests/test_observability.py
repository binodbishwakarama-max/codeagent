"""Unit and integration tests for src/prism/observability/.

Tests:
1. Langfuse ReviewTrace tree hierarchy, spans, metadata tagging, and rendering
2. Prometheus metrics recording and /metrics endpoint exposition
3. Feedback loop recording (reactions, resolutions, line touches) and acceptance rate
4. Structured JSON logging with correlation IDs
5. GitHub webhook feedback event handling
"""

import hashlib
import hmac
import json
import logging

import pytest
from fastapi.testclient import TestClient

from prism.api import app
from prism.config import Settings
from prism.db import (
    create_tables,
    get_feedback_records,
)
from prism.observability.feedback import (
    record_comment_reaction,
    record_line_touched,
    record_thread_resolved,
    refresh_acceptance_rate_metric,
)
from prism.observability.metrics import (
    get_metrics_payload,
    record_acceptance_rate,
    record_cache_stats,
    record_github_api_error,
    record_queue_depth,
    record_review_completed,
)
from prism.observability.structured_logging import (
    StructuredJSONFormatter,
    bind_log_context,
    clear_log_context,
    get_correlation_id,
    set_correlation_id,
)
from prism.observability.tracing import (
    ReviewTrace,
    compute_config_hash,
    get_current_trace,
)

TEST_DB = "sqlite+aiosqlite:///:memory:"


def make_sig(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


# ---------------------------------------------------------------------------
# 1. Tracing Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_trace_hierarchy_and_tags() -> None:
    """Verify ReviewTrace captures hierarchical spans, timing, and tags."""
    repo = "acme/backend"
    pr = 42
    sha = "abcdef123456"
    config_hash = compute_config_hash(standards_content="rule1: strict")

    async with ReviewTrace(
        repo=repo,
        pr=pr,
        head_sha=sha,
        config_hash=config_hash,
        delivery_id="deliv-999",
    ) as trace:
        assert get_current_trace() is trace
        assert f"repo:{repo}" in trace.tags
        assert f"pr:{pr}" in trace.tags

        # Child span 1: diff_fetch
        async with trace.span("diff_fetch", metadata={"files": 3}) as s1:
            assert s1.name == "diff_fetch"
            s1.metadata["bytes"] = 1024

        # Child span 2: retrieval
        async with trace.span("retrieval", span_type="retriever", metadata={"mode": "hybrid"}):
            pass

        # Child span 3: llm_call
        async with trace.span(
            "llm_call:chunk_0",
            span_type="generation",
            model="claude-3-5-sonnet",
            prompt_version="v1.2-standard",
        ) as s3:
            s3.input_tokens = 1500
            s3.output_tokens = 120
            s3.cost_usd = 0.006

        # Child span 4: postprocess
        async with trace.span("postprocess", metadata={"raw": 4, "final": 2}):
            pass

        # Child span 5: github_post
        async with trace.span("github_post", metadata={"comments": 2}):
            pass

    assert get_current_trace() is None
    assert trace.status == "completed"
    assert trace.duration_ms >= 0.0
    assert len(trace.spans) == 5

    # Check serialization
    trace_dict = trace.to_dict()
    assert trace_dict["repo"] == repo
    assert trace_dict["pr"] == pr
    assert trace_dict["delivery_id"] == "deliv-999"
    assert len(trace_dict["tree"]["children"]) == 5

    # Check text tree rendering
    tree_text = trace.render_tree()
    assert "diff_fetch" in tree_text
    assert "retrieval" in tree_text
    assert "llm_call:chunk_0" in tree_text
    assert "postprocess" in tree_text
    assert "github_post" in tree_text


# ---------------------------------------------------------------------------
# 2. Prometheus Metrics Tests
# ---------------------------------------------------------------------------


def test_prometheus_metrics_recording_and_endpoint() -> None:
    """Verify Prometheus instruments record values and /metrics endpoint outputs text format."""
    record_review_completed(
        status="completed", latency_seconds=1.25, cost_usd=0.0035, findings_count=3
    )
    record_cache_stats(hits=8, misses=2)
    record_queue_depth(depth=15)
    record_github_api_error(operation="create_check_run")
    record_acceptance_rate(rate=0.92)

    payload_bytes, content_type = get_metrics_payload()
    payload = payload_bytes.decode("utf-8")

    assert "reviews_total" in payload
    assert "review_latency_seconds" in payload
    assert "llm_cost_usd_total" in payload
    assert "cache_hit_ratio" in payload
    assert "queue_depth" in payload
    assert "findings_per_review" in payload
    assert "github_api_errors_total" in payload
    assert "acceptance_rate" in payload

    # Test via FastAPI test client
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "reviews_total" in resp.text


# ---------------------------------------------------------------------------
# 3. Developer Feedback Loop Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_recording_and_acceptance_rate() -> None:
    """Verify recording comment reactions, resolutions, and line touches."""
    db_url = TEST_DB
    await create_tables(db_url)

    # 1. Record thumbs up reaction
    r1 = await record_comment_reaction(
        database_url=db_url,
        repo="acme/repo",
        pr=10,
        comment_id="comment-101",
        reaction_content="+1",
        author="dev_alice",
        file_path="src/main.py",
        line_number=45,
    )
    assert r1 > 0

    # 2. Record resolved thread
    r2 = await record_thread_resolved(
        database_url=db_url,
        repo="acme/repo",
        pr=10,
        comment_id="comment-102",
        author="dev_bob",
        file_path="src/utils.py",
        line_number=12,
    )
    assert r2 > 0

    # 3. Record line touched in commit
    r3 = await record_line_touched(
        database_url=db_url,
        repo="acme/repo",
        pr=10,
        comment_id="comment-103",
        file_path="src/app.py",
        line_number=88,
        head_sha="sha_update_123",
        commit_author="dev_alice",
    )
    assert r3 > 0

    # 4. Record negative reaction
    r4 = await record_comment_reaction(
        database_url=db_url,
        repo="acme/repo",
        pr=10,
        comment_id="comment-104",
        reaction_content="-1",
        author="dev_charlie",
    )
    assert r4 > 0

    records = await get_feedback_records(database_url=db_url, repo="acme/repo")
    assert len(records) == 4

    # Acceptance rate = 3 positive (r1, r2, r3) / 4 total = 0.75
    rate = await refresh_acceptance_rate_metric(database_url=db_url)
    assert rate == 0.75


# ---------------------------------------------------------------------------
# 4. Structured JSON Logging Tests
# ---------------------------------------------------------------------------


def test_structured_json_logging() -> None:
    """Verify JSON formatter outputs correlation ID and contextual metadata."""
    formatter = StructuredJSONFormatter()
    logger = logging.getLogger("test_obs_logger")

    set_correlation_id("delivery-uuid-xyz-123")
    bind_log_context(repo="acme/repo", pr=101, head_sha="11223344")

    record = logger.makeRecord(
        name="test_obs_logger",
        level=logging.INFO,
        fn="test.py",
        lno=20,
        msg="Processing review step %s",
        args=("diff_fetch",),
        exc_info=None,
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["level"] == "INFO"
    assert data["message"] == "Processing review step diff_fetch"
    assert data["correlation_id"] == "delivery-uuid-xyz-123"
    assert data["delivery_id"] == "delivery-uuid-xyz-123"
    assert data["repo"] == "acme/repo"
    assert data["pr"] == 101
    assert data["head_sha"] == "11223344"

    clear_log_context()
    assert get_correlation_id() is None


# ---------------------------------------------------------------------------
# 5. Webhook Feedback Ingestion Tests
# ---------------------------------------------------------------------------


def test_webhook_feedback_events() -> None:
    """Verify incoming feedback webhook events (review threads, reactions)."""
    secret = "test-secret-webhook"
    app.dependency_overrides = {}

    settings = Settings(
        github_app_id="123",
        github_private_key="FAKE",
        github_webhook_secret=secret,
        database_url=TEST_DB,
        redis_url="redis://localhost:6379/0",
    )

    import prism.api

    prism.api._settings = settings

    client = TestClient(app)

    # 1. pull_request_review_thread (resolved)
    thread_payload = {
        "action": "resolved",
        "repository": {"full_name": "acme/repo"},
        "pull_request": {"number": 15},
        "sender": {"login": "alice"},
        "thread": {
            "id": "PRRT_123",
            "comments": [{"id": 555444}],
        },
    }
    raw = json.dumps(thread_payload).encode("utf-8")
    sig = make_sig(raw, secret)
    resp = client.post(
        "/webhook/github",
        content=raw,
        headers={
            "X-GitHub-Event": "pull_request_review_thread",
            "X-Hub-Signature-256": sig,
            "X-GitHub-Delivery": "deliv-thread-1",
        },
    )
    assert resp.status_code == 200
    assert resp.json().get("feedback") == "resolved"

    # 2. reaction (created)
    reaction_payload = {
        "action": "created",
        "repository": {"full_name": "acme/repo"},
        "pull_request": {"number": 15},
        "comment": {"id": 555444},
        "reaction": {"content": "+1"},
        "sender": {"login": "bob"},
    }
    raw_react = json.dumps(reaction_payload).encode("utf-8")
    sig_react = make_sig(raw_react, secret)
    resp_react = client.post(
        "/webhook/github",
        content=raw_react,
        headers={
            "X-GitHub-Event": "reaction",
            "X-Hub-Signature-256": sig_react,
            "X-GitHub-Delivery": "deliv-react-1",
        },
    )
    assert resp_react.status_code == 200
    assert resp_react.json().get("feedback") == "reaction"
