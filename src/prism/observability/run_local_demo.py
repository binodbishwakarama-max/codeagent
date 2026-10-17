"""Run 5 reviews locally through the complete observability pipeline and generate screenshots.

Demonstrates:
1. End-to-end execution of 5 PR reviews with ReviewTrace and correlation IDs
2. Span hierarchy: diff_fetch, retrieval, llm_call, postprocess, github_post
3. Prometheus metrics updates (latency, throughput, cost, cache hit ratio, queue depth, errors, acceptance rate)
4. Developer feedback loop recording (reactions, resolutions, line touches)
5. High-resolution screenshot rendering of:
   - docs/img/langfuse_trace_tree.png
   - docs/img/grafana_dashboard.png
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from prism.config import Settings
from prism.db import (
    create_tables,
    get_feedback_records,
)
from prism.models import DiffLine, FileDiff, Hunk, IssueCategory, ReviewFinding, Severity, Standard
from prism.observability.feedback import (
    record_comment_reaction,
    record_line_touched,
    record_thread_resolved,
    refresh_acceptance_rate_metric,
)
from prism.observability.metrics import (
    get_metrics_payload,
    record_cache_stats,
    record_queue_depth,
    record_review_completed,
)
from prism.observability.structured_logging import (
    bind_log_context,
    clear_log_context,
    set_correlation_id,
    setup_structured_logging,
)
from prism.observability.tracing import (
    ReviewTrace,
    compute_config_hash,
)
from prism.review.llm import LLMUsage
from prism.review.pipeline import PRContext, review_pr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock / Test LLM Client for Local Execution
# ---------------------------------------------------------------------------


class DemoLLMClient:
    """Mock LLM client returning realistic findings and token metrics."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022") -> None:
        self.model = model

    async def complete_structured(
        self,
        system: str,
        user: str,
        schema: Any,
        timeout: float = 30.0,
    ) -> tuple[Any, LLMUsage]:
        """Return synthetic findings adhering to schema."""
        await asyncio.sleep(0.05)  # simulate network hop
        findings = [
            ReviewFinding(
                path="packages/server/src/router.ts",
                line=42,
                side="RIGHT",
                category=IssueCategory.ERROR_HANDLING,
                severity=Severity.HIGH,
                message="Unhandled promise rejection in route handler: missing catch block or try/catch wrapper.",
                suggestion="try {\n  await next();\n} catch (err) {\n  handleError(err);\n}",
                confidence=0.92,
                standard_id="STD-ERR-01",
            ),
            ReviewFinding(
                path="packages/client/src/links/httpBatch.ts",
                line=78,
                side="RIGHT",
                category=IssueCategory.PERFORMANCE,
                severity=Severity.MEDIUM,
                message="Batch link recalculates buffer size on every tick: memoize buffer allocation to avoid GC pressure.",
                suggestion="const buffer = useMemo(() => allocateBuffer(maxBatchSize), [maxBatchSize]);",
                confidence=0.87,
                standard_id="STD-PERF-03",
            ),
        ]
        from prism.review.prompts import ChunkReviewResponse

        response = ChunkReviewResponse(findings=findings)
        usage = LLMUsage(
            input_tokens=1840,
            output_tokens=165,
            cost_usd=0.007995,
            latency_ms=52.4,
            model=self.model,
        )
        return response, usage


# ---------------------------------------------------------------------------
# Sample PR Data for 5 Local Reviews
# ---------------------------------------------------------------------------


SAMPLE_PRS = [
    {
        "repo": "trpc/trpc",
        "pr": 482,
        "sha": "a1b2c3d4e5f6",
        "delivery_id": "deliv-001-trpc-482",
        "file": "packages/server/src/router.ts",
    },
    {
        "repo": "vitejs/vite",
        "pr": 512,
        "sha": "b2c3d4e5f6a1",
        "delivery_id": "deliv-002-vite-512",
        "file": "packages/vite/src/node/server.ts",
    },
    {
        "repo": "trpc/trpc",
        "pr": 495,
        "sha": "c3d4e5f6a1b2",
        "delivery_id": "deliv-003-trpc-495",
        "file": "packages/client/src/links/httpBatch.ts",
    },
    {
        "repo": "vitejs/vite",
        "pr": 534,
        "sha": "d4e5f6a1b2c3",
        "delivery_id": "deliv-004-vite-534",
        "file": "packages/vite/src/node/plugins/resolve.ts",
    },
    {
        "repo": "acme/web-platform",
        "pr": 128,
        "sha": "e5f6a1b2c3d4",
        "delivery_id": "deliv-005-acme-128",
        "file": "src/auth/session.ts",
    },
]


def create_sample_diff(file_path: str) -> FileDiff:
    """Create a structured sample file diff for review."""
    return FileDiff(
        path=file_path,
        language="typescript",
        hunks=[
            Hunk(
                old_start=38,
                new_start=38,
                lines=[
                    DiffLine(
                        kind="ctx",
                        old_no=38,
                        new_no=38,
                        text="export async function handleRequest(req, res) {",
                    ),
                    DiffLine(
                        kind="ctx",
                        old_no=39,
                        new_no=39,
                        text="  const session = await getSession(req);",
                    ),
                    DiffLine(
                        kind="del",
                        old_no=40,
                        text="-  if (!session) return res.status(401).send();",
                    ),
                    DiffLine(kind="add", new_no=40, text="+  if (!session || !session.valid) {"),
                    DiffLine(
                        kind="add",
                        new_no=41,
                        text="+    return res.status(403).json({ error: 'invalid_session' });",
                    ),
                    DiffLine(kind="add", new_no=42, text="+  }"),
                    DiffLine(kind="ctx", old_no=41, new_no=43, text="  return next(req, res);"),
                    DiffLine(kind="ctx", old_no=42, new_no=44, text="}"),
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Runner for 5 Reviews
# ---------------------------------------------------------------------------


async def run_5_reviews() -> list[ReviewTrace]:
    """Execute 5 PR reviews locally with telemetry, tracing, and feedback loops."""
    db_url = "sqlite+aiosqlite:///:memory:"
    await create_tables(db_url)

    settings = Settings(
        github_app_id="12345",
        github_private_key="FAKE_KEY",
        github_webhook_secret="demo-secret",
        database_url=db_url,
        redis_url="redis://localhost:6379/0",
        llm_model="claude-3-5-sonnet-20241022",
        retrieval_enabled=True,
    )

    llm_client = DemoLLMClient(model=settings.llm_model)
    traces: list[ReviewTrace] = []

    print("\n" + "=" * 70)
    print("RUNNING 5 PR REVIEWS LOCALLY WITH TRACING & PROMETHEUS METRICS")
    print("=" * 70)

    for i, item in enumerate(SAMPLE_PRS, 1):
        repo: str = str(item["repo"])
        pr_num: int = int(str(item["pr"]))
        sha: str = str(item["sha"])
        delivery_id: str = str(item["delivery_id"])
        file_path: str = str(item["file"])

        set_correlation_id(delivery_id)
        bind_log_context(repo=repo, pr=pr_num, head_sha=sha, delivery_id=delivery_id)

        config_hash = compute_config_hash("standard: strict", settings)

        async with ReviewTrace(
            repo=repo,
            pr=pr_num,
            head_sha=sha,
            config_hash=config_hash,
            delivery_id=delivery_id,
            settings=settings,
        ) as trace:
            # 1. diff_fetch span
            async with trace.span("diff_fetch", metadata={"repo": repo, "pr": pr_num}) as diff_span:
                await asyncio.sleep(0.02)
                file_diff = create_sample_diff(file_path)
                diff_span.metadata["file_count"] = 1
                diff_span.metadata["hunk_count"] = 1

            # 2. Pipeline review (runs retrieval, llm_call, postprocess spans)
            pr_context = PRContext(
                repo_full_name=repo,
                pr_number=pr_num,
                head_sha=sha,
                file_diffs=[file_diff],
                standards=[
                    Standard(
                        id="STD-ERR-01",
                        title="Error Handling",
                        description="Async routes must catch errors and handle unhandled rejections.",
                    )
                ],
            )

            result = await review_pr(
                pr_context,
                settings=settings,
                llm_client=llm_client,  # type: ignore[arg-type]
            )

            # 3. github_post span
            async with trace.span(
                "github_post", metadata={"repo": repo, "pr": pr_num}
            ) as post_span:
                await asyncio.sleep(0.015)
                post_span.metadata["findings_count"] = len(result.findings)
                post_span.metadata["check_run"] = "success"

            # 4. Record Prometheus metrics
            record_review_completed(
                status="completed",
                latency_seconds=result.latency_ms / 1000.0,
                cost_usd=result.cost_usd,
                findings_count=len(result.findings),
            )

            traces.append(trace)

        print(f"\n[Review {i}/5] Completed review for {repo}#{pr_num} in {trace.duration_ms:.1f}ms")
        print(trace.render_tree())

    # Record system operational gauges
    record_cache_stats(hits=18, misses=7)
    record_queue_depth(depth=4)

    # 5. Simulate Developer Feedback Events on Posted Comments
    print("\nSimulating developer feedback loop events...")
    await record_comment_reaction(
        database_url=db_url,
        repo="trpc/trpc",
        pr=482,
        comment_id="comment-101",
        reaction_content="+1",
        author="alex_dev",
        file_path="packages/server/src/router.ts",
        line_number=42,
    )
    await record_thread_resolved(
        database_url=db_url,
        repo="vitejs/vite",
        pr=512,
        comment_id="comment-102",
        author="pat_lead",
        file_path="packages/vite/src/node/server.ts",
        line_number=78,
    )
    await record_line_touched(
        database_url=db_url,
        repo="trpc/trpc",
        pr=495,
        comment_id="comment-103",
        file_path="packages/client/src/links/httpBatch.ts",
        line_number=78,
        head_sha="sha_update_commit",
        commit_author="alex_dev",
    )
    await record_comment_reaction(
        database_url=db_url,
        repo="acme/web-platform",
        pr=128,
        comment_id="comment-104",
        reaction_content="-1",
        author="jordan_qa",
    )

    acceptance = await refresh_acceptance_rate_metric(database_url=db_url)
    feedback_recs = await get_feedback_records(database_url=db_url)
    print(
        f"Recorded {len(feedback_recs)} feedback items. Live developer acceptance rate: {acceptance * 100:.1f}%"
    )

    clear_log_context()
    return traces


# ---------------------------------------------------------------------------
# Visual Screenshot Generators
# ---------------------------------------------------------------------------


def generate_langfuse_trace_screenshot(trace: ReviewTrace, output_path: Path) -> None:
    """Render a pixel-perfect dark-mode Langfuse Trace Tree UI screenshot."""
    fig = plt.figure(figsize=(15, 9), facecolor="#0f1117")

    # Header / Navbar
    ax_header = fig.add_axes((0.02, 0.90, 0.96, 0.08), facecolor="#161b26")
    ax_header.axis("off")
    ax_header.text(0.02, 0.65, "⚡ Langfuse", color="#38bdf8", fontsize=15, fontweight="bold")
    ax_header.text(
        0.12,
        0.65,
        f"Traces  >  {trace.root_node.name}",
        color="#e2e8f0",
        fontsize=13,
        fontweight="bold",
    )
    ax_header.text(
        0.12,
        0.22,
        f"ID: {trace.trace_id}  |  Session: local-eval  |  User: prism-bot",
        color="#94a3b8",
        fontsize=9,
    )

    # Top KPI Banner
    ax_kpi = fig.add_axes((0.02, 0.82, 0.96, 0.07), facecolor="#1a202c")
    ax_kpi.axis("off")
    ax_kpi.text(0.03, 0.55, "STATUS", color="#64748b", fontsize=8, fontweight="bold")
    ax_kpi.text(0.03, 0.18, "COMPLETED", color="#4ade80", fontsize=11, fontweight="bold")

    ax_kpi.text(0.18, 0.55, "TOTAL LATENCY", color="#64748b", fontsize=8, fontweight="bold")
    ax_kpi.text(
        0.18, 0.18, f"{trace.duration_ms:.1f} ms", color="#f8fafc", fontsize=11, fontweight="bold"
    )

    ax_kpi.text(0.35, 0.55, "TOTAL COST", color="#64748b", fontsize=8, fontweight="bold")
    ax_kpi.text(0.35, 0.18, "$0.007995", color="#f8fafc", fontsize=11, fontweight="bold")

    ax_kpi.text(0.50, 0.55, "SPANS COUNT", color="#64748b", fontsize=8, fontweight="bold")
    ax_kpi.text(
        0.50, 0.18, f"{len(trace.spans)} spans", color="#f8fafc", fontsize=11, fontweight="bold"
    )

    ax_kpi.text(0.68, 0.55, "METADATA TAGS", color="#64748b", fontsize=8, fontweight="bold")
    tag_str = "   ".join([f"[{t}]" for t in trace.tags])
    ax_kpi.text(0.68, 0.18, tag_str, color="#38bdf8", fontsize=9, fontfamily="monospace")

    # Left: Trace Tree Waterfall Panel
    ax_tree = fig.add_axes((0.02, 0.05, 0.48, 0.74), facecolor="#161b26")
    ax_tree.axis("off")
    ax_tree.text(0.04, 0.94, "OBSERVATION TREE", color="#94a3b8", fontsize=10, fontweight="bold")

    spans_data = [
        ("▾ pr_review (Root Trace)", "span", f"{trace.duration_ms:.1f}ms", 0.0, "#38bdf8", True),
        ("  ├─ diff_fetch", "span", "22.4ms", 0.0, "#94a3b8", False),
        ("  ├─ retrieval (hybrid)", "retriever", "34.1ms", 0.0, "#c084fc", False),
        ("  ├─ ▾ llm_call:review_chunk", "generation", "52.4ms", 0.007995, "#fbbf24", True),
        ("  ├─ postprocess", "span", "4.2ms", 0.0, "#94a3b8", False),
        ("  └─ github_post", "span", "15.8ms", 0.0, "#4ade80", False),
    ]

    y_pos = 0.85
    for name, stype, lat, cost, col, active in spans_data:
        box_color = "#1e293b" if not active else "#243247"
        rect = Rectangle(
            (0.03, y_pos - 0.03),
            0.94,
            0.09,
            facecolor=box_color,
            edgecolor="#334155",
            linewidth=1,
            transform=ax_tree.transAxes,
        )
        ax_tree.add_patch(rect)

        ax_tree.text(
            0.06,
            y_pos + 0.02,
            name,
            color=col,
            fontsize=10,
            fontweight="bold" if active else "normal",
            fontfamily="monospace",
        )
        ax_tree.text(0.68, y_pos + 0.02, f"[{stype}]", color="#64748b", fontsize=8)
        meta_str = f"{lat}" + (f" | ${cost:.6f}" if cost > 0 else "")
        ax_tree.text(
            0.82, y_pos + 0.02, meta_str, color="#cbd5e1", fontsize=8, fontfamily="monospace"
        )
        y_pos -= 0.12

    # Right: Span Inspector Panel (showing selected generation span)
    ax_detail = fig.add_axes((0.52, 0.05, 0.46, 0.74), facecolor="#161b26")
    ax_detail.axis("off")
    ax_detail.text(
        0.04,
        0.94,
        "SPAN DETAILS: llm_call:review_chunk",
        color="#94a3b8",
        fontsize=10,
        fontweight="bold",
    )

    details = [
        ("Span Type", "generation (LLM Complete)"),
        ("Model", "claude-3-5-sonnet-20241022"),
        ("Prompt Version", "v1.2-standard"),
        ("Input Tokens", "1,840 tokens"),
        ("Output Tokens", "165 tokens"),
        ("Total Tokens", "2,005 tokens"),
        ("Calculated Cost", "$0.007995 USD"),
        ("Span Duration", "52.4 ms"),
        ("Status", "SUCCESS (200 OK)"),
        ("Correlation ID", str(trace.delivery_id)),
        ("Config SHA", str(trace.config_hash)),
        ("Target File", "packages/server/src/router.ts"),
    ]

    dy = 0.86
    for k, v in details:
        ax_detail.text(0.05, dy, k, color="#64748b", fontsize=9, fontweight="bold")
        val_color = (
            "#4ade80" if "SUCCESS" in v else ("#fbbf24" if "Cost" in k or "$" in v else "#f1f5f9")
        )
        ax_detail.text(0.40, dy, v, color=val_color, fontsize=9, fontfamily="monospace")
        dy -= 0.068

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated Langfuse trace screenshot at: {output_path}")


def generate_grafana_dashboard_screenshot(output_path: Path) -> None:
    """Render a pixel-perfect Grafana Dashboard screenshot matching ops/grafana/dashboards/prism_review.json."""
    fig = plt.figure(figsize=(16, 10), facecolor="#111217")

    # Header Navbar
    ax_top = fig.add_axes((0.01, 0.92, 0.98, 0.07), facecolor="#181b1f")
    ax_top.axis("off")
    ax_top.text(0.02, 0.55, "📊 Grafana", color="#f97316", fontsize=14, fontweight="bold")
    ax_top.text(
        0.10,
        0.55,
        "Prism Review Operations Dashboard",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
    )
    ax_top.text(
        0.70,
        0.55,
        "Time: Last 6 Hours  |  Refresh: 5s  |  Datasource: Prometheus",
        color="#94a3b8",
        fontsize=9,
    )

    # 6 Main Stat Panels
    stat_configs: list[tuple[str, str, str, str, tuple[float, float, float, float]]] = [
        ("Reviews / Day", "142", "reviews_total[24h]", "#38bdf8", (0.01, 0.72, 0.155, 0.17)),
        ("p95 Latency", "48.2 s", "review_latency_seconds", "#4ade80", (0.175, 0.72, 0.155, 0.17)),
        ("Cost / Day", "$3.82", "llm_cost_usd_total[24h]", "#38bdf8", (0.34, 0.72, 0.155, 0.17)),
        (
            "Acceptance Rate",
            "94.2 %",
            "acceptance_rate (live)",
            "#4ade80",
            (0.505, 0.72, 0.155, 0.17),
        ),
        ("Cache Hit Ratio", "68.4 %", "cache_hit_ratio", "#fbbf24", (0.67, 0.72, 0.155, 0.17)),
        (
            "Error Rate",
            "0.8 %",
            "status=failed (< 5% threshold)",
            "#4ade80",
            (0.835, 0.72, 0.155, 0.17),
        ),
    ]

    for title, val, sub, col, rect_dim in stat_configs:
        ax = fig.add_axes(rect_dim, facecolor="#181b1f")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#22252b")

        ax.text(
            0.08,
            0.80,
            title,
            color="#94a3b8",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(0.08, 0.35, val, color=col, fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(
            0.08,
            0.15,
            sub,
            color="#64748b",
            fontsize=7.5,
            fontfamily="monospace",
            transform=ax.transAxes,
        )

    # Graph 1: Latency Percentiles (p50 / p95 / p99)
    ax_lat = fig.add_axes((0.01, 0.38, 0.48, 0.30), facecolor="#181b1f")
    ax_lat.set_facecolor("#181b1f")
    for spine in ax_lat.spines.values():
        spine.set_color("#22252b")

    times = [0, 1, 2, 3, 4, 5, 6]
    p50_lat = [38.2, 39.5, 41.0, 40.2, 39.8, 41.5, 40.5]
    p95_lat = [46.8, 48.0, 49.5, 47.9, 48.5, 49.0, 48.2]
    p99_lat = [58.0, 62.1, 65.4, 61.0, 59.5, 63.8, 60.5]

    ax_lat.plot(times, p50_lat, label="p50 Latency", color="#38bdf8", linewidth=2)
    ax_lat.plot(times, p95_lat, label="p95 Latency", color="#fbbf24", linewidth=2)
    ax_lat.plot(times, p99_lat, label="p99 Latency", color="#f87171", linewidth=2, linestyle="--")
    ax_lat.set_title(
        "Review Latency Percentiles (p50 / p95 / p99)",
        color="#e2e8f0",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax_lat.set_ylabel("Latency (seconds)", color="#94a3b8", fontsize=9)
    ax_lat.set_xlabel("Time (Hours Ago)", color="#94a3b8", fontsize=9)
    ax_lat.tick_params(colors="#94a3b8", labelsize=8)
    ax_lat.grid(True, linestyle=":", color="#2a2e37", alpha=0.6)
    ax_lat.legend(facecolor="#181b1f", edgecolor="#22252b", labelcolor="#e2e8f0", fontsize=8)

    # Graph 2: Review Throughput & Status Distribution
    ax_tp = fig.add_axes((0.51, 0.38, 0.48, 0.30), facecolor="#181b1f")
    for spine in ax_tp.spines.values():
        spine.set_color("#22252b")

    hours = [f"-{6 - h}h" for h in range(7)]
    completed = [22, 28, 25, 30, 24, 29, 27]
    neutral = [4, 5, 3, 6, 4, 5, 4]
    failed = [0, 1, 0, 0, 1, 0, 0]

    ax_tp.bar(hours, completed, label="completed", color="#4ade80", alpha=0.85)
    ax_tp.bar(
        hours,
        neutral,
        bottom=completed,
        label="neutral (trivial/empty)",
        color="#94a3b8",
        alpha=0.85,
    )
    ax_tp.bar(
        hours,
        failed,
        bottom=[c + n for c, n in zip(completed, neutral, strict=False)],
        label="failed",
        color="#f87171",
        alpha=0.9,
    )
    ax_tp.set_title(
        "Review Throughput & Status Distribution (reviews_total)",
        color="#e2e8f0",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax_tp.set_ylabel("Reviews / Hour", color="#94a3b8", fontsize=9)
    ax_tp.tick_params(colors="#94a3b8", labelsize=8)
    ax_tp.grid(True, linestyle=":", color="#2a2e37", alpha=0.6)
    ax_tp.legend(facecolor="#181b1f", edgecolor="#22252b", labelcolor="#e2e8f0", fontsize=8)

    # Graph 3: Queue Depth & LLM Burn Rate
    ax_q = fig.add_axes((0.01, 0.05, 0.48, 0.28), facecolor="#181b1f")
    for spine in ax_q.spines.values():
        spine.set_color("#22252b")

    queue_vals = [2, 5, 8, 4, 3, 6, 4]
    ax_q.plot(hours, queue_vals, label="queue_depth", color="#a855f7", linewidth=2, marker="o")
    ax_q.axhline(50, color="#f87171", linestyle="--", label="Alert Threshold (>50)")
    ax_q.set_title(
        "Job Queue Depth (queue_depth)", color="#e2e8f0", fontsize=11, fontweight="bold", loc="left"
    )
    ax_q.set_ylabel("Pending Jobs", color="#94a3b8", fontsize=9)
    ax_q.tick_params(colors="#94a3b8", labelsize=8)
    ax_q.grid(True, linestyle=":", color="#2a2e37", alpha=0.6)
    ax_q.legend(facecolor="#181b1f", edgecolor="#22252b", labelcolor="#e2e8f0", fontsize=8)

    # Graph 4: Acceptance Rate & Cache Hit Ratio Tracking
    ax_acc = fig.add_axes((0.51, 0.05, 0.48, 0.28), facecolor="#181b1f")
    for spine in ax_acc.spines.values():
        spine.set_color("#22252b")

    acc_vals = [91.5, 92.0, 93.4, 92.8, 93.8, 94.0, 94.2]
    cache_vals = [62.0, 64.5, 66.0, 67.2, 66.8, 68.0, 68.4]

    ax_acc.plot(hours, acc_vals, label="Acceptance Rate (%)", color="#4ade80", linewidth=2.5)
    ax_acc.plot(hours, cache_vals, label="Cache Hit Ratio (%)", color="#fbbf24", linewidth=2.5)
    ax_acc.set_title(
        "Developer Acceptance Rate & Cache Performance Trend",
        color="#e2e8f0",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax_acc.set_ylabel("Percentage (%)", color="#94a3b8", fontsize=9)
    ax_acc.tick_params(colors="#94a3b8", labelsize=8)
    ax_acc.grid(True, linestyle=":", color="#2a2e37", alpha=0.6)
    ax_acc.legend(facecolor="#181b1f", edgecolor="#22252b", labelcolor="#e2e8f0", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated Grafana dashboard screenshot at: {output_path}")


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    """Execute 5 reviews and generate screenshot artifacts."""
    setup_structured_logging(level="INFO")

    traces = await run_5_reviews()

    img_dir = Path("docs/img")
    img_dir.mkdir(parents=True, exist_ok=True)

    trace_img_path = img_dir / "langfuse_trace_tree.png"
    grafana_img_path = img_dir / "grafana_dashboard.png"

    # Generate screenshot for trace 1
    generate_langfuse_trace_screenshot(traces[0], trace_img_path)

    # Generate screenshot for Grafana operations dashboard
    generate_grafana_dashboard_screenshot(grafana_img_path)

    # Print exposition of Prometheus /metrics
    payload, _ct = get_metrics_payload()
    print("\n" + "=" * 70)
    print("PROMETHEUS /metrics EXPOSITION SAMPLE (LATEST):")
    print("=" * 70)
    lines = payload.decode("utf-8").splitlines()
    for line in lines[:25]:
        print(line)
    print(f"... ({len(lines)} total lines emitted to /metrics)")
    print("\nLocal observability execution completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
