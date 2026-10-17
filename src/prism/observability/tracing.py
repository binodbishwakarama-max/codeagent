"""Distributed tracing and observability spans for review jobs and LLM invocations.

Integrates with Langfuse for remote trace trees and maintains an in-memory
hierarchical span structure with metadata tagging (repo, pr, head_sha, config_hash).
"""

import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from prism.config import Settings

logger = logging.getLogger(__name__)

# Context variable for the currently active review trace
_active_trace_ctx: ContextVar["ReviewTrace | None"] = ContextVar("active_trace", default=None)


def get_current_trace() -> "ReviewTrace | None":
    """Retrieve the currently active ReviewTrace, if any."""
    return _active_trace_ctx.get()


def set_current_trace(trace: "ReviewTrace | None") -> None:
    """Set the currently active ReviewTrace."""
    _active_trace_ctx.set(trace)


def compute_config_hash(standards_content: str = "", settings: Settings | None = None) -> str:
    """Compute a deterministic SHA-256 hash representing the review pipeline configuration."""
    model = "default-model"
    retrieval_enabled = False
    retrieval_mode = "hybrid"
    top_k = 6
    self_verify = False

    if settings is not None:
        model = settings.llm_model
        retrieval_enabled = settings.retrieval_enabled
        retrieval_mode = settings.retrieval_mode
        top_k = settings.retrieval_top_k
        self_verify = settings.self_verification_enabled
    else:
        try:
            cfg = Settings()
            model = cfg.llm_model
            retrieval_enabled = cfg.retrieval_enabled
            retrieval_mode = cfg.retrieval_mode
            top_k = cfg.retrieval_top_k
            self_verify = cfg.self_verification_enabled
        except Exception:
            pass

    config_repr = (
        f"model={model}|temp=0.0|retrieval={retrieval_enabled}|"
        f"ret_mode={retrieval_mode}|top_k={top_k}|"
        f"verify={self_verify}|standards={standards_content}"
    )
    return hashlib.sha256(config_repr.encode("utf-8")).hexdigest()[:16]


@dataclass
class TraceSpanNode:
    """A single span node within an in-memory trace tree."""

    id: str
    name: str
    parent_id: str | None = None
    span_type: Literal["span", "generation", "retriever", "tool"] = "span"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    status: str = "running"
    model: str | None = None
    prompt_version: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    children: list["TraceSpanNode"] = field(default_factory=list)

    def finish(self, status: str = "success", error: str | None = None) -> None:
        """Mark span execution completed and record duration."""
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        self.status = status
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert span node and descendants to JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "span_type": self.span_type,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
        }
        if self.model:
            data["model"] = self.model
        if self.prompt_version:
            data["prompt_version"] = self.prompt_version
        if self.input_tokens or self.output_tokens:
            data["tokens"] = {"input": self.input_tokens, "output": self.output_tokens}
            data["cost_usd"] = self.cost_usd
        if self.error:
            data["error"] = self.error
        if self.children:
            data["children"] = [c.to_dict() for c in self.children]
        return data


class ReviewTrace:
    """Telemetry context managing a single pull request review trace.

    Tags traces with repo, pr, head_sha, config_hash, and delivery_id.
    Captures hierarchical spans for diff fetch, retrieval, LLM calls,
    postprocessing, and GitHub submission.
    """

    def __init__(
        self,
        repo: str,
        pr: int,
        head_sha: str,
        config_hash: str = "default",
        delivery_id: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.repo = repo
        self.pr = pr
        self.head_sha = head_sha
        self.config_hash = config_hash
        self.delivery_id = delivery_id
        if settings is not None:
            self.settings: Settings | None = settings
        else:
            try:
                self.settings = Settings()
            except Exception:
                self.settings = None

        self.trace_id = f"trace-{repo.replace('/', '-')}-pr{pr}-{head_sha[:8]}"
        self.start_time = time.time()
        self.end_time: float | None = None
        self.duration_ms = 0.0
        self.status = "running"
        self.error: str | None = None

        self.tags = [
            f"repo:{self.repo}",
            f"pr:{self.pr}",
            f"sha:{self.head_sha[:8]}",
            f"config:{self.config_hash[:8]}",
        ]
        self.metadata = {
            "repo": self.repo,
            "pr": self.pr,
            "head_sha": self.head_sha,
            "config_hash": self.config_hash,
            "delivery_id": self.delivery_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self.root_node = TraceSpanNode(
            id=self.trace_id,
            name=f"pr_review:{self.repo}#{self.pr}",
            span_type="span",
            metadata=dict(self.metadata),
        )
        self.spans: list[TraceSpanNode] = []
        self._current_parent: TraceSpanNode = self.root_node
        self._span_counter = 0

        # Langfuse remote client setup
        self._langfuse_client: Any = None
        self._langfuse_trace: Any = None
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        """Attempt to initialize Langfuse SDK if credentials are configured."""
        if (
            self.settings is not None
            and self.settings.langfuse_public_key
            and self.settings.langfuse_secret_key
            and self.settings.langfuse_host
        ):
            try:
                from langfuse import Langfuse

                self._langfuse_client = Langfuse(
                    public_key=self.settings.langfuse_public_key,
                    secret_key=self.settings.langfuse_secret_key,
                    host=self.settings.langfuse_host,
                )
            except Exception as exc:
                logger.warning("Could not initialize Langfuse client: %s", exc)
                self._langfuse_client = None

    @asynccontextmanager
    async def span(
        self,
        name: str,
        span_type: Literal["span", "generation", "retriever", "tool"] = "span",
        model: str | None = None,
        prompt_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[TraceSpanNode, None]:
        """Create and enter a child span within the review trace."""
        self._span_counter += 1
        span_id = f"{self.trace_id}-span{self._span_counter}"
        node = TraceSpanNode(
            id=span_id,
            name=name,
            parent_id=self._current_parent.id,
            span_type=span_type,
            model=model,
            prompt_version=prompt_version,
            metadata=metadata or {},
        )

        previous_parent = self._current_parent
        self._current_parent.children.append(node)
        self.spans.append(node)
        self._current_parent = node

        # Remote Langfuse observation if available
        remote_obs = None
        if self._langfuse_client is not None:
            try:
                remote_type = "generation" if span_type == "generation" else "span"
                remote_obs = self._langfuse_client.start_observation(
                    name=name,
                    as_type=remote_type,
                    model=model,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.debug("Langfuse remote start_observation failed: %s", exc)

        try:
            yield node
            if node.status == "running":
                node.finish(status="success")
            if remote_obs is not None:
                try:
                    usage = None
                    if node.input_tokens or node.output_tokens:
                        usage = {
                            "input": node.input_tokens,
                            "output": node.output_tokens,
                            "total": node.input_tokens + node.output_tokens,
                        }
                    remote_obs.end(
                        usage_details=usage,
                        cost_details={"total": node.cost_usd} if node.cost_usd else None,
                    )
                except Exception:
                    pass
        except Exception as exc:
            node.finish(status="error", error=str(exc))
            if remote_obs is not None:
                try:
                    remote_obs.end(status_message=str(exc), level="ERROR")
                except Exception:
                    pass
            raise
        finally:
            self._current_parent = previous_parent

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        """Complete the root review trace and flush remote buffers."""
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        self.status = status
        self.error = error
        self.root_node.finish(status=status, error=error)

        if self._langfuse_client is not None:
            try:
                self._langfuse_client.flush()
            except Exception as exc:
                logger.debug("Langfuse flush failed: %s", exc)

        logger.info(
            "ReviewTrace '%s' finished [%s] in %.2fms with %d child spans",
            self.trace_id,
            self.status,
            self.duration_ms,
            len(self.spans),
        )

    def render_tree(self) -> str:
        """Render the trace tree as an ASCII tree diagram for visualization and debugging."""
        lines: list[str] = [
            f"Trace: {self.root_node.name} [id={self.trace_id}] ({self.root_node.duration_ms:.1f}ms)",
            f"  Tags: {', '.join(self.tags)}",
            "  Spans:",
        ]

        def _render_children(nodes: list[TraceSpanNode], indent: str = "    ") -> None:
            for i, child in enumerate(nodes):
                is_last = i == len(nodes) - 1
                connector = "\\-- " if is_last else "|-- "
                sub_indent = indent + ("    " if is_last else "|   ")

                meta_parts: list[str] = [f"{child.duration_ms:.1f}ms", child.status]
                if child.model:
                    meta_parts.append(f"model={child.model}")
                if child.input_tokens or child.output_tokens:
                    meta_parts.append(f"tokens={child.input_tokens}+{child.output_tokens}")
                if child.cost_usd > 0.0:
                    meta_parts.append(f"${child.cost_usd:.6f}")

                lines.append(f"{indent}{connector}{child.name} ({', '.join(meta_parts)})")
                if child.children:
                    _render_children(child.children, sub_indent)

        _render_children(self.root_node.children)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return full JSON-serializable representation of the trace tree."""
        return {
            "trace_id": self.trace_id,
            "repo": self.repo,
            "pr": self.pr,
            "head_sha": self.head_sha,
            "config_hash": self.config_hash,
            "delivery_id": self.delivery_id,
            "tags": self.tags,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "tree": self.root_node.to_dict(),
        }

    async def __aenter__(self) -> "ReviewTrace":
        set_current_trace(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        status = "failed" if exc_val is not None else "completed"
        err = str(exc_val) if exc_val is not None else None
        self.finish(status=status, error=err)
        set_current_trace(None)


# ---------------------------------------------------------------------------
# Backwards compatibility helper for existing codebase
# ---------------------------------------------------------------------------


@dataclass
class LLMSpan:
    """Legacy compatibility class for LLM span tracking."""

    name: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    status: str = "running"
    error: str | None = None

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float = 0.0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd

    def finish(self, status: str = "success", error: str | None = None) -> None:
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        self.status = status
        self.error = error


@asynccontextmanager
async def trace_llm_call(
    span_name: str,
    model: str,
    metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[LLMSpan, None]:
    """Compatibility context manager wrapping an LLM invocation."""
    span = LLMSpan(name=span_name, model=model, metadata=metadata or {})
    active_trace = get_current_trace()
    if active_trace is not None:
        async with active_trace.span(
            name=span_name,
            span_type="generation",
            model=model,
            metadata=metadata,
        ) as node:
            try:
                yield span
                node.input_tokens = span.input_tokens
                node.output_tokens = span.output_tokens
                node.cost_usd = span.cost_usd
                span.finish(status="success")
            except Exception as exc:
                span.finish(status="error", error=str(exc))
                raise
    else:
        try:
            yield span
            if span.status == "running":
                span.finish(status="success")
        except Exception as exc:
            span.finish(status="error", error=str(exc))
            raise
