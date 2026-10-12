"""Core review pipeline orchestrating diff chunking, retrieval augmentation,
LLM analysis, self-verification, and post-processing.
"""

import asyncio
import dataclasses
import logging
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prism.config import Settings
from prism.models import FileDiff, Hunk, ReviewFinding, ReviewResult, Standard
from prism.observability.tracing import get_current_trace
from prism.retrieval import (
    CodeChunk,
    RepositoryIndexer,
    build_search_query,
    chunk_source_file,
    format_retrieved_context,
    rerank_chunks,
    retrieve_context,
)
from prism.review.llm import LLMClient, LLMUsage, create_llm_client
from prism.review.postprocess import postprocess_findings
from prism.review.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    ChunkReviewResponse,
    build_repair_prompt,
    build_user_prompt,
    render_diff_with_line_numbers,
)
from prism.review.verify import verify_findings

logger = logging.getLogger(__name__)


class PRContext(BaseModel):
    """Contextual metadata, diff content, and repository files for a pull request under review."""

    model_config = ConfigDict(frozen=True)

    repo_full_name: str
    pr_number: int
    title: str = ""
    description: str = ""
    head_sha: str = ""
    file_diffs: list[FileDiff] = Field(default_factory=list)
    standards: list[Standard] = Field(default_factory=list)
    repo_files: dict[str, str] = Field(default_factory=dict)
    indexed_chunks: list[Any] = Field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Conservative token count estimation heuristic (~3 characters per token)."""
    return max(1, len(text) // 3)


def chunk_diffs(file_diffs: list[FileDiff], max_tokens: int) -> list[list[FileDiff]]:
    """Partition file diffs into chunks that each fit within max_tokens.

    Splits by file group first. If a single file diff exceeds max_tokens,
    its hunks are partitioned into smaller subgroup chunks.
    """
    if not file_diffs:
        return []

    # 1. Check if all files fit together in one chunk
    total_rendered = render_diff_with_line_numbers(file_diffs)
    if estimate_tokens(total_rendered) <= max_tokens:
        return [file_diffs]

    chunks: list[list[FileDiff]] = []
    current_chunk: list[FileDiff] = []
    current_tokens = 0

    for fd in file_diffs:
        fd_rendered = render_diff_with_line_numbers([fd])
        fd_tokens = estimate_tokens(fd_rendered)

        # If a single file diff exceeds the token limit, split by hunk groups
        if fd_tokens > max_tokens and len(fd.hunks) > 1:
            # Flush existing accumulated chunk first
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

            # Partition hunks of this large file
            hunk_subgroup: list[Hunk] = []
            for hunk in fd.hunks:
                candidate = FileDiff(
                    path=fd.path, language=fd.language, hunks=hunk_subgroup + [hunk]
                )
                cand_tokens = estimate_tokens(render_diff_with_line_numbers([candidate]))
                if cand_tokens > max_tokens and hunk_subgroup:
                    sub_fd = FileDiff(path=fd.path, language=fd.language, hunks=hunk_subgroup)
                    chunks.append([sub_fd])
                    hunk_subgroup = [hunk]
                else:
                    hunk_subgroup.append(hunk)

            if hunk_subgroup:
                chunks.append([FileDiff(path=fd.path, language=fd.language, hunks=hunk_subgroup)])
            continue

        # Normal file: accumulate into current chunk or start a new chunk
        if current_tokens + fd_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [fd]
            current_tokens = fd_tokens
        else:
            current_chunk.append(fd)
            current_tokens += fd_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def _review_chunk(
    chunk: list[FileDiff],
    pr_context: PRContext,
    llm_client: LLMClient,
    semaphore: asyncio.Semaphore,
    timeout: float,
    retrieved_context: str | None = None,
    prompt_variant: str = "standard",
    redis_client: Any = None,
) -> tuple[list[ReviewFinding], LLMUsage]:
    """Execute LLM review on a single chunk with Redis caching, retry,
    and repair prompt on malformed output.
    """
    import json

    from prism.queue.cache import (
        compute_standards_hash,
        get_cached_result,
        set_cached_result,
    )

    chunk_text = render_diff_with_line_numbers(chunk)
    if retrieved_context:
        chunk_text = f"{chunk_text}\n---retrieved---\n{retrieved_context}"
    prompt_ver = f"{PROMPT_VERSION}-{prompt_variant}"

    standards_text = json.dumps([s.model_dump() for s in pr_context.standards], sort_keys=True)
    standards_hash = compute_standards_hash(standards_text)
    model_name = getattr(llm_client, "model", "unknown")

    if redis_client is not None:
        cached = await get_cached_result(
            redis_client=redis_client,
            model=model_name,
            prompt_version=prompt_ver,
            standards_hash=standards_hash,
            chunk_text=chunk_text,
        )
        if cached is not None:
            findings = [ReviewFinding(**f) for f in cached.get("findings", [])]
            usage_dict = cached.get("usage", {})
            usage = LLMUsage(
                input_tokens=int(usage_dict.get("input_tokens", 0)),
                output_tokens=int(usage_dict.get("output_tokens", 0)),
                cost_usd=float(usage_dict.get("cost_usd", 0.0)),
                latency_ms=float(usage_dict.get("latency_ms", 0.0)),
                model=str(usage_dict.get("model", model_name)),
            )
            return findings, usage

    user_prompt = build_user_prompt(
        pr_title=pr_context.title,
        pr_description=pr_context.description,
        standards=pr_context.standards,
        file_diffs=chunk,
        retrieved_context=retrieved_context,
        prompt_variant=prompt_variant,
    )

    active_trace = get_current_trace()
    async with semaphore:
        if active_trace is not None:
            async with active_trace.span(
                name="llm_call:review_chunk",
                span_type="generation",
                model=model_name,
                prompt_version=prompt_ver,
                metadata={"chunk_files": [f.path for f in chunk]},
            ) as span_node:
                try:
                    response, usage = await llm_client.complete_structured(
                        system=SYSTEM_PROMPT,
                        user=user_prompt,
                        schema=ChunkReviewResponse,
                        timeout=timeout,
                    )
                    findings = response.findings
                except Exception as initial_error:
                    logger.warning(
                        "LLM review on chunk failed (%s); retrying with repair prompt",
                        initial_error,
                    )
                    repair_prompt = build_repair_prompt(
                        malformed_output=str(initial_error),
                        error_detail=str(initial_error),
                    )
                    full_prompt = f"{user_prompt}\n\n{repair_prompt}"
                    try:
                        repair_response, repair_usage = await llm_client.complete_structured(
                            system=SYSTEM_PROMPT,
                            user=full_prompt,
                            schema=ChunkReviewResponse,
                            timeout=timeout,
                        )
                        findings, usage = repair_response.findings, repair_usage
                    except Exception as repair_error:
                        logger.error(
                            "Repair prompt also failed (%s); returning empty findings for chunk",
                            repair_error,
                        )
                        fallback_usage = LLMUsage(
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=0.0,
                            latency_ms=0.0,
                            model=model_name,
                        )
                        return [], fallback_usage
                span_node.input_tokens = usage.input_tokens
                span_node.output_tokens = usage.output_tokens
                span_node.cost_usd = usage.cost_usd
        else:
            try:
                response, usage = await llm_client.complete_structured(
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    schema=ChunkReviewResponse,
                    timeout=timeout,
                )
                findings = response.findings
            except Exception as initial_error:
                logger.warning(
                    "LLM review on chunk failed (%s); retrying with repair prompt",
                    initial_error,
                )
                repair_prompt = build_repair_prompt(
                    malformed_output=str(initial_error),
                    error_detail=str(initial_error),
                )
                full_prompt = f"{user_prompt}\n\n{repair_prompt}"
                try:
                    repair_response, repair_usage = await llm_client.complete_structured(
                        system=SYSTEM_PROMPT,
                        user=full_prompt,
                        schema=ChunkReviewResponse,
                        timeout=timeout,
                    )
                    findings, usage = repair_response.findings, repair_usage
                except Exception as repair_error:
                    # Never raise on malformed output: return empty list for this chunk
                    logger.error(
                        "Repair prompt also failed (%s); returning empty findings for chunk",
                        repair_error,
                    )
                    fallback_usage = LLMUsage(
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        latency_ms=0.0,
                        model=model_name,
                    )
                    return [], fallback_usage

        if redis_client is not None:
            cache_payload = {
                "findings": [f.model_dump() for f in findings],
                "usage": dataclasses.asdict(usage),
            }
            await set_cached_result(
                redis_client=redis_client,
                model=model_name,
                prompt_version=prompt_ver,
                standards_hash=standards_hash,
                chunk_text=chunk_text,
                result=cache_payload,
            )

        return findings, usage


async def review_pr(
    pr_context: PRContext,
    settings: Settings | None = None,
    llm_client: LLMClient | None = None,
    max_concurrency: int = 5,
    redis_client: Any = None,
) -> ReviewResult:
    """Execute end-to-end pull request review against engineering standards.

    1. Retrieve repository context (if retrieval.enabled).
    2. Partition diff into chunks <= max_diff_tokens.
    3. Review chunks concurrently bounded by concurrency semaphore.
    4. Self-verify findings (if self_verification.enabled).
    5. Post-process, deduplicate, filter, rank, and cap findings.
    6. Compile and return complete ReviewResult.
    """
    start_time = time.time()
    cfg = settings or Settings()
    client = llm_client or create_llm_client(cfg)

    prompt_ver = (
        PROMPT_VERSION
        if cfg.prompt_variant == "standard"
        else f"{PROMPT_VERSION}-{cfg.prompt_variant}"
    )

    # If diff is empty, return immediate clean result
    if not pr_context.file_diffs:
        return ReviewResult(
            findings=[],
            model=getattr(client, "model", cfg.llm_model),
            prompt_version=prompt_ver,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0.0,
            cost_usd=0.0,
        )

    # 1. Retrieval & Context Augmentation (gated by config flag retrieval.enabled)
    retrieved_context_str: str | None = None
    if cfg.retrieval_enabled and pr_context.file_diffs:
        active_trace = get_current_trace()
        if active_trace is not None:
            async with active_trace.span(
                "retrieval",
                span_type="retriever",
                metadata={"mode": cfg.retrieval_mode},
            ) as ret_span:
                chunks_to_search: list[CodeChunk] = []
                if pr_context.indexed_chunks:
                    chunks_to_search = [
                        c for c in pr_context.indexed_chunks if isinstance(c, CodeChunk)
                    ]
                elif pr_context.repo_files:
                    indexer = RepositoryIndexer()
                    chunks_to_search = await indexer.index_repository(
                        repo=pr_context.repo_full_name,
                        commit_sha=pr_context.head_sha,
                        files=pr_context.repo_files,
                    )
                else:
                    for fd in pr_context.file_diffs:
                        code_lines = [
                            line.text for h in fd.hunks for line in h.lines if line.kind != "del"
                        ]
                        if code_lines:
                            chunks_to_search.extend(
                                chunk_source_file(fd.path, "\n".join(code_lines), fd.language)
                            )

                if chunks_to_search:
                    retrieved_candidates = retrieve_context(
                        file_diffs=pr_context.file_diffs,
                        chunks=chunks_to_search,
                        mode=cfg.retrieval_mode,
                        top_k=30,
                    )
                    candidate_chunks = [c for c, _score in retrieved_candidates]
                    if cfg.reranker_enabled and candidate_chunks:
                        query = build_search_query(pr_context.file_diffs)
                        scored = rerank_chunks(
                            query=query,
                            candidates=candidate_chunks,
                            top_k=cfg.retrieval_top_k,
                            model_name=cfg.reranker_model,
                        )
                    else:
                        scored = retrieved_candidates[: cfg.retrieval_top_k]

                    retrieved_context_str = format_retrieved_context(scored)
                    ret_span.metadata["candidates_count"] = len(scored)
                    logger.info("Retrieved %d repository context chunks for review", len(scored))
        else:
            chunks_to_search = []
            if pr_context.indexed_chunks:
                chunks_to_search = [
                    c for c in pr_context.indexed_chunks if isinstance(c, CodeChunk)
                ]
            elif pr_context.repo_files:
                indexer = RepositoryIndexer()
                chunks_to_search = await indexer.index_repository(
                    repo=pr_context.repo_full_name,
                    commit_sha=pr_context.head_sha,
                    files=pr_context.repo_files,
                )
            else:
                for fd in pr_context.file_diffs:
                    code_lines = [
                        line.text for h in fd.hunks for line in h.lines if line.kind != "del"
                    ]
                    if code_lines:
                        chunks_to_search.extend(
                            chunk_source_file(fd.path, "\n".join(code_lines), fd.language)
                        )

            if chunks_to_search:
                retrieved_candidates = retrieve_context(
                    file_diffs=pr_context.file_diffs,
                    chunks=chunks_to_search,
                    mode=cfg.retrieval_mode,
                    top_k=30,
                )
                candidate_chunks = [c for c, _score in retrieved_candidates]
                if cfg.reranker_enabled and candidate_chunks:
                    query = build_search_query(pr_context.file_diffs)
                    scored = rerank_chunks(
                        query=query,
                        candidates=candidate_chunks,
                        top_k=cfg.retrieval_top_k,
                        model_name=cfg.reranker_model,
                    )
                else:
                    scored = retrieved_candidates[: cfg.retrieval_top_k]

                retrieved_context_str = format_retrieved_context(scored)
                logger.info("Retrieved %d repository context chunks for review", len(scored))

    # 2. Chunk diffs
    chunks = chunk_diffs(pr_context.file_diffs, max_tokens=cfg.max_diff_tokens)
    logger.info(
        "Split PR diff into %d chunk(s) for review of %s#%d",
        len(chunks),
        pr_context.repo_full_name,
        pr_context.pr_number,
    )

    # 3. Concurrently review chunks
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [
        _review_chunk(
            chunk=c,
            pr_context=pr_context,
            llm_client=client,
            semaphore=semaphore,
            timeout=cfg.llm_timeout_seconds,
            retrieved_context=retrieved_context_str,
            prompt_variant=cfg.prompt_variant,
            redis_client=redis_client,
        )
        for c in chunks
    ]
    results = await asyncio.gather(*tasks)

    # 4. Aggregate raw findings and usage
    all_raw_findings: list[ReviewFinding] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    for findings, usage in results:
        all_raw_findings.extend(findings)
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        total_cost += usage.cost_usd

    # 5. Self-verification pass (gated by config flag self_verification.enabled)
    if cfg.self_verification_enabled and all_raw_findings:
        verified_findings, verify_usage = await verify_findings(
            file_diffs=pr_context.file_diffs,
            candidate_findings=all_raw_findings,
            llm_client=client,
            timeout=cfg.llm_timeout_seconds,
        )
        all_raw_findings = verified_findings
        total_input_tokens += verify_usage.input_tokens
        total_output_tokens += verify_usage.output_tokens
        total_cost += verify_usage.cost_usd

    # 6. Post-process findings
    active_trace = get_current_trace()
    if active_trace is not None:
        async with active_trace.span(
            "postprocess",
            metadata={"raw_findings_count": len(all_raw_findings)},
        ) as post_span:
            final_findings = postprocess_findings(
                findings=all_raw_findings,
                file_diffs=pr_context.file_diffs,
                confidence_threshold=cfg.confidence_threshold,
                max_comments_per_pr=cfg.max_comments_per_pr,
            )
            post_span.metadata["final_findings_count"] = len(final_findings)
    else:
        final_findings = postprocess_findings(
            findings=all_raw_findings,
            file_diffs=pr_context.file_diffs,
            confidence_threshold=cfg.confidence_threshold,
            max_comments_per_pr=cfg.max_comments_per_pr,
        )

    elapsed_ms = round((time.time() - start_time) * 1000.0, 2)
    model_name = getattr(client, "model", cfg.llm_model)

    return ReviewResult(
        findings=final_findings,
        model=model_name,
        prompt_version=prompt_ver,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        latency_ms=elapsed_ms,
        cost_usd=round(total_cost, 6),
    )
