"""Self-verification pass to filter false positives and hallucinations."""

import logging
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from prism.models import FileDiff, ReviewFinding
from prism.observability.tracing import trace_llm_call
from prism.review.llm import LLMClient, LLMUsage
from prism.review.prompts import render_diff_with_line_numbers

logger = logging.getLogger(__name__)


class FindingVerification(BaseModel):
    """Verification outcome for an individual candidate finding."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="File path of the finding")
    line: int = Field(description="Line number of the finding")
    decision: Literal["KEEP", "DROP"] = Field(
        description="KEEP if genuine actionable defect on changed lines; DROP if false positive, stylistic nit, or invalid"
    )
    reason: str = Field(description="Concrete reason justifying keep or drop decision")


class VerificationResponse(BaseModel):
    """Collection of verification decisions from the self-verification LLM pass."""

    model_config = ConfigDict(frozen=True)

    verifications: list[FindingVerification] = Field(
        default_factory=list,
        description="List of verification decisions for candidate findings",
    )


VERIFICATION_SYSTEM_PROMPT = """You are a Principal Software Engineer performing an impartial verification pass on automated pull request review comments.

Your primary objective is to aggressively eliminate false positives, hallucinated issues, and unhelpful nitpicks.

### Strict Verification Criteria
- KEEP:
  1. The issue is a genuine defect (logic bug, security risk, performance regression, standards violation).
  2. The issue occurs on or is directly introduced by the added/changed lines in this diff.
  3. The critique is concrete, correct, and actionable.

- DROP:
  1. The issue is speculative, stylistic bikeshedding, or a minor nitpick.
  2. The comment is on an unchanged context line that was not caused by the PR changes.
  3. The reviewer misunderstood the code logic, language semantics, or standard library.
  4. The issue is already handled elsewhere in the diff or surrounding context.
  5. The suggested change would introduce a new bug or syntax error.

Be rigorous and objective. When in doubt, prefer DROP to preserve high reviewer signal-to-noise ratio.
"""


def build_verification_prompt(
    file_diffs: list[FileDiff],
    candidate_findings: list[ReviewFinding],
) -> str:
    """Format PR diff and candidate findings for verification pass."""
    rendered_diff = render_diff_with_line_numbers(file_diffs)

    findings_text: list[str] = []
    for idx, f in enumerate(candidate_findings, start=1):
        sug_text = f"\n  Suggestion: {f.suggestion}" if f.suggestion else ""
        findings_text.append(
            f"[{idx}] {f.path}:{f.line} [{f.category}] (Severity: {f.severity}, Confidence: {f.confidence:.2f})\n"
            f"  Message: {f.message}{sug_text}"
        )

    return f"""## Pull Request Diff:
{rendered_diff}

## Candidate Review Findings to Verify:
{chr(10).join(findings_text)}

Evaluate each candidate finding against the diff.
Answer with a decision of KEEP or DROP and a clear reason for each finding.
"""


async def verify_findings(
    file_diffs: list[FileDiff],
    candidate_findings: list[ReviewFinding],
    llm_client: LLMClient,
    timeout: float = 30.0,
) -> tuple[list[ReviewFinding], LLMUsage]:
    """Execute second-pass self-verification to filter false positives.

    Every function that calls an LLM is traced, has a timeout, and returns typed output.
    """
    if not candidate_findings:
        model_name = getattr(llm_client, "model", "unknown")
        return [], LLMUsage(
            input_tokens=0, output_tokens=0, cost_usd=0.0, latency_ms=0.0, model=model_name
        )

    start_time = time.time()
    model_name = getattr(llm_client, "model", "unknown")
    user_prompt = build_verification_prompt(file_diffs, candidate_findings)

    async with trace_llm_call(f"verify:{model_name}", model=model_name) as span:
        try:
            response, usage = await llm_client.complete_structured(
                system=VERIFICATION_SYSTEM_PROMPT,
                user=user_prompt,
                schema=VerificationResponse,
                timeout=timeout,
            )
        except Exception as e:
            logger.warning(
                "Self-verification call failed (%s); keeping candidate findings as fallback", e
            )
            return candidate_findings, LLMUsage(
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=round((time.time() - start_time) * 1000.0, 2),
                model=model_name,
            )

        # Build lookup of dropped finding keys (path, line)
        drop_keys: set[tuple[str, int]] = set()
        for v in response.verifications:
            if v.decision == "DROP":
                drop_keys.add((v.path, v.line))
                logger.info(
                    "Self-verification dropped finding at %s:%d: %s", v.path, v.line, v.reason
                )

        kept: list[ReviewFinding] = []
        for f in candidate_findings:
            if (f.path, f.line) in drop_keys:
                continue
            kept.append(f)

        span.record_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )

        logger.info(
            "Self-verification pass finished: %d candidate findings -> %d kept (%d dropped)",
            len(candidate_findings),
            len(kept),
            len(candidate_findings) - len(kept),
        )
        return kept, usage
