"""Interactive multi-turn PR review thread resolution and conversational replies.

Enables developers to reply directly to @prism-review or inline review comments
with clarifications, justifications, or questions. Prism evaluates the thread context,
diff hunk, and developer arguments to either:
1. Concede false positives gracefully and resolve the finding.
2. Maintain the finding with clear technical rationale and patch suggestions.
3. Answer technical questions and offer refactoring guidance.
"""

import logging
import re
from enum import StrEnum

from pydantic import BaseModel, Field

from prism.review.llm import LLMClient

logger = logging.getLogger(__name__)

PRISM_REPLY_MARKER = "<!-- prism-reply -->"
PRISM_MENTION_REGEX = re.compile(r"@prism(-review)?\b", re.IGNORECASE)


class ThreadResolutionDecision(StrEnum):
    """Outcome of evaluating developer feedback on a review comment."""

    RESOLVED = "resolved"  # Developer context validated; bot concedes false positive
    MAINTAINED = "maintained"  # Bot maintains violation with technical rationale
    CLARIFICATION = "clarification"  # Answering question or providing code suggestion


class ThreadResolutionResult(BaseModel):
    """Structured resolution evaluation output from the LLM."""

    decision: ThreadResolutionDecision = Field(
        description=(
            "Whether the finding is conceded as resolved/false-positive, "
            "maintained as a valid violation, or answering a question/clarification."
        )
    )
    is_false_positive: bool = Field(
        default=False,
        description=(
            "True if the model agrees the original finding was a false positive "
            "based on developer explanation or broader context."
        ),
    )
    explanation: str = Field(
        description="Clear, technically rigorous markdown explanation responding directly to the developer."
    )
    suggested_patch: str | None = Field(
        default=None,
        description="Optional replacement code snippet or suggestion, or None if not applicable.",
    )


THREAD_RESOLVER_SYSTEM_PROMPT = """You are Prism Review, an expert senior principal software engineer and automated code reviewer responding to an inline discussion thread on a GitHub pull request.

A developer has replied to an automated review finding or tagged @prism-review with a clarification, objection, or question.

Your responsibility:
1. Evaluate the developer's message objectively against the code diff hunk and original review comment.
2. If the developer provides valid architectural context, explains that the issue is mitigated elsewhere (e.g. sanitization in middleware, upstream invariant, intentional business logic), or you realize the original finding was inaccurate:
   - Set `decision="resolved"` and `is_false_positive=true`.
   - Concede gracefully, apologize for the false alarm, and confirm that no changes are necessary.
3. If the developer's objection is mistaken or introduces real security/correctness/maintainability risks (e.g., they claim client-side validation is sufficient, or miss an edge-case concurrency race condition):
   - Set `decision="maintained"` and `is_false_positive=false`.
   - Be polite, constructive, and explain the concrete failure scenario or attack vector with evidence.
   - Provide a clean `suggested_patch` demonstrating the safe pattern if applicable.
4. If the developer simply asks a question or asks for guidance/refactoring:
   - Set `decision="clarification"` and `is_false_positive=false`.
   - Answer the question directly with clear rationale and code examples.

Maintain a collaborative, professional, and humble tone. Never argue defensively; prioritize engineering accuracy and developer velocity."""


def is_prism_mention(text: str) -> bool:
    """Check if text contains @prism or @prism-review mention."""
    return bool(PRISM_MENTION_REGEX.search(text))


def should_respond_to_comment(
    comment_body: str,
    parent_body: str | None = None,
    in_reply_to_id: int | str | None = None,
    sender_login: str | None = None,
    sender_type: str | None = None,
) -> tuple[bool, str]:
    """Determine if Prism should process and reply to this review comment."""
    if sender_type == "Bot" or (sender_login and sender_login.endswith("[bot]")):
        return False, "sender is a bot"

    clean_body = comment_body.strip()
    if not clean_body:
        return False, "empty comment body"

    # Fast-path for simple thumbs reactions
    if clean_body in ("+1", "-1", "👍", "👎", "thumbs_up", "thumbs_down"):
        return False, "reaction comment"

    # Explicit mention: @prism or @prism-review
    if is_prism_mention(clean_body):
        return True, "explicit mention"

    # Thread reply to an earlier Prism comment
    if in_reply_to_id and parent_body and ("<!-- prism" in parent_body or "Prism" in parent_body):
        return True, "reply to prism review comment"

    return False, "no mention or prism thread association"


def build_thread_resolution_prompt(
    developer_message: str,
    developer_author: str,
    parent_comment_body: str,
    file_path: str,
    diff_hunk: str | None = None,
    line_number: int | None = None,
    standards_context: str | None = None,
) -> str:
    """Construct structured user prompt for the thread resolver LLM."""
    sections: list[str] = [
        f"### Thread Context on `{file_path}`" + (f" (Line {line_number})" if line_number else ""),
        "",
        "#### Original Prism Review Finding:",
        parent_comment_body.strip() if parent_comment_body else "*(Initial inline comment)*",
        "",
    ]

    if diff_hunk:
        sections.extend(
            [
                "#### Relevant Diff Hunk:",
                "```diff",
                diff_hunk.strip(),
                "```",
                "",
            ]
        )

    if standards_context:
        sections.extend(
            [
                "#### Applicable Repository Standards:",
                standards_context.strip(),
                "",
            ]
        )

    sections.extend(
        [
            f"#### Developer Response from @{developer_author}:",
            developer_message.strip(),
            "",
            "Evaluate the developer's response against the code diff and provide your structured resolution.",
        ]
    )

    return "\n".join(sections)


def format_thread_reply_markdown(
    result: ThreadResolutionResult,
    developer_author: str,
) -> str:
    """Render structured resolution into formatted GitHub markdown."""
    lines: list[str] = [
        PRISM_REPLY_MARKER,
        "### 💬 Prism Review Response",
        "",
    ]

    if result.decision == ThreadResolutionDecision.RESOLVED:
        status_line = "✅ **Status**: Conceded (False Positive Accepted)"
    elif result.decision == ThreadResolutionDecision.MAINTAINED:
        status_line = "⚠️ **Status**: Finding Maintained"
    else:
        status_line = "ℹ️ **Status**: Clarification & Guidance"

    lines.append(status_line)
    lines.append("")
    lines.append(result.explanation.strip())

    if result.suggested_patch:
        lines.append("")
        lines.append("```suggestion")
        lines.append(result.suggested_patch.strip())
        lines.append("```")

    lines.append("")
    lines.append(
        f"> *Replied to @{developer_author}. "
        "If this addresses your inquiry, you may resolve this thread.*"
    )

    return "\n".join(lines)


async def resolve_thread_conversation(
    llm_client: LLMClient,
    developer_message: str,
    developer_author: str,
    parent_comment_body: str,
    file_path: str,
    diff_hunk: str | None = None,
    line_number: int | None = None,
    standards_context: str | None = None,
) -> tuple[ThreadResolutionResult, str]:
    """Execute LLM resolution and return both structured result and formatted markdown."""
    user_prompt = build_thread_resolution_prompt(
        developer_message=developer_message,
        developer_author=developer_author,
        parent_comment_body=parent_comment_body,
        file_path=file_path,
        diff_hunk=diff_hunk,
        line_number=line_number,
        standards_context=standards_context,
    )

    result, _ = await llm_client.complete_structured(
        system=THREAD_RESOLVER_SYSTEM_PROMPT,
        user=user_prompt,
        schema=ThreadResolutionResult,
    )

    markdown_reply = format_thread_reply_markdown(result, developer_author)
    return result, markdown_reply
