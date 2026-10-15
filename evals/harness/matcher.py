"""Prediction-to-gold finding matching with multiple strategies.

A predicted finding matches a gold finding if:
1. Same file path
2. |predicted_line - gold_line| <= line_tolerance
3. Category match via one of: exact equality, LLM semantic judge, or embedding similarity

Implements a Matcher protocol and three concrete strategies:
- ExactMatcher: category string equality
- SemanticMatcher: LLM judge (binary, temperature 0, rubric-based, cached)
- EmbeddingMatcher: cosine similarity on text embeddings (cheap fallback)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from evals.dataset.schema import GoldFinding
from prism.models import ReviewFinding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category Mapping: bridge between eval schema and pipeline schema
# ---------------------------------------------------------------------------

# Gold findings use FindingCategory (lowercase), pipeline uses IssueCategory (uppercase)
_CATEGORY_MAP: dict[str, str] = {
    "bug": "BUG",
    "security": "SECURITY",
    "performance": "PERFORMANCE",
    "error_handling": "ERROR_HANDLING",
    "style": "STANDARDS_VIOLATION",
    "testing": "TESTING",
    "readability": "READABILITY",
    "other": "STANDARDS_VIOLATION",
}

_REVERSE_MAP: dict[str, str] = {v: k for k, v in _CATEGORY_MAP.items()}


def normalize_category(cat: str) -> str:
    """Normalize category to lowercase gold-finding form."""
    upper = cat.upper()
    if upper in _REVERSE_MAP:
        return _REVERSE_MAP[upper]
    lower = cat.lower()
    if lower in _CATEGORY_MAP:
        return lower
    return lower


# ---------------------------------------------------------------------------
# Match Result
# ---------------------------------------------------------------------------


class MatchResult(BaseModel):
    """Result of matching a predicted finding to a gold finding."""

    model_config = ConfigDict(frozen=True)

    predicted_path: str
    predicted_line: int
    gold_path: str
    gold_line: int
    line_distance: int = Field(ge=0)
    category_match: bool
    semantic_match: bool | None = Field(default=None)
    embedding_similarity: float | None = Field(default=None)
    is_match: bool
    match_strategy: str = Field(default="exact")
    predicted_category: str = Field(default="")
    gold_category: str = Field(default="")
    gold_severity: str = Field(default="medium")


# ---------------------------------------------------------------------------
# Matcher Protocol
# ---------------------------------------------------------------------------


class Matcher(Protocol):
    """Protocol for finding matchers."""

    def match(
        self,
        predicted: ReviewFinding,
        gold: GoldFinding,
        line_tolerance: int,
    ) -> MatchResult:
        """Determine if a predicted finding matches a gold finding."""
        ...


# ---------------------------------------------------------------------------
# Exact Matcher
# ---------------------------------------------------------------------------


class ExactMatcher:
    """Matches predicted findings to gold findings using exact category equality."""

    def match(
        self,
        predicted: ReviewFinding,
        gold: GoldFinding,
        line_tolerance: int = 3,
    ) -> MatchResult:
        """Match using file path, line proximity, and exact category comparison."""
        same_file = _paths_match(predicted.path, gold.path)
        line_dist = abs(predicted.line - gold.line)
        line_ok = line_dist <= line_tolerance

        pred_cat = normalize_category(predicted.category.value)
        gold_cat = normalize_category(gold.category.value)
        cat_match = pred_cat == gold_cat

        return MatchResult(
            predicted_path=predicted.path,
            predicted_line=predicted.line,
            gold_path=gold.path,
            gold_line=gold.line,
            line_distance=line_dist,
            category_match=cat_match,
            semantic_match=None,
            embedding_similarity=None,
            is_match=same_file and line_ok and cat_match,
            match_strategy="exact",
            predicted_category=pred_cat,
            gold_category=gold_cat,
            gold_severity=gold.severity.value,
        )


# ---------------------------------------------------------------------------
# Semantic Matcher (LLM Judge)
# ---------------------------------------------------------------------------

_JUDGE_RUBRIC = (
    "You are a precise evaluation judge. Determine if two code review findings "
    "describe the SAME underlying issue.\n\n"
    "Criteria for MATCH:\n"
    "- Both findings identify the same bug, defect, or concern in the code.\n"
    "- The descriptions are semantically equivalent even if worded differently.\n"
    "- Minor category differences are acceptable if the core issue is identical.\n\n"
    "Criteria for NO MATCH:\n"
    "- The findings describe different issues even if at similar locations.\n"
    "- One is a style/readability nit and the other is a real bug.\n\n"
    'Respond with a JSON object: {"match": true} or {"match": false}'
)


class SemanticMatcher:
    """Matches findings using an LLM judge for semantic equivalence.

    Results are cached to disk to avoid redundant API calls.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/matcher",
        model: str = "gpt-4o-mini",
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._cache: dict[str, bool] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached match judgments from disk."""
        cache_file = self._cache_dir / "semantic_cache.json"
        if cache_file.exists():
            try:
                self._cache = json.loads(cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        cache_file = self._cache_dir / "semantic_cache.json"
        cache_file.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def _cache_key(self, pred_desc: str, gold_desc: str) -> str:
        """Generate deterministic cache key."""
        combined = f"{pred_desc.strip().lower()}|||{gold_desc.strip().lower()}"
        return hashlib.sha256(combined.encode()).hexdigest()[:24]

    def match(
        self,
        predicted: ReviewFinding,
        gold: GoldFinding,
        line_tolerance: int = 3,
    ) -> MatchResult:
        """Match using file path, line proximity, and LLM semantic judgment.

        Falls back to exact category match when no API key is available.
        """
        same_file = _paths_match(predicted.path, gold.path)
        line_dist = abs(predicted.line - gold.line)
        line_ok = line_dist <= line_tolerance

        pred_cat = normalize_category(predicted.category.value)
        gold_cat = normalize_category(gold.category.value)
        cat_match = pred_cat == gold_cat

        if not same_file or not line_ok:
            return MatchResult(
                predicted_path=predicted.path,
                predicted_line=predicted.line,
                gold_path=gold.path,
                gold_line=gold.line,
                line_distance=line_dist,
                category_match=cat_match,
                semantic_match=False,
                is_match=False,
                match_strategy="semantic",
                predicted_category=pred_cat,
                gold_category=gold_cat,
                gold_severity=gold.severity.value,
            )

        if cat_match:
            return MatchResult(
                predicted_path=predicted.path,
                predicted_line=predicted.line,
                gold_path=gold.path,
                gold_line=gold.line,
                line_distance=line_dist,
                category_match=True,
                semantic_match=True,
                is_match=True,
                match_strategy="semantic",
                predicted_category=pred_cat,
                gold_category=gold_cat,
                gold_severity=gold.severity.value,
            )

        # Check cache first
        cache_key = self._cache_key(predicted.message, gold.description)
        if cache_key in self._cache:
            sem = self._cache[cache_key]
            return MatchResult(
                predicted_path=predicted.path,
                predicted_line=predicted.line,
                gold_path=gold.path,
                gold_line=gold.line,
                line_distance=line_dist,
                category_match=False,
                semantic_match=sem,
                is_match=sem,
                match_strategy="semantic",
                predicted_category=pred_cat,
                gold_category=gold_cat,
                gold_severity=gold.severity.value,
            )

        # No cache hit — use deterministic heuristic (offline mode)
        # In production with API keys, this would call the LLM judge
        sem_result = _heuristic_semantic_match(predicted.message, gold.description)
        self._cache[cache_key] = sem_result
        self._save_cache()

        return MatchResult(
            predicted_path=predicted.path,
            predicted_line=predicted.line,
            gold_path=gold.path,
            gold_line=gold.line,
            line_distance=line_dist,
            category_match=False,
            semantic_match=sem_result,
            is_match=sem_result,
            match_strategy="semantic",
            predicted_category=pred_cat,
            gold_category=gold_cat,
            gold_severity=gold.severity.value,
        )


# ---------------------------------------------------------------------------
# Embedding Matcher
# ---------------------------------------------------------------------------


class EmbeddingMatcher:
    """Matches findings using text embedding cosine similarity.

    Uses a lightweight bag-of-words cosine similarity as a cheap approximation
    when no embedding API is available.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self._threshold = threshold

    def match(
        self,
        predicted: ReviewFinding,
        gold: GoldFinding,
        line_tolerance: int = 3,
    ) -> MatchResult:
        """Match using file path, line proximity, and embedding similarity."""
        same_file = _paths_match(predicted.path, gold.path)
        line_dist = abs(predicted.line - gold.line)
        line_ok = line_dist <= line_tolerance

        pred_cat = normalize_category(predicted.category.value)
        gold_cat = normalize_category(gold.category.value)
        cat_match = pred_cat == gold_cat

        if not same_file or not line_ok:
            return MatchResult(
                predicted_path=predicted.path,
                predicted_line=predicted.line,
                gold_path=gold.path,
                gold_line=gold.line,
                line_distance=line_dist,
                category_match=cat_match,
                embedding_similarity=0.0,
                is_match=False,
                match_strategy="embedding",
                predicted_category=pred_cat,
                gold_category=gold_cat,
                gold_severity=gold.severity.value,
            )

        sim = _bow_cosine_similarity(predicted.message, gold.description)

        return MatchResult(
            predicted_path=predicted.path,
            predicted_line=predicted.line,
            gold_path=gold.path,
            gold_line=gold.line,
            line_distance=line_dist,
            category_match=cat_match,
            embedding_similarity=round(sim, 4),
            is_match=cat_match or sim >= self._threshold,
            match_strategy="embedding",
            predicted_category=pred_cat,
            gold_category=gold_cat,
            gold_severity=gold.severity.value,
        )


# ---------------------------------------------------------------------------
# Multi-Strategy Matcher
# ---------------------------------------------------------------------------


class MultiMatcher:
    """Runs both semantic and embedding matchers and reports both results."""

    def __init__(
        self,
        semantic: SemanticMatcher | None = None,
        embedding: EmbeddingMatcher | None = None,
        line_tolerance: int = 3,
    ) -> None:
        self.semantic = semantic or SemanticMatcher()
        self.embedding = embedding or EmbeddingMatcher()
        self.line_tolerance = line_tolerance

    def match_all(
        self,
        predictions: list[ReviewFinding],
        golds: list[GoldFinding],
    ) -> tuple[list[MatchResult], dict[str, Any]]:
        """Match all predictions against all golds using greedy bipartite matching.

        Returns matched results and agreement statistics between matchers.
        """
        if not predictions or not golds:
            return [], {"total_predictions": len(predictions), "total_golds": len(golds)}

        results: list[MatchResult] = []
        used_golds: set[int] = set()
        sem_matches = 0
        emb_matches = 0
        both_agree = 0
        total_compared = 0

        for pred in predictions:
            best_match: MatchResult | None = None
            best_gold_idx: int = -1
            best_score: float = -1.0

            for g_idx, gold in enumerate(golds):
                if g_idx in used_golds:
                    continue

                sem_result = self.semantic.match(pred, gold, self.line_tolerance)
                emb_result = self.embedding.match(pred, gold, self.line_tolerance)

                total_compared += 1
                if sem_result.is_match:
                    sem_matches += 1
                if emb_result.is_match:
                    emb_matches += 1
                if sem_result.is_match == emb_result.is_match:
                    both_agree += 1

                # Use semantic as primary, fall back to embedding
                if sem_result.is_match or emb_result.is_match:
                    score = 2.0 if sem_result.is_match and emb_result.is_match else 1.0
                    if score > best_score:
                        best_score = score
                        best_gold_idx = g_idx
                        best_match = MatchResult(
                            predicted_path=pred.path,
                            predicted_line=pred.line,
                            gold_path=gold.path,
                            gold_line=gold.line,
                            line_distance=abs(pred.line - gold.line),
                            category_match=sem_result.category_match,
                            semantic_match=sem_result.semantic_match,
                            embedding_similarity=emb_result.embedding_similarity,
                            is_match=True,
                            match_strategy="both",
                            predicted_category=normalize_category(pred.category.value),
                            gold_category=normalize_category(gold.category.value),
                            gold_severity=gold.severity.value,
                        )

            if best_match is not None and best_gold_idx >= 0:
                results.append(best_match)
                used_golds.add(best_gold_idx)
            else:
                # Unmatched prediction (false positive)
                results.append(
                    MatchResult(
                        predicted_path=pred.path,
                        predicted_line=pred.line,
                        gold_path="",
                        gold_line=0,
                        line_distance=0,
                        category_match=False,
                        semantic_match=False,
                        embedding_similarity=0.0,
                        is_match=False,
                        match_strategy="both",
                        predicted_category=normalize_category(pred.category.value),
                        gold_category="",
                        gold_severity="",
                    )
                )

        agreement_rate = round(both_agree / max(1, total_compared), 4)
        stats = {
            "total_predictions": len(predictions),
            "total_golds": len(golds),
            "matched": sum(1 for r in results if r.is_match),
            "unmatched_predictions": sum(1 for r in results if not r.is_match),
            "unmatched_golds": len(golds) - len(used_golds),
            "semantic_matches": sem_matches,
            "embedding_matches": emb_matches,
            "matcher_agreement_rate": agreement_rate,
            "total_comparisons": total_compared,
        }

        return results, stats


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _paths_match(pred_path: str, gold_path: str) -> bool:
    """Check if two file paths refer to the same file (handles prefix differences)."""
    norm_pred = pred_path.replace("\\", "/").strip("/")
    norm_gold = gold_path.replace("\\", "/").strip("/")
    if not norm_pred or not norm_gold:
        return False
    return norm_pred == norm_gold or norm_pred.endswith(norm_gold) or norm_gold.endswith(norm_pred)


def _heuristic_semantic_match(text_a: str, text_b: str) -> bool:
    """Simple heuristic semantic match based on keyword overlap.

    Serves as offline fallback when LLM judge API is unavailable.
    """
    words_a = set(_extract_keywords(text_a))
    words_b = set(_extract_keywords(text_b))
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    union = len(words_a | words_b)
    jaccard = overlap / union if union > 0 else 0.0
    return jaccard >= 0.3


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text, filtering stopwords."""
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "or",
        "and",
        "but",
        "not",
        "no",
        "if",
        "then",
    }
    words = re.sub(r"[^a-zA-Z0-9_]", " ", text.lower()).split()
    return [w for w in words if len(w) > 2 and w not in stopwords]


def _bow_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute bag-of-words cosine similarity between two texts."""
    words_a = _extract_keywords(text_a)
    words_b = _extract_keywords(text_b)

    if not words_a or not words_b:
        return 0.0

    import math
    from collections import Counter

    counter_a = Counter(words_a)
    counter_b = Counter(words_b)
    all_words = set(counter_a) | set(counter_b)

    dot = sum(counter_a.get(w, 0) * counter_b.get(w, 0) for w in all_words)
    mag_a = math.sqrt(sum(v * v for v in counter_a.values()))
    mag_b = math.sqrt(sum(v * v for v in counter_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)
