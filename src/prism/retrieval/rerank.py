"""Cross-encoder reranker (e.g. BAAI/bge-reranker-base) for top 30 -> top 6 chunks."""

import logging
import re
from typing import Any

from prism.retrieval.indexer import CodeChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranker using FastEmbed TextCrossEncoder with deterministic fallback."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._reranker: Any = None
        self._initialized = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        import os

        if os.environ.get("PRISM_LOCAL_FAST_EMBEDDINGS") == "1":
            self._reranker = None
            return

        try:
            from fastembed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(model_name=self.model_name)
            logger.info("Loaded cross-encoder model '%s'", self.model_name)
        except Exception as e:
            logger.debug(
                "Fastembed cross-encoder unavailable (%s); using heuristic cross-scorer", e
            )
            self._reranker = None

    def _heuristic_score(self, query: str, chunk: CodeChunk) -> float:
        """Deterministic cross-scoring heuristic based on symbol and token overlap."""
        q_tokens = {tok.lower() for tok in re.split(r"[_./\s]+", query) if tok.strip()}
        if not q_tokens:
            return 0.0

        # Exact or substring match in symbol name
        sym_tokens = {
            tok.lower() for tok in re.split(r"[_./\s]+", chunk.symbol_name) if tok.strip()
        }
        sym_overlap = len(q_tokens & sym_tokens)
        if any(q in chunk.symbol_name.lower() for q in q_tokens if len(q) > 2):
            sym_overlap += 1

        # Match in path
        path_tokens = {tok.lower() for tok in re.split(r"[_./\s]+", chunk.file_path) if tok.strip()}
        path_overlap = len(q_tokens & path_tokens)

        # Match in code body
        code_tokens = {tok.lower() for tok in re.split(r"[_./\s]+", chunk.code) if tok.strip()}
        code_overlap = len(q_tokens & code_tokens)

        # Weighted composite score
        score = (sym_overlap * 3.0 + path_overlap * 2.0 + code_overlap * 0.5) / (
            len(q_tokens) * 3.0
        )
        return min(1.0, round(score, 4))

    def rerank(
        self,
        query: str,
        candidates: list[CodeChunk],
        top_k: int = 6,
    ) -> list[tuple[CodeChunk, float]]:
        """Rerank candidates down to top_k using cross-encoder scoring."""
        if not candidates:
            return []

        self._ensure_init()
        if self._reranker is not None:
            try:
                documents = [f"{c.file_path} {c.symbol_name}\n{c.code}" for c in candidates]
                results = list(self._reranker.rerank(query, documents))
                # Map scored results back to CodeChunk
                scored: list[tuple[CodeChunk, float]] = []
                for i, res in enumerate(results):
                    if hasattr(res, "index") and hasattr(res, "score"):
                        idx = int(res.index)
                        score = float(res.score)
                    elif isinstance(res, dict):
                        idx = int(res.get("index", i))
                        score = float(res.get("score", 0.0))
                    else:
                        idx = i
                        score = float(res)
                    if 0 <= idx < len(candidates):
                        scored.append((candidates[idx], round(score, 4)))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:top_k]
            except Exception as e:
                logger.warning("Cross-encoder rerank failed (%s); using heuristic fallback", e)

        # Fallback scoring
        scored_fallback = [(c, self._heuristic_score(query, c)) for c in candidates]
        scored_fallback.sort(key=lambda x: x[1], reverse=True)
        return scored_fallback[:top_k]


# Global singleton reranker
_GLOBAL_RERANKER: CrossEncoderReranker | None = None


def get_cross_encoder_reranker(model_name: str = "BAAI/bge-reranker-base") -> CrossEncoderReranker:
    """Return singleton cross-encoder reranker instance."""
    global _GLOBAL_RERANKER
    if _GLOBAL_RERANKER is None or _GLOBAL_RERANKER.model_name != model_name:
        _GLOBAL_RERANKER = CrossEncoderReranker(model_name=model_name)
    return _GLOBAL_RERANKER


def rerank_chunks(
    query: str,
    candidates: list[CodeChunk],
    top_k: int = 6,
    model_name: str = "BAAI/bge-reranker-base",
) -> list[tuple[CodeChunk, float]]:
    """Rerank top candidates from retrieval down to top_k using cross-encoder."""
    reranker = get_cross_encoder_reranker(model_name=model_name)
    return reranker.rerank(query=query, candidates=candidates, top_k=top_k)


def format_retrieved_context(scored_chunks: list[tuple[CodeChunk, float]]) -> str:
    """Render retrieved and reranked context chunks for inclusion in the LLM review prompt."""
    if not scored_chunks:
        return ""

    sections: list[str] = [
        "## Retrieved Repository Context:",
        "The following existing definitions from the repository provide additional context for this change:\n",
    ]

    for idx, (chunk, score) in enumerate(scored_chunks, start=1):
        rel_str = f" [Relevance: {score:.2f}]" if score > 0.0 else ""
        header = f"### [Context {idx}] File: {chunk.file_path}:{chunk.start_line}-{chunk.end_line} ({chunk.symbol_type}: {chunk.symbol_name}){rel_str}"
        sections.append(f"{header}\n```{chunk.language}\n{chunk.code.strip()}\n```\n")

    return "\n".join(sections)
