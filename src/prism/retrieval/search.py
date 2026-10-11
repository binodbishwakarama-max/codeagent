"""Hybrid retrieval: BM25 + vector search with Reciprocal Rank Fusion (RRF).

Queries are built from changed symbols in the PR diff rather than the raw diff text.
"""

import collections
import logging
import math
import re

from prism.models import FileDiff
from prism.retrieval.indexer import CodeChunk, EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)

# Common keywords to exclude from symbol extraction
KEYWORDS_TO_IGNORE = {
    # Python
    "def",
    "class",
    "self",
    "cls",
    "return",
    "yield",
    "raise",
    "try",
    "except",
    "finally",
    "if",
    "elif",
    "else",
    "for",
    "while",
    "break",
    "continue",
    "pass",
    "with",
    "as",
    "import",
    "from",
    "async",
    "await",
    "lambda",
    "global",
    "nonlocal",
    "assert",
    "del",
    "in",
    "is",
    "not",
    "and",
    "or",
    "true",
    "false",
    "none",
    # TypeScript / JavaScript
    "function",
    "const",
    "let",
    "var",
    "interface",
    "type",
    "enum",
    "namespace",
    "export",
    "default",
    "extends",
    "implements",
    "public",
    "private",
    "protected",
    "readonly",
    "static",
    "abstract",
    "override",
    "new",
    "this",
    "super",
    "null",
    "undefined",
    "void",
    "any",
    "unknown",
    "never",
    "string",
    "number",
    "boolean",
    "symbol",
    "object",
    "switch",
    "case",
    "throw",
    "catch",
    "instanceof",
    "typeof",
    # Common short abbreviations
    "str",
    "int",
    "bool",
    "dict",
    "list",
    "set",
    "tuple",
    "optional",
    "union",
}


# ---------------------------------------------------------------------------
# Changed Symbol Extraction from Diffs
# ---------------------------------------------------------------------------


def extract_changed_symbols(file_diffs: list[FileDiff]) -> list[str]:
    """Extract distinct identifiers and symbols referenced or defined in changed diff lines.

    Analyzes added/modified lines in hunks to extract class, function, method,
    and key variable names while excluding programming keywords.
    """
    symbol_counter: collections.Counter[str] = collections.Counter()

    # Regex patterns for definitions and symbol occurrences
    def_pattern = re.compile(
        r"(?:def|class|function|interface|type)\s+([a-zA-Z0-9_$]+)",
        re.IGNORECASE,
    )
    call_pattern = re.compile(r"([a-zA-Z0-9_$]{2,})\s*\(", re.ASCII)
    assign_pattern = re.compile(r"([a-zA-Z0-9_$]{2,})\s*=", re.ASCII)
    dotted_pattern = re.compile(r"\.([a-zA-Z0-9_$]{2,})", re.ASCII)
    import_pattern = re.compile(r"(?:import|from)\s+([a-zA-Z0-9_$,\s]+)", re.ASCII)

    for fd in file_diffs:
        # Include file basename as context symbol
        base_name = fd.path.rsplit("/", 1)[-1].split(".", 1)[0]
        if len(base_name) > 2 and base_name.lower() not in KEYWORDS_TO_IGNORE:
            symbol_counter[base_name] += 2

        for hunk in fd.hunks:
            for line in hunk.lines:
                if line.kind != "add":
                    continue
                text = line.text.strip()
                if not text or text.startswith(("#", "//", "/*", "*")):
                    continue

                # 1. Definitions (highest weight)
                for match in def_pattern.findall(text):
                    sym = match.strip()
                    if len(sym) >= 2 and sym.lower() not in KEYWORDS_TO_IGNORE:
                        symbol_counter[sym] += 5

                # 2. Imports (high weight)
                for imp in import_pattern.findall(text):
                    for sub in re.split(r"[,\s]+", imp):
                        s = sub.strip()
                        if len(s) >= 2 and s.lower() not in KEYWORDS_TO_IGNORE:
                            symbol_counter[s] += 3

                # 3. Function/method invocations
                for call in call_pattern.findall(text):
                    s = call.strip()
                    if len(s) >= 2 and s.lower() not in KEYWORDS_TO_IGNORE:
                        symbol_counter[s] += 2

                # 4. Assignments
                for ass in assign_pattern.findall(text):
                    s = ass.strip()
                    if len(s) >= 2 and s.lower() not in KEYWORDS_TO_IGNORE:
                        symbol_counter[s] += 1

                # 5. Dotted accesses
                for dot in dotted_pattern.findall(text):
                    s = dot.strip()
                    if len(s) >= 2 and s.lower() not in KEYWORDS_TO_IGNORE:
                        symbol_counter[s] += 1

    # Return unique symbols sorted by frequency descending
    symbols = [sym for sym, _count in symbol_counter.most_common(20)]
    return symbols


def build_search_query(file_diffs: list[FileDiff], max_symbols: int = 10) -> str:
    """Construct a clean, focused retrieval query from changed symbols in the diff."""
    symbols = extract_changed_symbols(file_diffs)
    if not symbols:
        # Fallback to file names
        file_names = [fd.path.rsplit("/", 1)[-1] for fd in file_diffs]
        return " ".join(file_names)
    return " ".join(symbols[:max_symbols])


# ---------------------------------------------------------------------------
# Vector Search (Cosine Similarity)
# ---------------------------------------------------------------------------


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two normalized or non-normalized vectors."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=False):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def search_vector(
    query: str,
    chunks: list[CodeChunk],
    top_k: int = 30,
    embedding_model: EmbeddingModel | None = None,
) -> list[tuple[CodeChunk, float]]:
    """Retrieve top_k code chunks using dense vector cosine similarity."""
    if not chunks or not query.strip():
        return []

    embedder = embedding_model or get_embedding_model()
    query_emb = embedder.embed_query(query)

    scored: list[tuple[CodeChunk, float]] = []
    for chunk in chunks:
        if not chunk.embedding:
            continue
        sim = _cosine_similarity(query_emb, chunk.embedding)
        scored.append((chunk, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# BM25 Keyword Search
# ---------------------------------------------------------------------------


class BM25Searcher:
    """In-memory BM25 index and ranking engine for code chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> list[str]:
        return [tok for tok in re.findall(r"\w+", text.lower()) if tok not in KEYWORDS_TO_IGNORE]

    def search(
        self,
        query: str,
        chunks: list[CodeChunk],
        top_k: int = 30,
    ) -> list[tuple[CodeChunk, float]]:
        """Rank code chunks against query terms using BM25 scoring."""
        if not chunks or not query.strip():
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        n_docs = len(chunks)
        doc_tokens: list[list[str]] = []
        doc_lengths: list[int] = []
        term_doc_counts: collections.Counter[str] = collections.Counter()

        for c in chunks:
            # Tokenize chunk representation: path + symbol + code
            doc_str = f"{c.file_path} {c.symbol_name} {c.code}"
            toks = self._tokenize(doc_str)
            doc_tokens.append(toks)
            doc_lengths.append(len(toks))
            for t in set(toks):
                term_doc_counts[t] += 1

        avg_doc_len = sum(doc_lengths) / max(1, n_docs)

        # Precompute IDF for query terms
        idf: dict[str, float] = {}
        for term in set(query_terms):
            df = term_doc_counts[term]
            # Standard Lucene/BM25 IDF
            idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        scored: list[tuple[CodeChunk, float]] = []
        for i, chunk in enumerate(chunks):
            toks = doc_tokens[i]
            d_len = doc_lengths[i]
            tf = collections.Counter(toks)
            score = 0.0

            for term in query_terms:
                if term in tf:
                    t_freq = tf[term]
                    numerator = t_freq * (self.k1 + 1.0)
                    denominator = t_freq + self.k1 * (1.0 - self.b + self.b * (d_len / avg_doc_len))
                    score += idf[term] * (numerator / denominator)

            if score > 0.0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def search_bm25(
    query: str,
    chunks: list[CodeChunk],
    top_k: int = 30,
) -> list[tuple[CodeChunk, float]]:
    """Retrieve top_k code chunks using BM25 keyword matching."""
    bm25 = BM25Searcher()
    return bm25.search(query=query, chunks=chunks, top_k=top_k)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF) & Hybrid Search
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    vector_results: list[tuple[CodeChunk, float]],
    bm25_results: list[tuple[CodeChunk, float]],
    rrf_k: int = 60,
    top_k: int = 30,
) -> list[tuple[CodeChunk, float]]:
    """Merge ranked results from dense vector and BM25 searches using RRF.

    RRF_score(d) = sum( 1.0 / (rrf_k + rank_i(d)) ) for each ranking list.
    """
    rrf_scores: dict[int, float] = collections.defaultdict(float)
    chunk_map: dict[int, CodeChunk] = {}

    # 1. Vector ranks
    for rank, (chunk, _score) in enumerate(vector_results, start=1):
        cid = id(chunk)
        chunk_map[cid] = chunk
        rrf_scores[cid] += 1.0 / (rrf_k + rank)

    # 2. BM25 ranks
    for rank, (chunk, _score) in enumerate(bm25_results, start=1):
        cid = id(chunk)
        chunk_map[cid] = chunk
        rrf_scores[cid] += 1.0 / (rrf_k + rank)

    sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [(chunk_map[cid], round(rrf_scores[cid], 6)) for cid in sorted_cids[:top_k]]


def hybrid_search(
    query: str,
    chunks: list[CodeChunk],
    top_k: int = 30,
    rrf_k: int = 60,
    embedding_model: EmbeddingModel | None = None,
) -> list[tuple[CodeChunk, float]]:
    """Execute hybrid search combining BM25 keyword and dense vector search via RRF."""
    vec_results = search_vector(
        query=query, chunks=chunks, top_k=top_k, embedding_model=embedding_model
    )
    bm25_results = search_bm25(query=query, chunks=chunks, top_k=top_k)
    return reciprocal_rank_fusion(
        vector_results=vec_results,
        bm25_results=bm25_results,
        rrf_k=rrf_k,
        top_k=top_k,
    )


def retrieve_context(
    file_diffs: list[FileDiff],
    chunks: list[CodeChunk],
    mode: str = "hybrid",  # "vector" or "hybrid"
    top_k: int = 30,
    embedding_model: EmbeddingModel | None = None,
) -> list[tuple[CodeChunk, float]]:
    """Top-level retrieval entrypoint: extracts symbols, builds query, runs search."""
    if not chunks or not file_diffs:
        return []

    query = build_search_query(file_diffs)
    if not query:
        return []

    if mode == "vector":
        return search_vector(
            query=query, chunks=chunks, top_k=top_k, embedding_model=embedding_model
        )
    return hybrid_search(query=query, chunks=chunks, top_k=top_k, embedding_model=embedding_model)
