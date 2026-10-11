"""Retrieval and semantic indexing module for engineering guidelines and codebase patterns."""

from prism.retrieval.indexer import (
    CodeChunk,
    EmbeddingModel,
    RepositoryIndexer,
    chunk_source_file,
    compute_file_hash,
    get_embedding_model,
)
from prism.retrieval.rerank import (
    CrossEncoderReranker,
    format_retrieved_context,
    get_cross_encoder_reranker,
    rerank_chunks,
)
from prism.retrieval.search import (
    build_search_query,
    extract_changed_symbols,
    hybrid_search,
    reciprocal_rank_fusion,
    retrieve_context,
    search_bm25,
    search_vector,
)

__all__ = [
    "CodeChunk",
    "CrossEncoderReranker",
    "EmbeddingModel",
    "RepositoryIndexer",
    "build_search_query",
    "chunk_source_file",
    "compute_file_hash",
    "extract_changed_symbols",
    "format_retrieved_context",
    "get_cross_encoder_reranker",
    "get_embedding_model",
    "hybrid_search",
    "reciprocal_rank_fusion",
    "rerank_chunks",
    "retrieve_context",
    "search_bm25",
    "search_vector",
]
