"""Unit and integration tests for retrieval (AST chunking, hybrid search, rerank) and self-verification."""

from __future__ import annotations

import pytest

from prism.config import Settings
from prism.models import DiffLine, FileDiff, Hunk, IssueCategory, ReviewFinding, Severity
from prism.retrieval.indexer import (
    CodeChunk,
    EmbeddingModel,
    RepositoryIndexer,
    chunk_source_file,
    compute_file_hash,
)
from prism.retrieval.rerank import (
    format_retrieved_context,
    rerank_chunks,
)
from prism.retrieval.search import (
    build_search_query,
    extract_changed_symbols,
    reciprocal_rank_fusion,
    search_bm25,
    search_vector,
)
from prism.review.llm import FakeLLMClient
from prism.review.pipeline import PRContext, review_pr
from prism.review.verify import FindingVerification, VerificationResponse, verify_findings

# ---------------------------------------------------------------------------
# 1. AST Chunking Tests
# ---------------------------------------------------------------------------


class TestASTChunker:
    """Tests for tree-sitter and AST-based source code chunking."""

    def test_chunk_python_functions_and_classes(self) -> None:
        code = (
            "import os\n\n"
            "def calculate_total(items: list[int]) -> int:\n"
            "    return sum(items)\n\n"
            "class OrderService:\n"
            "    def __init__(self, db):\n"
            "        self.db = db\n\n"
            "    def process_order(self, order_id: str) -> bool:\n"
            "        return True\n"
        )
        chunks = chunk_source_file("src/service.py", code, language="python")
        assert len(chunks) >= 3
        symbol_names = {c.symbol_name for c in chunks}
        assert "calculate_total" in symbol_names
        assert any("OrderService" in s for s in symbol_names)
        for c in chunks:
            assert c.file_path == "src/service.py"
            assert c.file_hash == compute_file_hash(code)
            assert c.start_line <= c.end_line

    def test_chunk_typescript_functions_and_classes(self) -> None:
        code = (
            "export interface User {\n"
            "  id: string;\n"
            "  name: string;\n"
            "}\n\n"
            "export function formatUser(u: User): string {\n"
            "  return u.name.trim();\n"
            "}\n\n"
            "export class UserManager {\n"
            "  getUser(id: string): User | null {\n"
            "    return null;\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_source_file("src/users.ts", code, language="typescript")
        assert len(chunks) >= 2
        symbols = {c.symbol_name for c in chunks}
        assert any("formatUser" in s for s in symbols)
        assert any("UserManager" in s for s in symbols)

    def test_empty_content_returns_empty_list(self) -> None:
        assert chunk_source_file("empty.py", "") == []
        assert chunk_source_file("empty.py", "   \n\n") == []

    def test_non_function_python_fallback(self) -> None:
        script = "x = 10\ny = 20\nprint(x + y)\n"
        chunks = chunk_source_file("script.py", script, language="python")
        assert len(chunks) == 1
        assert chunks[0].symbol_name == "module"


# ---------------------------------------------------------------------------
# 2. Embedding Model & Incremental Indexer Tests
# ---------------------------------------------------------------------------


class TestIndexerAndEmbeddings:
    """Tests for embedding generation and incremental indexing."""

    def test_embedding_dimension_and_normalization(self) -> None:
        embedder = EmbeddingModel(dim=384)
        vec = embedder.embed_query("authenticate user token")
        assert len(vec) == 384
        # Check normalized magnitude ~1.0
        norm = sum(x * x for x in vec) ** 0.5
        assert pytest.approx(norm, rel=1e-3) == 1.0

    @pytest.mark.asyncio
    async def test_incremental_indexing_skips_unchanged_files(self) -> None:
        indexer = RepositoryIndexer()
        files = {
            "src/a.py": "def foo():\n    return 1\n",
            "src/b.py": "def bar():\n    return 2\n",
        }

        # Initial indexing
        chunks_1 = await indexer.index_repository("org/repo", "sha1", files)
        assert len(chunks_1) >= 2

        # Second indexing with identical files (should hit cache)
        chunks_2 = await indexer.index_repository("org/repo", "sha2", files)
        assert len(chunks_2) == len(chunks_1)
        assert [c.symbol_name for c in chunks_1] == [c.symbol_name for c in chunks_2]

        # Modify one file
        files_modified = dict(files)
        files_modified["src/b.py"] = "def bar():\n    return 42\n"
        chunks_3 = await indexer.index_repository("org/repo", "sha3", files_modified)
        assert len(chunks_3) >= 2
        assert any(c.code.strip().endswith("return 42") for c in chunks_3)


# ---------------------------------------------------------------------------
# 3. Symbol Extraction & Search Tests
# ---------------------------------------------------------------------------


class TestSearchAndHybridRetrieval:
    """Tests for changed symbol extraction, BM25, vector search, and RRF."""

    def test_extract_changed_symbols_from_diff(self) -> None:
        diff = FileDiff(
            path="src/payments.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=10,
                    new_start=10,
                    lines=[
                        DiffLine(kind="ctx", old_no=10, new_no=10, text="import stripe"),
                        DiffLine(
                            kind="add", new_no=11, text="def process_payment(customer_id, amount):"
                        ),
                        DiffLine(
                            kind="add",
                            new_no=12,
                            text="    charge = stripe.Charge.create(amount=amount)",
                        ),
                        DiffLine(
                            kind="add", new_no=13, text="    return record_transaction(charge)"
                        ),
                    ],
                )
            ],
        )
        symbols = extract_changed_symbols([diff])
        assert "process_payment" in symbols
        assert "record_transaction" in symbols
        query = build_search_query([diff])
        assert "process_payment" in query

    def test_bm25_and_vector_search(self) -> None:
        embedder = EmbeddingModel()
        chunks = [
            CodeChunk(
                file_path="src/auth.py",
                symbol_name="verify_jwt_token",
                symbol_type="function",
                start_line=1,
                end_line=10,
                code="def verify_jwt_token(token: str):\n    return jwt.decode(token)",
                language="python",
                file_hash="h1",
            ),
            CodeChunk(
                file_path="src/db.py",
                symbol_name="get_database_pool",
                symbol_type="function",
                start_line=1,
                end_line=10,
                code="def get_database_pool():\n    return asyncpg.create_pool()",
                language="python",
                file_hash="h2",
            ),
        ]
        # Embed chunks
        for c in chunks:
            c.embedding = embedder.embed_query(f"{c.file_path} {c.symbol_name} {c.code}")

        # Search for JWT token verification
        query = "jwt token verify"
        bm25_res = search_bm25(query, chunks, top_k=2)
        assert len(bm25_res) > 0
        assert bm25_res[0][0].symbol_name == "verify_jwt_token"

        vec_res = search_vector(query, chunks, top_k=2, embedding_model=embedder)
        assert len(vec_res) > 0
        assert vec_res[0][0].symbol_name == "verify_jwt_token"

    def test_reciprocal_rank_fusion(self) -> None:
        c1 = CodeChunk("f1.py", "s1", "function", 1, 5, "code1", "python", "h1")
        c2 = CodeChunk("f2.py", "s2", "function", 1, 5, "code2", "python", "h2")

        v_res = [(c1, 0.9), (c2, 0.5)]
        b_res = [(c2, 10.0), (c1, 5.0)]

        rrf = reciprocal_rank_fusion(v_res, b_res, rrf_k=60, top_k=2)
        assert len(rrf) == 2
        # Both documents scored
        assert {r[0].symbol_name for r in rrf} == {"s1", "s2"}


# ---------------------------------------------------------------------------
# 4. Reranking & Context Formatting Tests
# ---------------------------------------------------------------------------


class TestReranker:
    """Tests for cross-encoder reranking and context prompt formatting."""

    def test_rerank_candidates_top_k(self) -> None:
        candidates = [
            CodeChunk(
                "src/utils.py",
                "irrelevant_helper",
                "function",
                1,
                5,
                "def irrelevant_helper(): pass",
                "python",
                "h1",
            ),
            CodeChunk(
                "src/auth.py",
                "login_handler",
                "function",
                10,
                25,
                "def login_handler(username, password):\n    return check_auth()",
                "python",
                "h2",
            ),
        ]
        query = "login credentials authentication"
        reranked = rerank_chunks(query, candidates, top_k=1)
        assert len(reranked) == 1
        assert reranked[0][0].symbol_name == "login_handler"

    def test_format_retrieved_context_markup(self) -> None:
        c = CodeChunk(
            "src/api.py",
            "get_status",
            "function",
            10,
            15,
            "def get_status(): return {'ok': True}",
            "python",
            "h1",
        )
        rendered = format_retrieved_context([(c, 0.92)])
        assert "## Retrieved Repository Context:" in rendered
        assert "src/api.py:10-15" in rendered
        assert "get_status" in rendered
        assert "[Relevance: 0.92]" in rendered


# ---------------------------------------------------------------------------
# 5. Self-Verification Pass Tests
# ---------------------------------------------------------------------------


class TestSelfVerification:
    """Tests for self-verification pass dropping false positives."""

    @pytest.mark.asyncio
    async def test_self_verification_drops_flagged_findings(self) -> None:
        diff = FileDiff(
            path="src/app.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=1,
                    new_start=1,
                    lines=[DiffLine(kind="add", new_no=1, text="def hello(): return 'world'")],
                )
            ],
        )
        findings = [
            ReviewFinding(
                path="src/app.py",
                line=1,
                category=IssueCategory.BUG,
                severity=Severity.HIGH,
                message="Real bug.",
                confidence=0.9,
            ),
            ReviewFinding(
                path="src/app.py",
                line=2,
                category=IssueCategory.READABILITY,
                severity=Severity.LOW,
                message="Nitpick about quote style.",
                confidence=0.6,
            ),
        ]

        # Mock LLM returns KEEP for #1 and DROP for #2
        mock_response = VerificationResponse(
            verifications=[
                FindingVerification(
                    path="src/app.py", line=1, decision="KEEP", reason="Actionable bug."
                ),
                FindingVerification(
                    path="src/app.py", line=2, decision="DROP", reason="Stylistic nitpick."
                ),
            ]
        )
        client = FakeLLMClient(responses=[mock_response])

        verified, usage = await verify_findings(
            file_diffs=[diff],
            candidate_findings=findings,
            llm_client=client,
            timeout=10.0,
        )
        assert len(verified) == 1
        assert verified[0].line == 1
        assert verified[0].category == IssueCategory.BUG


# ---------------------------------------------------------------------------
# 6. End-to-End Review Pipeline with Retrieval & Verification
# ---------------------------------------------------------------------------


class TestReviewPipelineRetrievalIntegration:
    """Integration test verifying review_pr executes with retrieval and self-verification enabled."""

    @pytest.mark.asyncio
    async def test_review_pr_with_retrieval_and_verification_flags(self) -> None:
        diff = FileDiff(
            path="src/app.py",
            language="python",
            hunks=[
                Hunk(
                    old_start=10,
                    new_start=10,
                    lines=[
                        DiffLine(kind="add", new_no=10, text="def handle(req):"),
                        DiffLine(
                            kind="add",
                            new_no=11,
                            text="    return execute_query(req.params.get('q'))",
                        ),
                    ],
                )
            ],
        )

        ctx = PRContext(
            repo_full_name="acme/api",
            pr_number=42,
            title="Update handler",
            file_diffs=[diff],
            repo_files={"src/db.py": "def execute_query(raw_sql):\n    return db.raw(raw_sql)\n"},
        )

        settings = Settings(
            github_app_id="12345",
            github_private_key="dummy_key",
            github_webhook_secret="dummy_secret",
            database_url="sqlite+aiosqlite:///test.db",
            redis_url="redis://localhost:6379/0",
            retrieval_enabled=True,
            retrieval_mode="hybrid",
            reranker_enabled=True,
            self_verification_enabled=True,
            prompt_variant="standard",
        )

        from evals.dataset.label import DeterministicMockAnnotator

        mock_client = DeterministicMockAnnotator(annotator_id="eval", model="gpt-4o-mini")

        result = await review_pr(
            pr_context=ctx,
            settings=settings,
            llm_client=mock_client,
        )

        assert isinstance(result.findings, list)
        assert len(result.findings) >= 1
        assert result.findings[0].category == IssueCategory.SECURITY
        assert result.cost_usd >= 0.0
