"""AST-based source code chunking, embedding generation, and incremental repository indexing."""

import ast
import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prism.db import CodeChunkRecord

logger = logging.getLogger(__name__)

# Tree-sitter parsers initialization (lazy / defensive)
_PY_PARSER: Any = None
_TS_PARSER: Any = None


def _get_tree_sitter_parsers() -> tuple[Any, Any]:
    """Initialize and return tree-sitter parsers for Python and TypeScript."""
    global _PY_PARSER, _TS_PARSER
    if _PY_PARSER is not None:
        return _PY_PARSER, _TS_PARSER

    try:
        import tree_sitter_python
        import tree_sitter_typescript
        from tree_sitter import Language, Parser

        py_lang = Language(tree_sitter_python.language())
        _PY_PARSER = Parser(py_lang)

        ts_lang = Language(tree_sitter_typescript.language_typescript())
        _TS_PARSER = Parser(ts_lang)
    except Exception as e:
        logger.debug("Tree-sitter initialization note: %s (using AST fallbacks)", e)
        _PY_PARSER = None
        _TS_PARSER = None

    return _PY_PARSER, _TS_PARSER


@dataclass
class CodeChunk:
    """A semantic AST chunk of code extracted from a source file."""

    file_path: str
    symbol_name: str
    symbol_type: str  # function, class, method, interface, module
    start_line: int
    end_line: int
    code: str
    language: str
    file_hash: str
    commit_sha: str = ""
    repo: str = ""
    embedding: list[float] | None = None

    def to_context_text(self) -> str:
        """Render chunk for LLM prompt context."""
        header = f"### File: {self.file_path}:{self.start_line}-{self.end_line} [{self.symbol_type}: {self.symbol_name}]"
        return f"{header}\n```{self.language}\n{self.code.strip()}\n```"


def compute_file_hash(content: str) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# AST Chunking: Python
# ---------------------------------------------------------------------------


def _chunk_python_tree_sitter(
    file_path: str,
    content: str,
    file_hash: str,
    parser: Any,
) -> list[CodeChunk]:
    """Chunk Python file using Tree-Sitter AST."""
    chunks: list[CodeChunk] = []
    lines = content.splitlines(keepends=True)
    tree = parser.parse(content.encode("utf-8"))
    root = tree.root_node

    def walk(node: Any, parent_class: str | None = None) -> None:
        if node.type in ("function_definition", "async_function_definition"):
            name = ""
            for child in node.children:
                if child.type == "identifier":
                    name = child.text.decode("utf-8", errors="replace")
                    break
            symbol_type = "method" if parent_class else "function"
            full_name = f"{parent_class}.{name}" if parent_class else name
            s_line = node.start_point[0] + 1
            e_line = node.end_point[0] + 1
            code_text = "".join(lines[node.start_point[0] : node.end_point[0] + 1])
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    symbol_name=full_name or "anonymous_function",
                    symbol_type=symbol_type,
                    start_line=s_line,
                    end_line=e_line,
                    code=code_text,
                    language="python",
                    file_hash=file_hash,
                )
            )
            return

        if node.type == "class_definition":
            class_name = ""
            for child in node.children:
                if child.type == "identifier":
                    class_name = child.text.decode("utf-8", errors="replace")
                    break
            s_line = node.start_point[0] + 1
            e_line = node.end_point[0] + 1
            code_text = "".join(lines[node.start_point[0] : node.end_point[0] + 1])
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    symbol_name=class_name or "AnonymousClass",
                    symbol_type="class",
                    start_line=s_line,
                    end_line=e_line,
                    code=code_text,
                    language="python",
                    file_hash=file_hash,
                )
            )
            # Walk children to extract individual methods
            for child in node.children:
                walk(child, parent_class=class_name)
            return

        for child in node.children:
            walk(child, parent_class)

    walk(root)
    return chunks


def _chunk_python_stdlib_ast(
    file_path: str,
    content: str,
    file_hash: str,
) -> list[CodeChunk]:
    """Fallback AST chunking for Python using stdlib `ast` module."""
    chunks: list[CodeChunk] = []
    lines = content.splitlines(keepends=True)

    try:
        parsed = ast.parse(content, filename=file_path)
    except SyntaxError:
        # Non-parseable Python; return whole file as module chunk
        return [
            CodeChunk(
                file_path=file_path,
                symbol_name="module",
                symbol_type="module",
                start_line=1,
                end_line=max(1, len(lines)),
                code=content,
                language="python",
                file_hash=file_hash,
            )
        ]

    for node in ast.iter_child_nodes(parsed):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s_line = node.lineno
            e_line = getattr(node, "end_lineno", node.lineno)
            code_text = "".join(lines[s_line - 1 : e_line])
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    symbol_name=node.name,
                    symbol_type="function",
                    start_line=s_line,
                    end_line=e_line,
                    code=code_text,
                    language="python",
                    file_hash=file_hash,
                )
            )
        elif isinstance(node, ast.ClassDef):
            s_line = node.lineno
            e_line = getattr(node, "end_lineno", node.lineno)
            code_text = "".join(lines[s_line - 1 : e_line])
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    symbol_name=node.name,
                    symbol_type="class",
                    start_line=s_line,
                    end_line=e_line,
                    code=code_text,
                    language="python",
                    file_hash=file_hash,
                )
            )
            # Extract methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_s = item.lineno
                    m_e = getattr(item, "end_lineno", item.lineno)
                    m_code = "".join(lines[m_s - 1 : m_e])
                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            symbol_name=f"{node.name}.{item.name}",
                            symbol_type="method",
                            start_line=m_s,
                            end_line=m_e,
                            code=m_code,
                            language="python",
                            file_hash=file_hash,
                        )
                    )

    return chunks


# ---------------------------------------------------------------------------
# AST Chunking: TypeScript / JavaScript
# ---------------------------------------------------------------------------


def _chunk_typescript_tree_sitter(
    file_path: str,
    content: str,
    file_hash: str,
    parser: Any,
) -> list[CodeChunk]:
    """Chunk TypeScript/JavaScript file using Tree-Sitter AST."""
    chunks: list[CodeChunk] = []
    lines = content.splitlines(keepends=True)
    tree = parser.parse(content.encode("utf-8"))
    root = tree.root_node

    target_types = {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
    }

    def walk(node: Any, parent_class: str | None = None) -> None:
        if node.type in target_types:
            sym_type = target_types[node.type]
            name = ""
            for child in node.children:
                if child.type in ("identifier", "type_identifier", "property_identifier"):
                    name = child.text.decode("utf-8", errors="replace")
                    break

            if sym_type == "method" and parent_class:
                full_name = f"{parent_class}.{name}"
            else:
                full_name = name or f"anonymous_{sym_type}"

            s_line = node.start_point[0] + 1
            e_line = node.end_point[0] + 1
            code_text = "".join(lines[node.start_point[0] : node.end_point[0] + 1])
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    symbol_name=full_name,
                    symbol_type=sym_type,
                    start_line=s_line,
                    end_line=e_line,
                    code=code_text,
                    language="typescript",
                    file_hash=file_hash,
                )
            )

            if sym_type == "class":
                for child in node.children:
                    walk(child, parent_class=name)
                return

        for child in node.children:
            walk(child, parent_class)

    walk(root)
    return chunks


def _chunk_regex_fallback(
    file_path: str,
    content: str,
    file_hash: str,
    language: str,
) -> list[CodeChunk]:
    """Lightweight regex chunker for languages without an active tree-sitter parser."""
    chunks: list[CodeChunk] = []
    lines = content.splitlines(keepends=True)
    pattern = re.compile(
        r"^(?:export\s+)?(?:async\s+)?(?:function|class|interface|type)\s+([a-zA-Z0-9_$]+)",
        re.MULTILINE,
    )

    matches = list(pattern.finditer(content))
    if not matches:
        return [
            CodeChunk(
                file_path=file_path,
                symbol_name="module",
                symbol_type="module",
                start_line=1,
                end_line=max(1, len(lines)),
                code=content,
                language=language,
                file_hash=file_hash,
            )
        ]

    for i, match in enumerate(matches):
        symbol_name = match.group(1)
        start_char = match.start()
        end_char = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        s_line = content[:start_char].count("\n") + 1
        e_line = content[:end_char].count("\n") + 1
        chunk_code = content[start_char:end_char]

        chunks.append(
            CodeChunk(
                file_path=file_path,
                symbol_name=symbol_name,
                symbol_type="function",
                start_line=s_line,
                end_line=e_line,
                code=chunk_code,
                language=language,
                file_hash=file_hash,
            )
        )

    return chunks


def chunk_source_file(
    file_path: str,
    content: str,
    language: str | None = None,
) -> list[CodeChunk]:
    """Chunk a source code file into discrete AST symbols.

    Extracts functions, classes, and methods with file path and symbol metadata.
    """
    if not content.strip():
        return []

    file_hash = compute_file_hash(content)
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    lang = language or (
        "python"
        if ext in ("py", "pyi")
        else "typescript"
        if ext in ("ts", "tsx", "js", "jsx")
        else "text"
    )

    py_parser, ts_parser = _get_tree_sitter_parsers()

    chunks: list[CodeChunk] = []
    if lang == "python":
        if py_parser is not None:
            try:
                chunks = _chunk_python_tree_sitter(file_path, content, file_hash, py_parser)
            except Exception as e:
                logger.debug("Tree-sitter python parsing failed (%s); using stdlib AST", e)
        if not chunks:
            chunks = _chunk_python_stdlib_ast(file_path, content, file_hash)
    elif lang in ("typescript", "javascript"):
        if ts_parser is not None:
            try:
                chunks = _chunk_typescript_tree_sitter(file_path, content, file_hash, ts_parser)
            except Exception as e:
                logger.debug("Tree-sitter TS parsing failed (%s); using regex fallback", e)
        if not chunks:
            chunks = _chunk_regex_fallback(file_path, content, file_hash, lang)
    else:
        chunks = _chunk_regex_fallback(file_path, content, file_hash, lang)

    # If no structural chunks detected, return whole file as module chunk
    if not chunks:
        chunks = [
            CodeChunk(
                file_path=file_path,
                symbol_name="module",
                symbol_type="module",
                start_line=1,
                end_line=max(1, content.count("\n") + 1),
                code=content,
                language=lang,
                file_hash=file_hash,
            )
        ]

    return chunks


# ---------------------------------------------------------------------------
# Embedding Generation (FastEmbed / BGE-small / Fallback)
# ---------------------------------------------------------------------------


class EmbeddingModel:
    """Embedding model supporting fastembed BGE-small with deterministic fallback."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384) -> None:
        self.model_name = model_name
        self.dim = dim
        self._embedder: Any = None
        self._initialized = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        import os

        if os.environ.get("PRISM_LOCAL_FAST_EMBEDDINGS") == "1":
            self._embedder = None
            return

        try:
            from fastembed import TextEmbedding

            # Lazy load model
            self._embedder = TextEmbedding(model_name=self.model_name)
            logger.info("Loaded fastembed model '%s'", self.model_name)
        except Exception as e:
            logger.debug(
                "Fastembed unavailable or failed to load (%s); using deterministic projection", e
            )
            self._embedder = None

    def _deterministic_embedding(self, text: str) -> list[float]:
        """Fast, deterministic normalized vector generator for tests/offline execution."""
        tokens = re.findall(r"\w+", text.lower())
        vec = [0.0] * self.dim
        if not tokens:
            return vec

        for tok in tokens:
            # Deterministic hash to dimension
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            weight = 1.0 + (h % 5) * 0.2
            vec[idx] += weight

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute normalized vector embeddings for a list of texts."""
        if not texts:
            return []
        self._ensure_init()
        if self._embedder is not None:
            try:
                embeddings = list(self._embedder.embed(texts))
                return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]
            except Exception as e:
                logger.warning(
                    "FastEmbed embedding call failed (%s); falling back to deterministic", e
                )

        return [self._deterministic_embedding(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        """Compute normalized vector embedding for a single query."""
        res = self.embed_texts([query])
        return res[0] if res else [0.0] * self.dim


# Global default embedding model singleton
_GLOBAL_EMBEDDER: EmbeddingModel | None = None


def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> EmbeddingModel:
    """Return singleton embedding model instance."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None or _GLOBAL_EMBEDDER.model_name != model_name:
        _GLOBAL_EMBEDDER = EmbeddingModel(model_name=model_name)
    return _GLOBAL_EMBEDDER


# ---------------------------------------------------------------------------
# Incremental Repository Indexer
# ---------------------------------------------------------------------------


class RepositoryIndexer:
    """Manages incremental repository indexing with file hash caching and pgvector storage."""

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.embedding_model = embedding_model or get_embedding_model()
        # In-memory index cache: repo -> file_path -> (file_hash, list[CodeChunk])
        self._mem_cache: dict[str, dict[str, tuple[str, list[CodeChunk]]]] = {}

    def get_indexed_chunks(self, repo: str) -> list[CodeChunk]:
        """Return all in-memory indexed chunks for a repository."""
        repo_files = self._mem_cache.get(repo, {})
        all_chunks: list[CodeChunk] = []
        for _fhash, chunks in repo_files.values():
            all_chunks.extend(chunks)
        return all_chunks

    async def index_repository(
        self,
        repo: str,
        commit_sha: str,
        files: dict[str, str],
        session: AsyncSession | None = None,
    ) -> list[CodeChunk]:
        """Index or incrementally re-index a repository at a commit SHA.

        Checks file hashes: unchanged files skip re-chunking and re-embedding.
        Persists to SQLAlchemy session (if provided) and local cache.
        """
        if repo not in self._mem_cache:
            self._mem_cache[repo] = {}

        repo_cache = self._mem_cache[repo]
        all_chunks: list[CodeChunk] = []
        files_to_index: list[tuple[str, str, str]] = []  # path, content, hash

        for file_path, content in files.items():
            fhash = compute_file_hash(content)
            # Check incremental memory cache
            if file_path in repo_cache and repo_cache[file_path][0] == fhash:
                # Cache hit: unchanged file hash
                all_chunks.extend(repo_cache[file_path][1])
                continue

            # Check database for existing file_hash if session provided
            if session is not None:
                stmt = select(CodeChunkRecord).where(
                    CodeChunkRecord.repo == repo,
                    CodeChunkRecord.file_path == file_path,
                    CodeChunkRecord.file_hash == fhash,
                )
                db_result = await session.execute(stmt)
                records = db_result.scalars().all()
                if records:
                    db_chunks: list[CodeChunk] = []
                    for r in records:
                        emb = r.embedding
                        if isinstance(emb, str):
                            import json

                            try:
                                emb = json.loads(emb)
                            except Exception:
                                emb = None
                        db_chunks.append(
                            CodeChunk(
                                file_path=r.file_path,
                                symbol_name=r.symbol_name,
                                symbol_type=r.symbol_type,
                                start_line=r.start_line,
                                end_line=r.end_line,
                                code=r.content,
                                language=r.language,
                                file_hash=r.file_hash,
                                commit_sha=r.commit_sha,
                                repo=r.repo,
                                embedding=emb,
                            )
                        )
                    repo_cache[file_path] = (fhash, db_chunks)
                    all_chunks.extend(db_chunks)
                    continue

            files_to_index.append((file_path, content, fhash))

        # Chunk and embed files that need indexing
        for file_path, content, fhash in files_to_index:
            chunks = chunk_source_file(file_path=file_path, content=content)
            for c in chunks:
                c.repo = repo
                c.commit_sha = commit_sha

            # Compute embeddings
            texts_to_embed = [f"{c.file_path} {c.symbol_name}\n{c.code}" for c in chunks]
            embeddings = self.embedding_model.embed_texts(texts_to_embed)
            for c, emb in zip(chunks, embeddings, strict=False):
                c.embedding = emb

            # Store in DB if session provided
            if session is not None:
                for c in chunks:
                    rec = CodeChunkRecord(
                        repo=repo,
                        commit_sha=commit_sha,
                        file_path=c.file_path,
                        file_hash=c.file_hash,
                        symbol_name=c.symbol_name,
                        symbol_type=c.symbol_type,
                        start_line=c.start_line,
                        end_line=c.end_line,
                        language=c.language,
                        content=c.code,
                        embedding=c.embedding,
                    )
                    session.add(rec)
                await session.flush()

            repo_cache[file_path] = (fhash, chunks)
            all_chunks.extend(chunks)

        logger.info(
            "Indexed repository %s@%s: %d total chunks (%d newly embedded files)",
            repo,
            commit_sha[:8] if commit_sha else "head",
            len(all_chunks),
            len(files_to_index),
        )
        return all_chunks
