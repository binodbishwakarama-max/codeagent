"""Interactive and automated repository standards initialization (prism init).

Scans repository files and manifests to auto-detect languages (Python, TypeScript/JavaScript,
Go, Rust) and frameworks (FastAPI, Django, Express, Next.js, Gin, Axum, Tokio), then generates
a tailored, production-ready `.prism/standards.yml` configuration and optional GitHub Action workflow.
"""

import json
from pathlib import Path
from typing import Any

import yaml

from prism.models import StandardsFile

# ---------------------------------------------------------------------------
# Pre-defined standards catalog by language and framework
# ---------------------------------------------------------------------------

UNIVERSAL_STANDARDS: list[dict[str, Any]] = [
    {
        "id": "STD-SEC-01",
        "title": "Zero Plaintext Credentials or API Secrets",
        "description": (
            "Never commit raw passwords, private keys, JWT secrets, or cloud API tokens into source code. "
            "Retrieve sensitive credentials exclusively from validated environment variables or external secret stores."
        ),
        "good_example": (
            "api_key = os.environ.get('STRIPE_API_KEY')\n"
            "if not api_key:\n"
            "    raise ConfigError('Missing STRIPE_API_KEY')"
        ),
        "bad_example": "client = Stripe(api_key='sk_live_9384729183748293')",
    },
]

PYTHON_STANDARDS: list[dict[str, Any]] = [
    {
        "id": "STD-PY-SQL-01",
        "title": "SQL Injection Prevention via Parameterized Queries",
        "description": (
            "Always use parameterized queries, prepared statements, or ORM parameter binding when assembling SQL. "
            "Never format, interpolate (f-strings), or concatenate untrusted variables into SQL statements."
        ),
        "good_example": "cursor.execute('SELECT id, email FROM users WHERE tenant_id = %s', (tenant_id,))",
        "bad_example": "cursor.execute(f\"SELECT id, email FROM users WHERE tenant_id = '{tenant_id}'\")",
    },
    {
        "id": "STD-PY-ERR-01",
        "title": "Explicit Exception Handling (No Bare Excepts)",
        "description": (
            "Never use bare 'except:' clauses or silent 'except Exception: pass'. Always catch specific exception types, "
            "log actionable context, and preserve tracebacks with 'raise ... from exc'."
        ),
        "good_example": (
            "try:\n"
            "    data = json.loads(payload)\n"
            "except json.JSONDecodeError as exc:\n"
            "    logger.warning('Malformed payload: %s', exc)\n"
            "    raise BadRequest('Invalid JSON') from exc"
        ),
        "bad_example": "try:\n    data = json.loads(payload)\nexcept:\n    pass",
    },
    {
        "id": "STD-PY-RES-01",
        "title": "Deterministic Context Manager Resource Cleanup",
        "description": (
            "Always use context managers ('with' or 'async with') for files, connections, sockets, and sessions "
            "to prevent leaked descriptors on error."
        ),
        "good_example": "with open(path, 'w', encoding='utf-8') as f:\n    f.write(data)",
        "bad_example": "f = open(path, 'w')\nf.write(data)\nf.close()",
    },
]

FASTAPI_STANDARDS: list[dict[str, Any]] = [
    {
        "id": "STD-FASTAPI-ASYNC-01",
        "title": "No Blocking Synchronous Calls in Async Route Handlers",
        "description": (
            "Do not call blocking synchronous I/O operations (e.g. requests.get, time.sleep, blocking DB calls) "
            "inside async def route handlers. Use non-blocking async equivalents (httpx.AsyncClient, asyncio.sleep) "
            "or offload to threadpool via run_in_threadpool."
        ),
        "good_example": (
            "async def get_user_status(client: httpx.AsyncClient) -> dict:\n"
            "    resp = await client.get('https://service/status', timeout=5.0)\n"
            "    return resp.json()"
        ),
        "bad_example": (
            "async def get_user_status() -> dict:\n"
            "    resp = requests.get('https://service/status')\n"
            "    return resp.json()"
        ),
    },
    {
        "id": "STD-FASTAPI-VAL-01",
        "title": "Strict Schema Validation at Service Boundaries",
        "description": (
            "Validate all incoming request bodies and query parameters with strict Pydantic models. "
            "Avoid untyped dict payloads or raw request parameter access."
        ),
        "good_example": (
            "class UpdateUserRequest(BaseModel):\n"
            "    email: EmailStr\n"
            "    age: int = Field(ge=18, le=120)"
        ),
        "bad_example": "async def update_user(payload: dict):\n    email = payload['email']",
    },
]

TYPESCRIPT_STANDARDS: list[dict[str, Any]] = [
    {
        "id": "STD-TS-NULL-01",
        "title": "Explicit Null & Undefined Guarding",
        "description": (
            "Always guard optional and nullable values before property dereferencing. "
            "Use optional chaining (?.) and nullish coalescing (??) instead of unsafe non-null assertions (!)."
        ),
        "good_example": "const displayName = user?.profile?.name ?? 'Guest';",
        "bad_example": "const displayName = user!.profile!.name;",
    },
    {
        "id": "STD-TS-ERR-01",
        "title": "Unhandled Promise Rejection Prevention",
        "description": (
            "All asynchronous operations and Promises must include explicit error handling (try/catch or .catch()). "
            "Never leave unhandled floating Promises."
        ),
        "good_example": (
            "try {\n"
            "  const res = await fetchClient.get(endpoint);\n"
            "  return res.data;\n"
            "} catch (err) {\n"
            "  logger.error('API request failed', { endpoint, err });\n"
            "  throw new ServiceError('Upstream failure', { cause: err });\n"
            "}"
        ),
        "bad_example": "fetchClient.get(endpoint).then(res => res.data); // Floating promise without catch",
    },
    {
        "id": "STD-TS-SQL-01",
        "title": "Parameterized Database & ORM Queries",
        "description": (
            "Never concatenate raw user input strings into database query templates. "
            "Use parameterized queries or ORM query builders (Prisma, TypeORM, Kysely)."
        ),
        "good_example": "await db.query('SELECT * FROM accounts WHERE org_id = $1', [orgId]);",
        "bad_example": "await db.query(`SELECT * FROM accounts WHERE org_id = '${orgId}'`);",
    },
]

GO_STANDARDS: list[dict[str, Any]] = [
    {
        "id": "STD-GO-ERR-01",
        "title": "Explicit Error Checking (No Blank Identifier Error Discarding)",
        "description": (
            "Never discard returned errors using blank identifiers ('_ = fn()'). "
            "Always inspect err != nil and handle or wrap with fmt.Errorf('...: %w', err)."
        ),
        "good_example": (
            "file, err := os.Open(path)\n"
            "if err != nil {\n"
            "    return fmt.Errorf('opening config %s: %w', path, err)\n"
            "}\n"
            "defer file.Close()"
        ),
        "bad_example": "file, _ := os.Open(path)\ndefer file.Close()",
    },
    {
        "id": "STD-GO-CTX-01",
        "title": "Context Propagation & Cancellation on I/O Operations",
        "description": (
            "All outbound network calls, database queries, and long-running goroutines must accept and respect "
            "context.Context for timely timeout cancellation."
        ),
        "good_example": (
            "req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)\n"
            "if err != nil {\n"
            "    return err\n"
            "}"
        ),
        "bad_example": "req, err := http.NewRequest(http.MethodGet, url, nil) // Lacks cancellation context",
    },
    {
        "id": "STD-GO-SQL-01",
        "title": "SQL Injection Prevention via Query Placeholders",
        "description": (
            "Always use database/sql placeholders ($1, ?, :) when executing SQL queries. "
            "Never construct queries with fmt.Sprintf."
        ),
        "good_example": "row := db.QueryRowContext(ctx, 'SELECT name FROM users WHERE id = $1', userID)",
        "bad_example": "row := db.QueryRowContext(ctx, fmt.Sprintf('SELECT name FROM users WHERE id = %s', userID))",
    },
]

RUST_STANDARDS: list[dict[str, Any]] = [
    {
        "id": "STD-RS-ERR-01",
        "title": "Defensive Result & Option Unwrapping (No Naked .unwrap())",
        "description": (
            "Avoid calling .unwrap() in production code paths where failures can occur at runtime. "
            "Use pattern matching, the '?' operator, .expect('actionable failure context'), or .unwrap_or_default()."
        ),
        "good_example": (
            "let file = File::open(&path)\n"
            "    .map_err(|e| AppError::ConfigNotFound(path.clone(), e))?;"
        ),
        "bad_example": "let file = File::open(&path).unwrap(); // Panic in production",
    },
    {
        "id": "STD-RS-ASYNC-01",
        "title": "Explicit Timeout Boundaries on Async Tasks",
        "description": (
            "All network requests and remote RPCs in Tokio/async tasks must declare explicit timeout wrappers "
            "to prevent unbounded hangs."
        ),
        "good_example": (
            "let res = tokio::time::timeout(Duration::from_secs(5), client.get(url).send()).await??;"
        ),
        "bad_example": "let res = client.get(url).send().await?; // Unbounded timeout",
    },
]


def detect_tech_stack(root_dir: Path) -> dict[str, Any]:
    """Inspect repository files and dependency manifests to identify languages and frameworks."""
    languages: list[str] = []
    frameworks: list[str] = []

    # 1. Python Detection
    py_indicators = [
        "pyproject.toml",
        "requirements.txt",
        "Pipfile",
        "setup.py",
        "setup.cfg",
        "uv.lock",
        "poetry.lock",
    ]
    has_python = any((root_dir / f).exists() for f in py_indicators) or any(root_dir.glob("*.py"))
    if has_python:
        languages.append("Python")

        # Inspect manifests for Python frameworks
        pyproject_path = root_dir / "pyproject.toml"
        req_path = root_dir / "requirements.txt"
        manifest_text = ""
        if pyproject_path.exists():
            manifest_text += pyproject_path.read_text(encoding="utf-8", errors="replace").lower()
        if req_path.exists():
            manifest_text += req_path.read_text(encoding="utf-8", errors="replace").lower()

        if "fastapi" in manifest_text:
            frameworks.append("FastAPI")
        if "django" in manifest_text:
            frameworks.append("Django")
        if "flask" in manifest_text:
            frameworks.append("Flask")
        if "sqlalchemy" in manifest_text or "asyncpg" in manifest_text:
            frameworks.append("SQLAlchemy")
        if "pydantic" in manifest_text:
            frameworks.append("Pydantic")

    # 2. TypeScript / JavaScript Detection
    ts_indicators = ["tsconfig.json", "package.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb"]
    has_ts = any((root_dir / f).exists() for f in ts_indicators) or any(root_dir.glob("*.ts"))
    if has_ts:
        languages.append("TypeScript")

        pkg_path = root_dir / "package.json"
        if pkg_path.exists():
            try:
                pkg_data = json.loads(pkg_path.read_text(encoding="utf-8", errors="replace"))
                deps = {
                    **pkg_data.get("dependencies", {}),
                    **pkg_data.get("devDependencies", {}),
                }
                dep_keys = {k.lower() for k in deps.keys()}
                if "next" in dep_keys:
                    frameworks.append("Next.js")
                if "express" in dep_keys:
                    frameworks.append("Express")
                if "react" in dep_keys:
                    frameworks.append("React")
                if "nest" in dep_keys or "@nestjs/core" in dep_keys:
                    frameworks.append("NestJS")
                if "prisma" in dep_keys or "@prisma/client" in dep_keys:
                    frameworks.append("Prisma")
            except Exception:
                pass

    # 3. Go Detection
    go_indicators = ["go.mod", "go.sum"]
    has_go = any((root_dir / f).exists() for f in go_indicators) or any(root_dir.glob("*.go"))
    if has_go:
        languages.append("Go")
        go_mod_path = root_dir / "go.mod"
        if go_mod_path.exists():
            text = go_mod_path.read_text(encoding="utf-8", errors="replace").lower()
            if "github.com/gin-gonic/gin" in text:
                frameworks.append("Gin")
            if "github.com/gofiber/fiber" in text:
                frameworks.append("Fiber")
            if "gorm.io/gorm" in text:
                frameworks.append("GORM")

    # 4. Rust Detection
    rust_indicators = ["Cargo.toml", "Cargo.lock"]
    has_rust = any((root_dir / f).exists() for f in rust_indicators) or any(root_dir.glob("*.rs"))
    if has_rust:
        languages.append("Rust")
        cargo_path = root_dir / "Cargo.toml"
        if cargo_path.exists():
            text = cargo_path.read_text(encoding="utf-8", errors="replace").lower()
            if "actix" in text:
                frameworks.append("Actix-web")
            if "axum" in text:
                frameworks.append("Axum")
            if "tokio" in text:
                frameworks.append("Tokio")

    return {
        "languages": languages or ["Generic"],
        "frameworks": frameworks,
    }


def assemble_standards(languages: list[str], frameworks: list[str]) -> list[dict[str, Any]]:
    """Assemble a curated list of standards tailored to the detected tech stack."""
    standards: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add_rules(rules: list[dict[str, Any]]) -> None:
        for r in rules:
            if r["id"] not in seen_ids:
                standards.append(r)
                seen_ids.add(r["id"])

    # Always include baseline security standards
    add_rules(UNIVERSAL_STANDARDS)

    # Add language-specific rules
    if "Python" in languages:
        add_rules(PYTHON_STANDARDS)
        if "FastAPI" in frameworks:
            add_rules(FASTAPI_STANDARDS)

    if "TypeScript" in languages:
        add_rules(TYPESCRIPT_STANDARDS)

    if "Go" in languages:
        add_rules(GO_STANDARDS)

    if "Rust" in languages:
        add_rules(RUST_STANDARDS)

    return standards


def generate_standards_yaml_content(standards: list[dict[str, Any]], stack_summary: str) -> str:
    """Format standards list into human-readable YAML with metadata comments."""
    header = f"""# ==============================================================================
# Prism Engineering Standards Configuration
# Auto-generated by: prism init
# Detected stack: {stack_summary}
# Learn more: https://github.com/binodbishwakarama-max/codeagent
# ==============================================================================

version: "1.0"
standards:
"""
    yaml_body = yaml.dump(
        standards,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    # Indent standards under the standards key
    indented = "\n".join(f"  {line}" if line.strip() else "" for line in yaml_body.splitlines())
    return header + indented + "\n"


WORKFLOW_TEMPLATE = """# ==============================================================================
# Prism Automated Pull Request Review Workflow
# Automatically generated by: prism init --workflow
# Reference: https://github.com/binodbishwakarama-max/codeagent
# ==============================================================================

name: Prism Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write
  checks: write

jobs:
  review:
    name: Standards Review
    runs-on: ubuntu-latest
    steps:
      - name: Checkout PR Branch
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Run Prism Code Review
        uses: binodbishwakarama-max/codeagent@v0.1.0
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          standards-file: ".prism/standards.yml"
          fail-on-severity: "high"
"""


def scaffold_prism(
    target_dir: Path,
    force: bool = False,
    include_workflow: bool = False,
    stack_override: str | None = None,
) -> tuple[Path, Path | None, list[str], list[str], int]:
    """Scaffold .prism/standards.yml and optional GitHub workflow in the target directory."""
    prism_dir = target_dir / ".prism"
    standards_path = prism_dir / "standards.yml"

    if standards_path.exists() and not force:
        raise FileExistsError(
            f"Standards file already exists at '{standards_path}'. Use '--force' to overwrite."
        )

    # 1. Detect or override tech stack
    if stack_override:
        languages = [stack_override.capitalize()]
        frameworks: list[str] = []
    else:
        detected = detect_tech_stack(target_dir)
        languages = detected["languages"]
        frameworks = detected["frameworks"]

    # 2. Assemble tailored standards
    standards_list = assemble_standards(languages, frameworks)

    # 3. Create .prism directory and write standards.yml
    prism_dir.mkdir(parents=True, exist_ok=True)
    summary_label = f"{', '.join(languages)}" + (f" ({', '.join(frameworks)})" if frameworks else "")
    yaml_content = generate_standards_yaml_content(standards_list, summary_label)
    standards_path.write_text(yaml_content, encoding="utf-8")

    # Validate that generated YAML is parseable by Prism
    StandardsFile.from_file(standards_path)

    # 4. Optional workflow generation
    workflow_path: Path | None = None
    if include_workflow:
        workflows_dir = target_dir / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        workflow_path = workflows_dir / "prism.yml"
        workflow_path.write_text(WORKFLOW_TEMPLATE, encoding="utf-8")

    return standards_path, workflow_path, languages, frameworks, len(standards_list)
