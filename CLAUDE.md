# CLAUDE.md - Engineering Standards & Operational Guidelines

## Core Principles

1. **Think before coding**: State assumptions, surface tradeoffs, ask when unclear.
2. **Simplicity first**: Minimum code, no speculative features or abstractions.
3. **Surgical changes**: Touch only what's needed, match existing style.
4. **Goal-driven execution**: State verifiable success criteria and a short plan for every multi-step task.
5. **Every function that calls an LLM must be traced, have a timeout, and return typed output.**
6. **No secrets in code**: All config via environment variables validated at startup.
7. **Every PR-affecting change must keep `make test` and `make eval-smoke` green.**

---

## Architectural Conventions

### LLM Client Constraints
- Always use asynchronous clients with explicit timeout deadlines (e.g. `timeout=30.0`).
- Wrap every LLM call with observability spans (tracing / metadata recording).
- Strictly parse and validate LLM outputs using Pydantic models with typed error handling.

### Configuration & Secrets
- All system configurations must reside in `src/prism/config.py` using `pydantic-settings`.
- Do not commit secrets, tokens, or private keys. Load secrets strictly from environment variables or mounted secret files.
- Fail fast at application startup if critical credentials (e.g., `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY_PATH`, `DATABASE_URL`) are missing or malformed.

### Code Style & Quality Gates
- **Formatter & Linter**: `ruff format` and `ruff check`.
- **Type Checking**: `mypy --strict` on all production code in `src/`, `evals/`, and `tests/`.
- **Testing**: `pytest` covering unit logic, integration webhooks, and smoke evals.
- **Verification Workflow**:
  - `make lint`
  - `make typecheck`
  - `make test`
  - `make eval-smoke`
