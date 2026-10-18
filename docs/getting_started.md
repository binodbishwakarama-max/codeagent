# Getting Started with Prism Review

## Prerequisites

- [uv](https://github.com/astral-sh/uv) (v0.12+)
- Python 3.11+ (Python 3.12 recommended)
- Docker & Docker Compose (for PostgreSQL + pgvector and Redis)
- Make (or run underlying `uv` commands)

## Installation & Setup

1. **Clone the repository and install dependencies:**
   ```bash
   make install
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Fill in GitHub App credentials and provider keys as needed.

3. **Start local infrastructure:**
   ```bash
   docker compose up -d
   ```

## Development Commands

- Run linting checks:
  ```bash
  make lint
  ```
- Run strict type checking:
  ```bash
  make typecheck
  ```
- Run unit test suite:
  ```bash
  make test
  ```
- Run evaluation smoke suite:
  ```bash
  make eval-smoke
  ```
- Run full evaluation harness:
  ```bash
  make eval-full
  ```
- Start the API server:
  ```bash
  make run
  ```
- Start background worker:
  ```bash
  make worker
  ```
