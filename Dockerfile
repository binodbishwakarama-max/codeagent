# ==============================================================================
# Stage 1: Build virtual environment with uv
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official pre-built image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project manifest and lockfile
COPY pyproject.toml uv.lock ./

# Install production dependencies only into /app/.venv
RUN uv sync --frozen --no-install-project --no-dev


# ==============================================================================
# Stage 2: Minimal, non-root runtime container
# ==============================================================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Install minimal runtime packages (curl for healthchecks, ca-certificates for TLS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged prism user and group (UID 10001, GID 10001)
RUN groupadd -g 10001 prism && \
    useradd -u 10001 -g prism -m -s /bin/bash prism

# Copy virtual environment from builder
COPY --from=builder --chown=prism:prism /app/.venv /app/.venv

# Configure execution environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src:$PYTHONPATH"

# Copy application sources, migration manifests, and entrypoint
COPY --chown=prism:prism src/ /app/src/
COPY --chown=prism:prism alembic.ini /app/
COPY --chown=prism:prism pyproject.toml /app/
COPY --chown=prism:prism docker-entrypoint.sh /app/

RUN chmod +x /app/docker-entrypoint.sh

# Liveness healthcheck (verifies API responds on /healthz)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run as non-root user
USER prism

EXPOSE 8000 7860

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["web"]
