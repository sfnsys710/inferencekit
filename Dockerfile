# Build stage: it is important to use same python version as in pyproject.toml, as we deactivate python download
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# UV_PYTHON_DOWNLOADS to use system python, not download a new one
# UV_COMPILE_BYTECODE to compile bytecode, faster startup
ENV UV_PYTHON_DOWNLOADS=0 UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install dependencies first (no project installation)
COPY uv.lock ./
COPY pyproject.toml ./
COPY README.md ./
RUN uv sync --locked --no-install-project --no-dev --group api

# Install project now
COPY src ./src
COPY scripts ./scripts
COPY api ./api
RUN uv sync --locked --no-dev --group api

# Runtime stage: it is important to use same python version as in pyproject.toml and the builder
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy virtual environment and application
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/api ./api

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Expose port for Cloud Run (Cloud Run sets PORT env var automatically)
EXPOSE 8080

# Run FastAPI server with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
