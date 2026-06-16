FROM python:3.12-slim

WORKDIR /app

# Create non-root user early so we can own /app
RUN useradd --create-home appuser && chown appuser:appuser /app

# Install curl (for healthcheck) and uv (fast Python package manager)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser uv.lock* .
COPY --chown=appuser:appuser requirements-ml.txt .

# Switch to non-root user before installing deps
USER appuser

# Install core dependencies AND the ML dependencies
RUN uv sync --frozen --no-dev
RUN uv pip install -r requirements-ml.txt

# Copy application code
COPY --chown=appuser:appuser app/ app/
