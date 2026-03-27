# ── FinMentor AI — Production Dockerfile ─────────────────────────────────────
# Multi-stage build: keeps the final image lean (~200MB vs ~1.2GB)
#
# Build:  docker build -t finmentor-ai:latest .
# Run:    docker run -p 8080:8080 \
#           -e ANTHROPIC_API_KEY=sk-ant-... \
#           -e FINMENTOR_ENV=production \
#           finmentor-ai:latest

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix we'll copy into final image
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r finmentor && useradd -r -g finmentor finmentor

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=finmentor:finmentor . /app

# Create writable directories for logs and memory store
RUN mkdir -p /app/logs /app/memory_store /app/saved_models \
    && chown -R finmentor:finmentor /app/logs /app/memory_store /app/saved_models

USER finmentor

# ── Environment defaults (override at runtime) ────────────────────────────────
ENV FINMENTOR_ENV=production \
    FINMENTOR__SERVER__HOST=0.0.0.0 \
    FINMENTOR__SERVER__PORT=8080 \
    FINMENTOR__SERVER__WORKERS=4 \
    FINMENTOR__AGENT__MEMORY_BACKEND=json \
    FINMENTOR__RL_MODEL__BACKEND=mock \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

# Healthcheck — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Start uvicorn with production settings
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]
