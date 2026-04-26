# =============================================================================
# Serving API — Production Dockerfile
# Multi-stage build: minimal final image (~800MB with torch CPU)
# =============================================================================

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
COPY requirements/serving.txt requirements/serving.txt
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements/serving.txt

# Pre-download the embedding model weights (baked into image for fast cold start)
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN /opt/venv/bin/python -c "\
    from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('${EMBEDDING_MODEL}', cache_folder='/opt/model_cache')"

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim as runtime

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r bmr && useradd -r -g bmr -d /app bmr

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/model_cache /opt/model_cache

# Copy application source
COPY embedding/ ./embedding/
COPY serving/ ./serving/
COPY mlops/ ./mlops/

# Ownership
RUN chown -R bmr:bmr /app
USER bmr

# Activate virtualenv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/opt/model_cache
ENV SENTENCE_TRANSFORMERS_HOME=/opt/model_cache

# Default environment
ENV SERVING_HOST=0.0.0.0
ENV SERVING_PORT=8000
ENV SERVING_WORKERS=2
ENV SERVING_LOG_LEVEL=info
ENV ENVIRONMENT=production

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start with gunicorn + uvicorn workers
CMD ["gunicorn", "serving.api.main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
