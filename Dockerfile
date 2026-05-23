# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps into a prefix we can copy later
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="satishkumarreddy595@gmail.com"
LABEL description="Real-Time LLM Inference Service"

# Needed for torch / transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user for security
RUN useradd --uid 1000 --create-home appuser
WORKDIR /app

# Copy source
COPY api/       ./api/
COPY inference/ ./inference/
COPY cache/     ./cache/
COPY monitoring/ ./monitoring/

RUN chown -R appuser:appuser /app
USER appuser

# HuggingFace cache dir
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

EXPOSE 8000

# Default: 4 Uvicorn workers
CMD ["uvicorn", "api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--timeout-keep-alive", "30"]
