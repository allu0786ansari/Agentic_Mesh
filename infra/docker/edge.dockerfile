# ── Stage 1: Builder ─────────────────────────────────────────────
# Compiles all C extensions. build-essential stays here — never in final image.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         libpq-dev     && rm -rf /var/lib/apt/lists/*

# Copy uv from its official image — zero install overhead
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# ── Install CPU-only torch first (separate layer for cache efficiency) ──
# CPU wheel is ~180MB vs ~2.5GB for CUDA wheel
# This single change saves ~5-6GB in the final image
RUN uv pip install --system --no-cache     torch==2.6.0+cpu     torchvision==0.21.0+cpu     --index-url https://download.pytorch.org/whl/cpu

# ── Install remaining edge dependencies ──────────────────────────
COPY requirements.edge.txt .
RUN uv pip install --system --no-cache -r requirements.edge.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────
# Clean slate — only Python packages copied from builder, no compilers
FROM python:3.12-slim AS runtime

LABEL maintainer="Allaudin Ansari"
LABEL description="Agentic Mesh — Edge Node (CPU-only, multi-stage)"
LABEL version="0.2.0"

# Runtime system deps only — no build tools
RUN apt-get update && apt-get install -y --no-install-recommends         libpq5     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder — compilers NOT included
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY edge/       ./edge/
COPY params.yaml .
COPY .env.example .env

RUN useradd -m -u 1001 edgeuser && chown -R edgeuser:edgeuser /app
USER edgeuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "edge.main"]