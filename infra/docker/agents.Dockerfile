# ── Stage 1: Builder ─────────────────────────────────────────────
# NOTE: No torch installed here — agents call Ollama via HTTP.
# Removing torch saves ~5-6GB from this image.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         libpq-dev     && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY requirements.agents.txt .
RUN uv pip install --system --no-cache -r requirements.agents.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Allaudin Ansari"
LABEL description="Agentic Mesh — Agent Mesh (no torch, multi-stage)"
LABEL version="0.2.0"

RUN apt-get update && apt-get install -y --no-install-recommends         libpq5     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

COPY agents/       ./agents/
COPY knowledge_base/ ./knowledge_base/
COPY params.yaml .
COPY .env.example .env

RUN useradd -m -u 1001 agentuser && chown -R agentuser:agentuser /app
USER agentuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8001
CMD ["python", "-m", "agents.graph"]