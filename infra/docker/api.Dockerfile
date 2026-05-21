# ── Stage 1: Builder ─────────────────────────────────────────────
# Smallest image — no torch, no flwr, pure FastAPI + routing layer
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         libpq-dev     && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY requirements.api.txt .
RUN uv pip install --system --no-cache -r requirements.api.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Allaudin Ansari"
LABEL description="Agentic Mesh — FastAPI Services (no torch, multi-stage)"
LABEL version="0.2.0"

RUN apt-get update && apt-get install -y --no-install-recommends         libpq5     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

COPY api/ ./api/
COPY params.yaml .
COPY .env.example .env

RUN useradd -m -u 1001 apiuser && chown -R apiuser:apiuser /app
USER apiuser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000
CMD ["uvicorn", "api.node_api:app", "--host", "0.0.0.0", "--port", "8000"]