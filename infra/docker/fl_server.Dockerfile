# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         libpq-dev     && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# CPU-only torch — FL server aggregates model weights, no GPU needed
RUN uv pip install --system --no-cache     torch==2.6.0+cpu     torchvision==0.21.0+cpu     --index-url https://download.pytorch.org/whl/cpu

COPY requirements.fl.txt .
RUN uv pip install --system --no-cache -r requirements.fl.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Allaudin Ansari"
LABEL description="Agentic Mesh — Flower FL Server (CPU-only, multi-stage)"
LABEL version="0.2.0"

RUN apt-get update && apt-get install -y --no-install-recommends         libpq5     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

COPY federated/ ./federated/
COPY params.yaml .
COPY .env.example .env

RUN useradd -m -u 1001 fluser && chown -R fluser:fluser /app
USER fluser

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080
CMD ["python", "-m", "federated.server"]