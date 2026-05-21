FROM python:3.12-slim
LABEL description="Agentic Mesh — Flower FL Server"

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential libpq-dev   && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip  && pip install --no-cache-dir -r requirements.txt

COPY federated/ ./federated/
COPY params.yaml .
COPY .env.example .env

RUN useradd -m -u 1001 fluser
USER fluser
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python", "-m", "federated.server"]