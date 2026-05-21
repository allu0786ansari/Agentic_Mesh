FROM python:3.12-slim
LABEL description="Agentic Mesh — Agent Mesh"

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential libpq-dev   && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip  && pip install --no-cache-dir -r requirements.txt

COPY agents/ ./agents/
COPY knowledge_base/ ./knowledge_base/
COPY params.yaml .
COPY .env.example .env

RUN useradd -m -u 1001 agentuser
USER agentuser
ENV PYTHONUNBUFFERED=1
EXPOSE 8001
CMD ["python", "-m", "agents.graph"]