FROM python:3.12-slim
LABEL description="Agentic Mesh — FastAPI Services"

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential libpq-dev   && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip  && pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY params.yaml .
COPY .env.example .env

RUN useradd -m -u 1001 apiuser
USER apiuser
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn","api.node_api:app","--host","0.0.0.0","--port","8000"]