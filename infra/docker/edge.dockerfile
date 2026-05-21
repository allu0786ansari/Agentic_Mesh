FROM python:3.12-slim
LABEL description="Agentic Mesh — Edge Node"

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential libpq-dev curl   && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip  && pip install --no-cache-dir -r requirements.txt

COPY edge/ ./edge/
COPY params.yaml .
COPY .env.example .env

RUN useradd -m -u 1001 edgeuser
USER edgeuser
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "edge.main"]