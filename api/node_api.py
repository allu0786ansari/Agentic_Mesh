"""
api/node_api.py
FastAPI edge node endpoint — Week 1 stub.
Full embedding query endpoint added in Week 6.
Run: uvicorn api.node_api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
from fastapi import FastAPI

app = FastAPI(
    title="Agentic Mesh — Node API",
    description="Edge node embedding query and status endpoint",
    version="0.1.0",
)

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "node_api", "version": "0.1.0"}

@app.get("/")
async def root() -> dict:
    return {"message": "Agentic Mesh Node API — see /docs"}