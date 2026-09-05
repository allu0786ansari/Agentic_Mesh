from __future__ import annotations

import json
from typing import Any


class LocalPhiModel:
    """Structured JSON adapter for a local Ollama Phi model."""

    def __init__(self, model: str = "agmesh-phi3.5", base_url: str = "http://localhost:11434", timeout: int = 30):
        if not model.strip():
            raise ValueError("model must not be empty")
        try:
            from langchain_community.llms import Ollama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("langchain-community is required for LocalPhiModel") from exc
        self.model = model
        self._llm = Ollama(
            model=model,
            base_url=base_url,
            temperature=0.0,
            format="json",
            timeout=timeout,
            system="Return only valid JSON. Do not include markdown fences or explanatory text.",
        )

    def invoke_json(self, prompt: str) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        raw = self._llm.invoke(prompt)
        if not isinstance(raw, str):
            raise TypeError("Ollama returned a non-string response")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Ollama returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Ollama JSON response must be an object")
        return payload


__all__ = ["LocalPhiModel"]
