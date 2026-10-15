"""The one LLM call per incident (Proposal §8.4; ADR 0012 §4).

The model is Ollama's ``/api/chat`` with ``format`` set to a JSON schema whose
``technique_id`` is an enumeration of the candidate IDs.

* **Settings:** temperature 0, seed 0, ``num_ctx`` 2048. The system prompt is the
  G0 benchmark's.
* **The raw content is returned unparsed.** :func:`validate` judges it the way the
  benchmark did (``feasibility.llm_bench.validate_output``) and adds the confidence
  check.
* **Failures are surfaced:** :class:`LLMUnavailableError` covers a timeout and an
  unreachable server. Attribution then falls back to retrieval.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from agmesh.feasibility.llm_bench import SYSTEM_PROMPT, response_schema, validate_output


class LLMUnavailableError(Exception):
    pass


@dataclass(frozen=True)
class LLMReply:
    content: str
    seconds: float
    model: str


class LLM(Protocol):
    def choose(self, user_message: str, candidates: tuple[str, ...]) -> LLMReply: ...


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float,
        num_ctx: int = 2048,
        temperature: float = 0.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url, self.model = base_url.rstrip("/"), model
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "seed": 0}
        self.http = http or httpx.Client(timeout=timeout_s)

    def choose(self, user_message: str, candidates: tuple[str, ...]) -> LLMReply:
        started = time.perf_counter()
        try:
            response = self.http.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "format": response_schema(candidates),
                    "stream": False,
                    "options": self.options,
                    "keep_alive": "30m",
                },
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        content = str(body.get("message", {}).get("content", ""))
        return LLMReply(content, time.perf_counter() - started, self.model)


def validate(content: str, candidates: tuple[str, ...], min_confidence: float) -> dict[str, Any]:
    """Benchmark checks plus the confidence floor; ``valid`` decides the fallback."""
    v = validate_output(content, candidates)
    conf = v["confidence"]
    v["confident"] = conf is not None and conf >= min_confidence
    v["valid"] = bool(v["schema_valid"] and v["grounded"] and v["confident"])
    if not v["json_valid"]:
        v["reason"] = "output is not JSON"
    elif not v["schema_valid"]:
        v["reason"] = "output does not match the schema"
    elif not v["grounded"]:
        v["reason"] = f"technique {v['technique_id']!r} is not a candidate"
    elif not v["confident"]:
        v["reason"] = f"confidence {conf} below {min_confidence}"
    else:
        v["reason"] = None
    return v


def user_message(evidence_text: str, candidates: list[tuple[str, str, list[str]]]) -> str:
    """Rendered evidence plus one line per candidate: ID, name, tactics."""
    lines = ["Evidence:", evidence_text, "", "Candidates:"]
    lines += [
        f"- {tid} {name} (tactics: {', '.join(tactics)})" for tid, name, tactics in candidates
    ]
    lines += [
        "",
        "Return technique_id (one of the candidate IDs), tactic, confidence between 0 and 1, "
        "and a rationale of at most 40 words.",
    ]
    return "\n".join(lines)


def scripted(content: dict[str, Any] | str) -> str:
    """Helper for tests: a reply body as the model would send it."""
    return content if isinstance(content, str) else json.dumps(content)
