"""The one LLM call per incident (Proposal §8.4; ADR 0012 §4, amended by ADR 0013).

The model is Ollama's ``/api/chat`` with ``format`` set to a JSON schema whose
``technique_id`` is an enumeration of the candidate IDs.

* **Settings:** G0's settings, which are temperature 0, seed 0 and ``num_predict``
  256 (ADR 0013 §3.1). ``num_ctx`` depends on the mode (§3.2). The system prompt is
  the G0 benchmark's.
* **Streaming, with the timeout as a total deadline (ADR 0013 §3.4).** At the
  deadline the client closes the connection, which cancels the request on the
  server. Without this, an abandoned request could delay the next one.
* **The raw content is returned unparsed.** :func:`validate` judges it the way the
  benchmark did (``feasibility.llm_bench.validate_output``) and adds the confidence
  check.
* **Failures are surfaced.** :class:`LLMUnavailableError` carries the error class
  and the elapsed seconds. Attribution then falls back to retrieval (``full``) or
  abstains (``llm_no_retrieval``).
* **Token counts and durations.** Ollama's final chunk carries them, and they are
  returned in :attr:`LLMReply.stats` (§3.5).
* **Recovery.** :meth:`OllamaLLM.recover` waits until the server answers again
  after a failure (§3.4).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from agmesh.feasibility.llm_bench import SYSTEM_PROMPT, response_schema, validate_output

Clock = Callable[[], float]


class LLMUnavailableError(Exception):
    """The call produced no reply: timeout, deadline, unreachable server or bad stream."""

    def __init__(self, message: str, *, kind: str = "Error", seconds: float | None = None):
        super().__init__(message)
        self.kind = kind
        self.seconds = seconds


@dataclass(frozen=True)
class LLMReply:
    content: str
    seconds: float
    model: str
    stats: dict[str, Any] = field(default_factory=dict)


class LLM(Protocol):
    def choose(self, user_message: str, candidates: tuple[str, ...]) -> LLMReply: ...


def _s(ns: Any) -> float | None:
    return round(float(ns) / 1e9, 4) if isinstance(ns, int | float) else None


def stats_of(final: dict[str, Any]) -> dict[str, Any]:
    """Token counts and durations from Ollama's final chunk."""
    return {
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_s": _s(final.get("prompt_eval_duration")),
        "eval_count": final.get("eval_count"),
        "eval_s": _s(final.get("eval_duration")),
        "load_s": _s(final.get("load_duration")),
        "done_reason": final.get("done_reason"),
    }


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float,
        num_ctx: int = 2048,
        num_predict: int = 256,
        temperature: float = 0.0,
        http: httpx.Client | None = None,
        clock: Clock = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url, self.model, self.timeout_s = base_url.rstrip("/"), model, timeout_s
        self.options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "seed": 0,
        }
        self.http = http or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0))
        self.clock, self.sleep = clock, sleep

    def _fail(self, exc: Exception, started: float) -> LLMUnavailableError:
        return LLMUnavailableError(
            f"{type(exc).__name__}: {exc}",
            kind=type(exc).__name__,
            seconds=self.clock() - started,
        )

    def choose(self, user_message: str, candidates: tuple[str, ...]) -> LLMReply:
        started = self.clock()
        deadline = started + self.timeout_s
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "format": response_schema(candidates),
            "stream": True,
            "options": self.options,
            "keep_alive": "30m",
        }
        parts: list[str] = []
        final: dict[str, Any] = {}
        try:
            # Leaving the block early closes the connection, which cancels the request
            # on the server (ADR 0013 §3.4).
            with self.http.stream("POST", f"{self.base_url}/api/chat", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if self.clock() > deadline:
                        raise LLMUnavailableError(
                            f"Deadline: no complete reply within {self.timeout_s} s",
                            kind="Deadline",
                            seconds=self.clock() - started,
                        )
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise LLMUnavailableError(
                            f"OllamaError: {chunk['error']}",
                            kind="OllamaError",
                            seconds=self.clock() - started,
                        )
                    parts.append(str(chunk.get("message", {}).get("content", "")))
                    if chunk.get("done"):
                        final = chunk
                        break
        except LLMUnavailableError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise self._fail(exc, started) from exc
        if not final:
            raise LLMUnavailableError(
                "Incomplete: the stream ended before the final chunk",
                kind="Incomplete",
                seconds=self.clock() - started,
            )
        return LLMReply("".join(parts), self.clock() - started, self.model, stats_of(final))

    def recover(self, max_wait_s: float = 120.0, poll_s: float = 2.0) -> dict[str, Any]:
        """After a failure: wait for ``/api/tags``, then one probe call (ADR 0013 §3.4).

        The probe uses the same options (one output token), so the model is not
        reloaded, and it queues behind any work still running on the server.
        """
        started = self.clock()
        end = started + max_wait_s
        reachable, probe_ok, error = False, False, None
        while True:
            try:
                self.http.get(f"{self.base_url}/api/tags", timeout=5.0).raise_for_status()
                reachable = True
                break
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
            if self.clock() + poll_s >= end:
                break
            self.sleep(poll_s)
        if reachable:
            try:
                self.http.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "ok"}],
                        "stream": False,
                        "options": {**self.options, "num_predict": 1},
                        "keep_alive": "30m",
                    },
                    timeout=max(1.0, end - self.clock()),
                ).raise_for_status()
                probe_ok, error = True, None
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
        return {
            "seconds": round(self.clock() - started, 3),
            "reachable": reachable,
            "probe_ok": probe_ok,
            "error": error,
        }


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
