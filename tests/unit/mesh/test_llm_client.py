"""The streaming Ollama client: G0 options, deadline, failure classes, recovery (ADR 0013)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from agmesh.mesh.llm import LLMUnavailableError, OllamaLLM

CANDS = ("T0806", "T0814")
REPLY = {"technique_id": "T0806", "tactic": "x", "confidence": 0.8, "rationale": "r"}


class Clock:
    """Advances ``step`` seconds per reading."""

    def __init__(self, step: float) -> None:
        self.t, self.step = 0.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def chunks(text: str, final: dict[str, Any] | None = None) -> list[bytes]:
    out = [json.dumps({"message": {"content": c}, "done": False}).encode() + b"\n" for c in text]
    done = {
        "message": {"content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 480,
        "prompt_eval_duration": 5_000_000_000,
        "eval_count": 60,
        "eval_duration": 8_500_000_000,
        "load_duration": 40_000_000,
    }
    return [*out, json.dumps(final or done).encode() + b"\n"]


def client(
    handler: Callable[[httpx.Request], httpx.Response], clock: Clock | None = None, **kw: Any
) -> OllamaLLM:
    return OllamaLLM(
        "http://ollama",
        "llama3.2:3b",
        timeout_s=kw.pop("timeout_s", 21.0),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=clock or Clock(0.01),
        sleep=lambda _s: None,
        **kw,
    )


def test_streamed_reply_is_joined_with_g0_options_and_stats() -> None:
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, content=iter(chunks(json.dumps(REPLY))))

    reply = client(handler, num_ctx=4096).choose("evidence", CANDS)
    assert json.loads(reply.content) == REPLY
    body = seen[0]
    assert body["stream"] is True
    assert body["options"] == {"temperature": 0.0, "num_ctx": 4096, "num_predict": 256, "seed": 0}
    assert body["format"]["properties"]["technique_id"]["enum"] == list(CANDS)
    assert reply.stats == {
        "prompt_eval_count": 480,
        "prompt_eval_s": 5.0,
        "eval_count": 60,
        "eval_s": 8.5,
        "load_s": 0.04,
        "done_reason": "stop",
    }


def test_deadline_stops_reading_the_stream() -> None:
    sent: list[int] = []

    def body() -> Iterator[bytes]:
        for i, c in enumerate(chunks(json.dumps(REPLY))):
            sent.append(i)
            yield c

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    with pytest.raises(LLMUnavailableError) as err:
        client(handler, Clock(1.0), timeout_s=5.0).choose("e", CANDS)
    assert err.value.kind == "Deadline" and err.value.seconds is not None
    assert len(sent) < len(chunks(json.dumps(REPLY)))  # the rest was never read


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (httpx.ReadTimeout("timed out"), "ReadTimeout"),
        (httpx.ConnectError("[Errno 111] Connection refused"), "ConnectError"),
    ],
)
def test_transport_failures_keep_their_class(exc: Exception, kind: str) -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(LLMUnavailableError) as err:
        client(handler).choose("e", CANDS)
    assert err.value.kind == kind and err.value.seconds is not None


def test_error_chunk_and_incomplete_stream() -> None:
    def error(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=iter([b'{"error": "out of memory"}\n']))

    with pytest.raises(LLMUnavailableError) as err:
        client(error).choose("e", CANDS)
    assert err.value.kind == "OllamaError"

    def cut(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=iter(chunks("{")[:-1]))

    with pytest.raises(LLMUnavailableError) as err:
        client(cut).choose("e", CANDS)
    assert err.value.kind == "Incomplete"


def test_recover_waits_for_the_server_then_probes_with_the_same_options() -> None:
    tags, probes = [0], []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            tags[0] += 1
            if tags[0] <= 2:
                raise httpx.ConnectError("refused")
            return httpx.Response(200, json={"models": []})
        probes.append(json.loads(req.content))
        return httpx.Response(200, json={"done": True})

    rec = client(handler, num_ctx=4096).recover(max_wait_s=120.0)
    assert rec["reachable"] and rec["probe_ok"] and rec["error"] is None
    assert tags[0] == 3 and len(probes) == 1
    assert probes[0]["options"] == {
        "temperature": 0.0,
        "num_ctx": 4096,
        "num_predict": 1,
        "seed": 0,
    }


def test_recover_gives_up_after_the_cap() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    rec = client(handler, Clock(1.0)).recover(max_wait_s=10.0, poll_s=2.0)
    assert not rec["reachable"] and not rec["probe_ok"]
    assert rec["error"].startswith("ConnectError")
