"""Tests for the LLM benchmark, using a mocked Ollama server (no model needed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from agmesh.common.artifacts import read_artifact
from agmesh.feasibility.llm_bench import (
    CallResult,
    build_prompts,
    is_installed,
    main,
    response_schema,
    summarise,
    validate_output,
)

REPO_PARAMS = Path(__file__).resolve().parents[2] / "configs" / "params.yaml"


def test_prompts_are_deterministic_and_grounded() -> None:
    first = build_prompts(30, seed=3)
    second = build_prompts(30, seed=3)
    assert first == second
    for prompt in first:
        assert len(prompt.candidates) == 5
        assert len(set(prompt.candidates)) == 5
        assert prompt.scenario.intended in prompt.candidates


def test_schema_restricts_ids_to_candidates() -> None:
    schema = response_schema(("T0855", "T0836"))
    assert schema["properties"]["technique_id"]["enum"] == ["T0855", "T0836"]


@pytest.mark.parametrize(
    ("content", "json_ok", "schema_ok", "grounded"),
    [
        (
            '{"technique_id": "T0855", "tactic": "x", "confidence": 0.8, "rationale": "r"}',
            True,
            True,
            True,
        ),
        (
            '{"technique_id": "T9999", "tactic": "x", "confidence": 0.8, "rationale": "r"}',
            True,
            True,
            False,
        ),
        (
            '{"technique_id": "T0855", "tactic": "x", "confidence": 1.7, "rationale": "r"}',
            True,
            False,
            True,
        ),
        ('{"technique_id": "T0855", "confidence": true}', True, False, True),
        ("not json", False, False, False),
        ("[1, 2]", False, False, False),
    ],
)
def test_validate_output(content: str, json_ok: bool, schema_ok: bool, grounded: bool) -> None:
    checked = validate_output(content, ("T0855", "T0836"))
    assert checked["json_valid"] is json_ok
    assert checked["schema_valid"] is schema_ok
    assert checked["grounded"] is grounded


def test_is_installed_handles_latest_tag() -> None:
    assert is_installed("llama3.2", ["llama3.2:latest"])
    assert is_installed("llama3.2:3b", ["llama3.2:3b"])
    assert not is_installed("llama3.2:3b", ["llama3.2:1b"])


def test_summary_statistics() -> None:
    results = [
        CallResult(
            i,
            ok=True,
            latency_s=float(i + 1),
            json_valid=True,
            schema_valid=True,
            grounded=i != 0,
            confidence=0.9,
            eval_count=50,
            eval_duration_s=5.0,
        )
        for i in range(10)
    ]
    summary = summarise(results, min_confidence=0.5)
    assert summary["n_ok"] == 10
    assert summary["latency_s"]["max"] == 10.0
    assert summary["decode_tokens_per_s"] == 10.0
    assert summary["grounded_rate"] == 0.9
    assert summary["suggested_l_llm_s"] >= 10
    assert summary["meets_provisional_20s"] is True


def _fake_ollama(installed: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0-test"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in installed]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": installed[0], "size": 2**31}]})
        if request.url.path == "/api/chat":
            body: dict[str, Any] = json.loads(request.content)
            choice = body["format"]["properties"]["technique_id"]["enum"][0]
            content = json.dumps(
                {"technique_id": choice, "tactic": "t", "confidence": 0.7, "rationale": "r"}
            )
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": content},
                    "eval_count": 40,
                    "eval_duration": 2_000_000_000,
                    "prompt_eval_count": 300,
                    "prompt_eval_duration": 1_000_000_000,
                    "load_duration": 0,
                },
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_main_writes_artifact(tmp_path: Path) -> None:
    out = tmp_path / "llm.json"
    http = httpx.Client(transport=_fake_ollama(["model-a:latest"]))
    code = main(
        [
            "--params",
            str(REPO_PARAMS),
            "--models",
            "model-a",
            "model-b",
            "--n-prompts",
            "4",
            "--output",
            str(out),
            "--no-memory",
        ],
        http=http,
    )
    assert code == 0
    document = read_artifact(out)
    assert document["kind"] == "feasibility.llm_bench"
    assert document["data_source"] == "runtime"
    reports = {r["model"]: r for r in document["models"]}
    assert reports["model-b"]["status"] == "not_installed"
    summary = reports["model-a"]["summary"]
    assert summary["n_ok"] == 4
    assert summary["schema_valid_rate"] == 1.0
    assert summary["grounded_rate"] == 1.0
    assert summary["decode_tokens_per_s"] == 20.0
    assert reports["model-a"]["memory"]["ollama_reported_size_mb"] == 2048


def test_main_fails_when_no_model_is_installed(tmp_path: Path) -> None:
    http = httpx.Client(transport=_fake_ollama(["other:latest"]))
    code = main(
        [
            "--params",
            str(REPO_PARAMS),
            "--models",
            "model-a",
            "--n-prompts",
            "2",
            "--output",
            str(tmp_path / "x.json"),
            "--no-memory",
        ],
        http=http,
    )
    assert code == 1


def test_main_refuses_remote_endpoint(tmp_path: Path) -> None:
    code = main(
        [
            "--params",
            str(REPO_PARAMS),
            "--base-url",
            "https://api.example.com",
            "--output",
            str(tmp_path / "x.json"),
        ]
    )
    assert code == 2
