"""Timeout calibration and the runner's use of it (ADR 0013 §3.3, §3.5)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agmesh.common.config import load_params
from agmesh.mesh import llm_calibrate as cal
from agmesh.mesh.attribution import AttributionAgent
from agmesh.mesh.llm import LLMReply, LLMUnavailableError
from agmesh.mesh.runner import _llm_block, llm_settings

REPLY = json.dumps({"technique_id": "T0806", "tactic": "x", "confidence": 0.8, "rationale": "r"})


def _db(path: Path, n: int) -> Path:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE session_steps "
        "(session_id TEXT, step TEXT, status TEXT, payload TEXT, at TEXT)"
    )
    for i in range(n):
        payload = {"evidence_text": f"ev {i}", "candidates": ["T0806", "T0814"]}
        con.execute(
            "INSERT INTO session_steps VALUES (?, 'attribute', 'done', ?, ?)",
            (f"s{i:03d}", json.dumps(payload), f"2026-10-15T00:{i:02d}:00"),
        )
    con.execute(
        "INSERT INTO session_steps VALUES ('empty', 'attribute', 'done', ?, '2026-10-15T01:00:00')",
        (json.dumps({"evidence_text": "", "candidates": []}),),
    )
    con.commit()
    con.close()
    return path


def test_rule_is_ceil_of_one_and_a_half_p95() -> None:
    p95, timeout = cal.timeout_rule([10.0] * 19 + [20.0])
    assert p95 == pytest.approx(10.5) and timeout == 16
    assert cal.timeout_rule([14.0] * 20)[1] == 21  # G0: p95 13.6 s -> 21 s


def test_prompts_come_from_the_stored_sessions_and_the_sample_is_fixed(tmp_path: Path) -> None:
    prompts = cal.stored_prompts(_db(tmp_path / "s.db", 40))
    assert len(prompts) == 40 and prompts[0] == ("ev 0", ["T0806", "T0814"])
    a, b = cal.sample(prompts, 15, 0), cal.sample(prompts, 15, 0)
    assert a == b and len(a) == 15 and len(set(map(str, a))) == 15
    with pytest.raises(SystemExit):
        cal.sample(prompts, 41, 0)


def test_first_run_is_the_earliest(tmp_path: Path) -> None:
    for stamp in ("20261016T021330", "20261015T234425"):
        d = tmp_path / f"site_b__full__val__{stamp}"
        d.mkdir()
        (d / "storage.db").touch()
    assert cal.first_run(tmp_path, "site_b", "full").name.endswith("20261015T234425")


class Timed:
    def __init__(self, seconds: list[float | None]) -> None:
        self.seconds = seconds

    def choose(self, _msg: str, _cands: tuple[str, ...]) -> LLMReply:
        s = self.seconds.pop(0)
        if s is None:
            raise LLMUnavailableError("x", kind="ReadTimeout", seconds=600.0)
        return LLMReply(REPLY, s, "m", {"prompt_eval_count": 500, "eval_count": 60})


def test_calibrate_excludes_the_warm_up_and_counts_failures() -> None:
    agent = AttributionAgent(None, None, mitigations={})
    prompts = [("site_a", "ev", ["T0806", "T0814"])] * 3
    records = cal.calibrate(
        Timed([99.0, 10.0, None, 12.0]), agent, prompts, 0.5, log=lambda _m: None
    )
    assert [r["seconds"] for r in records] == [10.0, 600.0, 12.0]
    s = cal.summary(records)
    assert s["ok"] == 2 and s["failures"] == {"ReadTimeout": 1} and s["valid_rate"] == 1.0


def _lock(tmp: Path, **entry: Any) -> Path:
    p = tmp / "lock.json"
    params = load_params()
    base = {
        "timeout_s": 30,
        "model": params.llm.model,
        "num_ctx": params.llm.ctx_for("full"),
        "num_predict": params.llm.num_predict,
        "temperature": params.llm.temperature,
    }
    p.write_text(json.dumps({"modes": {"full": {**base, **entry}}}))
    return p


def test_runner_needs_a_current_lock(tmp_path: Path) -> None:
    params = load_params()
    with pytest.raises(SystemExit, match="make llm-calibrate"):
        llm_settings(params, "llm_no_retrieval", _lock(tmp_path))
    with pytest.raises(SystemExit, match="other settings"):
        llm_settings(params, "full", _lock(tmp_path, num_ctx=1024))
    cfg = llm_settings(params, "full", _lock(tmp_path))
    assert cfg["timeout_s"] == 30.0 and cfg["num_predict"] == 256
    assert params.llm.ctx_for("llm_no_retrieval") == 4096 and params.llm.ctx_for("full") == 2048


def test_summary_counts_failures_by_class() -> None:
    ok = {
        "llm": {
            "content": REPLY,
            "seconds": 14.0,
            "stats": {"prompt_eval_count": 480, "eval_count": 60, "done_reason": "stop"},
            "validation": {"grounded": True, "valid": True},
        }
    }
    bad = {"llm": {"error": "ReadTimeout: timed out", "kind": "ReadTimeout", "seconds": 21.0}}
    down = {"llm": {"error": "ConnectError", "kind": "ConnectError", "seconds": 0.01}}
    b = _llm_block([ok, bad, bad, down, {}])
    assert b["calls"] == 4 and b["answered"] == 1
    assert b["failures"] == {"ReadTimeout": 2, "ConnectError": 1}
    assert b["prompt_tokens"]["p50"] == 480.0 and b["done_reason"] == {"stop": 1}


def test_stopping_rule_flags_slow_decode_and_high_p95() -> None:
    healthy: list[dict[str, Any]] = [{"decode_tok_s": 5.5}, {"decode_tok_s": None}]
    assert cal.slow_spell(healthy, 65.0) == []
    assert cal.decode_rate({"eval_count": 67, "eval_s": 349.9}) == 0.191
    reasons = cal.slow_spell([*healthy, {"decode_tok_s": 0.19}], 387.0)
    assert reasons == ["1 call(s) decoded under 1 token/s", "p95 387.0 s above 120 s"]


def test_runner_refuses_a_stopped_mode(tmp_path: Path) -> None:
    p = tmp_path / "lock.json"
    p.write_text(json.dumps({"modes": {}, "stopped": {"llm_no_retrieval": {"reasons": ["x"]}}}))
    with pytest.raises(SystemExit, match="stopped by ADR 0013"):
        llm_settings(load_params(), "llm_no_retrieval", p)


def test_progress_line_names_status_mode_latency_and_memory() -> None:
    from agmesh.mesh.runner import progress

    state = {
        "status": "COMPLETED",
        "attribution": {
            "attribution_mode": "retrieval_fallback",
            "technique_id": "T0814",
            "llm": {"kind": "Deadline", "seconds": 26.4},
        },
        "latency_ms": {"incident_to_dispatch": 26900.0},
    }
    line = progress(7, state, {"mem_available_mb": 2100, "llm_server_rss_mb": 2600})
    assert line == (
        "[   7] COMPLETED | retrieval_fallback T0814 | dispatch 26.9s | llm 26.4s Deadline"
        " | mem avail 2100 MB, llm server 2600 MB"
    )
