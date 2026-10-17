"""LLM timeout per mode, by G0's rule on the real prompts (ADR 0013 §3.3).

    python -m agmesh.mesh.llm_calibrate --mode full
    make llm-calibrate MODE=llm_no_retrieval

**Prompts.**

* **Source:** the first ADR 0012 run of the mode, per site (the earliest
  ``results/mesh/<site>__<mode>__val__*/storage.db``).
* **Sample:** 15 sessions per site, drawn with seed 0 from the sessions with
  evidence.
* **Rebuilt prompt:** each session's stored evidence text and candidate list, with
  the candidate lines and user message the agent builds.

**Calls.**

* **Warm-up:** one call on the first prompt, excluded from the result.
* **Each prompt is sent once.** There is no client timeout, only a 600 s safety cap.
* **Options:** ``num_predict`` and the mode's ``num_ctx`` (§3.1-3.2).

**Result.**

* **Rule:** ``timeout_s = ceil(1.5 * p95)``. The p95 is taken over every call's
  seconds; a failed call counts with its elapsed time.
* **Where it goes:**
  * the full record goes to ``results/llm_calibration/<mode>__val__<time>.json``;
  * the timeout, model and options go to ``configs/mesh/llm_timeouts.lock.json``.
* **What is not compared:** answers against ground truth. Only latency, token counts
  and output validity are recorded.

**Once per mode.** A mode that is already in the lock is refused unless ``--force``
is given. Re-measuring until a timeout looks good would be tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH, load_params
from agmesh.common.env import load_env
from agmesh.common.sysmem import snapshot
from agmesh.mesh.attribution import AttributionAgent
from agmesh.mesh.llm import LLMUnavailableError, OllamaLLM, user_message, validate

LLM_MODES = ("full", "llm_no_retrieval")
LOCK = Path("configs/mesh/llm_timeouts.lock.json")
SAFETY_CAP_S = 600.0


def first_run(root: Path, site: str, mode: str) -> Path:
    runs = sorted(p for p in root.glob(f"{site}__{mode}__val__*") if (p / "storage.db").is_file())
    if not runs:
        raise SystemExit(f"no {mode} validation run with storage.db for {site} under {root}")
    return runs[0]


def stored_prompts(db: Path) -> list[tuple[str, list[str]]]:
    """(evidence text, candidates) of every session with evidence, in session order."""
    con = sqlite3.connect(db)
    try:
        rows = con.execute(
            "SELECT payload FROM session_steps WHERE step = 'attribute' ORDER BY at, session_id"
        ).fetchall()
    finally:
        con.close()
    out = []
    for (payload,) in rows:
        a = json.loads(payload)
        if a.get("evidence_text") and a.get("candidates"):
            out.append((str(a["evidence_text"]), [str(c) for c in a["candidates"]]))
    return out


def sample(prompts: list[Any], k: int, seed: int) -> list[Any]:
    if len(prompts) < k:
        raise SystemExit(f"only {len(prompts)} sessions with evidence; {k} needed")
    idx = sorted(random.Random(seed).sample(range(len(prompts)), k))  # noqa: S311 - reproducible sample
    return [prompts[i] for i in idx]


def timeout_rule(seconds: list[float]) -> tuple[float, int]:
    p95 = float(np.percentile(np.asarray(seconds), 95))
    return p95, math.ceil(1.5 * p95)


def _q(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def decode_rate(stats: dict[str, Any]) -> float | None:
    n, s = stats.get("eval_count"), stats.get("eval_s")
    if not isinstance(n, int | float) or not isinstance(s, int | float) or s <= 0:
        return None
    return round(float(n) / float(s), 3)


def slow_spell(records: list[dict[str, Any]], p95_s: float) -> list[str]:
    """ADR 0013 §6 stopping rule: any decode under 1 token/s, or p95 above 120 s."""
    reasons = []
    slow = [r for r in records if r.get("decode_tok_s") is not None and r["decode_tok_s"] < 1.0]
    if slow:
        reasons.append(f"{len(slow)} call(s) decoded under 1 token/s")
    if p95_s > 120.0:
        reasons.append(f"p95 {p95_s:.1f} s above 120 s")
    return reasons


def calibrate(
    llm: Any,
    agent: AttributionAgent,
    prompts: list[tuple[str, str, list[str]]],
    min_confidence: float,
    log: Any = print,
) -> list[dict[str, Any]]:
    """One call per prompt (after one warm-up); returns one record per call."""

    def call(site: str, text: str, cands: list[str]) -> dict[str, Any]:
        message = user_message(text, [agent.line(t) for t in cands])
        try:
            reply = llm.choose(message, tuple(cands))
        except LLMUnavailableError as exc:
            return {
                "site": site,
                "ok": False,
                "seconds": exc.seconds,
                "kind": exc.kind,
                **snapshot(),
            }
        v = validate(reply.content, tuple(cands), min_confidence)
        return {
            "site": site,
            "ok": True,
            "seconds": reply.seconds,
            "valid": v["valid"],
            "grounded": v["grounded"],
            "prompt_chars": len(message),
            **reply.stats,
            "decode_tok_s": decode_rate(reply.stats),
            **snapshot(),
        }

    site, text, cands = prompts[0]
    warm = call(site, text, cands)
    log(f"warm-up: {warm['seconds']:.1f} s (excluded)")
    records = []
    for i, (site, text, cands) in enumerate(prompts, 1):
        r = call(site, text, cands)
        records.append(r)
        log(
            f"{i:2d}/{len(prompts)} {site} {float(r['seconds'] or 0.0):.1f} s ok={r['ok']} "
            f"prompt={r.get('prompt_eval_count')} out={r.get('eval_count')} "
            f"decode={r.get('decode_tok_s')} tok/s | mem avail {r.get('mem_available_mb')} MB, "
            f"llm server {r.get('llm_server_rss_mb')} MB"
        )
    return records


def summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r["ok"]]

    def ints(key: str) -> list[float]:
        return [float(r[key]) for r in ok if isinstance(r.get(key), int | float)]

    return {
        "calls": len(records),
        "ok": len(ok),
        "failures": {
            k: sum(1 for r in records if not r["ok"] and r["kind"] == k)
            for k in sorted({r["kind"] for r in records if not r["ok"]})
        },
        "valid_rate": float(np.mean([r["valid"] for r in ok])) if ok else None,
        "seconds": _q([float(r["seconds"] or 0.0) for r in records]),
        "prompt_tokens": _q(ints("prompt_eval_count")),
        "output_tokens": _q(ints("eval_count")),
        "prompt_eval_s": _q(ints("prompt_eval_s")),
        "eval_s": _q(ints("eval_s")),
        "done_reason": {
            d: sum(1 for r in ok if r.get("done_reason") == d)
            for d in sorted({str(r.get("done_reason")) for r in ok})
        },
        "by_site": {
            s: _q([float(r["seconds"] or 0.0) for r in records if r["site"] == s])
            for s in sorted({r["site"] for r in records})
        },
    }


def read_lock(path: Path = LOCK) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "1.0", "adr": "docs/adr/0013-llm-path-fix.md", "modes": {}}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description="LLM timeout per mode by G0's rule (ADR 0013)")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--mode", choices=LLM_MODES, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("results/mesh"))
    parser.add_argument("--per-site", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--out", type=Path, default=Path("results/llm_calibration"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    params = load_params(args.params)
    lock = read_lock(args.lock)
    if args.mode in lock["modes"] and not args.force:
        raise SystemExit(
            f"{args.mode} is already calibrated in {args.lock}; ADR 0013 measures once "
            "(--force only to replace a broken measurement, and say why in the ADR)"
        )
    from agmesh.mesh.runner import check_llm

    check_llm(params)
    sources, prompts = {}, []
    for site in ("site_a", "site_b"):
        run = first_run(args.source_root, site, args.mode)
        sources[site] = run.as_posix()
        prompts += [
            (site, t, c)
            for t, c in sample(stored_prompts(run / "storage.db"), args.per_site, args.seed)
        ]
    num_ctx = params.llm.ctx_for(args.mode)
    llm = OllamaLLM(
        params.llm.base_url,
        params.llm.model,
        timeout_s=SAFETY_CAP_S,
        num_ctx=num_ctx,
        num_predict=params.llm.num_predict,
        temperature=params.llm.temperature,
    )
    agent = AttributionAgent(None, None, mitigations={})
    records = calibrate(llm, agent, prompts, params.mesh.attribution.min_confidence)
    result = summary(records)
    p95, timeout_s = timeout_rule([float(r["seconds"] or 0.0) for r in records])
    stopped = slow_spell(records, p95)
    entry = {
        "timeout_s": timeout_s,
        "p95_s": round(p95, 3),
        "rule": "ceil(1.5 x p95), G0 decision 7",
        "model": params.llm.model,
        "num_ctx": num_ctx,
        "num_predict": params.llm.num_predict,
        "temperature": params.llm.temperature,
        "calls": len(records),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = write_artifact(
        args.out / f"{args.mode}__val__{stamp}.json",
        kind="mesh.llm_calibration",
        data_source="runtime",
        payload={
            "protocol": "docs/adr/0013-llm-path-fix.md",
            "mode": args.mode,
            "split": "val",
            "sources": sources,
            "per_site": args.per_site,
            "seed": args.seed,
            "safety_cap_s": SAFETY_CAP_S,
            "lock_entry": None if stopped else entry,
            "stopped_by_rule": stopped or None,
            "summary": result,
            "calls": records,
        },
        params_path=args.params,
    )
    if stopped:
        # ADR 0013 §6: the mode is not run; the runner refuses it from now on.
        lock["modes"].pop(args.mode, None)
        lock.setdefault("stopped", {})[args.mode] = {
            "reasons": stopped,
            "record": path.as_posix(),
        }
    else:
        lock["modes"][args.mode] = {**entry, "record": path.as_posix()}
        lock.get("stopped", {}).pop(args.mode, None)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if stopped:
        print(
            f"{args.mode}: STOPPED by ADR 0013 §6 ({'; '.join(stopped)}); not run on this "
            f"machine.\n  -> {args.lock}\n  -> {path}"
        )
        return 0
    print(
        f"{args.mode}: p95 {p95:.1f} s over {len(records)} calls -> timeout {timeout_s} s "
        f"(ok {result['ok']}/{result['calls']}, valid {result['valid_rate']})\n"
        f"  -> {args.lock}\n  -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
