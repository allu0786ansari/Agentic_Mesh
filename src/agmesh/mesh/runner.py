"""Mesh runner: bus → triage → graph per incident → summary (Plan Week 10 task 6; ADR 0012 §6).

    python -m agmesh.mesh.runner --site site_b --mode retrieval_only --split val
    make mesh SITE=site_b MODE=full SPLIT=val

**What the runner does.**

1. It copies the latest edge run's ``storage.db`` (its alerts) into a fresh run
   directory, so every mode consumes the same alerts.
2. It claims ``alerts.<site>`` under leases and feeds Triage in data-time order.
3. When an incident closes, it runs the graph in a worker thread. It acknowledges
   the incident's alerts after the session has committed (at-least-once, ADR 0009
   §5).

**Requirements.**

* ``full`` and ``llm_no_retrieval`` need Ollama with ``llm.model``, and a calibrated
  timeout in ``configs/mesh/llm_timeouts.lock.json`` (``make llm-calibrate``; ADR 0013
  §3.3). The runner checks both before starting.
* After a failed LLM call, the runner waits for Ollama to recover before the next
  incident (ADR 0013 §3.4).
* Every mode except ``rule_only`` needs the knowledge base (``make kb``).

**Output.** ``summary.json`` holds:

* sessions by status and attribution mode;
* the technique distribution and the grounding rate;
* LLM failures by error class, token counts and recoveries (ADR 0013 §3.5);
* investigation and incident-to-dispatch latency.

It holds no ground-truth accuracy; that is X4.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.common.artifacts import write_artifact, write_event
from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.common.env import load_env
from agmesh.common.sysmem import llm_server_info, snapshot
from agmesh.mesh.attribution import MODES, AttributionAgent
from agmesh.mesh.breaker import CircuitBreaker
from agmesh.mesh.graph import Mesh
from agmesh.mesh.investigator import Investigator
from agmesh.mesh.llm_calibrate import LLM_MODES, LOCK, read_lock
from agmesh.mesh.nodeclient import LocalNodeClient, NodeClient
from agmesh.mesh.sessions import SessionStore
from agmesh.mesh.triage import ClosedIncident, Triage


def latest_edge_run(root: Path, site: str, split: str) -> Path:
    runs = sorted(p for p in root.glob(f"{site}__{split}__*") if (p / "storage.db").is_file())
    if not runs:
        raise SystemExit(f"no edge run with storage.db for {site} {split} under {root} (make edge)")
    return runs[-1]


def _q(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    a = np.asarray(values)
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def build_mesh(
    params: Params,
    store: SessionStore,
    bus: SqliteBus,
    client: NodeClient,
    agent: AttributionAgent,
    events: Path | None = None,
    run_id: str = "",
) -> Mesh:
    m = params.mesh

    def transition(node: str, frm: str, to: str) -> None:
        if events is not None:
            write_event(
                events,
                "breaker",
                {"node": node, "from": frm, "to": to},
                run_id=run_id,
                data_source="runtime",
            )

    breakers = {
        n: CircuitBreaker(
            n,
            failure_threshold=m.breaker.failure_threshold,
            open_seconds=m.breaker.open_seconds,
            on_transition=transition,
        )
        for n in ("a1", "a2", "a3", "a4", "b1", "b2", "b3")
    }
    inv = Investigator(
        client,
        store,
        max_hops=m.max_hops,
        connect_timeout_s=m.connect_timeout_s,
        read_timeout_s=m.read_timeout_s,
        max_retries=m.max_retries,
        deadline_s=m.investigation_deadline_s,
        breakers=breakers,
    )
    return Mesh(store, bus, inv, agent)


def consume(
    mesh: Mesh,
    bus: SqliteBus,
    triage: Triage,
    site: str,
    mode: str,
    run_id: str,
    *,
    consumer: str = "mesh",
    batch: int = 100,
    worker_timeout_s: float = 120.0,
    on_session: Any = None,
) -> list[dict[str, Any]]:
    """Drain ``alerts.<site>``: every closed incident runs the graph, then its alerts are acked."""
    topic = f"alerts.{site}"
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:

        def run(inc: ClosedIncident) -> None:
            fut = pool.submit(mesh.run, site, inc.summary(), mode, run_id)
            state = fut.result(timeout=worker_timeout_s)
            for msg_id in inc.msg_ids:
                bus.ack(msg_id, consumer)
            results.append(dict(state))
            if on_session is not None:
                on_session(state)

        seen: set[str] = set()  # a lease that expires while its incident is open is re-claimed
        while True:
            msgs = bus.claim(
                topic, consumer, lease_s=max(3600.0, worker_timeout_s * 4), limit=batch
            )
            fresh = [m for m in msgs if m.msg_id not in seen]
            if not fresh:
                break
            for m in fresh:
                seen.add(m.msg_id)
                for inc in triage.feed(m.payload, m.msg_id):
                    run(inc)
        for inc in triage.drain():
            run(inc)
    return results


def _llm_block(atts: list[dict[str, Any]]) -> dict[str, Any]:
    """Answered calls, failures by error class, tokens and durations (ADR 0013 §3.5)."""
    answered = [a for a in atts if a.get("llm") and "validation" in (a.get("llm") or {})]
    failed = [a for a in atts if a.get("llm") and "error" in (a.get("llm") or {})]
    grounded = [bool(a["llm"]["validation"]["grounded"]) for a in answered]
    stats = [a["llm"].get("stats") or {} for a in answered]

    def nums(key: str) -> list[float]:
        return [float(s[key]) for s in stats if isinstance(s.get(key), int | float)]

    return {
        "calls": len(answered) + len(failed),
        "answered": len(answered),
        "raw_grounding_rate": float(np.mean(grounded)) if grounded else None,
        "valid_rate": float(np.mean([a["llm"]["validation"]["valid"] for a in answered]))
        if answered
        else None,
        "seconds": _q([float(a["llm"]["seconds"]) for a in answered]),
        "failures": dict(
            collections.Counter(str(a["llm"].get("kind", "Error")) for a in failed).most_common()
        ),
        "failed_seconds": _q(
            [float(a["llm"]["seconds"]) for a in failed if a["llm"].get("seconds") is not None]
        ),
        "prompt_tokens": _q(nums("prompt_eval_count")),
        "output_tokens": _q(nums("eval_count")),
        "prompt_eval_s": _q(nums("prompt_eval_s")),
        "eval_s": _q(nums("eval_s")),
        "load_s_max": max(nums("load_s"), default=None),
        "done_reason": dict(collections.Counter(str(s.get("done_reason")) for s in stats)),
    }


def progress(n: int, state: dict[str, Any], mem: dict[str, int | None]) -> str:
    """One line per incident, so a long run shows that it is moving (ADR 0013 §6)."""
    att = state.get("attribution") or {}
    llm = att.get("llm") or {}
    i2d = (state.get("latency_ms") or {}).get("incident_to_dispatch")
    parts = [
        f"[{n:4d}] {state.get('status')}",
        f"{att.get('attribution_mode')} {att.get('technique_id') or '-'}",
        f"dispatch {i2d / 1000:.1f}s" if i2d is not None else "no dispatch",
    ]
    if llm:
        parts.append(f"llm {float(llm.get('seconds') or 0.0):.1f}s {llm.get('kind', 'ok')}")
    parts.append(
        f"mem avail {mem.get('mem_available_mb')} MB, llm server {mem.get('llm_server_rss_mb')} MB"
    )
    return " | ".join(parts)


def summarise(
    store: SessionStore,
    results: list[dict[str, Any]],
    recoveries: list[dict[str, Any]] | None = None,
    l_llm_s: float = 24.0,
) -> dict[str, Any]:
    sessions = store.all_sessions()
    by_status = collections.Counter(s.status for s in sessions)
    atts = [r.get("attribution") or {} for r in results]
    by_mode = collections.Counter(a.get("attribution_mode", "none") for a in atts)
    techniques = collections.Counter(a.get("technique_id") for a in atts if a.get("technique_id"))
    rec = recoveries or []
    return {
        "sessions": len(sessions),
        "by_status": dict(by_status),
        "by_attribution_mode": dict(by_mode),
        "techniques": dict(techniques.most_common()),
        "llm": _llm_block(atts),
        "llm_recoveries": {
            "n": len(rec),
            "probe_ok": sum(1 for r in rec if r.get("probe_ok")),
            "seconds": _q([float(r["seconds"]) for r in rec]),
        },
        "hops": {
            "max_hop_count": max((s.hop_count for s in sessions), default=0),
            "total": store.count("hops"),
        },
        "actions": {
            "total": store.count("actions"),
            "by_status": dict(collections.Counter(a["status"] for a in store.actions())),
        },
        "latency_ms": {
            "investigation": _q(
                [
                    r["latency_ms"]["investigation"]
                    for r in results
                    if "investigation" in r.get("latency_ms", {})
                ]
            ),
            "incident_to_dispatch": _q(
                [
                    r["latency_ms"]["incident_to_dispatch"]
                    for r in results
                    if "incident_to_dispatch" in r.get("latency_ms", {})
                ]
            ),
        },
        "targets_ms": {
            "investigation_p95": 2000.0,
            "incident_to_dispatch_p95_retrieval_only": 3000.0,
            "incident_to_dispatch_p95_full": l_llm_s * 1000.0,  # PT7, unchanged (ADR 0013 §3.6)
        },
    }


def llm_settings(params: Params, mode: str, lock_path: Path = LOCK) -> dict[str, Any]:
    """The calibrated timeout and options of an LLM mode; refuses a missing or stale lock."""
    lock = read_lock(lock_path)
    entry = lock["modes"].get(mode)
    if entry is None and mode in lock.get("stopped", {}):
        why = "; ".join(lock["stopped"][mode]["reasons"])
        raise SystemExit(f"{mode} was stopped by ADR 0013 §6 on this machine ({why})")
    if entry is None:
        raise SystemExit(
            f"{mode} has no calibrated LLM timeout in {lock_path}: make llm-calibrate MODE={mode} "
            "(ADR 0013 §3.3)"
        )
    want = {
        "model": params.llm.model,
        "num_ctx": params.llm.ctx_for(mode),
        "num_predict": params.llm.num_predict,
        "temperature": params.llm.temperature,
    }
    stale = {k: (entry.get(k), v) for k, v in want.items() if entry.get(k) != v}
    if stale:
        raise SystemExit(f"{lock_path} was measured with other settings for {mode}: {stale}")
    return {**want, "timeout_s": float(entry["timeout_s"]), "lock": lock_path.as_posix()}


def _agent(
    params: Params,
    mode: str,
    kb: Path,
    retrieval_version: str | None,
    llm_cfg: dict[str, Any] | None = None,
) -> tuple[AttributionAgent, Any]:
    from agmesh.kb.retriever import version

    ver = version(retrieval_version)
    rank = None
    if mode in ("full", "retrieval_only"):
        from agmesh.kb.embed import SentenceEncoder
        from agmesh.kb.retriever import Retriever
        from agmesh.kb.store import QdrantStore

        retriever = Retriever(
            QdrantStore(kb / "qdrant"),
            SentenceEncoder(),
            exclude_only_tactics=ver["exclude_only_tactics"],
        )
        rank = retriever.rank
    llm = None
    if llm_cfg is not None:
        from agmesh.mesh.llm import OllamaLLM

        check_llm(params)
        llm = OllamaLLM(
            params.llm.base_url,
            params.llm.model,
            timeout_s=llm_cfg["timeout_s"],
            num_ctx=llm_cfg["num_ctx"],
            num_predict=llm_cfg["num_predict"],
            temperature=llm_cfg["temperature"],
        )
    agent = AttributionAgent(
        rank,
        llm,
        k=int(ver["candidates"]),
        exclude_only_tactics=ver["exclude_only_tactics"],
        min_confidence=params.mesh.attribution.min_confidence,
    )
    return agent, llm


def check_llm(params: Params) -> None:
    import httpx

    from agmesh.feasibility.llm_bench import is_installed

    try:
        tags = httpx.get(f"{params.llm.base_url}/api/tags", timeout=5).json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SystemExit(
            f"Ollama is not reachable at {params.llm.base_url} ({exc}); start it (ollama serve)"
        ) from exc
    names = [str(m.get("name")) for m in tags.get("models", [])]
    if not is_installed(params.llm.model, names):
        raise SystemExit(
            f"model {params.llm.model} is not installed: ollama pull {params.llm.model}"
        )


def main(argv: list[str] | None = None) -> int:
    load_env()
    from agmesh.kb.retriever import version

    parser = argparse.ArgumentParser(description="Run the agent mesh over an edge run's alerts")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--site", choices=("site_a", "site_b"), required=True)
    parser.add_argument("--mode", choices=MODES, default=None)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--edge-root", type=Path, default=Path("results/edge"))
    parser.add_argument("--edge-run", type=Path, default=None)
    parser.add_argument("--kb", type=Path, default=Path("data/kb"))
    parser.add_argument("--retrieval-version", default=None)
    parser.add_argument("--out", type=Path, default=Path("results/mesh"))
    parser.add_argument("--llm-lock", type=Path, default=LOCK)
    args = parser.parse_args(argv)
    params = load_params(args.params)
    mode = args.mode or params.mesh.attribution.mode
    llm_cfg = llm_settings(params, mode, args.llm_lock) if mode in LLM_MODES else None
    edge = args.edge_run or latest_edge_run(args.edge_root, args.site, args.split)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = args.out / f"{args.site}__{mode}__{args.split}__{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(edge / "storage.db", out / "storage.db")
    run_id = f"mesh__{args.site}__{mode}__{args.split}__{stamp}"
    agent, llm = _agent(params, mode, args.kb, args.retrieval_version, llm_cfg)
    store = SessionStore(out / "storage.db", max_hops=params.mesh.max_hops)
    bus = SqliteBus(out / "storage.db")
    events = out / "events.jsonl"
    mesh = build_mesh(params, store, bus, LocalNodeClient(edge), agent, events, run_id)
    cfg = params.fusion.site_b
    triage = Triage(
        cfg.delta_t_s, exclude_latency=cfg.exclude_latency and args.site == "site_b", k=cfg.k_of_n
    )

    def logged(state: dict[str, Any]) -> None:
        att = state.get("attribution") or {}
        mem = snapshot()
        write_event(
            events,
            "session",
            {
                "session_id": state.get("session_id"),
                "status": state.get("status"),
                "attribution_mode": att.get("attribution_mode"),
                "technique_id": att.get("technique_id"),
                "latency_ms": state.get("latency_ms"),
                "llm_seconds": (att.get("llm") or {}).get("seconds"),
                **mem,
            },
            run_id=run_id,
            data_source="runtime",
        )
        done[0] += 1
        print(progress(done[0], state, mem), flush=True)
        if llm is not None and "error" in (att.get("llm") or {}):
            # Between incidents, so it is not in any incident's latency (ADR 0013 §3.4).
            rec = llm.recover()
            recoveries.append(rec)
            write_event(events, "llm_recovery", rec, run_id=run_id, data_source="runtime")

    recoveries: list[dict[str, Any]] = []
    done = [0]
    server: dict[str, Any] | None = None
    if llm is not None:
        # ADR 0013 §7: load the model before the first incident, with the run's own
        # options, so incident 1 does not pay the cold start. Excluded from every metric.
        warm = llm.recover()
        server = llm_server_info()
        write_event(
            events, "llm_warmup", {**warm, "server": server}, run_id=run_id, data_source="runtime"
        )
        print(f"warm-up {warm['seconds']:.1f}s probe_ok={warm['probe_ok']}", flush=True)
        if server is None or server.get("cache_ram") != "0":
            print(
                "WARNING: the LLM server's prompt cache is not visibly off "
                f"(cache_ram={None if server is None else server.get('cache_ram')}); "
                "ADR 0013 §7 runs need LLAMA_ARG_CACHE_RAM=0",
                flush=True,
            )
    print(f"{args.site} {mode}: running; one line per incident", flush=True)
    started = time.perf_counter()
    results = consume(
        mesh,
        bus,
        triage,
        args.site,
        mode,
        run_id,
        on_session=logged,
        worker_timeout_s=params.mesh.investigation_deadline_s
        + (llm_cfg["timeout_s"] if llm_cfg else 0.0)
        + 30,
    )
    payload = {
        "run_id": run_id,
        "site": args.site,
        "mode": mode,
        "split": args.split,
        "edge_run": edge.as_posix(),
        "retrieval_version": version(args.retrieval_version)["name"],
        "llm_settings": llm_cfg,
        "llm_server": server,
        "wall_seconds": round(time.perf_counter() - started, 1),
        **summarise(store, results, recoveries, params.mesh.attribution.l_llm_s),
    }
    path = write_artifact(
        out / "summary.json",
        kind="mesh.summary",
        data_source="runtime",
        payload=payload,
        params_path=args.params,
    )
    print(f"{args.site} {mode}: {payload['sessions']} sessions {payload['by_status']} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
