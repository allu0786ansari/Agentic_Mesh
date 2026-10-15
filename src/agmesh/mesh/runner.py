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

* ``full`` and ``llm_no_retrieval`` need Ollama with ``llm.model``. The runner checks
  this before starting.
* Every mode except ``rule_only`` needs the knowledge base (``make kb``).

**Output.** ``summary.json`` holds:

* sessions by status and attribution mode;
* the technique distribution and the grounding rate;
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
from agmesh.mesh.attribution import MODES, AttributionAgent
from agmesh.mesh.breaker import CircuitBreaker
from agmesh.mesh.graph import Mesh
from agmesh.mesh.investigator import Investigator
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


def summarise(store: SessionStore, results: list[dict[str, Any]]) -> dict[str, Any]:
    sessions = store.all_sessions()
    by_status = collections.Counter(s.status for s in sessions)
    atts = [r.get("attribution") or {} for r in results]
    by_mode = collections.Counter(a.get("attribution_mode", "none") for a in atts)
    techniques = collections.Counter(a.get("technique_id") for a in atts if a.get("technique_id"))
    llm_calls = [a for a in atts if a.get("llm") and "validation" in (a.get("llm") or {})]
    grounded = [bool(a["llm"]["validation"]["grounded"]) for a in llm_calls]
    llm_s = [float(a["llm"]["seconds"]) for a in llm_calls]
    return {
        "sessions": len(sessions),
        "by_status": dict(by_status),
        "by_attribution_mode": dict(by_mode),
        "techniques": dict(techniques.most_common()),
        "llm": {
            "calls": len(llm_calls),
            "raw_grounding_rate": float(np.mean(grounded)) if grounded else None,
            "valid_rate": float(np.mean([a["llm"]["validation"]["valid"] for a in llm_calls]))
            if llm_calls
            else None,
            "seconds": _q(llm_s),
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
        },
    }


def _agent(params: Params, mode: str, kb: Path, retrieval_version: str | None) -> AttributionAgent:
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
    if mode in ("full", "llm_no_retrieval"):
        from agmesh.mesh.llm import OllamaLLM

        llm = OllamaLLM(
            params.llm.base_url,
            params.llm.model,
            timeout_s=params.mesh.attribution.llm_timeout_s,
            num_ctx=params.llm.num_ctx,
            temperature=params.llm.temperature,
        )
        _check_llm(params)
    return AttributionAgent(
        rank,
        llm,
        k=int(ver["candidates"]),
        exclude_only_tactics=ver["exclude_only_tactics"],
        min_confidence=params.mesh.attribution.min_confidence,
    )


def _check_llm(params: Params) -> None:
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
    args = parser.parse_args(argv)
    params = load_params(args.params)
    mode = args.mode or params.mesh.attribution.mode
    edge = args.edge_run or latest_edge_run(args.edge_root, args.site, args.split)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out = args.out / f"{args.site}__{mode}__{args.split}__{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(edge / "storage.db", out / "storage.db")
    run_id = f"mesh__{args.site}__{mode}__{args.split}__{stamp}"
    agent = _agent(params, mode, args.kb, args.retrieval_version)
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
        write_event(
            events,
            "session",
            {
                "session_id": state.get("session_id"),
                "status": state.get("status"),
                "attribution_mode": att.get("attribution_mode"),
                "technique_id": att.get("technique_id"),
                "latency_ms": state.get("latency_ms"),
            },
            run_id=run_id,
            data_source="runtime",
        )

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
        + params.mesh.attribution.llm_timeout_s
        + 30,
    )
    payload = {
        "run_id": run_id,
        "site": args.site,
        "mode": mode,
        "split": args.split,
        "edge_run": edge.as_posix(),
        "retrieval_version": args.retrieval_version or "default",
        "wall_seconds": round(time.perf_counter() - started, 1),
        **summarise(store, results),
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
