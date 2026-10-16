"""Retrieval evaluation: Recall@5 and MRR on validation incidents (Plan Week 9 task 6).

    python -m agmesh.evaluation.retrieval_eval --split val --queries-only   # render queries
    python -m agmesh.evaluation.retrieval_eval --split val                  # needs `make kb`
    make retrieval-eval

The protocol is ADR 0010 §4-5.

**Queries.**

* Each validation incident is rendered from the evidence records that the edge
  runtime stored for it:
  * Site A: the 15 events with ``partition = val_attack`` in the frozen mapping;
  * Site B: episodes 0-3.
* The labels set only the incident's time window. A missed incident has no records,
  produces no query, and is counted.
* The 30 paraphrases in ``configs/kb/paraphrases.yaml`` inherit their incident's
  acceptable set.

**Relevance.** A retrieved technique is relevant if its canonical ID is in the
incident's acceptable set.

**Outputs.**

* The first run with the reference encoder writes ``results/retrieval/val__pt6.json``,
  the PT6 measurement.
* Later runs write ``val__exploratory__<time>.json``.
* The hashing encoder is refused unless ``--fixture``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from agmesh.common.artifacts import write_artifact
from agmesh.common.env import load_env
from agmesh.common.schemas import EvidenceRecord
from agmesh.data.manifest import read_manifest
from agmesh.edge.evidence import iso
from agmesh.edge.evidence_store import EvidenceStore
from agmesh.evaluation import attack_ics
from agmesh.evaluation.ground_truth import OUT_DIR
from agmesh.kb.render import render
from agmesh.mesh.fusion import parse_iso

PARAPHRASES = Path("configs/kb/paraphrases.yaml")
KS = (1, 3, 5, 10)
TOLERANCE_S = 60.0


@dataclass(frozen=True)
class Query:
    query_id: str
    incident: str  # "site_a/<segment>" or "site_b/<episode>"
    site: str
    kind: str  # "rendered" | "paraphrase"
    text: str
    relevant: tuple[str, ...]
    stratum: str | None = None  # Site A: "sp" or "co" led (primary T0836 or not)


def _latest(root: Path, site: str, split: str) -> Path:
    runs = sorted(p for p in root.glob(f"{site}__{split}__*") if any(p.glob("evidence_*.db")))
    if not runs:
        raise SystemExit(f"no edge run with evidence stores for {site} {split} under {root}")
    return runs[-1]


def _records(run: Path, node: str, start_s: float, end_s: float) -> list[EvidenceRecord]:
    path = run / f"evidence_{node}.db"
    if not path.is_file():
        return []
    store = EvidenceStore(path)
    try:
        found = store.between(node, iso(start_s), iso(end_s + 3600))
    finally:
        store.close()
    # windows overlapping [start, end]
    return [
        r for r in found if parse_iso(r.window_end) > start_s and parse_iso(r.window_start) < end_s
    ]


def incidents(
    edge_root: Path, partitions: Path, split: str = "val"
) -> tuple[list[Query], dict[str, Any]]:
    """Rendered queries for the validation incidents, and the count of missed ones."""
    if split != "val":
        raise SystemExit("retrieval queries are built from validation incidents only (ADR 0010 §4)")
    queries: list[Query] = []
    missed: list[str] = []
    gt_a = json.loads((OUT_DIR / "mapping_site_a.json").read_text(encoding="utf-8"))["events"]
    run_a = _latest(edge_root, "site_a", split)
    for seg, e in sorted(gt_a.items()):
        if e["partition"] != "val_attack":
            continue
        start = parse_iso(e["start"].replace(" ", "T") + "Z")
        recs = _records(run_a, e["client"], start, start + e["labelled_s"] + TOLERANCE_S)
        inc = f"site_a/{seg}"
        if not recs:
            missed.append(inc)
            continue
        queries.append(
            Query(
                f"r:{inc}",
                inc,
                "site_a",
                "rendered",
                render("site_a", recs),
                tuple(e["acceptable"]),
                "sp" if e["primary"] == "T0836" else "co",
            )
        )
    gt_b = json.loads((OUT_DIR / "mapping_site_b.json").read_text(encoding="utf-8"))["events"]
    run_b = _latest(edge_root, "site_b", split)
    episodes = {str(ep["episode"]): ep for ep in read_manifest(partitions / "site_b")["episodes"]}
    for epi in ("0", "1", "2", "3"):
        ep = episodes[epi]
        start, end = parse_iso(ep["start"]), parse_iso(ep["end"]) + TOLERANCE_S
        recs = [r for node in ("b1", "b2", "b3") for r in _records(run_b, node, start, end)]
        inc = f"site_b/{epi}"
        if not recs:
            missed.append(inc)
            continue
        queries.append(
            Query(
                f"r:{inc}",
                inc,
                "site_b",
                "rendered",
                render("site_b", recs),
                tuple(gt_b[epi]["acceptable"]),
            )
        )
    meta = {
        "edge_runs": {"site_a": run_a.as_posix(), "site_b": run_b.as_posix()},
        "missed_incidents": missed,
    }
    return queries, meta


def paraphrases(rendered: list[Query], path: Path = PARAPHRASES) -> list[Query]:
    if not path.is_file():
        return []
    by_incident = {q.incident: q for q in rendered}
    out = []
    for item in yaml.safe_load(path.read_text(encoding="utf-8"))["paraphrases"]:
        base = by_incident.get(item["incident"])
        if base is None:  # the incident was missed in this run: no relevant evidence to paraphrase
            continue
        out.append(
            Query(
                f"p:{item['id']}",
                base.incident,
                base.site,
                "paraphrase",
                item["text"].strip(),
                base.relevant,
                base.stratum,
            )
        )
    return out


def ranks(ranked_ids: list[str], relevant: tuple[str, ...]) -> int | None:
    """1-based rank of the first relevant technique (canonical IDs), or None."""
    rel = {attack_ics.canonical(t) for t in relevant}
    for i, tid in enumerate(ranked_ids, start=1):
        if attack_ics.canonical(tid) in rel:
            return i
    return None


def metrics(rank_list: list[int | None]) -> dict[str, float | int]:
    n = len(rank_list)
    out: dict[str, float | int] = {"queries": n}
    for k in KS:
        out[f"recall@{k}"] = (
            float(np.mean([r is not None and r <= k for r in rank_list])) if n else float("nan")
        )
    out["mrr"] = float(np.mean([1.0 / r if r else 0.0 for r in rank_list])) if n else float("nan")
    return out


def evaluate(queries: list[Query], rank_fn: Any) -> dict[str, Any]:
    per_query = []
    for q in queries:
        hits = rank_fn(q.text)
        r = ranks([h.technique_id for h in hits], q.relevant)
        per_query.append(
            {
                **asdict(q),
                "rank": r,
                "top5": [h.technique_id for h in hits[:5]],
                "top5_scores": [round(h.score, 4) for h in hits[:5]],
            }
        )

    def subset(pred: Any) -> dict[str, float | int]:
        return metrics([p["rank"] for p in per_query if pred(p)])

    return {
        "all": subset(lambda p: True),
        "by_kind": {k: subset(lambda p, k=k: p["kind"] == k) for k in ("rendered", "paraphrase")},
        "by_site": {s: subset(lambda p, s=s: p["site"] == s) for s in ("site_a", "site_b")},
        "site_a_by_stratum": {
            s: subset(lambda p, s=s: p["site"] == "site_a" and p["stratum"] == s)
            for s in ("sp", "co")
        },
        "per_query": per_query,
    }


def main(argv: list[str] | None = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(description="Retrieval evaluation (Recall@5, MRR)")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--edge-root", type=Path, default=Path("results/edge"))
    parser.add_argument("--partitions", type=Path, default=Path("data/partitions"))
    parser.add_argument("--kb", type=Path, default=Path("data/kb"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--encoder", choices=("reference", "hashing"), default="reference")
    parser.add_argument(
        "--queries-only", action="store_true", help="write the rendered queries and stop"
    )
    parser.add_argument(
        "--version", default="v1", help="retrieval version in configs/kb/retrieval.yaml (v1: PT6)"
    )
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    rendered, meta = incidents(args.edge_root, args.partitions, args.split)
    queries = rendered + paraphrases(rendered)
    out_dir = args.results / "retrieval"
    if args.queries_only:
        path = write_artifact(
            out_dir / f"queries_{args.split}.json",
            kind="retrieval.queries",
            data_source="fixture" if args.fixture else "runtime",
            payload={**meta, "queries": [asdict(q) for q in queries]},
        )
        print(
            f"{len(rendered)} rendered + {len(queries) - len(rendered)} paraphrase queries, "
            f"{len(meta['missed_incidents'])} missed -> {path}"
        )
        return 0
    if args.encoder == "hashing" and not args.fixture:
        raise SystemExit(
            "the hashing encoder is for tests only; it cannot report a result (ADR 0010 §5)"
        )
    from agmesh.kb.embed import HashingEncoder, SentenceEncoder
    from agmesh.kb.retriever import Retriever, version
    from agmesh.kb.store import QdrantStore

    ver = version(args.version)
    encoder = SentenceEncoder() if args.encoder == "reference" else HashingEncoder()
    store = QdrantStore(args.kb / "qdrant")
    retriever = Retriever(store, encoder, exclude_only_tactics=ver["exclude_only_tactics"])
    result = evaluate(queries, retriever.rank)
    store.close()
    pt6 = out_dir / f"{args.split}__pt6.json"
    official = (
        ver["name"] == "v1"
        and args.encoder == "reference"
        and not args.fixture
        and not pt6.exists()
    )
    tag = "" if ver["name"] == "v1" else f"{ver['name']}__"
    path = (
        pt6
        if official
        else out_dir / f"{args.split}__{tag}exploratory__{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    path = write_artifact(
        path,
        kind="retrieval.eval",
        data_source="fixture" if args.fixture else "runtime",
        payload={
            **meta,
            "protocol": ver["adr"],
            "retrieval_version": ver,
            "candidates_excluded": sorted(retriever.excluded),
            "pt6_measurement": official,
            "target": (
                {"pt6": {"recall@5": 0.85}}
                if ver["name"] == "v1"
                else {
                    "pt6_v2": {"recall@10": 0.85},
                    "measured_on": "test rendered queries, once after g4-frozen (prereg. 3b)",
                    "this_run": "exploratory" if not official else "PT6",
                }
            ),
            "encoder": encoder.identity(),
            "kb_manifest": (args.results / "kb" / "manifest.json").as_posix(),
            **result,
        },
    )
    a = result["all"]
    print(
        f"{ver['name']}: Recall@5 {a['recall@5']:.3f}  Recall@10 {a['recall@10']:.3f}  "
        f"MRR {a['mrr']:.3f}  ({a['queries']} queries; PT6 measurement: {official}) -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
