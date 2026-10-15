"""Edge runtime: replay → score → evidence → alert (Plan Week 8; ADR 0009).

    python -m agmesh.edge.runtime --site site_b --model <dir> --split val
    make edge SITE=site_b SPLIT=val MODEL=<dir>

For one site, the runtime:

1. **Loads the frozen global model** (``GlobalModel``) and checks its version.
2. **Fits each node's local calibration** on that node's own data:
   - references and threshold;
   - calibrator and magnitude buckets;
   - Site B's request-rate bucket.

   The calibration is written to ``calibration_<node>.json``.
3. **Replays the split**, merging all nodes' windows in time order (``replay.Pacer``).
4. **Scores each window on its own** (``stream.NodeScorer``). Above the node's
   threshold it:
   - builds the minimised evidence record and stores it in that node's store
     (``evidence_<node>.db``);
   - publishes the alert to ``alerts.<site>`` in ``storage.db``;
   - writes a ``detection`` metric event with the detection latency.
5. **Writes ``summary.json``.** It reports:
   - per-node window, alert and false-alarm counts;
   - the streaming-versus-batch check;
   - detection latency p50 and p95;
   - evidence statistics;
   - for Site B, the descriptive fusion sensitivity table (ADR 0009 §7).

Labels are read only to count, in the summary, which alerts fell in attack windows.
They never reach a record or an alert. The ``test`` split needs ``--final``, which
requires the ``g4-frozen`` tag and is logged.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agmesh.bus.sqlite_bus import SqliteBus
from agmesh.common.artifacts import write_artifact, write_event
from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.common.schemas import check_features
from agmesh.data.access import Site, load_partition
from agmesh.data.manifest import read_manifest
from agmesh.edge import evidence as ev
from agmesh.edge.bundle import GlobalModel
from agmesh.edge.calibrate import calibrate_node
from agmesh.edge.evidence_store import EvidenceStore
from agmesh.edge.replay import Pacer
from agmesh.edge.stream import NodeScorer, WindowSource, merged
from agmesh.evaluation.calibration import IsotonicCalibrator
from agmesh.evaluation.sealed import FinalAccess, add_final_argument
from agmesh.experiments import x1_baselines as x1
from agmesh.mesh import fusion

SPLITS = {"val": ("val_normal", "val_attack"), "test": ("test",)}


def _constants(params: Params, site: Site, root: Path) -> dict[str, float]:
    if site != "site_a":
        return {}
    manifest = read_manifest(root / site)
    return {k: float(v) for k, v in manifest["features"]["training_constant"].items()}


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    a = np.asarray(values)
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def run(
    params: Params,
    site: Site,
    model_dir: Path,
    partitions: Path,
    out_root: Path,
    *,
    split: str = "val",
    speed: float = 0.0,
    max_gap_s: float = 60.0,
    final: FinalAccess | None = None,
    data_source: str = "runtime",
    params_path: Path = DEFAULT_PARAMS_PATH,
) -> Path:
    if split == "test" and final is None:
        raise SystemExit("the test split needs --final (and the g4-frozen tag)")
    started = time.strftime("%Y%m%dT%H%M%S")
    model = GlobalModel.load(model_dir)
    if model.site != site:
        raise SystemExit(f"{model_dir} is a {model.site} model, not {site}")
    run_id = f"edge__{site}__{split}__{model.version[7:19]}__{started}"
    out = out_root / f"{site}__{split}__{started}"
    out.mkdir(parents=True, exist_ok=True)
    net = model.network()
    constants = _constants(params, site, partitions)
    sources: list[WindowSource] = []
    scorers: dict[str, NodeScorer] = {}
    calibrators: dict[str, IsotonicCalibrator] = {}
    calibrations = {}
    batch: dict[tuple[str, str], Any] = {}
    for node in x1._clients(params, site):
        data = x1.NodeData(
            {
                p: load_partition(params, site, node, p, root=partitions)
                for p in ("train", "val_normal", "val_attack")
            }
        )
        fitted = calibrate_node(site, node, model, data, params, constants, net)
        fitted.calibration.save(out / f"calibration_{node}.json")
        calibrations[node] = fitted.calibration
        calibrators[node] = IsotonicCalibrator.from_dict(fitted.calibration.calibrator)
        scorers[node] = NodeScorer(site, fitted.calibration, net, params)
        for part in SPLITS[split]:
            partition = (
                data.parts[part]
                if part in data.parts
                else load_partition(params, site, node, part, root=partitions, final=final)
            )
            batch[(node, part)] = fitted.batch.get(part)
            consts = None
            if site == "site_a":
                frame = pd.read_parquet(
                    partitions / site / node / f"{part}.parquet", columns=list(constants)
                )
                consts = frame[list(constants)].to_numpy(dtype=np.float64)
            sources.append(WindowSource(site, node, part, partition, fitted.scaler, params, consts))

    bus = SqliteBus(out / "storage.db")
    stores = {node: EvidenceStore(out / f"evidence_{node}.db") for node in scorers}
    events_path = out / "events.jsonl"
    topic = f"alerts.{site}"
    pacer = Pacer(speed, max_gap_s)
    stream_scores = {(s.node, s.part): np.full(len(s), np.nan) for s in sources}
    stream_ok = {(s.node, s.part): np.zeros(len(s), bool) for s in sources}
    counts: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    latencies: list[float] = []
    processing: list[float] = []
    magnitudes: collections.Counter[str] = collections.Counter()
    feature_freq: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    fusion_alerts: dict[str, list[fusion.FusionAlert]] = collections.defaultdict(list)
    allowed = {node: [*calibrations[node].features, *constants] for node in scorers}
    wall_start = time.perf_counter()
    for src, k in merged(sources):
        emitted = pacer.emit(src.t_end(k))
        w = src.window(k)
        s = scorers[src.node].score(w, src.stride)
        key = (src.node, src.part)
        stream_scores[key][k] = s.score
        stream_ok[key][k] = s.scored
        c = counts[key]
        c["windows"] += 1
        c["scored"] += int(s.scored)
        is_attack = src.label(k)
        c["attack_windows"] += int(is_attack)
        cal = calibrations[src.node]
        if s.scored and s.score > cal.threshold:
            feats = ev.top_features(s.feature_errors, s.residuals, cal.features, cal.buckets)
            summary = (
                ev.protocol_summary(w.counters, cal.request_rate)  # type: ignore[arg-type]
                if site == "site_b"
                else None
            )
            changed = (
                ev.constants_changed(w.constants, list(constants), constants)
                if site == "site_a" and w.constants is not None
                else []
            )
            prob = float(calibrators[src.node].predict([s.score])[0])
            record, alert = ev.build(
                site=site,
                node=src.node,
                window_start_s=w.t_start,
                window_end_s=w.t_end,
                model_version=model.version,
                score=s.score,
                threshold=cal.threshold,
                calibrated_prob=prob,
                features=feats,
                run_id=run_id,
                data_source=data_source,
                summary=summary,
                changed=changed,
            )
            check_features(record, allowed[src.node])
            stores[src.node].put(record)
            bus.publish(topic, alert.model_dump(), alert.alert_id)
            published = time.perf_counter()
            latency_ms = 1000.0 * (published - emitted)
            latencies.append(latency_ms)
            write_event(
                events_path,
                "detection",
                {
                    "node": src.node,
                    "part": src.part,
                    "alert_id": alert.alert_id,
                    "latency_ms": latency_ms,
                },
                run_id=run_id,
                data_source=data_source,  # type: ignore[arg-type]
            )
            c["alerts"] += 1
            c["alerts_on_attack"] += int(is_attack)
            c["alerts_empty_features"] += int(not feats)
            c["alerts_latency_only"] += int(alert.latency_only)
            c["alerts_constants_changed"] += int(bool(changed))
            magnitudes.update(f.magnitude for f in feats)
            feature_freq[src.part].update(f.name for f in feats)
            fusion_alerts[src.part].append(
                fusion.FusionAlert(alert.alert_id, src.node, w.t_end, prob, alert.latency_only)
            )
        processing.append(1000.0 * (time.perf_counter() - emitted))
    wall = time.perf_counter() - wall_start

    per_stream: dict[str, Any] = {}
    agreement: dict[str, Any] = {}
    for (node, part), c in sorted(counts.items()):
        normal = c["windows"] - c["attack_windows"]
        per_stream[f"{node}/{part}"] = {
            **dict(c),
            "false_alarms": c["alerts"] - c["alerts_on_attack"],
            "window_fpr_normal": (c["alerts"] - c["alerts_on_attack"]) / normal if normal else None,
        }
        ref = batch.get((node, part))
        if ref is not None:
            ss, ok = stream_scores[(node, part)], stream_ok[(node, part)]
            both = ok & ref.scored
            diff = np.abs(ss[both] - ref.scores[both])
            rel = diff / np.maximum(np.abs(ref.scores[both]), 1e-12)
            thr = calibrations[node].threshold
            s_alert = ok & (np.nan_to_num(ss, nan=-np.inf) > thr)
            b_alert = ref.scored & (np.nan_to_num(ref.scores, nan=-np.inf) > thr)
            flipped = s_alert != b_alert
            margin = np.abs(ref.scores[flipped] - thr) / max(abs(thr), 1e-12)
            agreement[f"{node}/{part}"] = {
                "windows": int(ss.size),
                "scored_mask_mismatches": int((ok != ref.scored).sum()),
                "max_abs_diff": float(diff.max()) if diff.size else 0.0,
                "max_rel_diff": float(rel.max()) if rel.size else 0.0,
                "alert_decision_mismatches": int(flipped.sum()),
                # how far each flipped window's batch score was from the threshold
                "mismatch_max_rel_margin": float(margin.max()) if margin.size else None,
            }
    fusion_block: dict[str, Any] = {}
    if site == "site_b":
        cfg = params.fusion.site_b
        for part, alerts in fusion_alerts.items():
            found = fusion.incidents(alerts, cfg.delta_t_s, exclude_latency=cfg.exclude_latency)
            fusion_block[part] = {
                "configured": {
                    "delta_t_s": cfg.delta_t_s,
                    "k": cfg.k_of_n,
                    "exclude_latency": cfg.exclude_latency,
                },
                "alerts": len(alerts),
                "incidents": len(found),
                "confirmed_k": sum(i.confirmed(cfg.k_of_n) for i in found),
                "nodes_per_incident": dict(collections.Counter(i.n_nodes for i in found)),
                "sensitivity_descriptive_only": fusion.sensitivity(
                    alerts, exclude_latency=cfg.exclude_latency
                ).rows,
            }
    bus_stats = bus.stats(topic)
    payload = {
        "run_id": run_id,
        "site": site,
        "split": split,
        "model": {
            "path": model_dir.as_posix(),
            "model_version": model.version,
            "source": model.source,
        },
        "replay": {
            "speed": speed,
            "max_gap_s": max_gap_s,
            "wall_seconds": round(wall, 2),
            "slept_s": round(pacer.slept_s, 2),
        },
        "nodes": {
            n: {
                "threshold": c.threshold,
                "threshold_val_fpr": c.threshold_val_fpr,
                "shift": c.shift,
            }
            for n, c in calibrations.items()
        },
        "streams": per_stream,
        "stream_vs_batch": agreement,
        "detection_latency_ms": _quantiles(latencies),
        "window_processing_ms": _quantiles(processing),
        "latency_target_ms": 1000.0,
        "evidence": {
            "records": {n: st.count() for n, st in stores.items()},
            "magnitudes": dict(magnitudes),
            "top_feature_frequency": {p: dict(c.most_common(10)) for p, c in feature_freq.items()},
        },
        "bus": bus_stats,
        "fusion": fusion_block,
        "outputs": {"dir": out.as_posix(), "storage": "storage.db", "events": "events.jsonl"},
    }
    for st in stores.values():
        st.close()
    bus.close()
    return write_artifact(
        out / "summary.json",
        kind="edge.summary",
        data_source=data_source,  # type: ignore[arg-type]
        payload=payload,
        params_path=params_path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Edge runtime: replay, score, evidence, alerts")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--site", choices=("site_a", "site_b"), required=True)
    parser.add_argument("--model", type=Path, required=True, help="a --save-model directory")
    parser.add_argument("--split", choices=tuple(SPLITS), default="val")
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/edge"))
    parser.add_argument("--speed", type=float, default=0.0, help="0 = as fast as possible")
    parser.add_argument("--max-gap-s", type=float, default=60.0)
    parser.add_argument("--fixture", action="store_true")
    add_final_argument(parser)
    args = parser.parse_args(argv)
    params = load_params(args.params)
    final = None
    if args.final:
        final = FinalAccess(f"edge__{args.site}", 0, params_path=args.params)
    path = run(
        params,
        args.site,
        args.model,
        args.partitions or params.paths.partitions_dir,
        args.out,
        split="test" if args.final else args.split,
        speed=args.speed,
        max_gap_s=args.max_gap_s,
        final=final,
        data_source="fixture" if args.fixture else "runtime",
        params_path=args.params,
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    lat = summary["detection_latency_ms"]
    alerts = sum(s.get("alerts", 0) for s in summary["streams"].values())
    print(
        f"{args.site} {summary['split']}: {alerts} alerts, detection latency p50 {lat['p50']} ms "
        f"p95 {lat['p95']} ms -> {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
