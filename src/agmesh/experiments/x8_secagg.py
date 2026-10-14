"""X8: SecAgg+ on Fed-A, a systems experiment (Plan Week 7; Proposal §7.4).

    python -m agmesh.experiments.x8_secagg            # -> results/X8/secagg/seed_<n>.json

Three arms on Fed-A (Site A, 4 clients, DP-SGD at ε = 10, ``--rounds`` rounds; the
privacy plan is calibrated for those rounds):

* ``plaintext``    Flower's default fit workflow;
* ``secagg``       ``SecAggPlusWorkflow``: 4 shares, reconstruction threshold 3;
* ``secagg_drop``  as ``secagg``, with client a4 failing during round 2 after the key
  exchange.

For every round of every arm, the federated aggregate is compared with the uniform
mean of that round's plaintext client updates, which the clients write to a temporary
side channel. Reported:

* **correctness:** the maximum and mean absolute error against the quantisation step
  2c/(Q-1), and the share of parameters outside the clipping range ±c (those would
  be clipped);
* **overhead:** seconds per round, and payload bytes per client per round by stage,
  against plaintext;
* **dropout tolerance:** whether the round with a drop aggregates the survivors' mean.

The normalisation round runs in-process first; it is DP and accounted. Its pooled
scaler is handed to the Flower clients. Validation data is not read: X8 measures
aggregation, not detection.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.common.provenance import sha256_file, utc_now_iso
from agmesh.data.access import Site
from agmesh.evaluation.results import write_result
from agmesh.experiments import x1_baselines as x1
from agmesh.federated import secagg
from agmesh.federated.backends import ClientSettings, InProcessPool
from agmesh.federated.server import initial_weights, pooled_normalisation

SITE: Site = "site_a"
CLIP = 8.0
QUANT = 2**22
DROP = ["a4", 2]


def _plaintext_mean(record: Path, rnd: int, nodes: list[str]) -> tuple[list[np.ndarray], int]:
    updates = []
    for node in nodes:
        path = record / f"update_r{rnd}_{node}.npz"
        if path.is_file():
            with np.load(path) as z:
                updates.append([z[k] for k in sorted(z.files, key=lambda k: int(k.split("_")[1]))])
    mean = [np.mean(np.stack([u[i] for u in updates]), axis=0) for i in range(len(updates[0]))]
    return mean, len(updates)


def _bytes(record: Path, nodes: list[str]) -> dict[str, Any]:
    per_stage: dict[str, list[int]] = collections.defaultdict(list)
    for node in nodes:
        path = record / f"bytes_{node}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            per_stage[row["stage"]].append(int(row["in"]) + int(row["out"]))
    total = sum(sum(v) for v in per_stage.values())
    return {"total": total, "by_stage": {k: int(sum(v)) for k, v in sorted(per_stage.items())}}


def run_arm(
    params: Params,
    params_path: Path,
    partitions: Path,
    *,
    arm: str,
    rounds: int,
    seed: int,
    privacy_root: Path,
) -> dict[str, Any]:
    nodes = x1._clients(params, SITE)
    run_id = f"X8__{arm}__seed{seed}__{utc_now_iso().replace(':', '')}"
    record = Path(tempfile.mkdtemp(prefix=f"x8_{arm}_"))
    try:
        pool = InProcessPool(
            ClientSettings(
                params,
                SITE,
                partitions,
                run_id,
                10.0,
                seed,
                epochs=rounds,
                privacy_root=privacy_root,
            ),
            nodes,
        )
        hello = [pool.hello(n) for n in nodes]
        norm = pooled_normalisation(hello)
        np.savez(record / "norm.npz", centre=norm.centre, scale=norm.scale)
        initial = initial_weights(hello[0].spec, seed)
        settings = secagg.SecAggSettings(
            str(params_path),
            SITE,
            str(partitions),
            run_id,
            10.0,
            seed,
            nodes,
            rounds,
            str(record),
            str(privacy_root),
            arm != "plaintext",
            [DROP] if arm == "secagg_drop" else [],
            num_shares=len(nodes),
            reconstruction_threshold=len(nodes) - 1,
            clipping_range=CLIP,
            quantization_range=QUANT,
        )
        res = params.federated.client_resources
        out = secagg.run(
            settings,
            initial,
            num_cpus=res.num_cpus,
            total_cpus=max(int(res.num_cpus), os.cpu_count() or 2),
        )
        step = 2 * CLIP / (QUANT - 1)
        per_round = []
        for rnd in range(1, rounds + 1):
            agg = out["aggregates"].get(rnd)
            mean, n_updates = _plaintext_mean(record, rnd, nodes)
            row: dict[str, Any] = {
                "round": rnd,
                "updates": n_updates,
                "aggregated": agg is not None,
            }
            if agg is not None:
                err = np.concatenate(
                    [np.abs(a - b).ravel() for a, b in zip(agg, mean, strict=True)]
                )
                flat = np.concatenate([m.ravel() for m in mean])
                row.update(
                    max_abs_error=float(err.max()),
                    mean_abs_error=float(err.mean()),
                    outside_clip_share=float((np.abs(flat) > CLIP).mean()),
                )
            row["seconds"] = float(out["round_seconds"].get(rnd, math.nan))
            per_round.append(row)
        return {
            "arm": arm,
            "run_id": run_id,
            "rounds": per_round,
            "quantisation_step": step,
            "bytes": _bytes(record, nodes),
            "wall_seconds": out["wall_seconds"],
            "parameters": int(sum(a.size for a in initial)),
        }
    finally:
        shutil.rmtree(record, ignore_errors=True)


def summarise(arms: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    plain, sec = arms["plaintext"], arms["secagg"]
    step = sec["quantisation_step"]
    errs = [
        r["max_abs_error"]
        for a in ("secagg", "secagg_drop")
        for r in arms[a]["rounds"]
        if "max_abs_error" in r
    ]
    plain_err = max(r.get("max_abs_error", 0.0) for r in plain["rounds"])
    drop_round = next(r for r in arms["secagg_drop"]["rounds"] if r["round"] == DROP[1])
    t_plain = np.nanmean([r["seconds"] for r in plain["rounds"]])
    t_sec = np.nanmean([r["seconds"] for r in sec["rounds"]])
    return {
        "max_abs_error": float(max(errs)) if errs else None,
        "quantisation_step": step,
        "error_within_step": float(max(errs) <= step) if errs else 0.0,
        "plaintext_max_abs_error": float(plain_err),
        "seconds_per_round_plaintext": float(t_plain),
        "seconds_per_round_secagg": float(t_sec),
        "time_overhead_ratio": float(t_sec / t_plain) if t_plain else None,
        "bytes_per_round_plaintext": plain["bytes"]["total"] / len(plain["rounds"]),
        "bytes_per_round_secagg": sec["bytes"]["total"] / len(sec["rounds"]),
        "bytes_overhead_ratio": (sec["bytes"]["total"] / plain["bytes"]["total"])
        if plain["bytes"]["total"]
        else None,
        "dropout_recovered": float(
            bool(drop_round.get("aggregated"))
            and drop_round.get("updates") == 3
            and drop_round.get("max_abs_error", math.inf) <= step
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X8: SecAgg+ systems experiment on Fed-A")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    params = load_params(args.params)
    partitions = args.partitions or params.paths.partitions_dir
    seed = params.project.seeds[0] if args.seed is None else args.seed
    started = utc_now_iso()
    arms = {
        arm: run_arm(
            params,
            args.params,
            partitions,
            arm=arm,
            rounds=args.rounds,
            seed=seed,
            privacy_root=args.results / "privacy",
        )
        for arm in ("plaintext", "secagg", "secagg_drop")
    }
    metrics = summarise(arms)
    path = write_result(
        args.results,
        experiment="X8",
        config_id="secagg",
        seed=seed,
        split="val",
        data_source="fixture" if args.fixture else "runtime",
        metrics=metrics,
        manifest_sha256=sha256_file(partitions / SITE / "manifest.json"),
        started_at=started,
        extra={
            "site": SITE,
            "arms": arms,
            "secagg": {
                "num_shares": 4,
                "reconstruction_threshold": 3,
                "clipping_range": CLIP,
                "quantization_range": QUANT,
                "max_weight": 1.0,
                "dropped": DROP,
            },
        },
        params_path=args.params,
    )
    print(json.dumps(metrics, indent=1))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
