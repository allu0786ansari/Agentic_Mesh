"""Robustness suite: synthetic perturbations of normal validation (ADR 0017 §7).

    python -m agmesh.experiments.robustness                  # make robustness, five seeds
    python -m agmesh.experiments.robustness --models results/models --seeds 11 --out <dir>

**Perturbations.** Four kinds are injected into one node's normal validation rows,
20 per kind and node. Their placement is fixed by seed 2026 and is the same for every
model seed.

| Kind | What it does |
|---|---|
| spike | +6 sd on one feature for one window (60 rows on Site A, one row on Site B) |
| drift | a linear ramp to +3 sd on one feature over 10 min |
| stuck | one feature frozen at its last value for 10 min |
| replay | every feature: the 10 min just before replayed |

**Conventions.**

* sd is the feature's standard deviation over the node's training rows, in the
  recorded units, before ``log1p`` or scaling. The perturbed rows then go through
  the same input pipeline.
* The feature is drawn uniformly among the node's model inputs.
* Injections in one stream are spaced by the duration, plus the R3 history on
  Site A, plus 2 min, so one cannot shift another's baseline. A node whose stream
  cannot hold 20 at that spacing gets several streams.

**Scoring.** The batch path of the edge calibration (``x1.Fitted.shifted``), which
the runtime's streaming scores match. It uses the node's threshold at 1% FPR on
normal validation.

**Reported, descriptive only:**

* the detection rate per kind, where any alert window meets
  ``[start, end + 60 s)``;
* beside it, the same rule on the unperturbed stream at the same placements. That is
  the rate benign alerts alone give: a 10 min span holds about 60 Site B windows, so
  chance is far from zero at a 1% window FPR;
* the window FPR on windows that contain no perturbed row.

Everything here is synthetic.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import Params, load_params
from agmesh.common.frames import LabelFreeFrame
from agmesh.data.access import Partition, Site, load_partition
from agmesh.edge.bundle import GlobalModel
from agmesh.edge.calibrate import Calibrated, calibrate_node
from agmesh.edge.runtime import _constants
from agmesh.experiments import x1_baselines as x1
from agmesh.experiments import x3_models
from agmesh.models import inputs as mi

KINDS = ("spike", "drift", "stuck", "replay")
PER_KIND = 20
SEED = 2026
DURATION_S = 600.0
MARGIN_S = 120.0
SPIKE_SD, DRIFT_SD = 6.0, 3.0

F32 = npt.NDArray[np.float32]


def _period(site: Site, params: Params) -> float:
    return (
        1.0 / params.data.hai.sample_rate_hz
        if site == "site_a"
        else float(params.data.windows.site_b.length_s)
    )


def model_columns(site: Site, part: Partition, params: Params) -> list[int]:
    return mi.site_a_columns(part, params) if site == "site_a" else mi.site_b_columns(part, params)


def plan(
    site: Site, node: str, part: Partition, params: Params, kind: str
) -> list[list[dict[str, Any]]]:
    """Injections of one kind on one node, grouped into streams."""
    period = _period(site, params)
    hist = float(params.models.scoring.shift_window.history_s) if site == "site_a" else 0.0
    spike_s = float(params.data.windows.site_a.length_s) if site == "site_a" else period
    duration = spike_s if kind == "spike" else DURATION_S
    spacing = DURATION_S + hist + MARGIN_S
    t = part.labels.time_s
    group = part.labels.group
    cols = model_columns(site, part, params)
    rng = np.random.default_rng([SEED, KINDS.index(kind), sum(map(ord, node))])
    # a start must leave its duration and the replay source inside one segment, after
    # the R3 history has filled
    ok = np.zeros(t.size, bool)
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        ts = t[idx]
        lead = max(hist, DURATION_S) + MARGIN_S
        ok[idx] = (ts >= ts.min() + lead) & (ts + duration <= ts.max() - MARGIN_S)
    starts: list[float] = []
    streams: list[list[float]] = []
    for i in rng.permutation(np.flatnonzero(ok)):
        s = float(t[i])
        for stream in streams:
            if all(abs(s - x) >= spacing for x in stream):
                stream.append(s)
                break
        else:
            if len(streams) < 4:
                streams.append([s])
            else:
                continue
        starts.append(s)
        if len(starts) == PER_KIND:
            break
    if len(starts) < PER_KIND:
        raise ValueError(f"{site} {node} {kind}: only {len(starts)} injections fit")
    out = []
    for stream in streams:
        out.append(
            [
                {
                    "kind": kind,
                    "start_s": s,
                    "end_s": s + duration,
                    "feature": None
                    if kind == "replay"
                    else part.features.columns[cols[int(rng.integers(len(cols)))]],
                }
                for s in sorted(stream)
            ]
        )
    return [s for s in out if s]


def perturb(
    part: Partition, injections: Sequence[dict[str, Any]], sd: dict[str, float]
) -> tuple[Partition, npt.NDArray[np.bool_]]:
    """``part`` with the injections applied, and the mask of the rows changed."""
    values = np.array(part.features.values, dtype=np.float32, copy=True)
    t = part.labels.time_s
    cols = list(part.features.columns)
    touched = np.zeros(t.size, bool)
    for inj in injections:
        rows = np.flatnonzero((t >= inj["start_s"]) & (t < inj["end_s"]))
        touched[rows] = True
        if inj["kind"] == "replay":
            src = np.flatnonzero((t >= inj["start_s"] - DURATION_S) & (t < inj["start_s"]))
            n = min(rows.size, src.size)
            values[rows[:n]] = values[src[:n]]
            continue
        j = cols.index(inj["feature"])
        if inj["kind"] == "spike":
            values[rows, j] += SPIKE_SD * sd[inj["feature"]]
        elif inj["kind"] == "drift":
            ramp = (np.arange(rows.size) + 1) / rows.size
            values[rows, j] += (DRIFT_SD * sd[inj["feature"]] * ramp).astype(np.float32)
        elif inj["kind"] == "stuck":
            values[rows, j] = values[rows[0] - 1, j]
    frame = LabelFreeFrame(part.features.columns, values)
    return dataclasses.replace(part, features=frame), touched


def alerts_of(site: Site, fitted: Calibrated, part: Partition, params: Params) -> Any:
    inp = x1._inputs(site, part, fitted.scaler, params, evaluation=True)
    shifted = fitted.fitted.shifted(
        inp,
        x1.site_scoring(site, params),
        x1.site_shift(site, params),
        params.models.scoring.shift_window,
    )
    thr = fitted.calibration.threshold
    alert = shifted.scored & (np.nan_to_num(shifted.scores, nan=-np.inf) > thr)
    return inp, shifted, alert


def evaluate(
    site: Site,
    fitted: Calibrated,
    part: Partition,
    streams: list[list[dict[str, Any]]],
    sd: dict[str, float],
    params: Params,
    clean: tuple[Any, Any],
) -> dict[str, Any]:
    """Detection of each injection, and the same rule on the clean stream (chance)."""
    tol = float(params.evaluation.event_tolerance_s)
    c_inp, c_alert = clean
    detected, chance, untouched_alerts, untouched_scored = [], [], 0, 0
    for stream in streams:
        new, touched = perturb(part, stream, sd)
        inp, shifted, alert = alerts_of(site, fitted, new, params)
        ws = inp.window_start
        we = ws + inp.window_length_s
        times = part.labels.time_s[touched]
        # a window is untouched when none of its rows was changed
        lo = np.searchsorted(times, ws, side="left")
        hi = np.searchsorted(times, we, side="left")
        clean = hi == lo
        untouched_alerts += int((alert & clean).sum())
        untouched_scored += int((shifted.scored & clean).sum())
        for inj in stream:
            end = inj["end_s"]
            hit = alert & (ws < end + tol) & (we > inj["start_s"])
            detected.append(bool(hit.any()))
            cs = c_inp.window_start
            ce = cs + c_inp.window_length_s
            chance.append(bool((c_alert & (cs < end + tol) & (ce > inj["start_s"])).any()))
    return {
        "injections": len(detected),
        "detected": int(sum(detected)),
        "rate": float(np.mean(detected)),
        "clean_stream_rate": float(np.mean(chance)),
        "untouched_window_fpr": untouched_alerts / untouched_scored if untouched_scored else None,
        "streams": len(streams),
    }


def verified_seeds(seeds: Sequence[int], problems: Sequence[str]) -> dict[str, Any]:
    """Per site, the seeds whose retrained model reproduces X1 (ADR 0017 §1)."""
    out: dict[str, Any] = {}
    for site in x3_models.SITES:
        cid = x3_models.CONFIG.format(site=site)
        bad = {s: [p for p in problems if p.startswith(f"{cid} seed {s}:")] for s in seeds}
        out[site] = {
            "used": [s for s in seeds if not bad[s]],
            "excluded": {str(s): v for s, v in bad.items() if v},
        }
    return out


def run(
    params: Params,
    models: Path,
    seeds: Sequence[int],
    partitions: Path,
    *,
    dev: bool,
    by_site: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "protocol": "docs/adr/0017-x3-x6-week13.md §7",
        "synthetic": True,
        "development_run": dev,
        "seeds": list(seeds),
        "seeds_by_site": by_site
        or {site: {"used": list(seeds), "excluded": {}} for site in x3_models.SITES},
        "placement_seed": SEED,
        "per_kind_and_node": PER_KIND,
        "sites": {},
    }
    for site in x3_models.SITES:
        site_t: Site = site  # type: ignore[assignment]
        constants = _constants(params, site_t, partitions)
        nodes: dict[str, Any] = {}
        for node in x1._clients(params, site_t):
            data = x1.NodeData(
                {
                    p: load_partition(params, site_t, node, p, root=partitions)
                    for p in ("train", "val_normal", "val_attack")
                }
            )
            normal = data.parts["val_normal"]
            train = data.parts["train"]
            cols = model_columns(site_t, train, params)
            sd = {
                train.features.columns[c]: float(
                    np.std(train.features.values[:, c], dtype=np.float64)
                )
                for c in cols
            }
            plans = {k: plan(site_t, node, normal, params, k) for k in KINDS}
            per_seed: dict[str, Any] = {}
            for seed in out["seeds_by_site"][site]["used"]:
                model = GlobalModel.load(x3_models.model_dir(site, seed, models))
                fitted = calibrate_node(site_t, node, model, data, params, constants)
                clean_inp, clean_shift, clean_alert = alerts_of(site_t, fitted, normal, params)
                row: dict[str, Any] = {
                    "clean_window_fpr": float(clean_alert.sum() / clean_shift.scored.sum()),
                }
                for k in KINDS:
                    row[k] = evaluate(
                        site_t, fitted, normal, plans[k], sd, params, (clean_inp, clean_alert)
                    )
                per_seed[str(seed)] = row
            nodes[node] = {
                "injections": {k: [i for s in v for i in s] for k, v in plans.items()},
                "per_seed": per_seed,
                "seed_mean": {
                    k: {
                        "rate": float(np.mean([per_seed[s][k]["rate"] for s in per_seed])),
                        "clean_stream_rate": float(
                            np.mean([per_seed[s][k]["clean_stream_rate"] for s in per_seed])
                        ),
                        "untouched_window_fpr": float(
                            np.mean([per_seed[s][k]["untouched_window_fpr"] for s in per_seed])
                        ),
                    }
                    for k in KINDS
                }
                | {
                    "clean_window_fpr": float(
                        np.mean([per_seed[s]["clean_window_fpr"] for s in per_seed])
                    )
                },
            }
        out["sites"][site] = nodes
    return out


def markdown(rep: dict[str, Any]) -> str:
    dev = " (DEVELOPMENT RUN, not a result)" if rep["development_run"] else ""
    lines = [
        f"# Robustness suite — synthetic perturbations of normal validation{dev}",
        "",
        "Seeds per site (only models that reproduce X1, ADR 0017 §1): "
        + "; ".join(
            f"{site} {v['used']}"
            + (f", excluded {sorted(v['excluded'], key=int)}" if v["excluded"] else "")
            for site, v in rep["seeds_by_site"].items()
        )
        + ".",
        "",
        f"{rep['per_kind_and_node']} injections per kind and node "
        f"(placement seed {rep['placement_seed']}). Descriptive; nothing is selected or "
        "tuned on these numbers. SYNTHETIC.",
        "",
        "Per kind (seeds averaged): detection rate / the same rule on the unperturbed "
        "stream at the same placements (chance, from benign alerts at the 1% window "
        "FPR) / window FPR on untouched windows. Clean window FPR for reference.",
        "",
        "| Site | Node | " + " | ".join(KINDS) + " | clean FPR |",
        "|---|---|" + "---|" * (len(KINDS) + 1),
    ]
    for site, nodes in rep["sites"].items():
        for node, v in nodes.items():
            m = v["seed_mean"]
            cells = [
                f"{m[k]['rate']:.2f} / {m[k]['clean_stream_rate']:.2f} / "
                f"{m[k]['untouched_window_fpr']:.4f}"
                for k in KINDS
            ]
            lines.append(
                f"| {site} | {node} | " + " | ".join(cells) + f" | {m['clean_window_fpr']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Robustness suite (ADR 0017 §7)")
    parser.add_argument("--models", type=Path, default=x3_models.MODELS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(x3_models.SEEDS))
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/robustness"))
    args = parser.parse_args(argv)
    params = load_params()
    dev = args.models != x3_models.MODELS or sorted(args.seeds) != sorted(x3_models.SEEDS)
    by_site = None
    if not dev:
        by_site = verified_seeds(args.seeds, x3_models.verify())
        for site, v in by_site.items():
            print(f"{site}: seeds used {v['used']}; excluded {sorted(v['excluded'], key=int)}")
            if not v["used"]:
                raise SystemExit(f"{site}: no seed reproduces X1 (ADR 0017 §1)")
    rep = run(
        params,
        args.models,
        args.seeds,
        args.partitions or params.paths.partitions_dir,
        dev=dev,
        by_site=by_site,
    )
    write_artifact(
        args.out / "robustness_val.json", kind="robustness", data_source="synthetic", payload=rep
    )
    text = markdown(rep)
    (args.out / "robustness_val.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
