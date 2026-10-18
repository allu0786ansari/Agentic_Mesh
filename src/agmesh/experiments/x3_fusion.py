"""X3: single node, 2-of-3 and noisy-OR on Site B validation (ADR 0017 §2-3; RQ2).

    python -m agmesh.experiments.x3_fusion                  # make x3, the five seeds
    python -m agmesh.experiments.x3_fusion --models results/models --seeds 11 --out <dir>

**Scoring.** Every window is scored by the edge runtime's own path:

* calibration per node (``edge.calibrate.calibrate_node``), with the threshold at 1%
  FPR on normal validation and the isotonic calibrator;
* ``edge.stream.NodeScorer`` window by window;
* the evidence rule for latency-only alerts.

Site B uses R0, so a window's score does not depend on the others. An injected
window can therefore be rescored alone.

**Detectors.** Each one counts incidents built by the Δt chaining of
``mesh.fusion``, with latency-only alerts excluded from counting, as configured.

* **Single node n:** node n's alerts alone. An alarm needs one counting alert.
* **2-of-3:** all alerts; an alarm needs counting alerts from two nodes.
* **Noisy-OR:** all alerts; an alarm needs a score of at least τ.

An alarm is raised at the alert that meets its condition. It is in force from that
window's start to the incident's end.

**Outputs.** ``results/X3/fusion_val.json`` (H2a, PT4, H2b) and
``results/X3/synthetic_val.json`` (H2c, synthetic), each with a ``.md``.

Five-seed models are used only when ``x3_models.verify`` reports no problem for
them. A run on other models is marked as a development run.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import Params, load_params
from agmesh.common.frames import LabelFreeFrame
from agmesh.data.access import Partition, Site, load_partition
from agmesh.edge.bundle import GlobalModel, NodeCalibration
from agmesh.edge.calibrate import calibrate_node
from agmesh.edge.evidence import LATENCY_FEATURES, top_features
from agmesh.edge.stream import NodeScorer, WindowSource
from agmesh.evaluation.calibration import IsotonicCalibrator
from agmesh.evaluation.events import events_from_episodes
from agmesh.experiments import x1_baselines as x1
from agmesh.experiments import x3_models
from agmesh.mesh import fusion

SITE: Site = "site_b"
BLOCK_S = 3600.0
RESAMPLES = 10_000
SEED = 2026
PT4_REDUCTION = 0.30
REFERENCE_NODE = "b1"  # IED1A, the only node that observes the attack (ADR 0017 §2)
W_PRIMARY, W_SENSITIVITY = 5, (3, 10)
BURSTS, BURST_WINDOWS, MIN_GAP_S = 24, 6, 1800.0
PAIRS = (("b1", "b2"), ("b2", "b3"), ("b1", "b3"))

F64 = npt.NDArray[np.float64]


# ------------------------------------------------------------------------ scoring


@dataclass(frozen=True)
class WindowAlert:
    node: str
    t_start: float
    t_end: float
    prob: float
    latency_only: bool

    def fusion(self) -> fusion.FusionAlert:
        return fusion.FusionAlert(
            f"{self.node}:{self.t_start:.0f}", self.node, self.t_end, self.prob, self.latency_only
        )


@dataclass
class NodeRig:
    node: str
    calibration: NodeCalibration
    scorer: NodeScorer
    calibrator: IsotonicCalibrator
    source: dict[str, WindowSource]  # val_normal, val_attack


def rigs(params: Params, model_dir: Path, partitions: Path) -> dict[str, NodeRig]:
    if params.models.scoring.site_b.shift != "r0":
        raise ValueError("X3 rescoring of injected windows assumes R0 on Site B")
    model = GlobalModel.load(model_dir)
    if model.site != SITE:
        raise ValueError(f"{model_dir} is a {model.site} model")
    net = model.network()
    out: dict[str, NodeRig] = {}
    for node in x1._clients(params, SITE):
        data = x1.NodeData(
            {
                p: load_partition(params, SITE, node, p, root=partitions)
                for p in ("train", "val_normal", "val_attack")
            }
        )
        fitted = calibrate_node(SITE, node, model, data, params, {}, net)
        cal = fitted.calibration
        out[node] = NodeRig(
            node,
            cal,
            NodeScorer(SITE, cal, net, params),
            IsotonicCalibrator.from_dict(cal.calibrator),
            {
                p: WindowSource(SITE, node, p, data.parts[p], fitted.scaler, params)
                for p in ("val_normal", "val_attack")
            },
        )
    return out


def score(rig: NodeRig, src: WindowSource, rows: Sequence[int] | None = None) -> dict[int, Any]:
    """Alerts of the given windows (all by default), keyed by window index."""
    cal = rig.calibration
    out: dict[int, Any] = {}
    for k in range(len(src)) if rows is None else rows:
        w = src.window(k)
        s = rig.scorer.score(w, src.stride)
        if not (s.scored and s.score > cal.threshold):
            out[k] = None
            continue
        feats = top_features(s.feature_errors, s.residuals, cal.features, cal.buckets)
        prob = float(np.clip(rig.calibrator.predict([s.score])[0], 0.0, 1.0))
        latency_only = bool(feats) and all(f.name in LATENCY_FEATURES for f in feats)
        out[k] = WindowAlert(rig.node, w.t_start, w.t_end, prob, latency_only)
    return out


# ---------------------------------------------------------------------- detectors


@dataclass(frozen=True)
class Alarm:
    t_s: float  # the alert that met the condition: its window end
    start_s: float  # that window's start
    end_s: float  # the incident's end


def alarm(inc: fusion.Incident, rule: str, tau: float, length_s: float) -> Alarm | None:
    """When ``inc`` raises an alarm under ``rule`` (``single``, ``2of3``, ``noisy_or``)."""
    best: dict[str, float] = {}
    for a in inc.counting:  # in window-end order
        best[a.node_id] = max(best.get(a.node_id, 0.0), a.prob)
        if (
            rule == "single"
            or (rule == "2of3" and len(best) >= 2)
            or (rule == "noisy_or" and fusion.noisy_or(best.values()) >= tau)
        ):
            return Alarm(a.t_end_s, a.t_end_s - length_s, inc.end_s)
    return None


def alarms(alerts: Sequence[WindowAlert], params: Params, tau: float) -> dict[str, list[Alarm]]:
    """Alarms of every detector over one stream of alerts."""
    cfg = params.fusion.site_b
    length = float(params.data.windows.site_b.length_s)
    fa = [a.fusion() for a in alerts]
    out: dict[str, list[Alarm]] = {}
    for node in sorted({a.node for a in alerts} | set(x1._clients(params, SITE))):
        found = (
            fusion.incidents(
                [a for a in fa if a.node_id == node],
                cfg.delta_t_s,
                exclude_latency=cfg.exclude_latency,
            )
            if any(a.node_id == node for a in fa)
            else []
        )
        out[node] = [x for i in found if (x := alarm(i, "single", tau, length))]
    joint = fusion.incidents(fa, cfg.delta_t_s, exclude_latency=cfg.exclude_latency) if fa else []
    out["2of3"] = [x for i in joint if (x := alarm(i, "2of3", tau, length))]
    out["noisy_or"] = [x for i in joint if (x := alarm(i, "noisy_or", tau, length))]
    return out


def matched_tau(alerts: Sequence[WindowAlert], params: Params, hours: float) -> dict[str, Any]:
    """τ: the smallest incident noisy-OR at which noisy-OR's FA/h ≤ the best node's."""
    cfg = params.fusion.site_b
    found = alarms(alerts, params, 1.0)
    singles = {n: v for n, v in found.items() if n not in ("2of3", "noisy_or")}
    best = min(sorted(singles), key=lambda n: len(singles[n]))
    target = len(singles[best]) / hours
    fa = [a.fusion() for a in alerts]
    joint = fusion.incidents(fa, cfg.delta_t_s, exclude_latency=cfg.exclude_latency)
    values = np.sort([i.noisy_or for i in joint])
    tau = math.inf
    for v in np.unique(values):
        if (values >= v).sum() / hours <= target:
            tau = float(v)
            break
    return {"tau": tau, "best_node": best, "best_fa_per_h": target}


# ------------------------------------------------------------------------- spans


def benign_span(rigs_: dict[str, NodeRig]) -> tuple[float, float]:
    """The hours of normal validation that every node covers."""
    starts = [r.source["val_normal"].partition.labels.time_s for r in rigs_.values()]
    length = next(iter(rigs_.values())).source["val_normal"].window_length_s
    return max(float(s.min()) for s in starts), min(float(s.max()) for s in starts) + length


def in_span(alerts: dict[int, Any], span: tuple[float, float]) -> list[WindowAlert]:
    return [a for a in alerts.values() if a and a.t_start >= span[0] and a.t_end <= span[1]]


def episodes(rigs_: dict[str, NodeRig]) -> dict[str, tuple[float, float]]:
    """Each validation episode's labelled windows on any node, as one span."""
    spans: dict[str, list[tuple[float, float]]] = {}
    for r in rigs_.values():
        src = r.source["val_attack"]
        lab = src.partition.labels
        group = lab.group.astype(np.int64)
        for e in np.unique(group[(lab.target > 0) & (group >= 0)]):
            sel = group == e
            spans.setdefault(str(e), []).extend(
                events_from_episodes(
                    lab.time_s[sel], group[sel], src.window_length_s, lab.target[sel]
                )
            )
    return {e: (min(a for a, _ in v), max(b for _, b in v)) for e, v in sorted(spans.items())}


def detect(
    events: dict[str, tuple[float, float]], found: list[Alarm], tolerance_s: float
) -> dict[str, float | None]:
    """Time to detect per event (None if missed), with the event-recall tolerance."""
    out: dict[str, float | None] = {}
    for key, (a, b) in events.items():
        hits = [x for x in found if x.start_s < b + tolerance_s and x.end_s >= a]
        out[key] = max(0.0, min(x.t_s for x in hits) - a) if hits else None
    return out


# ------------------------------------------------------------------- synthetic


def bursts(rigs_: dict[str, NodeRig], span: tuple[float, float]) -> list[dict[str, Any]]:
    """The fixed scenario (ADR 0017 §3): 24 bursts of 6 windows on two devices."""
    step = float(next(iter(rigs_.values())).source["val_normal"].stride)
    have = [set(r.source["val_normal"].partition.labels.time_s.tolist()) for r in rigs_.values()]
    common = sorted(set.intersection(*have))
    ok = [
        t
        for t in common
        if t >= span[0]
        and t + BURST_WINDOWS * step <= span[1]
        and all(t + j * step in h for h in have for j in range(BURST_WINDOWS))
    ]
    rng = np.random.default_rng(SEED)
    chosen: list[float] = []
    for i in rng.permutation(len(ok)):
        t = float(ok[i])
        if all(abs(t - c) >= MIN_GAP_S for c in chosen):
            chosen.append(t)
            if len(chosen) == BURSTS:
                break
    if len(chosen) < BURSTS:
        raise ValueError(f"only {len(chosen)} bursts fit the benign span")
    chosen.sort()
    return [
        {
            "event": f"s{i:02d}",
            "start_s": t,
            "end_s": t + BURST_WINDOWS * step,
            "nodes": list(PAIRS[i % len(PAIRS)]),
        }
        for i, t in enumerate(chosen)
    ]


def injected(part: Partition, rows: Sequence[int], w: float) -> Partition:
    """``part`` with ``req_write`` set to W and ``req_recv`` raised by W in ``rows``.

    ``hmi_requests`` is also raised by W in the scenario; it is not a model input, so
    it does not enter the partition's features.
    """
    cols = list(part.features.columns)
    values = np.array(part.features.values, dtype=np.float32, copy=True)
    idx = np.asarray(rows, dtype=np.int64)
    values[idx, cols.index("req_write")] = w
    values[idx, cols.index("req_recv")] += w
    return dataclasses.replace(part, features=LabelFreeFrame(part.features.columns, values))


def synthetic_alerts(
    rig: NodeRig, clean: dict[int, Any], scenario: list[dict[str, Any]], w: float
) -> dict[int, Any]:
    src = rig.source["val_normal"]
    times = src.partition.labels.time_s
    index = {float(t): i for i, t in enumerate(times.tolist())}
    rows = [
        index[b["start_s"] + j * src.stride]
        for b in scenario
        if rig.node in b["nodes"]
        for j in range(BURST_WINDOWS)
    ]
    if not rows:
        return dict(clean)
    new_src = WindowSource(
        SITE, rig.node, "val_normal", injected(src.partition, rows, w), src.scaler, src.params
    )
    out = dict(clean)
    out.update(score(rig, new_src, rows))
    return out


# ------------------------------------------------------------------- statistics


def _blocks(found: list[Alarm], span: tuple[float, float], n: int) -> F64:
    counts = np.zeros(n)
    for x in found:
        counts[min(n - 1, int((x.t_s - span[0]) // BLOCK_S))] += 1
    return counts


def pt4(per_seed: list[dict[str, Any]], hours: F64) -> dict[str, Any]:
    """Reduction of FA/h, 2-of-3 against the best single node; seeds averaged."""
    two = np.array([s["blocks"]["2of3"] for s in per_seed])
    best = np.array([s["blocks"][s["tau"]["best_node"]] for s in per_seed])
    nor = np.array([s["blocks"]["noisy_or"] for s in per_seed])

    def reduction(a: F64, b: F64, idx: npt.NDArray[np.int64] | slice) -> float:
        h = hours[idx].sum()
        fb = (b[:, idx].sum(axis=1) / h).mean()
        return float(1.0 - (a[:, idx].sum(axis=1) / h).mean() / fb) if fb > 0 else math.nan

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, hours.size, size=(RESAMPLES, hours.size))
    r2 = np.array([reduction(two, best, d) for d in draws])
    rn = np.array([reduction(nor, best, d) for d in draws])
    full = slice(None)
    return {
        "blocks": int(hours.size),
        "hours": float(hours.sum()),
        "reduction_2of3": {
            "estimate": reduction(two, best, full),
            "low": float(np.nanquantile(r2, 0.025)),
            "high": float(np.nanquantile(r2, 0.975)),
            "p_one_sided_H0_le_0.30": float(np.mean(~(r2 > PT4_REDUCTION))),
        },
        "reduction_noisy_or_descriptive": {
            "estimate": reduction(nor, best, full),
            "low": float(np.nanquantile(rn, 0.025)),
            "high": float(np.nanquantile(rn, 0.975)),
        },
        "pt4_met_on_validation": bool(np.mean(~(r2 > PT4_REDUCTION)) < 0.05),
    }


def recall_block(
    per_seed: list[dict[str, float | None]], keys: list[str], label: str
) -> dict[str, Any]:
    """Recall and median time to detect over events, seeds averaged; event bootstrap."""
    hit = np.array([[s[k] is not None for k in keys] for s in per_seed], float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(keys), size=(RESAMPLES, len(keys)))
    boot = hit[:, draws].mean(axis=2).mean(axis=0)
    ttd = [float(v) for s in per_seed for k in keys if (v := s[k]) is not None]
    return {
        "detector": label,
        "events": len(keys),
        "recall": float(hit.mean()),
        "low": float(np.quantile(boot, 0.025)),
        "high": float(np.quantile(boot, 0.975)),
        "ttd_median_s": float(np.median(ttd)) if ttd else None,
        "per_event_detected_seeds": {k: int(hit[:, i].sum()) for i, k in enumerate(keys)},
    }


def paired_loss(
    a: list[dict[str, float | None]], b: list[dict[str, float | None]], keys: list[str]
) -> dict[str, float]:
    """recall(a) - recall(b), paired over events, seeds averaged."""
    ha = np.array([[s[k] is not None for k in keys] for s in a], float).mean(axis=0)
    hb = np.array([[s[k] is not None for k in keys] for s in b], float).mean(axis=0)
    d = ha - hb
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, d.size, size=(RESAMPLES, d.size))
    m = d[draws].mean(axis=1)
    return {
        "loss": float(d.mean()),
        "low": float(np.quantile(m, 0.025)),
        "high": float(np.quantile(m, 0.975)),
    }


# ------------------------------------------------------------------------ driver


def req_write_max(params: Params, partitions: Path) -> dict[str, float]:
    out = {}
    for n in x1._clients(params, SITE):
        f = load_partition(params, SITE, n, "val_normal", root=partitions).features
        out[n] = float(f.values[:, list(f.columns).index("req_write")].max())
    return out


def run(
    params: Params, models: Path, seeds: Sequence[int], partitions: Path, *, dev: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    tol = float(params.evaluation.event_tolerance_s)
    per_seed: list[dict[str, Any]] = []
    attack: dict[str, list[dict[str, float | None]]] = {}
    synth: dict[float, dict[str, list[dict[str, float | None]]]] = {}
    scenario: list[dict[str, Any]] = []
    span = (0.0, 0.0)
    eps: dict[str, tuple[float, float]] = {}
    for seed in seeds:
        mdir = x3_models.model_dir(SITE, seed, models)
        r = rigs(params, mdir, partitions)
        span = benign_span(r)
        n_blocks = math.ceil((span[1] - span[0]) / BLOCK_S)
        hours_blocks = np.array(
            [
                (min(span[0] + BLOCK_S * (i + 1), span[1]) - (span[0] + BLOCK_S * i)) / 3600.0
                for i in range(n_blocks)
            ]
        )
        clean = {n: score(rig, rig.source["val_normal"]) for n, rig in r.items()}
        benign = [a for n in clean for a in in_span(clean[n], span)]
        tau = matched_tau(benign, params, float(hours_blocks.sum()))
        found = alarms(benign, params, tau["tau"])
        per_seed.append(
            {
                "seed": seed,
                "model": mdir.as_posix(),
                "model_version": GlobalModel.load(mdir).version,
                "thresholds": {n: rig.calibration.threshold for n, rig in r.items()},
                "threshold_val_fpr": {n: rig.calibration.threshold_val_fpr for n, rig in r.items()},
                "benign_alerts": {n: len(in_span(clean[n], span)) for n in clean},
                "benign_latency_only_alerts": {
                    n: sum(a.latency_only for a in in_span(clean[n], span)) for n in clean
                },
                "tau": tau,
                "fa_per_h": {d: len(v) / float(hours_blocks.sum()) for d, v in found.items()},
                "blocks": {d: _blocks(v, span, n_blocks).tolist() for d, v in found.items()},
            }
        )
        eps = episodes(r)
        att = [a for rig in r.values() for a in score(rig, rig.source["val_attack"]).values() if a]
        for d, v in alarms(att, params, tau["tau"]).items():
            attack.setdefault(d, []).append(detect(eps, v, tol))
        scenario = bursts(r, span)
        events = {b["event"]: (b["start_s"], b["end_s"]) for b in scenario}
        for w in (W_PRIMARY, *W_SENSITIVITY):
            alerts = [
                a
                for n, rig in r.items()
                for a in in_span(synthetic_alerts(rig, clean[n], scenario, float(w)), span)
            ]
            for d, v in alarms(alerts, params, tau["tau"]).items():
                synth.setdefault(float(w), {}).setdefault(d, []).append(detect(events, v, tol))

    hours = np.array(
        [
            (min(span[0] + BLOCK_S * (i + 1), span[1]) - (span[0] + BLOCK_S * i)) / 3600.0
            for i in range(len(per_seed[0]["blocks"]["2of3"]))
        ]
    )
    ep_keys = sorted(eps)
    detectors = [*x1._clients(params, SITE), "2of3", "noisy_or"]
    common = {
        "protocol": "docs/adr/0017-x3-x6-week13.md §2-3",
        "split": "val",
        "seeds": list(seeds),
        "development_run": dev,
        "tolerance_s": tol,
        "resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "fusion": params.fusion.site_b.model_dump(),
    }
    seed_mean = {d: float(np.mean([s["fa_per_h"][d] for s in per_seed])) for d in detectors}
    fusion_rep = {
        **common,
        "benign_span_s": list(span),
        "per_seed": [{k: v for k, v in s.items() if k != "blocks"} for s in per_seed],
        "h2a_fa_per_h_seed_mean": seed_mean,
        "pt4": pt4(per_seed, hours),
        "h2b": {
            "episodes": {k: list(v) for k, v in eps.items()},
            "reference_node": REFERENCE_NODE,
            "detectors": {d: recall_block(attack[d], ep_keys, d) for d in detectors},
            "recall_loss_2of3_vs_reference": paired_loss(
                attack[REFERENCE_NODE], attack["2of3"], ep_keys
            ),
            "recall_loss_noisy_or_vs_reference": paired_loss(
                attack[REFERENCE_NODE], attack["noisy_or"], ep_keys
            ),
        },
        "development_evidence": True,
    }
    synth_rep: dict[str, Any] = {
        **common,
        "synthetic": True,
        "scenario": {
            "bursts": scenario,
            "windows_per_burst": BURST_WINDOWS,
            "min_gap_s": MIN_GAP_S,
            "seed": SEED,
            "injection": "req_write := W; req_recv += W (hmi_requests += W, not a model input)",
        },
        "benign_req_write_max": req_write_max(params, partitions),
        "by_w": {},
    }
    for wv, by_d in synth.items():
        block: dict[str, Any] = {"primary": wv == W_PRIMARY, "detectors": {}}
        for d in detectors:
            ks = [
                b["event"]
                for b in scenario
                if d not in x1._clients(params, SITE) or d in b["nodes"]
            ]
            block["detectors"][d] = recall_block(by_d[d], ks, d)
        block["recall_loss_2of3_vs_best_single"] = {
            n: paired_loss(by_d[n], by_d["2of3"], [b["event"] for b in scenario if n in b["nodes"]])
            for n in x1._clients(params, SITE)
        }
        synth_rep["by_w"][f"W={wv:g}"] = block
    return fusion_rep, synth_rep


def _f(x: float | None, nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def markdown(fr: dict[str, Any], sr: dict[str, Any]) -> str:
    dev = " (DEVELOPMENT RUN, not a result)" if fr["development_run"] else ""
    lines = [
        f"# X3 — fusion on Site B validation{dev}",
        "",
        f"Seeds {fr['seeds']}; Δt {fr['fusion']['delta_t_s']} s, latency-only excluded: "
        f"{fr['fusion']['exclude_latency']}. Benign span {fr['pt4']['hours']:.1f} h in "
        f"{fr['pt4']['blocks']} blocks of 60 min.",
        "",
        "## H2a — false alarms per hour (benign validation, seeds averaged)",
        "",
        "| Detector | FA/h |",
        "|---|---|",
    ]
    for d, v in fr["h2a_fa_per_h_seed_mean"].items():
        lines.append(f"| {d} | {_f(v)} |")
    lines += [
        "",
        "| Seed | τ | Best single node | its FA/h | thresholds' val FPR |",
        "|---|---|---|---|---|",
    ]
    for s in fr["per_seed"]:
        fpr = ", ".join(f"{n} {v:.4f}" for n, v in s["threshold_val_fpr"].items())
        lines.append(
            f"| {s['seed']} | {_f(s['tau']['tau'], 4)} | {s['tau']['best_node']} | "
            f"{_f(s['tau']['best_fa_per_h'])} | {fpr} |"
        )
    p = fr["pt4"]["reduction_2of3"]
    n = fr["pt4"]["reduction_noisy_or_descriptive"]
    lines += [
        "",
        f"**PT4 (validation):** 2-of-3 reduction of FA/h against the best single "
        f"node {_f(p['estimate'])} [{_f(p['low'])}, {_f(p['high'])}]; one-sided p "
        f"(H0: reduction ≤ 0.30) = {_f(p['p_one_sided_H0_le_0.30'], 4)}. "
        f"Noisy-OR (descriptive, matched by construction): {_f(n['estimate'])} "
        f"[{_f(n['low'])}, {_f(n['high'])}].",
        "",
        "## H2b — real attack day, validation episodes",
        "",
        "| Detector | Recall [95% CI] | Median TTD (s) | Detected seeds per episode |",
        "|---|---|---|---|",
    ]
    for d, v in fr["h2b"]["detectors"].items():
        lines.append(
            f"| {d} | {_f(v['recall'])} [{_f(v['low'])}, {_f(v['high'])}] | "
            f"{_f(v['ttd_median_s'], 0)} | {v['per_event_detected_seeds']} |"
        )
    lo = fr["h2b"]["recall_loss_2of3_vs_reference"]
    lines += [
        "",
        f"Recall loss of 2-of-3 against {fr['h2b']['reference_node']}: "
        f"{_f(lo['loss'])} [{_f(lo['low'])}, {_f(lo['high'])}].",
        "",
        "## H2c — synthetic cross-device bursts (SYNTHETIC)",
        "",
    ]
    for w, block in sr["by_w"].items():
        tag = " (primary)" if block["primary"] else " (sensitivity)"
        lines += [
            f"### {w}{tag}",
            "",
            "| Detector | Events | Recall [95% CI] | Median TTD (s) |",
            "|---|---|---|---|",
        ]
        for d, v in block["detectors"].items():
            lines.append(
                f"| {d} | {v['events']} | {_f(v['recall'])} [{_f(v['low'])}, "
                f"{_f(v['high'])}] | {_f(v['ttd_median_s'], 0)} |"
            )
        lines.append("")
    lines.append(f"Benign `req_write` maximum per node: {sr['benign_req_write_max']}.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="X3 fusion (ADR 0017 §2-3)")
    parser.add_argument("--models", type=Path, default=x3_models.MODELS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(x3_models.SEEDS))
    parser.add_argument("--partitions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/X3"))
    args = parser.parse_args(argv)
    params = load_params()
    dev = args.models != x3_models.MODELS or sorted(args.seeds) != sorted(x3_models.SEEDS)
    if not dev:
        problems = [p for p in x3_models.verify() if SITE in p]
        if problems:
            for p in problems:
                print(p)
            raise SystemExit("x3_models verify reports problems: X3 stops (ADR 0017 §1)")
    fr, sr = run(
        params, args.models, args.seeds, args.partitions or params.paths.partitions_dir, dev=dev
    )
    write_artifact(
        args.out / "fusion_val.json", kind="X3.fusion", data_source="runtime", payload=fr
    )
    write_artifact(
        args.out / "synthetic_val.json", kind="X3.synthetic", data_source="synthetic", payload=sr
    )
    text = markdown(fr, sr)
    (args.out / "fusion_val.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
