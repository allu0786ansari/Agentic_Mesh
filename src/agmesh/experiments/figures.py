"""Figures and the per-client privacy table, from artifacts only (ADR 0016 §6).

    python -m agmesh.experiments.figures          # make figures  -> results/figures/
    python -m agmesh.experiments.figures --split test   # -> results/figures/test/

**What it reads.** ``results/X1/analysis_val.json``, ``results/X2/analysis_val.json``
(event-level CIs; ``make stats`` first) and the result artifacts. No number is
typed in.

**Figures.** Each one is written as PNG and SVG:

* ``privacy_utility_<site>_<metric>``: AUROC and AUPRC against ε for FedAvg and
  FedProx, with 95% event-level CIs. The baselines are drawn as labelled reference
  lines: centralised and local-only, non-private, and through the same DP pipeline.
* ``convergence_<site>``: validation loss per round at ε = 3, mean over seeds with
  the seeds' range, for the homogeneous and heterogeneous arms.
* ``heterogeneity_<site>``: AUROC with CI for FedAvg and FedProx in each arm (plus
  IID on Site A).

**The privacy table.** ``privacy_table.csv`` and ``.md`` hold, per configuration and
client, the mean over seeds of ε, δ, the DP-SGD noise multiplier, the steps and the
training windows.

**Encoding.** Two categorical slots identify the strategy: FedAvg ``#2a78d6`` and
FedProx ``#eb6834``, both validated. The arm is shown by line style. References are
neutral grey and labelled directly, so colour is never the only cue. There is one
axis per chart.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SERIES = {"fedavg": "#2a78d6", "fedprox": "#eb6834"}
NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx (μ 0.01)"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EPS = ("1", "3", "10", "inf")
MODEL = {"site_a": "vae", "site_b": "ae"}
SITE = {"site_a": "Site A (HAI, 4 clients)", "site_b": "Site B (Modbus, 3 clients)"}


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: Any, out: Path, name: str) -> list[Path]:
    paths = []
    for ext in ("png", "svg"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        paths.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


#: The split the figures are drawn from, and how they are labelled (set by ``main``).
SPLIT = {"split": "val", "label": "validation",
         "note": "Development evidence; claims are made on the test split after G4."}  # fmt: skip
NOTES = {
    "val": ("validation", "Development evidence; claims are made on the test split after G4."),
    "test": ("test, after G4", "Test split, evaluated once after g4-frozen (ADR 0019)."),
}


def _ci(analysis: dict[str, Any], key: str, metric: str) -> dict[str, Any] | None:
    entry = (analysis.get("configs") or {}).get(key) or {}
    found: dict[str, Any] | None = entry.get(metric)
    return found


def _load(results: Path, exp: str, cid: str) -> list[dict[str, Any]]:
    docs = []
    for p in sorted((results / exp / cid).glob("seed_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("split") == SPLIT["split"]:
            docs.append(d)
    return docs


def _mean_metric(results: Path, exp: str, cid: str, metric: str) -> float | None:
    vals = [d["metrics"].get(metric) for d in _load(results, exp, cid)]
    finite = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def privacy_utility(
    results: Path, x1: dict[str, Any], out: Path, site: str, metric: str
) -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _style(ax)
    xs = np.arange(len(EPS))
    for k, strat in enumerate(("fedavg", "fedprox")):
        pts = []
        for i, e in enumerate(EPS):
            c = _ci(x1, f"X1/{site}__{strat}__eps{e}", metric)
            if c and c.get("observed") is not None:
                pts.append((i + (k - 0.5) * 0.08, c["observed"], c["low"], c["high"]))
        if not pts:
            continue
        x, y, lo, hi = (np.array(v, dtype=float) for v in zip(*pts, strict=True))
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=SERIES[strat], linewidth=2,
                    marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    capsize=3, label=NAMES[strat])  # fmt: skip
    refs: list[tuple[str, str, Any]] = [
        (f"{site}__central__{MODEL[site]}", "centralised, non-private", "--"),
        (f"{site}__local__{MODEL[site]}", "local-only, non-private", ":"),
        (f"{site}__central__{MODEL[site]}__eps3", "centralised, DP pipeline, ε 3", "-."),
        (
            f"{site}__local__{MODEL[site]}__eps3",
            "local-only, DP pipeline, ε 3",
            (0, (6, 2, 1, 2, 1, 2)),
        ),
    ]
    from matplotlib.lines import Line2D

    ref_handles = []
    for cid, label, dash in refs:
        v = _mean_metric(results, "X1", cid, metric)
        if v is None:
            continue
        ax.axhline(v, color=INK2, linewidth=1.2, linestyle=dash)
        ref_handles.append(
            Line2D([], [], color=INK2, linewidth=1.2, linestyle=dash, label=f"{label}: {v:.3f}")
        )  # fmt: skip
    ax.set_xticks(xs, [f"ε = {e}" if e != "inf" else "ε = ∞ (no DP)" for e in EPS])
    ax.set_ylabel(metric.upper(), color=INK, fontsize=10)
    ax.set_title(f"{SITE[site]}: {metric.upper()} against privacy ({SPLIT['label']}, 5 seeds)",
                 color=INK, fontsize=10.5, loc="left")  # fmt: skip
    series = [
        Line2D([], [], color=SERIES[k], linewidth=2, marker="o", markersize=8,
               markeredgecolor=SURFACE, markeredgewidth=2, label=NAMES[k])
        for k in ("fedavg", "fedprox")
    ]  # fmt: skip
    ax.legend(handles=series + ref_handles, frameon=False, fontsize=8, labelcolor=INK,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), title="mean over 5 seeds",
              title_fontsize=8)  # fmt: skip
    fig.text(0.0, -0.04, "Bars: 95% event-level CIs (ADR 0005). " + SPLIT["note"],
             fontsize=7, color=INK2)  # fmt: skip
    return _save(fig, out, f"privacy_utility_{site}_{metric}")


def convergence(results: Path, out: Path, site: str) -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _style(ax)
    arms = (("X1", "", "-", "homogeneous"), ("X2", "__het__drop0.2", "--", "heterogeneous"))
    for exp, suffix, dash, arm in arms:
        for strat in ("fedavg", "fedprox"):
            docs = _load(results, exp, f"{site}__{strat}__eps3{suffix}")
            curves = [[r["val_loss"] for r in d["federation"]["round_log"]] for d in docs]
            if not curves:
                continue
            arr = np.array(curves, dtype=float)
            x = np.arange(1, arr.shape[1] + 1)
            ax.fill_between(x, arr.min(axis=0), arr.max(axis=0), color=SERIES[strat],
                            alpha=0.12, linewidth=0)  # fmt: skip
            ax.plot(x, arr.mean(axis=0), color=SERIES[strat], linewidth=2, linestyle=dash,
                    label=f"{NAMES[strat]}, {arm}")  # fmt: skip
    ax.set_xlabel("round", color=INK, fontsize=10)
    ax.set_ylabel("validation loss (global model)", color=INK, fontsize=10)
    ax.set_title(f"{SITE[site]}: convergence at ε = 3 (mean, band = seed range)",
                 color=INK, fontsize=10.5, loc="left")  # fmt: skip
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    return _save(fig, out, f"convergence_{site}")


def heterogeneity(x1: dict[str, Any], x2: dict[str, Any], out: Path, site: str) -> list[Path]:
    import matplotlib.pyplot as plt

    # X2's analysis carries the CIs of both arms (the homogeneous arm is X1's runs)
    arms = [("homogeneous", "X1", "", x2), ("heterogeneous", "X2", "__het__drop0.2", x2)]
    if site == "site_a":
        arms.append(("IID partitions", "X2", "__iid", x2))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    _style(ax)
    lows: list[float] = []
    for i, (_arm, exp, suffix, analysis) in enumerate(arms):
        for k, strat in enumerate(("fedavg", "fedprox")):
            c = _ci(analysis, f"{exp}/{site}__{strat}__eps3{suffix}", "auroc")
            if not c or c.get("observed") is None:
                continue
            x = i + (k - 0.5) * 0.25
            lows.append(float(c["low"]))
            ax.errorbar([x], [c["observed"]], yerr=[[c["observed"] - c["low"]],
                        [c["high"] - c["observed"]]], color=SERIES[strat], marker="o",
                        markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, capsize=3,
                        linewidth=2, label=NAMES[strat] if i == 0 else None)  # fmt: skip
    ax.set_xticks(range(len(arms)), [a[0] for a in arms])
    ax.set_xlim(-0.6, len(arms) - 0.4)
    # an honest range: never zoom a sub-0.01 difference to fill the axis
    floor = min([*lows, 1.0])
    ax.set_ylim(min(0.95, np.floor((floor - 0.01) * 50) / 50), 1.002)
    ax.set_ylabel("AUROC", color=INK, fontsize=10)
    ax.set_title(f"{SITE[site]}: FedProx against FedAvg at ε = 3 ({SPLIT['label']}, 95% CI)",
                 color=INK, fontsize=10.5, loc="left")  # fmt: skip
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    return _save(fig, out, f"heterogeneity_{site}")


def privacy_table(results: Path, out: Path) -> list[Path]:
    rows: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for exp in ("X1", "X2"):
        for p in sorted((results / exp).glob("*/seed_*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("split") != SPLIT["split"] or d.get("target_epsilon") in (None, "inf"):
                continue
            for client, v in ((d.get("privacy") or {}).get("clients") or {}).items():
                plan = v.get("plan") or {}
                rows[(exp, d["config_id"], client)].append(
                    {
                        "epsilon": float(v["epsilon"]),
                        "target": float(v["target_epsilon"]),
                        "delta": float(v["delta"]),
                        "noise_multiplier": float(plan.get("noise_multiplier", np.nan)),
                        "steps": float(plan.get("steps", np.nan)),
                        "n_windows": float(plan.get("n_windows", np.nan)),
                    }
                )
    cols = ("epsilon", "target", "delta", "noise_multiplier", "steps", "n_windows")
    csv_path, md_path = out / "privacy_table.csv", out / "privacy_table.md"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "config", "client", "seeds", *[f"mean_{c}" for c in cols],
                    "max_epsilon"])  # fmt: skip
        lines = [
            "| Experiment | Config | Client | Seeds | ε (mean) | ε (max) | target | δ "
            "| noise multiplier | steps | windows |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for (exp, cid, client), vs in sorted(rows.items()):
            mean = {c: float(np.mean([v[c] for v in vs])) for c in cols}
            mx = max(v["epsilon"] for v in vs)
            w.writerow([exp, cid, client, len(vs), *[f"{mean[c]:.6g}" for c in cols], f"{mx:.6g}"])
            lines.append(
                f"| {exp} | {cid} | {client} | {len(vs)} | {mean['epsilon']:.4f} | {mx:.4f} "
                f"| {mean['target']:g} | {mean['delta']:.2e} | {mean['noise_multiplier']:.3f} "
                f"| {mean['steps']:.0f} | {mean['n_windows']:.0f} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "agmesh"})
    parser = argparse.ArgumentParser(description="Figures from artifacts (ADR 0016 §6)")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args(argv)
    label, note = NOTES[args.split]
    SPLIT.update(split=args.split, label=label, note=note)
    root = args.results / "test" if args.split == "test" else args.results
    out = args.out or (
        Path("results/figures/test") if args.split == "test" else Path("results/figures")
    )
    out.mkdir(parents=True, exist_ok=True)
    x1p = root / "X1" / f"analysis_{args.split}.json"
    x2p = root / "X2" / f"analysis_{args.split}.json"
    if not x1p.is_file() or not x2p.is_file():
        raise SystemExit(f"run the analyses first ({x1p}, {x2p})")
    x1, x2 = json.loads(x1p.read_text()), json.loads(x2p.read_text())
    written: list[Path] = []
    for site in ("site_a", "site_b"):
        for metric in ("auroc", "auprc"):
            written += privacy_utility(root, x1, out, site, metric)
        if args.split == "val":  # training curves are a validation quantity
            written += convergence(root, out, site)
        written += heterogeneity(x1, x2, out, site)
    written += privacy_table(root, out)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
