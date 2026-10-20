"""Headline models for Week 13, five seeds, retrained and checked (ADR 0017 §1).

    make x3-models       # retrain FedProx ε 3, both sites, five seeds, with --save-model
    python -m agmesh.experiments.x3_models verify
    python -m agmesh.experiments.x3_models sensitivity   # ADR 0018 §1

Each seed is retrained in its own process into ``results/x13_rerun``. Its model is
saved to ``results/models/x3/<config>/seed_<n>``. ``verify`` then checks two things:

* every rerun artifact reproduces its X1 artifact, every detection metric to 1e-9;
* every model exists.

A mismatch is listed.

**Acceptance (ADR 0018 §1).** A seed is usable when it meets the Plan's reproduction
tolerance: the privacy accounting is identical (ε exactly) and AUROC is within
±0.005 of X1. ``accept`` checks that. The exact comparison (1e-9, ADR 0017 §1) is
still run and reported, but it no longer excludes a seed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SITES = ("site_a", "site_b")
SEEDS = (11, 23, 37, 41, 53)
CONFIG = "{site}__fedprox__eps3"
RERUN = Path("results/x13_rerun")
MODELS = Path("results/models/x3")
TOL = 1e-9
AUROC_TOL = 0.005  # Plan Week 14 task 2: "ε exactly; AUROC ± 0.005 for the same seed"


def model_dir(site: str, seed: int, root: Path = MODELS) -> Path:
    return root / CONFIG.format(site=site) / f"seed_{seed}"


def train(results: Path = RERUN, models: Path = MODELS) -> int:
    for site in SITES:
        for seed in SEEDS:
            if (model_dir(site, seed, models) / "model.json").is_file():
                print(f"{site} seed {seed}: model exists, skipped")
                continue
            cmd = [
                sys.executable, "-m", "agmesh.federated.run", "--site", site,
                "--strategy", "fedprox", "--epsilon", "3", "--seeds", str(seed),
                "--results", str(results), "--save-model", str(models),
            ]  # fmt: skip
            print(f"{site} seed {seed}: training", flush=True)
            if subprocess.call(cmd) != 0:
                print(f"{site} seed {seed}: FAILED", flush=True)
                return 1
    return 0


def verify(
    results: Path = RERUN, reference: Path = Path("results"), models: Path = MODELS
) -> list[str]:
    problems = []
    for site in SITES:
        cid = CONFIG.format(site=site)
        for seed in SEEDS:
            new = results / "X1" / cid / f"seed_{seed}.json"
            ref = reference / "X1" / cid / f"seed_{seed}.json"
            if not (model_dir(site, seed, models) / "model.json").is_file():
                problems.append(f"{cid} seed {seed}: no model")
            if not new.is_file() or not ref.is_file():
                problems.append(f"{cid} seed {seed}: artifact missing")
                continue
            a = json.loads(new.read_text())["metrics"]
            b = json.loads(ref.read_text())["metrics"]
            for k in sorted(set(a) | set(b)):
                x, y = a.get(k), b.get(k)
                if x is None and y is None:
                    continue
                if x is None or y is None or abs(float(x) - float(y)) > TOL:
                    problems.append(f"{cid} seed {seed}: {k} {x} != X1 {y}")
    return problems


def accept(
    results: Path = RERUN, reference: Path = Path("results"), models: Path = MODELS
) -> list[str]:
    """Problems under the Plan's reproduction tolerance (ADR 0018 §1)."""
    problems = []
    for site in SITES:
        cid = CONFIG.format(site=site)
        for seed in SEEDS:
            new = results / "X1" / cid / f"seed_{seed}.json"
            ref = reference / "X1" / cid / f"seed_{seed}.json"
            if not (model_dir(site, seed, models) / "model.json").is_file():
                problems.append(f"{cid} seed {seed}: no model")
            if not new.is_file() or not ref.is_file():
                problems.append(f"{cid} seed {seed}: artifact missing")
                continue
            a, b = json.loads(new.read_text()), json.loads(ref.read_text())
            if a["privacy"] != b["privacy"]:
                problems.append(f"{cid} seed {seed}: privacy accounting differs from X1")
            d = abs(float(a["metrics"]["auroc"]) - float(b["metrics"]["auroc"]))
            if d > AUROC_TOL:
                problems.append(f"{cid} seed {seed}: AUROC differs from X1 by {d:.6f}")
    return problems


def sensitivity(
    results: Path = RERUN, reference: Path = Path("results"), out: Path | None = None
) -> dict[str, Any]:
    """The Week 12 X1 and X2 analyses with the reruns in place of the mismatched seeds.

    The committed analyses are not changed. A copy of ``results/X1`` and
    ``results/X2`` is built in a temporary directory, the seeds that differ exactly
    are replaced by their reruns (artifact and score file), and both analyses are
    run on it. Every test is compared with the committed analysis (ADR 0018 §1).
    """
    import shutil
    import tempfile

    from agmesh.common.artifacts import write_artifact
    from agmesh.experiments import x1x2_analysis

    swapped = sorted({p.split(":")[0] for p in verify(results, reference)})
    report: dict[str, Any] = {
        "protocol": "docs/adr/0018-week13-decisions.md §1",
        "substituted": swapped,
        "experiments": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for exp in ("X1", "X2"):
            shutil.copytree(reference / exp, root / exp, symlinks=True)
        for item in swapped:
            cid, _, seed = item.rpartition(" seed ")
            for suffix in (".json", ".scores.npz"):
                src = results / "X1" / cid / f"seed_{seed}{suffix}"
                shutil.copyfile(src, root / "X1" / cid / f"seed_{seed}{suffix}")
        for exp in ("X1", "X2"):
            new = x1x2_analysis.run(root, exp)
            old = json.loads((reference / exp / "analysis_val.json").read_text())
            rows = []
            for a, b in zip(old["tests"], new["tests"], strict=True):
                rows.append(
                    {
                        "label": a["label"],
                        "family": a.get("family"),
                        "difference": [a.get("observed_difference"), b.get("observed_difference")],
                        "ci": [a.get("ci"), b.get("ci")],
                        "p_one_sided": [a.get("p_one_sided"), b.get("p_one_sided")],
                        "p_holm": [a.get("p_holm"), b.get("p_holm")],
                        "identical": a == b,
                    }
                )
            report["experiments"][exp] = rows

    def biggest(key: str) -> float:
        out_ = 0.0
        for rows_ in report["experiments"].values():
            for r in rows_:
                a, b = r[key]
                if a is None or b is None:
                    continue
                if isinstance(a, list):
                    out_ = max(out_, *(abs(x - y) for x, y in zip(a, b, strict=True)))
                else:
                    out_ = max(out_, abs(a - b))
        return out_

    report["max_change"] = {k: biggest(k) for k in ("difference", "ci", "p_one_sided", "p_holm")}
    if out is not None:
        write_artifact(out, kind="X1.rerun_sensitivity", data_source="runtime", payload=report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 13 headline models (ADR 0017 §1)")
    parser.add_argument("command", choices=("train", "verify", "sensitivity"))
    args = parser.parse_args(argv)
    if args.command == "sensitivity":
        rep = sensitivity(out=Path("results/X1/rerun_sensitivity_val.json"))
        print(f"substituted: {rep['substituted']}")
        print(f"largest change: {rep['max_change']}")
        return 0
    if args.command == "train" and train() != 0:
        return 1
    exact = verify()
    for p in exact:
        print(f"  - {p}")
    print(f"verify: {len(exact)} exact mismatch(es) over {len(SITES) * len(SEEDS)} models (1e-9)")
    problems = accept()
    for p in problems:
        print(f"  ! {p}")
    print(f"accept: {len(problems)} problem(s) under the Plan tolerance (ADR 0018 §1)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
