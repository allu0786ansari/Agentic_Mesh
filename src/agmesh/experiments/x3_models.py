"""Headline models for Week 13, five seeds, retrained and checked (ADR 0017 §1).

    make x3-models       # retrain FedProx ε 3, both sites, five seeds, with --save-model
    python -m agmesh.experiments.x3_models verify

Each seed is retrained in its own process into ``results/x13_rerun``. Its model is
saved to ``results/models/x3/<config>/seed_<n>``. ``verify`` then checks two things:

* every rerun artifact reproduces its X1 artifact, every detection metric to 1e-9;
* every model exists.

A mismatch is listed, and X3 must not use that seed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SITES = ("site_a", "site_b")
SEEDS = (11, 23, 37, 41, 53)
CONFIG = "{site}__fedprox__eps3"
RERUN = Path("results/x13_rerun")
MODELS = Path("results/models/x3")
TOL = 1e-9


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 13 headline models (ADR 0017 §1)")
    parser.add_argument("command", choices=("train", "verify"))
    args = parser.parse_args(argv)
    if args.command == "train" and train() != 0:
        return 1
    problems = verify()
    for p in problems:
        print(f"  - {p}")
    print(f"verify: {len(problems)} problem(s) over {len(SITES) * len(SEEDS)} models")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
