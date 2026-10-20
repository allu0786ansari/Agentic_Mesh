"""Clean-checkout reproduction (ADR 0019 §2; Plan Week 14 task 2; CC7).

    make reproduce                                    # in a fresh clone, raw data copied
    make reproduce REF=~/Agentic_Mesh/data/partitions # also compare every partition file

**Steps.** Each one is recorded in ``results/reproduce/reproduction.json``.

1. **Checksums:** ``scripts/verify_checksums.py --verify``. The raw data is the one
   in ``data/raw_checksums.json``.
2. **Partitions:** ``agmesh.data.pipeline prepare``, ``partition`` and ``check``
   rebuild them from raw. Each site's ``manifest.json`` must hash to the
   ``manifest_sha256`` recorded in the committed headline X1 artifact. With ``REF``,
   every partition file's SHA-256 must also equal the reference manifest's.
3. **Headline models:** FedProx ε 3, seed 11, is retrained on both sites into
   ``results/reproduce/run``.
4. **Baselines:** ``central__vae`` (Site A) and ``central__ae`` (Site B), seed 11.

**Tolerance.** For 3 and 4, the Plan's: identical privacy accounting, and AUROC
within ±0.005 of the committed artifact. Whether every metric is exact is reported
too.

**Pass:** every step holds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import load_params
from agmesh.common.provenance import sha256_file
from agmesh.evaluation.results import compare_validation, result_path

OUT = Path("results/reproduce")
HEADLINE = {"site_a": "site_a__fedprox__eps3", "site_b": "site_b__fedprox__eps3"}
BASELINE = {"site_a": ("vae", "site_a__central__vae"), "site_b": ("ae", "site_b__central__ae")}
SEED = 11


def _call(cmd: list[str]) -> dict[str, Any]:
    print("$", " ".join(cmd), flush=True)
    code = subprocess.call(cmd)
    return {"cmd": cmd, "returncode": code, "ok": code == 0}


def manifests(partitions: Path, reference: Path | None, results: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for site, cid in HEADLINE.items():
        path = partitions / site / "manifest.json"
        ref_doc = json.loads(result_path(results, "X1", cid, SEED).read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "manifest_sha256": sha256_file(path) if path.is_file() else None,
            "expected": ref_doc["provenance"]["manifest_sha256"],
        }
        row["manifest_matches"] = row["manifest_sha256"] == row["expected"]
        if reference is not None and path.is_file():
            new = json.loads(path.read_text(encoding="utf-8"))
            old = json.loads((reference / site / "manifest.json").read_text(encoding="utf-8"))
            files = {}
            for node, v in old["nodes"].items():
                for part, info in v["partitions"].items():
                    got = new["nodes"].get(node, {}).get("partitions", {}).get(part, {})
                    files[f"{node}/{part}"] = got.get("sha256") == info["sha256"]
            row["files_match"] = all(files.values())
            row["files"] = files
        row["ok"] = row["manifest_matches"] and row.get("files_match", True)
        out[site] = row
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean-checkout reproduction (ADR 0019 §2)")
    parser.add_argument("--reference-partitions", type=Path, default=None)
    parser.add_argument("--skip-data", action="store_true", help="partitions already rebuilt")
    args = parser.parse_args(argv)
    params = load_params()
    py = sys.executable
    steps: dict[str, Any] = {}
    steps["checksums"] = _call([py, "scripts/verify_checksums.py", "--verify"])
    if not args.skip_data:
        steps["data"] = {
            s: _call([py, "-m", "agmesh.data.pipeline", s])
            for s in ("prepare", "partition", "check")
        }
    parts = params.paths.partitions_dir
    steps["partitions"] = manifests(parts, args.reference_partitions, Path("results"))
    run_root = OUT / "run"
    models: dict[str, Any] = {}
    for site, cid in HEADLINE.items():
        call = _call([
            py, "-m", "agmesh.federated.run", "--site", site, "--strategy", "fedprox",
            "--epsilon", "3", "--seeds", str(SEED), "--results", str(run_root),
        ])  # fmt: skip
        models[cid] = {"run": call, **_compare(run_root, cid)}
    for site, (kind, cid) in BASELINE.items():
        call = _call([
            py, "-m", "agmesh.experiments.x1_baselines", "--sites", site, "--models", kind,
            "--seeds", str(SEED), "--results", str(run_root),
        ])  # fmt: skip
        models[cid] = {"run": call, **_compare(run_root, cid)}
    steps["models"] = models
    ok = (
        steps["checksums"]["ok"]
        and all(v["ok"] for v in steps.get("data", {}).values())
        and all(v["ok"] for v in steps["partitions"].values())
        and all(v["run"]["ok"] and v["comparison"]["accepted"] for v in models.values())
    )
    path = write_artifact(OUT / "reproduction.json", kind="reproduction", data_source="runtime",
                          payload={"protocol": "docs/adr/0019-g4-final-test-protocol-week14.md §2",
                                   "seed": SEED, "steps": steps, "passed": ok})  # fmt: skip
    for cid, v in models.items():
        c = v["comparison"]
        print(f"{cid}: accepted {c['accepted']} exact {c.get('exact')} "
              f"|dAUROC| {c.get('auroc_abs_difference')}")  # fmt: skip
    for site, v in steps["partitions"].items():
        print(f"{site} partitions: manifest {v['manifest_matches']} files {v.get('files_match')}")
    print(f"reproduction {'PASSED' if ok else 'FAILED'} -> {path}")
    return 0 if ok else 1


def _compare(run_root: Path, cid: str) -> dict[str, Any]:
    path = result_path(run_root, "X1", cid, SEED)
    if not path.is_file():
        return {"comparison": {"accepted": False, "reason": "no artifact"}}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {
        "artifact": path.as_posix(),
        "comparison": compare_validation(
            Path("results"), "X1", cid, SEED, doc["metrics"], doc.get("privacy")
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
