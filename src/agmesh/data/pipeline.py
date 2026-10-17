"""Command-line entry point for the Week 2 data stages.

    python -m agmesh.data.pipeline prepare     # parse Modbus captures -> data/interim
    python -m agmesh.data.pipeline partition   # Site A and Site B partitions + manifests
    python -m agmesh.data.pipeline check       # integrity and label checks -> results/data

``prepare`` and ``partition`` are the DVC stages of the same names (``dvc.yaml``).
``check`` writes ``results/data/partition_check.json``, a small artifact with
provenance that is committed to Git as evidence that the partitions are sound.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params
from agmesh.data import checks, site_a, site_b
from agmesh.data.access import load_partition
from agmesh.data.manifest import read_manifest
from agmesh.evaluation.diagnostics import label_proxies

DEFAULT_EXPECTATIONS = Path("configs/partition_expectations.yaml")
DEFAULT_CHECK_OUTPUT = Path("results/data/partition_check.json")


def _counts(manifest: dict[str, Any]) -> dict[str, Any]:
    """The per-node counts from a manifest, without hashes or file details."""
    nodes = {}
    for client, node in manifest["nodes"].items():
        nodes[client] = {
            part: {k: v for k, v in summary.items() if k != "sha256"}
            for part, summary in node["partitions"].items()
        }
    return nodes


def _label_proxies(
    params: Params, root: Path, manifests: dict[str, dict[str, Any]]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Features that reproduce or perfectly separate the label, on attack validation only."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for site, manifest in manifests.items():
        out[site] = {}
        for client in manifest["nodes"]:
            part = load_partition(params, site, client, "val_attack", root=root)  # type: ignore[arg-type]
            out[site][client] = [
                p.to_dict() for p in label_proxies(part.features, part.labels.target)
            ]
    return out


def cmd_prepare(params: Params, args: argparse.Namespace) -> int:
    modbus_dir = args.modbus_dir or params.paths.modbus_dir
    interim = args.interim_dir or params.paths.interim_dir
    print(f"Parsing Modbus captures in {modbus_dir}")
    manifest = site_b.prepare(params, modbus_dir, interim)
    total = sum(
        e["packets"] for split in manifest["captures"].values() for d in split.values() for e in d
    )
    print(f"Parsed {total:,} packets into {interim / 'modbus'}")
    return 0


def cmd_partition(params: Params, args: argparse.Namespace) -> int:
    out = args.out or params.paths.partitions_dir
    if args.scheme is not None:  # X2's IID arm (ADR 0016 §3), into its own directory
        if args.out is None or args.site != "a":
            raise SystemExit("--scheme needs --site a and an explicit --out directory")
        raw = params.model_dump()
        raw["data"]["partition"]["site_a"]["scheme"] = args.scheme
        params = Params.model_validate(raw)
    if args.site in ("a", "all"):
        hai_dir = args.hai_dir or params.paths.hai_dir
        print(f"Site A from {hai_dir}")
        m = site_a.partition(params, hai_dir, out / "site_a")
        print(
            f"Site A: {m['features']['n_features']} features; events {m['events']['total']} "
            f"(validation {m['events']['val_attack']}, test {m['events']['test']})"
        )
    if args.site in ("b", "all"):
        interim = args.interim_dir or params.paths.interim_dir
        print(f"Site B from {interim / 'modbus'}")
        m = site_b.partition(params, interim, out / "site_b")
        per_node = {n["device"]: n["attack_day_attack_windows"] for n in m["nodes"].values()}
        print(f"Site B: {len(m['episodes'])} attacker episodes; attack windows {per_node}")
    return 0


def cmd_check(params: Params, args: argparse.Namespace) -> int:
    root = args.out or params.paths.partitions_dir
    dirs = checks.partition_dirs(root)
    if not dirs:
        print(f"No manifests found under {root}; run the partition stage first", file=sys.stderr)
        return 1
    manifests = {site: read_manifest(path) for site, path in dirs.items()}
    expected = None
    if not args.no_expectations and args.expectations.is_file():
        expected = yaml.safe_load(args.expectations.read_text(encoding="utf-8"))
    results = checks.run_checks(manifests, params, expected)
    proxies = _label_proxies(params, root, manifests)
    failed = [c for c in results if not c["ok"]]
    write_artifact(
        args.output,
        kind="data.partition_check",
        data_source="fixture" if args.fixture else "runtime",
        payload={
            "passed": len(results) - len(failed),
            "total": len(results),
            "checks": results,
            "manifest_hashes": {
                site: {"features": m["features"], "sources": m["sources"]}
                for site, m in manifests.items()
            },
            "counts": {site: _counts(m) for site, m in manifests.items()},
            "label_proxies_val_attack": proxies,
        },
        params_path=args.params,
    )
    print(f"Partition checks: {len(results) - len(failed)}/{len(results)} passed -> {args.output}")
    for site, per_client in proxies.items():
        for client, found in per_client.items():
            exact = [p["feature"] for p in found if p["identical_to_label"]]
            split = [
                p["feature"] for p in found if p["perfect_split"] and not p["identical_to_label"]
            ]
            if exact or split:
                print(
                    f"  note: {site}/{client} label proxies: identical {exact}, "
                    f"perfect split {split}"
                )
    for c in failed:
        print(f"  FAILED {c['check']}: {c['detail']}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week 2 data pipeline")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="parse Modbus captures into data/interim")
    prep.add_argument("--modbus-dir", type=Path, default=None)
    prep.add_argument("--interim-dir", type=Path, default=None)

    part = sub.add_parser("partition", help="build Site A and Site B partitions")
    part.add_argument("--site", choices=("a", "b", "all"), default="all")
    part.add_argument("--hai-dir", type=Path, default=None)
    part.add_argument("--interim-dir", type=Path, default=None)
    part.add_argument("--out", type=Path, default=None)
    part.add_argument(
        "--scheme", choices=("time", "iid"), default=None, help="override Site A's scheme"
    )

    chk = sub.add_parser("check", help="integrity and label checks on the manifests")
    chk.add_argument("--out", type=Path, default=None, help="partitions directory")
    chk.add_argument("--expectations", type=Path, default=DEFAULT_EXPECTATIONS)
    chk.add_argument("--no-expectations", action="store_true", help="structural checks only")
    chk.add_argument("--fixture", action="store_true", help="mark the artifact as fixture data")
    chk.add_argument("--output", type=Path, default=DEFAULT_CHECK_OUTPUT)

    args = parser.parse_args(argv)
    params = load_params(args.params)
    commands = {"prepare": cmd_prepare, "partition": cmd_partition, "check": cmd_check}
    return commands[args.command](params, args)


if __name__ == "__main__":
    raise SystemExit(main())
