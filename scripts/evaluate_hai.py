from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.inference import evaluate_node


def resolve_checkpoint_root(root: Path, nodes: list[str]) -> Path:
    """Use an explicit root or discover the newest complete isolated training run."""
    if all((root / f"{node_id}.pt").exists() for node_id in nodes):
        return root
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and all((path / f"{node_id}.pt").exists() for node_id in nodes)
    ] if root.exists() else []
    if not candidates:
        expected = ", ".join(f"{node_id}.pt" for node_id in nodes)
        raise FileNotFoundError(
            f"No complete checkpoint run found under {root}. Expected {expected}."
        )
    return sorted(candidates)[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained HAI VAE checkpoints")
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-root", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--nodes", nargs="+", default=["node_01", "node_02", "node_03"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--threshold-method", choices=("quantile", "pot"), default="quantile")
    parser.add_argument("--overlap-strides", nargs="+", type=int, default=[10, 30, 60])
    parser.add_argument("--scalers-root", type=Path, default=Path("results/scalers"))
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    checkpoint_root = resolve_checkpoint_root(args.checkpoint_root, args.nodes)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for node_id in args.nodes:
        node_root = args.partitions_root / node_id
        scaler_path = args.scalers_root / f"{node_id}.json"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler artifact: {scaler_path}. Run scripts.prepare_hai_scalers first.")
        result = evaluate_node(
            checkpoint_root / f"{node_id}.pt",
            node_root / "train.parquet",
            node_root / "test.parquet",
            batch_size=args.batch_size,
            validation_quantile=args.threshold_quantile,
            plot_path=run_dir / f"{node_id}_curves.png",
            scaler_path=scaler_path,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            threshold_method=args.threshold_method,
        )
        overlap_results = {}
        for overlap_stride in args.overlap_strides:
            overlap = evaluate_node(
                checkpoint_root / f"{node_id}.pt",
                node_root / "train.parquet",
                node_root / "test.parquet",
                stride=overlap_stride,
                batch_size=args.batch_size,
                validation_quantile=args.threshold_quantile,
                scaler_path=scaler_path,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                threshold_method=args.threshold_method,
            )
            overlap_results[str(overlap_stride)] = {
                "validation_windows": overlap["validation"]["threshold_windows"],
                "test_windows": overlap["test"]["metrics"]["positive_windows"] + overlap["test"]["metrics"]["negative_windows"],
                "roc_auc": overlap["test"]["metrics"]["roc_auc"],
                "pr_auc": overlap["test"]["metrics"]["pr_auc"],
                "fixed_fpr": overlap["test"]["fixed_fpr_operating_points"],
            }
        result["overlap_sensitivity"] = overlap_results
        result["run_id"] = run_id
        result["node_id"] = node_id
        results.append(result)
        (run_dir / f"{node_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {"run_id": run_id, "nodes": results}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "results": str(run_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
