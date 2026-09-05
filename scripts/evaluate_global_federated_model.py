from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from edge.data_loader import HAIWindowDataset
from evaluation.inference import (
    bootstrap_confidence_intervals,
    error_distribution,
    evaluate_scores,
    event_level_metrics,
    fixed_fpr_operating_points,
    load_checkpoint_model,
    reconstruction_errors,
)
from federated.model_loader import GlobalModelLoader


def _merge_global_core(model: torch.nn.Module, global_parameters: list[np.ndarray]) -> None:
    state = model.state_dict()
    names = tuple(name for name in state if not name.startswith("encoder.0.") and not name.startswith("decoder.4."))
    if len(names) != len(global_parameters):
        raise ValueError(f"Global core has {len(global_parameters)} tensors; local model expects {len(names)}")
    for name, array in zip(names, global_parameters):
        tensor = torch.as_tensor(array, dtype=state[name].dtype, device=state[name].device)
        if tuple(tensor.shape) != tuple(state[name].shape):
            raise ValueError(f"Global tensor shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(state[name].shape)}")
        state[name] = tensor
    model.load_state_dict(state, strict=True)


def evaluate_global_model(
    global_summary: Path,
    *,
    checkpoints_root: Path,
    partitions_root: Path,
    scalers_root: Path,
    output_root: Path,
    nodes: list[str],
    batch_size: int,
    bootstrap_samples: int,
    max_windows: int | None,
    seed: int,
) -> dict[str, Any]:
    global_parameters, summary = GlobalModelLoader(global_summary.parent.parent).load(global_summary.parent.name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    node_results: list[dict[str, Any]] = []
    for node_id in nodes:
        checkpoint_path = checkpoints_root / f"{node_id}.pt"
        model, checkpoint = load_checkpoint_model(checkpoint_path)
        _merge_global_core(model, global_parameters)
        node_root = partitions_root / node_id
        scaler_path = scalers_root / f"{node_id}.json"
        validation = HAIWindowDataset(node_root / "train.parquet", normal_only=True, start_fraction=0.8, end_fraction=1.0, max_windows=max_windows, scaler_path=scaler_path)
        test = HAIWindowDataset(node_root / "test.parquet", normal_only=False, max_windows=max_windows, scaler_path=scaler_path)
        validation_scores = reconstruction_errors(model, validation, batch_size=batch_size)
        test_scores = reconstruction_errors(model, test, batch_size=batch_size)
        labels = test.window_labels.astype(np.int8)
        threshold = float(np.quantile(validation_scores, 0.95))
        metrics = evaluate_scores(test_scores, labels, threshold=threshold)
        node_result = {
            "node_id": node_id,
            "source_global_run": str(global_summary),
            "local_checkpoint": str(checkpoint_path),
            "checkpoint_epsilon": checkpoint.get("epsilon"),
            "validation": {"windows": len(validation), "threshold": threshold, "errors": error_distribution(validation_scores)},
            "test": {
                "windows": len(test),
                "attack_windows": int(labels.sum()),
                "metrics": metrics,
                "bootstrap_confidence_intervals": bootstrap_confidence_intervals(test_scores, labels, threshold, seed=seed, samples=bootstrap_samples),
                "event_level_metrics": event_level_metrics(test.labels, test.window_starts, test.window_size, test_scores, threshold),
                "fixed_fpr_operating_points": fixed_fpr_operating_points(validation_scores, test_scores, labels),
            },
        }
        node_results.append(node_result)
        (output_dir / f"{node_id}.json").write_text(json.dumps(node_result, indent=2), encoding="utf-8")
    aucs = [item["test"]["metrics"]["roc_auc"] for item in node_results if item["test"]["metrics"]["roc_auc"] is not None]
    f1s = [item["test"]["metrics"]["f1_score"] for item in node_results]
    result = {
        "run_id": run_id,
        "source_global_run": summary.get("run_id"),
        "strategy": summary.get("strategy"),
        "nodes": node_results,
        "aggregate": {"macro_roc_auc": float(np.mean(aucs)) if aucs else None, "macro_f1": float(np.mean(f1s)) if f1s else None},
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "summary": str(output_dir / "summary.json")}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved federated VAE core on held-out HAI attack labels")
    parser.add_argument("--global-summary", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints/20260821T024039Z"))
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--scalers-root", type=Path, default=Path("results/scalers"))
    parser.add_argument("--results-root", type=Path, default=Path("results/week9_global_evaluation"))
    parser.add_argument("--nodes", nargs="+", default=["node_01", "node_02", "node_03"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate_global_model(args.global_summary, checkpoints_root=args.checkpoints_root, partitions_root=args.partitions_root, scalers_root=args.scalers_root, output_root=args.results_root, nodes=args.nodes, batch_size=args.batch_size, bootstrap_samples=args.bootstrap_samples, max_windows=args.max_windows, seed=args.seed)


if __name__ == "__main__":
    main()
