from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from edge.config import Config
from edge.data_loader import HAIWindowDataset
from edge.models.vae import VAE
from edge.training.dp_trainer import DPVAETrainer
from edge.training.trainer import VAETrainer
from evaluation.inference import evaluate_node


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DP versus non-private multi-seed HAI study")
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--results-root", type=Path, default=Path("results/comparative"))
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--nodes", nargs="+", default=["node_01", "node_02", "node_03"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--variants", nargs="+", choices=["dp", "non_private"], default=["dp", "non_private"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = Config.from_env()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_root / run_id
    checkpoint_root = args.checkpoint_root or run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict] = []

    for node_id in args.nodes:
        train_path = args.partitions_root / node_id / "train.parquet"
        test_path = args.partitions_root / node_id / "test.parquet"
        scaler_path = Path(config.scaler_root) / f"{node_id}.json"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler artifact: {scaler_path}. Run scripts.prepare_hai_scalers first.")
        train_dataset = HAIWindowDataset(
            train_path,
            window_size=config.hai_window_size,
            stride=config.hai_window_stride,
            normal_only=True,
            start_fraction=0.0,
            end_fraction=0.8,
            scaler_path=scaler_path,
        )
        for seed in args.seeds:
            for variant in args.variants:
                set_seed(seed)
                model = VAE(train_dataset.input_dim, latent_dim=config.latent_dim, hidden_dim=config.hidden_dim)
                checkpoint = checkpoint_root / variant / f"{node_id}_seed_{seed}.pt"
                epochs = args.epochs or config.local_epochs
                if variant == "dp":
                    trainer = DPVAETrainer(
                        model,
                        learning_rate=config.learning_rate,
                        max_grad_norm=config.max_grad_norm,
                        secure_rng=config.secure_rng,
                        device="cpu",
                    )
                    result = trainer.train(
                        train_dataset,
                        epochs=epochs,
                        batch_size=config.batch_size,
                        epsilon=config.epsilon,
                        delta=config.delta,
                        checkpoint_path=checkpoint,
                    )
                else:
                    trainer = VAETrainer(model, learning_rate=config.learning_rate, device="cpu")
                    result = trainer.train(
                        train_dataset,
                        epochs=epochs,
                        batch_size=config.batch_size,
                        checkpoint_path=checkpoint,
                    )
                evaluation = evaluate_node(
                    checkpoint,
                    train_path,
                    test_path,
                    window_size=config.hai_window_size,
                    stride=config.hai_window_stride,
                    batch_size=256,
                    plot_path=run_dir / "plots" / f"{variant}_{node_id}_seed_{seed}.png",
                    scaler_path=scaler_path,
                    seed=seed,
                )
                metrics = evaluation["test"]["metrics"]
                records.append(
                    {
                        "run_id": run_id,
                        "node_id": node_id,
                        "variant": variant,
                        "seed": seed,
                        "epochs": result.epochs,
                        "checkpoint": str(checkpoint),
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "false_positive_rate": metrics["false_positive_rate"],
                        "fixed_fpr": evaluation["test"]["fixed_fpr_operating_points"],
                        "overlap_stride": config.hai_window_stride,
                    }
                )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[f"{record['node_id']}::{record['variant']}"] .append(record)
    aggregate = []
    metric_names = ["roc_auc", "pr_auc", "precision", "recall", "f1_score", "false_positive_rate"]
    for key, group in grouped.items():
        node_id, variant = key.split("::")
        row = {"node_id": node_id, "variant": variant, "runs": len(group)}
        for metric in metric_names:
            values = np.asarray([item[metric] for item in group], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(row)

    output = {"run_id": run_id, "seeds": args.seeds, "records": records, "aggregate": aggregate}
    (run_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "summary": str(run_dir / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
