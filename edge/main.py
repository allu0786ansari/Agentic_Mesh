from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

from agents.metrics import log_event
from edge.config import Config
from edge.data_loader import HAIWindowDataset
from edge.models.vae import VAE
from edge.training.dp_trainer import DPVAETrainer


def train_node(config: Config, *, partitions_root: Path, checkpoint_root: Path, max_windows: int | None, run_id: str | None = None) -> object:
    run_id = run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    node_dir = partitions_root / config.node_id
    train_path = node_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training partition does not exist: {train_path}")
    scaler_path = Path(config.scaler_root) / f"{config.node_id}.json"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler artifact does not exist: {scaler_path}. Run scripts.prepare_hai_scalers first."
        )
    dataset = HAIWindowDataset(
        train_path,
        window_size=config.hai_window_size,
        stride=config.hai_window_stride,
        normal_only=True,
        max_windows=max_windows,
        scaler_path=scaler_path,
    )
    model = VAE(dataset.input_dim, latent_dim=config.latent_dim, hidden_dim=config.hidden_dim)
    checkpoint = checkpoint_root / f"{config.node_id}.pt"
    run_started = time.time()
    trainer = DPVAETrainer(
        model,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        secure_rng=config.secure_rng,
        device="cpu",
        logger=lambda epoch, loss, epsilon, epoch_time: log_event(
            "vae_epoch",
            {
                "node_id": config.node_id,
                "run_id": run_id,
                "epoch": epoch,
                "loss": loss,
                "epsilon": epsilon,
                "epoch_time_seconds": epoch_time,
                "sample_count": len(dataset),
                "input_dim": dataset.input_dim,
            },
            config.metrics_log_path,
        ),
    )
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("agentic-mesh-week4")
    with mlflow.start_run(run_name=f"{config.node_id}-dp-vae"):
        mlflow.log_params(
            {
                "node_id": config.node_id,
                "run_id": run_id,
                "window_size": config.hai_window_size,
                "window_stride": config.hai_window_stride,
                "latent_dim": config.latent_dim,
                "hidden_dim": config.hidden_dim,
                "batch_size": config.batch_size,
                "epochs": config.local_epochs,
                "target_epsilon": config.epsilon,
                "delta": config.delta,
                "seed": config.seed,
                "secure_rng": config.secure_rng,
            }
        )
        result = trainer.train(
            dataset,
            epochs=config.local_epochs,
            batch_size=config.batch_size,
            epsilon=config.epsilon,
            delta=config.delta,
            checkpoint_path=checkpoint,
        )
        for epoch, loss in enumerate(result.losses, start=1):
            mlflow.log_metric("train_loss", loss, step=epoch)
        for epoch, epoch_time in enumerate(result.epoch_times_seconds or [], start=1):
            mlflow.log_metric("epoch_time_seconds", epoch_time, step=epoch)
        for epoch, epsilon in enumerate(result.epsilon_per_epoch or [], start=1):
            mlflow.log_metric("epsilon_per_epoch", epsilon, step=epoch)
        mlflow.log_metric("epsilon", result.epsilon or 0.0)
        mlflow.log_metric("sample_count", result.samples)
        mlflow.log_metric("duration_seconds", time.time() - run_started)
        mlflow.log_artifact(str(checkpoint))
    
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    total_duration = time.time() - run_started
    
    # Compute convergence metrics
    losses = result.losses
    initial_loss = losses[0] if losses else 0
    final_loss = losses[-1] if losses else 0
    convergence_pct = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
    
    # Find epoch where loss stabilized (within 2% of final)
    stable_after = len(losses)
    if final_loss > 0:
        threshold = final_loss * 1.02
        for i, loss in enumerate(losses):
            if loss <= threshold:
                stable_after = i + 1
                break
    
    summary = {
        "node_id": config.node_id,
                "run_id": run_id,
        "source_partition": str(train_path),
        "scaler": str(scaler_path),
        "dataset_rows": len(dataset),
        "input_dim": dataset.input_dim,
        "feature_count": len(dataset.feature_columns),
        "window_size": config.hai_window_size,
        "window_stride": config.hai_window_stride,
        "training_config": {
            "epochs": result.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "target_epsilon": config.epsilon,
            "delta": config.delta,
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "seed": config.seed,
            "secure_rng": config.secure_rng,
        },
        "training_results": {
            "losses": result.losses,
            "epsilon_per_epoch": result.epsilon_per_epoch or [],
            "epoch_times_seconds": result.epoch_times_seconds or [],
            "total_duration_seconds": total_duration,
            "convergence_metric": {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "convergence_percent": round(convergence_pct, 2),
                "stable_after_epoch": stable_after,
            }
        },
        "privacy_accounting": {
            "final_epsilon": result.epsilon,
            "delta": config.delta,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            "size_mb": round(checkpoint.stat().st_size / (1024**2), 2),
        }
    }
    manifest_path = Path(config.run_manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError(f"Training manifest must contain a JSON list: {manifest_path}")
    existing = [item for item in existing if item.get("node_id") != config.node_id]
    existing.append(summary)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    log_event("vae_run_completed", summary, config.metrics_log_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local Agentic Mesh VAE")
    parser.add_argument("--mode", choices=("train",), required=True)
    parser.add_argument("--partitions-root", type=Path, default=Path("data/partitions"))
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "train":
        config = Config.from_env()
        checkpoint_root = args.checkpoint_root or Path(config.checkpoint_root)
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        result = train_node(
            config,
            partitions_root=args.partitions_root,
            checkpoint_root=checkpoint_root,
            max_windows=args.max_windows,
            run_id=run_id,
        )
        print(
            {
                "node_id": config.node_id,
                "run_id": run_id,
                "epochs": result.epochs,
                "samples": result.samples,
                "epsilon": result.epsilon,
                "checkpoint": str(result.checkpoint_path),
            }
        )


if __name__ == "__main__":
    main()



