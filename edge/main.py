"""Command line entrypoint for edge node training and inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import yaml
from loguru import logger

from edge.config import load_config
from edge.data_loader import WindowedDataset
from edge.models.vae import VAE
from edge.training.dp_trainer import DPVAETrainer
from edge.training.trainer import VAETrainer


ROOT = Path(__file__).resolve().parents[1]


def load_params() -> dict:
    """Load project hyperparameters from params.yaml."""
    with (ROOT / "params.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train(use_dp: bool = True) -> None:
    """Train the configured node model."""
    config = load_config()
    params = load_params()
    if config.model_type != "vae":
        raise NotImplementedError("Week 4 edge/main.py supports VAE nodes only")

    dataset = WindowedDataset(config.train_path)
    input_dim = dataset.input_dim
    model = VAE(input_dim=input_dim, latent_dim=int(params.get("vae_latent_dim", 32)))
    checkpoint = ROOT / "models" / f"{config.node_id}_vae_v1.pt"

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment("local-edge-training")
    with mlflow.start_run(run_name=f"{config.node_id}-vae"):
        mlflow.log_params(
            {
                "node_id": config.node_id,
                "input_dim": input_dim,
                "latent_dim": params.get("vae_latent_dim", 32),
                "dp_enabled": use_dp,
            }
        )
        if use_dp:
            trainer = DPVAETrainer(
                model=model,
                dataset=dataset,
                checkpoint_path=checkpoint,
                epsilon=config.epsilon,
                delta=config.delta,
                max_grad_norm=config.max_grad_norm,
                batch_size=int(params.get("vae_batch_size", 64)),
                learning_rate=float(params.get("vae_lr", 1e-3)),
                epochs=int(params.get("vae_epochs", 20)),
            )
        else:
            trainer = VAETrainer(
                model=model,
                dataset=dataset,
                checkpoint_path=checkpoint,
                batch_size=int(params.get("vae_batch_size", 64)),
                learning_rate=float(params.get("vae_lr", 1e-3)),
                epochs=int(params.get("vae_epochs", 20)),
            )

        result = trainer.train()
        mlflow.log_metric("best_val_loss", result.best_val_loss)
        if result.epsilon is not None:
            mlflow.log_metric("epsilon_final", result.epsilon)
        mlflow.log_artifact(str(result.checkpoint_path))

    logger.success("saved checkpoint to {}", result.checkpoint_path)


def main() -> None:
    """Parse CLI arguments and dispatch the selected mode."""
    parser = argparse.ArgumentParser(description="Edge node entrypoint")
    parser.add_argument("--mode", choices=["train"], default="train")
    parser.add_argument("--no-dp", action="store_true", help="Disable DP-SGD for debugging")
    args = parser.parse_args()

    if args.mode == "train":
        train(use_dp=not args.no_dp)


if __name__ == "__main__":
    main()
