"""Unit tests for the Opacus DP trainer."""

from __future__ import annotations

import pandas as pd
import torch

from edge.data_loader import WindowedDataset
from edge.models.vae import VAE
from edge.training.dp_trainer import DPVAETrainer


def test_dp_trainer_returns_positive_epsilon(tmp_path) -> None:
    rows = 100
    input_dim = 20
    df = pd.DataFrame(torch.rand(rows, input_dim).numpy(), columns=[f"feature_{i}" for i in range(input_dim)])
    df["label"] = 0
    parquet_path = tmp_path / "train.parquet"
    df.to_parquet(parquet_path, index=False)

    dataset = WindowedDataset(parquet_path)
    model = VAE(input_dim=input_dim, latent_dim=4)
    trainer = DPVAETrainer(
        model=model,
        dataset=dataset,
        checkpoint_path=tmp_path / "model.pt",
        epsilon=5.0,
        delta=1e-5,
        max_grad_norm=1.0,
        batch_size=16,
        learning_rate=1e-3,
        epochs=2,
    )

    result = trainer.train()

    assert result.checkpoint_path.exists()
    assert result.epsilon is not None
    assert result.epsilon > 0
    assert torch.isfinite(torch.tensor(result.epsilon))
