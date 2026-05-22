"""Differentially private VAE trainer built on Opacus."""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
from loguru import logger
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, random_split

from edge.data_loader import WindowedDataset
from edge.models.vae import VAE, vae_loss
from edge.training.trainer import TrainingResult


class DPVAETrainer:
    """Train a VAE with DP-SGD and report consumed epsilon."""

    def __init__(
        self,
        model: VAE,
        dataset: WindowedDataset,
        checkpoint_path: str | Path,
        epsilon: float = 2.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        beta: float = 1.0,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.checkpoint_path = Path(checkpoint_path)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = beta
        self.privacy_engine: PrivacyEngine | None = None

    def train(self) -> TrainingResult:
        """Run DP-SGD local training and save the best checkpoint."""
        val_size = max(1, int(len(self.dataset) * 0.2))
        train_size = len(self.dataset) - val_size
        train_set, val_set = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.privacy_engine = PrivacyEngine()
        self.model, optimizer, train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=self.epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )

        best_val_loss = float("inf")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_loss = self._evaluate_loss(val_loader)
            consumed_epsilon = self.privacy_engine.get_epsilon(self.delta)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("epsilon_consumed", consumed_epsilon, step=epoch)
            logger.info(
                "epoch={} train_loss={:.6f} val_loss={:.6f} epsilon={:.4f}",
                epoch,
                train_loss,
                val_loss,
                consumed_epsilon,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)

        consumed_epsilon = self.privacy_engine.get_epsilon(self.delta)
        return TrainingResult(self.checkpoint_path, best_val_loss, self.epochs, consumed_epsilon)

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        total_rows = 0
        for batch, _labels in loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction, mu, log_var = self.model(batch)
            loss = vae_loss(reconstruction, batch, mu, log_var, self.beta)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch)
            total_rows += len(batch)
        return total_loss / max(total_rows, 1)

    @torch.no_grad()
    def _evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_rows = 0
        for batch, _labels in loader:
            reconstruction, mu, log_var = self.model(batch)
            loss = vae_loss(reconstruction, batch, mu, log_var, self.beta)
            total_loss += float(loss) * len(batch)
            total_rows += len(batch)
        return total_loss / max(total_rows, 1)
