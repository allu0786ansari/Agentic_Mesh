"""Local model training loops for edge nodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from loguru import logger
from torch.utils.data import DataLoader, random_split

from edge.data_loader import WindowedDataset
from edge.models.vae import VAE, vae_loss


@dataclass
class TrainingResult:
    """Summary emitted after local training."""

    checkpoint_path: Path
    best_val_loss: float
    epochs_ran: int
    epsilon: float | None = None


class VAETrainer:
    """Train a VAE on one edge node partition."""

    def __init__(
        self,
        model: VAE,
        dataset: WindowedDataset,
        checkpoint_path: str | Path,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        beta: float = 1.0,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.checkpoint_path = Path(checkpoint_path)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = beta

    def _split_loaders(self) -> tuple[DataLoader, DataLoader]:
        val_size = max(1, int(len(self.dataset) * 0.2))
        train_size = len(self.dataset) - val_size
        train_set, val_set = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_set, batch_size=self.batch_size),
        )

    def train(self) -> TrainingResult:
        """Run standard non-private local VAE training."""
        train_loader, val_loader = self._split_loaders()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_loss = self.evaluate_loss(val_loader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            logger.info("epoch={} train_loss={:.6f} val_loss={:.6f}", epoch, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)

        return TrainingResult(self.checkpoint_path, best_val_loss, self.epochs)

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
    def evaluate_loss(self, loader: DataLoader) -> float:
        """Compute average validation loss."""
        self.model.eval()
        total_loss = 0.0
        total_rows = 0
        for batch, _labels in loader:
            reconstruction, mu, log_var = self.model(batch)
            loss = vae_loss(reconstruction, batch, mu, log_var, self.beta)
            total_loss += float(loss) * len(batch)
            total_rows += len(batch)
        return total_loss / max(total_rows, 1)
