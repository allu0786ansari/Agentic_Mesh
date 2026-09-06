from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset

from edge.models.vae import VAE, vae_loss


@dataclass(frozen=True)
class TrainingResult:
    epochs: int
    samples: int
    losses: list[float]
    checkpoint_path: Path | None
    epsilon: float | None = None
    epoch_times_seconds: list[float] | None = None
    epsilon_per_epoch: list[float] | None = None


class VAETrainer:
    """Reusable non-private VAE training loop with checkpoint persistence."""

    def __init__(
        self,
        model: VAE,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        device: str = "cpu",
        logger: Callable[[int, float], None] | None = None,
    ):
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.beta = beta
        self.device = torch.device(device)
        self.logger = logger

    def train(
        self,
        dataset: Dataset[torch.Tensor],
        *,
        epochs: int = 1,
        batch_size: int = 64,
        checkpoint_path: str | Path | None = None,
    ) -> TrainingResult:
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        losses: list[float] = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            sample_count = 0
            for batch in loader:
                inputs = _batch_inputs(batch).to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = self.model(inputs)
                loss, _, _ = vae_loss(output, inputs, beta=self.beta)
                loss.backward()
                optimizer.step()
                batch_size_actual = inputs.shape[0]
                total_loss += float(loss.detach()) * batch_size_actual
                sample_count += batch_size_actual
            epoch_loss = total_loss / max(sample_count, 1)
            losses.append(epoch_loss)
            if self.logger:
                self.logger(epoch, epoch_loss)
        saved_path = self._save_checkpoint(checkpoint_path, losses[-1] if losses else None)
        return TrainingResult(epochs=epochs, samples=sample_count, losses=losses, checkpoint_path=saved_path)

    def _save_checkpoint(self, path: str | Path | None, loss: float | None) -> Path | None:
        if path is None:
            return None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.model.input_dim,
                "latent_dim": self.model.latent_dim,
                "hidden_dim": self.model.hidden_dim,
                "architecture": getattr(self.model, "architecture", "fully_connected"),
                "window_size": getattr(self.model, "window_size", None),
                "feature_count": getattr(self.model, "feature_count", None),
                "loss": loss,
                "variant": "non_private",
            },
            destination,
        )
        return destination


def _batch_inputs(batch: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], torch.Tensor):
        return batch[0]
    raise TypeError("VAE datasets must yield a tensor or a tensor-first tuple/list")

