from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, Dataset

from edge.models.vae import VAE, vae_loss
from edge.training.trainer import TrainingResult


class DPVAETrainer:
    """VAE trainer using Opacus DP-SGD with explicit epsilon accounting."""

    def __init__(
        self,
        model: VAE,
        *,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
        max_grad_norm: float = 1.0,
        secure_rng: bool = False,
        device: str = "cpu",
        logger: Callable[..., None] | None = None,
    ):
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.model = model.to(device)
        self.input_dim = model.input_dim
        self.latent_dim = model.latent_dim
        self.hidden_dim = model.hidden_dim
        self.learning_rate = learning_rate
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.secure_rng = secure_rng
        self.device = torch.device(device)
        self.logger = logger
        self.privacy_engine: PrivacyEngine | None = None

    def train(
        self,
        dataset: Dataset[torch.Tensor],
        *,
        epochs: int = 1,
        batch_size: int = 64,
        epsilon: float = 2.0,
        delta: float = 1e-5,
        checkpoint_path: str | Path | None = None,
    ) -> TrainingResult:
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if epsilon <= 0 or not 0 < delta < 1:
            raise ValueError("epsilon must be positive and delta must be between 0 and 1")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.privacy_engine = PrivacyEngine(secure_mode=self.secure_rng)
        self.model, optimizer, loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs=epochs,
            max_grad_norm=self.max_grad_norm,
        )
        losses: list[float] = []
        epoch_times_seconds: list[float] = []
        epsilon_per_epoch: list[float] = []
        sample_count = 0
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
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
                count = inputs.shape[0]
                total_loss += float(loss.detach()) * count
                sample_count += count
            epoch_loss = total_loss / max(sample_count, 1)
            epoch_time = time.time() - epoch_start
            epoch_times_seconds.append(epoch_time)
            current_epsilon = self.epsilon(delta)
            losses.append(epoch_loss)
            epsilon_per_epoch.append(current_epsilon)
            if self.logger:
                self.logger(epoch, epoch_loss, current_epsilon, epoch_time)
        final_epsilon = self.epsilon(delta)
        destination = self._save_checkpoint(checkpoint_path, losses[-1] if losses else None, final_epsilon, delta)
        return TrainingResult(
            epochs=epochs,
            samples=len(dataset),
            losses=losses,
            checkpoint_path=destination,
            epsilon=final_epsilon,
            epoch_times_seconds=epoch_times_seconds,
            epsilon_per_epoch=epsilon_per_epoch,
        )
    def epsilon(self, delta: float) -> float:
        if self.privacy_engine is None:
            raise RuntimeError("train must be called before epsilon")
        return float(self.privacy_engine.get_epsilon(delta))

    def _save_checkpoint(
        self,
        path: str | Path | None,
        loss: float | None,
        epsilon: float,
        delta: float,
    ) -> Path | None:
        if path is None:
            return None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "loss": loss,
                "epsilon": epsilon,
                "delta": delta,
                "max_grad_norm": self.max_grad_norm,
                "architecture": getattr(self.model, "architecture", "fully_connected"),
                "window_size": getattr(self.model, "window_size", None),
                "feature_count": getattr(self.model, "feature_count", None),
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
