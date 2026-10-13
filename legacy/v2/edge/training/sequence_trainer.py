from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from edge.models.sequence_vae import RecurrentFlowVAE, sequence_vae_loss
from edge.training.trainer import TrainingResult


class SequenceVAETrainer:
    """Non-private trainer for the recurrent-flow VAE upgrade."""

    def __init__(self, model: RecurrentFlowVAE, *, learning_rate: float = 1e-3, beta: float = 0.3, free_bits: float = 0.05, device: str = "cpu"):
        self.model = model.to(device)
        self.learning_rate = learning_rate
        self.beta = beta
        self.free_bits = free_bits
        self.device = torch.device(device)

    def train(self, dataset: Dataset[torch.Tensor], *, epochs: int = 1, batch_size: int = 64, checkpoint_path: str | Path | None = None) -> TrainingResult:
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        losses: list[float] = []
        samples = 0
        for _ in range(epochs):
            self.model.train()
            total = 0.0
            samples = 0
            for inputs in loader:
                inputs = inputs.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                output = self.model(inputs)
                loss, _, _ = sequence_vae_loss(output, inputs, beta=self.beta, free_bits=self.free_bits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                count = inputs.shape[0]
                total += float(loss.detach()) * count
                samples += count
            losses.append(total / max(samples, 1))
        destination = None
        if checkpoint_path is not None:
            destination = Path(checkpoint_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "architecture": self.model.architecture,
                "feature_count": self.model.feature_count,
                "latent_dim": self.model.latent_dim,
                "hidden_dim": self.model.hidden_dim,
                "graph_embed_dim": self.model.graph_embed_dim,
                "flow_layers": self.model.flow_layers,
                "beta": self.beta,
                "free_bits": self.free_bits,
                "loss": losses[-1],
                "variant": "non_private",
            }, destination)
        return TrainingResult(epochs=epochs, samples=samples, losses=losses, checkpoint_path=destination)
