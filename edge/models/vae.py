"""Variational autoencoder used by time-series edge nodes."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VAE(nn.Module):
    """Feed-forward VAE for flattened windowed telemetry."""

    def __init__(self, input_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, latent_dim)
        self.log_var_head = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return latent mean and log variance for a batch."""
        hidden = self.encoder(x)
        return self.mu_head(hidden), self.log_var_head(hidden)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample latent vector with the reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Reconstruct an input vector from a latent sample."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return reconstruction, latent mean, and latent log variance."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @torch.no_grad()
    def anomaly_score(self, x: Tensor) -> Tensor:
        """Compute per-row reconstruction error."""
        reconstruction, _, _ = self.forward(x)
        return F.mse_loss(reconstruction, x, reduction="none").mean(dim=1)


def vae_loss(reconstruction: Tensor, x: Tensor, mu: Tensor, log_var: Tensor, beta: float = 1.0) -> Tensor:
    """Compute reconstruction plus beta-weighted KL loss."""
    recon_loss = F.mse_loss(reconstruction, x, reduction="mean")
    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_divergence
