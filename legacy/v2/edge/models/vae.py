from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class VAEOutput(NamedTuple):
    reconstruction: Tensor
    mu: Tensor
    logvar: Tensor


class VAE(nn.Module):
    """Compact fully connected VAE for flattened multivariate HAI windows."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        if input_dim < 1 or latent_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim, latent_dim, and hidden_dim must be positive")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        encoded_dim = hidden_dim // 2
        self.mu = nn.Linear(encoded_dim, latent_dim)
        self.logvar = nn.Linear(encoded_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(inputs)
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + standard_deviation * torch.randn_like(standard_deviation)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def forward(self, inputs: Tensor) -> VAEOutput:
        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar)
        return VAEOutput(self.decode(latent), mu, logvar)


class _ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs + self.block(inputs))


class TemporalVAE(nn.Module):
    """Residual dilated-convolutional VAE for multivariate temporal windows.

    Unlike the fully connected baseline, this model preserves the time/feature
    layout through the encoder and decoder. The latent vector summarizes the
    complete window, while dilated residual blocks capture short and medium-range
    process dynamics without requiring recurrent state.
    """

    architecture = "temporal_residual"

    def __init__(
        self,
        window_size: int,
        feature_count: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.05,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if window_size < 2 or feature_count < 1 or latent_dim < 1 or hidden_dim < 8:
            raise ValueError("window_size, feature_count, latent_dim, and hidden_dim are invalid")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        if output_activation not in {"sigmoid", "linear"}:
            raise ValueError("output_activation must be sigmoid or linear")
        self.window_size = window_size
        self.feature_count = feature_count
        self.input_dim = window_size * feature_count
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.output_activation = output_activation
        self.input_projection = nn.Conv1d(feature_count, hidden_dim, kernel_size=1)
        self.encoder_blocks = nn.Sequential(
            _ResidualTemporalBlock(hidden_dim, dilation=1, dropout=dropout),
            _ResidualTemporalBlock(hidden_dim, dilation=2, dropout=dropout),
            _ResidualTemporalBlock(hidden_dim, dilation=4, dropout=dropout),
        )
        self.encoder_pool = nn.AdaptiveAvgPool1d(1)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_projection = nn.Linear(latent_dim, hidden_dim * window_size)
        self.decoder_blocks = nn.Sequential(
            _ResidualTemporalBlock(hidden_dim, dilation=4, dropout=dropout),
            _ResidualTemporalBlock(hidden_dim, dilation=2, dropout=dropout),
            _ResidualTemporalBlock(hidden_dim, dilation=1, dropout=dropout),
        )
        self.output_projection = nn.Conv1d(hidden_dim, feature_count, kernel_size=1)

    def _as_sequence(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 2:
            if inputs.shape[1] != self.input_dim:
                raise ValueError(f"Expected flattened input dimension {self.input_dim}, got {inputs.shape[1]}")
            return inputs.reshape(inputs.shape[0], self.window_size, self.feature_count).transpose(1, 2)
        if inputs.ndim == 3 and inputs.shape[1:] == (self.window_size, self.feature_count):
            return inputs.transpose(1, 2)
        if inputs.ndim == 3 and inputs.shape[1:] == (self.feature_count, self.window_size):
            return inputs
        raise ValueError("TemporalVAE inputs must be [batch, input_dim] or [batch, time, features]")

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.input_projection(self._as_sequence(inputs))
        hidden = self.encoder_blocks(hidden)
        pooled = self.encoder_pool(hidden).squeeze(-1)
        return self.mu(pooled), self.logvar(pooled)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        standard_deviation = torch.exp(0.5 * logvar.clamp(min=-20.0, max=10.0))
        return mu + standard_deviation * torch.randn_like(standard_deviation)

    def decode(self, latent: Tensor) -> Tensor:
        hidden = self.decoder_projection(latent).reshape(-1, self.hidden_dim, self.window_size)
        hidden = self.decoder_blocks(hidden)
        output = self.output_projection(hidden)
        if self.output_activation == "sigmoid":
            output = torch.sigmoid(output)
        return output.transpose(1, 2).reshape(-1, self.input_dim)

    def forward(self, inputs: Tensor) -> VAEOutput:
        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar)
        return VAEOutput(self.decode(latent), mu, logvar)


def vae_loss(output: VAEOutput, inputs: Tensor, beta: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, reconstruction, and KL losses for a VAE batch."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    reconstruction_loss = nn.functional.mse_loss(output.reconstruction, inputs, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + output.logvar - output.mu.pow(2) - output.logvar.exp())
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


def reconstruction_error(output: VAEOutput, inputs: Tensor) -> Tensor:
    """Return one mean-squared reconstruction error per input sample."""
    return torch.mean((output.reconstruction - inputs).pow(2), dim=1)
