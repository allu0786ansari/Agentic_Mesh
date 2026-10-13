from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from edge.models.graph_encoder import SensorGraphEncoder


class SequenceVAEOutput(NamedTuple):
    reconstruction: Tensor
    posterior_mu: Tensor
    posterior_logvar: Tensor
    prior_mu: Tensor
    prior_logvar: Tensor
    log_det: Tensor


class PlanarFlow(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.u = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.w = nn.Parameter(torch.randn(latent_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        linear = z @ self.w + self.b
        activation = torch.tanh(linear)
        transformed = z + activation.unsqueeze(-1) * self.u
        psi = (1.0 - activation.pow(2)).unsqueeze(-1) * self.w
        log_det = torch.log(torch.abs(1.0 + (psi * self.u).sum(dim=-1)) + 1e-6)
        return transformed, log_det


class PlanarFlowStack(nn.Module):
    def __init__(self, latent_dim: int, layers: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.layers = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(layers)])

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det
        return z, total_log_det


class RecurrentFlowVAE(nn.Module):
    """Graph-attention recurrent VAE with conditional prior and planar flow posterior."""

    architecture = "recurrent_flow"

    def __init__(
        self,
        feature_count: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        graph_embed_dim: int = 32,
        flow_layers: int = 2,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if feature_count < 1 or latent_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_count, latent_dim, and hidden_dim must be positive")
        if graph_embed_dim % attention_heads:
            raise ValueError("graph_embed_dim must be divisible by attention_heads")
        self.feature_count = feature_count
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.graph_embed_dim = graph_embed_dim
        self.flow_layers = flow_layers
        self.graph_encoder = SensorGraphEncoder(feature_count, graph_embed_dim, attention_heads)
        self.rnn = nn.GRUCell(graph_embed_dim, hidden_dim)
        self.prior_net = nn.Linear(hidden_dim, latent_dim * 2)
        self.posterior_net = nn.Linear(hidden_dim + graph_embed_dim, latent_dim * 2)
        self.flow = PlanarFlowStack(latent_dim, flow_layers)
        self.decoder_rnn = nn.GRUCell(latent_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, feature_count)

    def forward(self, inputs: Tensor) -> SequenceVAEOutput:
        if inputs.ndim == 2:
            if inputs.shape[1] % self.feature_count:
                raise ValueError("flattened input does not match feature_count")
            inputs = inputs.reshape(inputs.shape[0], -1, self.feature_count)
        if inputs.ndim != 3 or inputs.shape[-1] != self.feature_count:
            raise ValueError("inputs must have shape [batch, time, feature_count]")
        batch, time, _ = inputs.shape
        embedded = self.graph_encoder(inputs)
        hidden = torch.zeros(batch, self.hidden_dim, device=inputs.device, dtype=inputs.dtype)
        decoder_hidden = torch.zeros_like(hidden)
        reconstructions, posterior_mus, posterior_logvars, prior_mus, prior_logvars, log_dets = [], [], [], [], [], []
        for step in range(time):
            prior_mu, prior_logvar = self.prior_net(hidden).chunk(2, dim=-1)
            posterior_mu, posterior_logvar = self.posterior_net(torch.cat([hidden, embedded[:, step]], dim=-1)).chunk(2, dim=-1)
            posterior_logvar = posterior_logvar.clamp(-12.0, 8.0)
            z0 = posterior_mu + torch.exp(0.5 * posterior_logvar) * torch.randn_like(posterior_mu)
            z, log_det = self.flow(z0)
            hidden = self.rnn(embedded[:, step], hidden)
            decoder_hidden = self.decoder_rnn(z, decoder_hidden + hidden)
            reconstructions.append(self.output_head(decoder_hidden))
            posterior_mus.append(posterior_mu)
            posterior_logvars.append(posterior_logvar)
            prior_mus.append(prior_mu)
            prior_logvars.append(prior_logvar.clamp(-12.0, 8.0))
            log_dets.append(log_det)
        return SequenceVAEOutput(
            torch.stack(reconstructions, dim=1),
            torch.stack(posterior_mus, dim=1),
            torch.stack(posterior_logvars, dim=1),
            torch.stack(prior_mus, dim=1),
            torch.stack(prior_logvars, dim=1),
            torch.stack(log_dets, dim=1),
        )

    def encode(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        output = self.forward(inputs)
        return output.posterior_mu[:, -1], output.posterior_logvar[:, -1]


def sequence_vae_loss(
    output: SequenceVAEOutput,
    targets: Tensor,
    *,
    beta: float = 0.3,
    free_bits: float = 0.05,
) -> tuple[Tensor, Tensor, Tensor]:
    if targets.ndim == 2:
        targets = targets.reshape(targets.shape[0], -1, output.reconstruction.shape[-1])
    if beta < 0 or free_bits < 0:
        raise ValueError("beta and free_bits must be non-negative")
    reconstruction = nn.functional.smooth_l1_loss(output.reconstruction, targets, reduction="mean")
    variance = output.posterior_logvar.exp()
    prior_variance = output.prior_logvar.exp()
    kl = 0.5 * (output.prior_logvar - output.posterior_logvar + (variance + (output.posterior_mu - output.prior_mu).pow(2)) / prior_variance - 1.0)
    kl = torch.clamp(kl, min=free_bits).mean() - output.log_det.mean() / targets.shape[1]
    return reconstruction + beta * kl, reconstruction, kl
