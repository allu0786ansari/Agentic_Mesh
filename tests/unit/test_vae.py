"""Unit tests for the VAE model."""

from __future__ import annotations

import torch

from edge.models.vae import VAE, vae_loss


def test_vae_forward_shapes() -> None:
    model = VAE(input_dim=1500, latent_dim=32)
    x = torch.rand(4, 1500)

    reconstruction, mu, log_var = model(x)

    assert reconstruction.shape == x.shape
    assert mu.shape == (4, 32)
    assert log_var.shape == (4, 32)


def test_vae_loss_is_scalar() -> None:
    model = VAE(input_dim=1500, latent_dim=32)
    x = torch.rand(4, 1500)
    reconstruction, mu, log_var = model(x)

    loss = vae_loss(reconstruction, x, mu, log_var)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
