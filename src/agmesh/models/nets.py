"""The VAE (Site A) and dense autoencoder (Site B), training and scoring.

Both networks use only ``Linear`` layers and ReLU activations, which Opacus supports
for per-sample gradients (no batch normalisation), so the same modules are trained
with DP-SGD from Week 5. The anomaly score of a window is its mean squared
reconstruction error over the inputs; for the VAE the reconstruction uses the
posterior mean (no sampling) so scores are deterministic.

Non-private training (this module) uses Adam, shuffled mini-batches from a seeded
generator, and early stopping on the normal-validation loss, keeping the best state.
"""

from __future__ import annotations

import io
import itertools
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from agmesh.common.provenance import sha256_bytes

F32 = npt.NDArray[np.float32]


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(itertools.pairwise(sizes)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class VAE(nn.Module):
    def __init__(self, n_inputs: int, hidden: list[int], latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(_mlp([n_inputs, *hidden]), nn.ReLU())
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.decoder = _mlp([latent, *reversed(hidden), n_inputs])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10.0, 10.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar


class Autoencoder(nn.Module):
    def __init__(self, n_inputs: int, hidden: list[int]) -> None:
        super().__init__()
        self.encoder = _mlp([n_inputs, *hidden])
        self.decoder = _mlp([hidden[-1], *reversed(hidden[:-1]), n_inputs])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon = self.decoder(torch.relu(self.encoder(x)))
        zero = torch.zeros((), dtype=x.dtype)
        return recon, zero, zero


Kind = Literal["vae", "ae"]


def build(
    kind: Kind, n_inputs: int, hidden: list[int], latent: int = 16, seed: int = 0
) -> nn.Module:
    """A freshly initialised network; ``seed`` fixes the initial weights."""
    torch.manual_seed(seed)
    return VAE(n_inputs, hidden, latent) if kind == "vae" else Autoencoder(n_inputs, hidden)


def loss_terms(
    model: nn.Module, x: torch.Tensor, kl_weight: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample loss (reconstruction + weighted KL) and per-sample reconstruction MSE."""
    recon, mu, logvar = model(x)
    mse = ((recon - x) ** 2).mean(dim=1)
    if isinstance(model, VAE):
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1) / x.shape[1]
        return mse + kl_weight * kl, mse
    return mse, mse


@dataclass
class TrainResult:
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def fit(
    model: nn.Module,
    x_train: F32,
    x_val: F32,
    *,
    lr: float,
    batch: int,
    max_epochs: int,
    patience: int,
    seed: int,
    kl_weight: float = 1.0,
) -> TrainResult:
    """Train with early stopping on the normal-validation loss; keeps the best state."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train = torch.from_numpy(np.ascontiguousarray(x_train))
    val = torch.from_numpy(np.ascontiguousarray(x_val))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch, stale = 0, 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(train.shape[0], generator=generator)
        total = 0.0
        for i in range(0, train.shape[0], batch):
            xb = train[order[i : i + batch]]
            loss, _mse = loss_terms(model, xb, kl_weight)
            batch_loss = loss.mean()
            optimiser.zero_grad()
            batch_loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            total += float(batch_loss.detach()) * xb.shape[0]
        val_loss = evaluate_loss(model, val, kl_weight)
        history.append({"epoch": epoch, "train_loss": total / train.shape[0], "val_loss": val_loss})
        if val_loss < best - 1e-6:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(epoch, best_epoch, best, time.perf_counter() - started, history)


@torch.no_grad()
def evaluate_loss(model: nn.Module, x: torch.Tensor, kl_weight: float, batch: int = 4096) -> float:
    model.eval()
    total = 0.0
    for i in range(0, x.shape[0], batch):
        loss, _ = loss_terms(model, x[i : i + batch], kl_weight)
        total += float(loss.sum())
    return total / max(x.shape[0], 1)


@torch.no_grad()
def reconstruction_errors(model: nn.Module, x: F32, batch: int = 4096) -> F32:
    """Squared error per input, shape (windows, inputs), using the posterior mean."""
    model.eval()
    data = torch.from_numpy(np.ascontiguousarray(x))
    parts = []
    for i in range(0, data.shape[0], batch):
        xb = data[i : i + batch]
        recon, _mu, _lv = model(xb)
        parts.append(((recon - xb) ** 2).numpy())
    out: F32 = (np.concatenate(parts) if parts else np.empty((0, x.shape[1]))).astype(np.float32)
    return out


def state_sha256(model: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save({k: v.cpu() for k, v in sorted(model.state_dict().items())}, buffer)
    return sha256_bytes(buffer.getvalue())


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


OPACUS_SAFE_LAYERS = (nn.Linear, nn.ReLU, nn.Sequential, VAE, Autoencoder)
