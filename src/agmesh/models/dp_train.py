"""DP-SGD training of the VAE and autoencoder (Proposal §7.3; Plan Week 5 task 5).

* **Per-sample clipping and Gaussian noise** come from Opacus (``GradSampleModule`` and
  ``DPOptimizer``), with the clipping norm and the noise multiplier of the client's
  :class:`~agmesh.privacy.calibrate.PrivacyPlan`.
* **Poisson sampling** is done here, exactly as the accountant assumes. Each step
  includes every training window independently with probability ``q``. An empty
  batch is still a step: it releases noise only, and is accounted for.
* **The step count is the plan's.** There is no early stopping on private data and no
  best-checkpoint selection: the model after the last planned step is the model.
  Selecting an epoch by validation loss would be a further, unaccounted use of the
  client's data.
* **Every step goes through the budget guard** and is recorded in the persistent
  accountant store *before* the optimiser applies it, so a crash can only over-count.
  If the guard refuses, training stops and
  :class:`~agmesh.privacy.budget_guard.BudgetExhausted` propagates; a federated
  client turns it into a declined ``fit``.

Randomness (initial weights, Poisson masks, the VAE's reparameterisation and the DP
noise) is seeded, so a run is reproducible. Opacus runs with ``secure_mode=False``:
its noise comes from torch's PRNG, not a cryptographically secure source. That is
standard for research and is recorded in every artifact.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from agmesh.models.nets import F32, loss_terms
from agmesh.privacy.budget_guard import BudgetGuard
from agmesh.privacy.calibrate import Mechanism, PrivacyPlan

SECURE_MODE = False


@dataclass
class DpTrainResult:
    steps: int
    epsilon: float
    train_seconds: float
    mean_batch: float
    history: list[dict[str, float]] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "epsilon": self.epsilon,
            "train_seconds": round(self.train_seconds, 2),
            "mean_batch": round(self.mean_batch, 2),
            "secure_mode": SECURE_MODE,
            "history": self.history,
        }


def fit_private(
    model: nn.Module,
    x_train: F32,
    *,
    plan: PrivacyPlan,
    guard: BudgetGuard,
    lr: float,
    seed: int,
    kl_weight: float = 1.0,
    log_every: int = 0,
) -> DpTrainResult:
    """Train ``model`` in place with DP-SGD (Adam update) for ``plan.steps`` steps."""
    from opacus import GradSampleModule
    from opacus.optimizers import DPOptimizer

    if x_train.shape[0] != plan.n_windows:
        raise ValueError(
            f"the plan was calibrated for {plan.n_windows} windows, not {x_train.shape[0]}"
        )
    torch.manual_seed(seed)
    sampler = torch.Generator().manual_seed(seed)
    noise = torch.Generator().manual_seed(seed + 1)
    data = torch.from_numpy(np.ascontiguousarray(x_train))
    gsm = GradSampleModule(model)
    optimiser = DPOptimizer(
        torch.optim.Adam(gsm.parameters(), lr=lr),
        noise_multiplier=plan.noise_multiplier,
        max_grad_norm=plan.max_grad_norm,
        expected_batch_size=max(1, round(plan.sample_rate * plan.n_windows)),
        loss_reduction="mean",
        generator=noise,
        secure_mode=SECURE_MODE,
    )
    step = Mechanism("dpsgd", plan.noise_multiplier, plan.sample_rate, 1)
    per_epoch = max(1, round(1.0 / plan.sample_rate))
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    batch_sizes: list[int] = []
    running = 0.0
    gsm.train()
    for i in range(plan.steps):
        guard.spend(step)  # checked and recorded before the update exists (conservative)
        mask = torch.rand(data.shape[0], generator=sampler) < plan.sample_rate
        xb = data[mask]
        batch_sizes.append(int(xb.shape[0]))
        optimiser.zero_grad(set_to_none=True)
        with warnings.catch_warnings():
            # Opacus' backward hooks warn when the input does not require grad; harmless
            warnings.simplefilter("ignore", UserWarning)
            loss, _mse = loss_terms(gsm, xb, kl_weight)
            batch_loss = loss.mean() if xb.shape[0] else loss.sum()
            batch_loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()
        if xb.shape[0]:
            running += float(batch_loss.detach())
        if (i + 1) % per_epoch == 0 or i + 1 == plan.steps:
            done = (i + 1) // per_epoch if (i + 1) % per_epoch == 0 else (i + 1) / per_epoch
            history.append({"epoch": float(done), "train_loss": running / per_epoch})
            running = 0.0
            if log_every and len(history) % log_every == 0:
                print(f"  epoch {done:g}: train loss {history[-1]['train_loss']:.4f}", flush=True)
    gsm.remove_hooks()  # the trained ``model`` is scored as a plain module
    model.eval()
    return DpTrainResult(
        steps=plan.steps,
        epsilon=guard.store.epsilon(),
        train_seconds=time.perf_counter() - started,
        mean_batch=float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        history=history,
    )


def unwrap(model: nn.Module) -> nn.Module:
    """The plain module inside an Opacus ``GradSampleModule`` (for scoring and hashing)."""
    inner = getattr(model, "_module", None)
    return inner if isinstance(inner, nn.Module) else model


def fit_reference(
    model: nn.Module,
    x_train: F32,
    *,
    steps: int,
    sample_rate: float,
    lr: float,
    seed: int,
    kl_weight: float = 1.0,
) -> DpTrainResult:
    """The ε = ∞ reference: the same Poisson batches and step count, no clipping, no noise.

    It isolates the cost of clipping and noise from everything else in the private
    pipeline (fixed steps, final model, histogram normalisation).
    """
    torch.manual_seed(seed)
    sampler = torch.Generator().manual_seed(seed)
    data = torch.from_numpy(np.ascontiguousarray(x_train))
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    per_epoch = max(1, round(1.0 / sample_rate))
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    running, sizes = 0.0, []
    model.train()
    for i in range(steps):
        xb = data[torch.rand(data.shape[0], generator=sampler) < sample_rate]
        sizes.append(int(xb.shape[0]))
        if xb.shape[0]:
            optimiser.zero_grad(set_to_none=True)
            loss, _ = loss_terms(model, xb, kl_weight)
            batch_loss = loss.mean()
            batch_loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
            running += float(batch_loss.detach())
        if (i + 1) % per_epoch == 0:
            history.append(
                {"epoch": float((i + 1) // per_epoch), "train_loss": running / per_epoch}
            )
            running = 0.0
    model.eval()
    return DpTrainResult(
        steps=steps,
        epsilon=float("inf"),
        train_seconds=time.perf_counter() - started,
        mean_batch=float(np.mean(sizes)) if sizes else 0.0,
        history=history,
    )
