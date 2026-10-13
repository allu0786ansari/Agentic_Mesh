from __future__ import annotations

import numpy as np
import torch

from edge.models.sequence_vae import RecurrentFlowVAE


def flow_anomaly_scores(
    model: RecurrentFlowVAE,
    inputs: torch.Tensor,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Score recurrent-flow windows by conditional posterior/prior divergence."""
    model.eval()
    with torch.no_grad():
        output = model(inputs.to(device))
        posterior_variance = output.posterior_logvar.exp()
        prior_variance = output.prior_logvar.exp()
        kl = 0.5 * (
            output.prior_logvar
            - output.posterior_logvar
            + (posterior_variance + (output.posterior_mu - output.prior_mu).pow(2)) / prior_variance
            - 1.0
        )
        scores = kl.sum(dim=-1).mean(dim=-1) - output.log_det.mean(dim=-1)
    values = scores.cpu().numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Flow anomaly scoring produced non-finite values")
    return values
