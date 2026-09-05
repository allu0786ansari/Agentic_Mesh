from __future__ import annotations

import numpy as np
import torch

from edge.models.vae import VAE
from scripts.evaluate_global_federated_model import _merge_global_core


def test_merge_global_core_preserves_local_input_and_output_heads() -> None:
    first = VAE(input_dim=12, latent_dim=2, hidden_dim=8)
    second = VAE(input_dim=20, latent_dim=2, hidden_dim=8)
    before_input = second.encoder[0].weight.detach().clone()
    before_output = second.decoder[4].weight.detach().clone()
    parameters = [tensor.detach().cpu().numpy().copy() for name, tensor in first.state_dict().items() if not name.startswith("encoder.0.") and not name.startswith("decoder.4.")]

    _merge_global_core(second, parameters)

    assert torch.equal(second.encoder[0].weight, before_input)
    assert torch.equal(second.decoder[4].weight, before_output)
    assert np.array_equal(second.encoder[2].weight.detach().cpu().numpy(), first.encoder[2].weight.detach().cpu().numpy())
