import torch

from edge.models.vae import VAE, reconstruction_error, vae_loss


def test_vae_forward_shapes_and_loss() -> None:
    model = VAE(input_dim=12, latent_dim=4, hidden_dim=16)
    inputs = torch.randn(3, 12)
    output = model(inputs)
    total, reconstruction, kl = vae_loss(output, inputs)

    assert output.reconstruction.shape == inputs.shape
    assert output.mu.shape == (3, 4)
    assert output.logvar.shape == (3, 4)
    assert total.ndim == reconstruction.ndim == kl.ndim == 0
    assert reconstruction_error(output, inputs).shape == (3,)


def test_vae_rejects_invalid_dimensions() -> None:
    try:
        VAE(input_dim=0)
    except ValueError:
        pass
    else:
        raise AssertionError("VAE accepted an invalid input dimension")
