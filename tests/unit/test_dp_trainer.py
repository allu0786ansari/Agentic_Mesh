import torch
from torch.utils.data import TensorDataset

from edge.models.vae import VAE
from edge.training.dp_trainer import DPVAETrainer


def test_dp_trainer_returns_epsilon_and_checkpoint(tmp_path) -> None:
    dataset = TensorDataset(torch.randn(12, 8))
    trainer = DPVAETrainer(VAE(input_dim=8, latent_dim=3, hidden_dim=12), max_grad_norm=1.0)

    result = trainer.train(
        dataset,
        epochs=1,
        batch_size=4,
        epsilon=5.0,
        delta=1e-5,
        checkpoint_path=tmp_path / "vae.pt",
    )

    assert result.epsilon is not None
    assert result.epsilon > 0
    assert result.samples == 12
    assert result.checkpoint_path == tmp_path / "vae.pt"
    assert result.checkpoint_path.exists()


def test_dp_trainer_rejects_invalid_privacy_parameters() -> None:
    dataset = TensorDataset(torch.randn(4, 8))
    trainer = DPVAETrainer(VAE(input_dim=8, latent_dim=3, hidden_dim=12))
    try:
        trainer.train(dataset, epsilon=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("DP trainer accepted an invalid epsilon")
