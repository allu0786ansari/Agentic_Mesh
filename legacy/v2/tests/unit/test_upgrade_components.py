import numpy as np
import pandas as pd
import torch

from edge.models.graph_encoder import SensorGraphEncoder
from edge.models.sequence_vae import RecurrentFlowVAE, sequence_vae_loss
from evaluation.threshold import pot_threshold
from evaluation.latent_likelihood import flow_anomaly_scores
from scripts.feature_selection import screen_features
from scripts.synthetic_anomaly_injection import inject_anomalies


def test_feature_screen_and_synthetic_validation() -> None:
    frame = pd.DataFrame({"signal": np.arange(100, dtype=float), "label": [0] * 50 + [1] * 50})
    report = screen_features(frame, ["signal"], min_auc=0.55, min_pr_auc=0.5)
    assert bool(report.iloc[0]["selected"])
    injected, labels = inject_anomalies(frame.drop(columns="label"), ["signal"], event_count=2, event_length=5)
    assert injected.shape == frame.drop(columns="label").shape
    assert labels.sum() > 0


def test_recurrent_flow_vae_forward_and_loss() -> None:
    model = RecurrentFlowVAE(feature_count=4, latent_dim=3, hidden_dim=16, graph_embed_dim=8, attention_heads=2)
    inputs = torch.rand(3, 6, 4)
    output = model(inputs)
    total, reconstruction, kl = sequence_vae_loss(output, inputs)
    assert output.reconstruction.shape == inputs.shape
    assert output.posterior_mu.shape == (3, 6, 3)
    assert torch.isfinite(total)
    assert torch.isfinite(reconstruction)
    assert torch.isfinite(kl)


def test_pot_threshold_is_finite() -> None:
    threshold = pot_threshold(np.linspace(0.0, 1.0, 100))
    assert np.isfinite(threshold)


def test_graph_encoder_shape() -> None:
    encoder = SensorGraphEncoder(feature_count=4, embed_dim=8, heads=2)
    output = encoder(torch.rand(2, 6, 4))
    assert output.shape == (2, 6, 8)


def test_flow_anomaly_score_is_finite() -> None:
    model = RecurrentFlowVAE(feature_count=3, latent_dim=2, hidden_dim=8, graph_embed_dim=4, attention_heads=2)
    scores = flow_anomaly_scores(model, torch.rand(2, 5, 3))
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
