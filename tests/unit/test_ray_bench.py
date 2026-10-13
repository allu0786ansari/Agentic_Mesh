"""Tests for the Flower/Ray benchmark helpers (needs the 'fl' dependency group)."""

from __future__ import annotations

import pytest

pytest.importorskip("flwr", reason="needs the 'fl' dependency group: uv sync --group fl")

from agmesh.feasibility.ray_bench import (
    BenchSettings,
    client_data,
    initial_arrays,
    layer_shapes,
)

pytestmark = pytest.mark.fl


def _settings() -> BenchSettings:
    return BenchSettings(
        clients=2,
        rounds=1,
        num_cpus_per_client=1,
        total_cpus=2,
        backend="numpy",
        input_dim=60 * 79,
        hidden=(256, 128),
        latent=16,
        samples_per_client=50,
        local_steps=1,
        batch_size=8,
        lr=1e-3,
        strategy="fedavg",
        proximal_mu=0.0,
        object_store_mb=100,
        seed=11,
        server_record_path="unused",
    )


def test_layer_shapes_mirror_the_encoder() -> None:
    assert layer_shapes(_settings()) == [
        (4740, 256),
        (256, 128),
        (128, 16),
        (16, 128),
        (128, 256),
        (256, 4740),
    ]


def test_initial_arrays_size_matches_vae_scale() -> None:
    n_params = sum(a.size for a in initial_arrays(_settings()))
    assert 2_000_000 < n_params < 3_000_000


def test_client_data_is_deterministic_and_differs_by_client() -> None:
    settings = _settings()
    a0, a0_again, a1 = client_data(settings, 0), client_data(settings, 0), client_data(settings, 1)
    assert (a0 == a0_again).all()
    assert a0.shape == (50, 4740)
    assert abs(float(a1.mean() - a0.mean())) > 0.05


def test_settings_round_trip_through_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from agmesh.feasibility.ray_bench import SETTINGS_ENV

    settings = _settings()
    monkeypatch.setenv(SETTINGS_ENV, settings.to_env())
    assert BenchSettings.from_env() == settings
