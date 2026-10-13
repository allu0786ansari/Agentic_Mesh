"""Tests for the params.yaml loader and its cross-checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from agmesh.common.config import DEFAULT_PARAMS_PATH, Params, load_params

REPO_PARAMS = Path(__file__).resolve().parents[2] / DEFAULT_PARAMS_PATH


def _raw() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(REPO_PARAMS.read_text(encoding="utf-8"))
    return data


def test_repository_params_load() -> None:
    params = load_params(REPO_PARAMS)
    assert len(params.data.partition.site_a.clients) == 4
    assert len(params.data.partition.site_b.clients) == 3
    assert params.federated.min_fit_clients.site_a == 3
    assert params.federated.min_fit_clients.site_b == 2
    assert params.privacy.is_private


def test_overlapping_training_windows_are_rejected() -> None:
    raw = _raw()
    raw["data"]["windows"]["site_a"]["train_stride_s"] = 10
    with pytest.raises(ValidationError, match="must not overlap"):
        Params.model_validate(raw)


def test_min_fit_clients_cannot_exceed_federation_size() -> None:
    raw = _raw()
    raw["federated"]["min_fit_clients"]["site_b"] = 4
    with pytest.raises(ValidationError, match="site_b"):
        Params.model_validate(raw)


def test_unknown_keys_are_rejected() -> None:
    raw = _raw()
    raw["privacy"]["epsilon_per_round"] = 2.0
    with pytest.raises(ValidationError):
        Params.model_validate(raw)


def test_infinite_epsilon_means_non_private() -> None:
    raw = _raw()
    raw["privacy"]["target_epsilon"] = math.inf
    assert not Params.model_validate(raw).privacy.is_private


@pytest.mark.parametrize("n", [1, 10, 1_000, 10_000, 1_000_000])
def test_delta_is_below_one_over_n(n: int) -> None:
    delta = load_params(REPO_PARAMS).privacy.resolve_delta(n)
    assert delta < 1 / n
    assert delta <= 1e-5


def test_delta_requires_positive_window_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        load_params(REPO_PARAMS).privacy.resolve_delta(0)
