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


def test_dataset_decisions_are_encoded() -> None:
    params = load_params(REPO_PARAMS)
    site_a = params.data.partition.site_a
    assert site_a.scheme == "time"
    assert site_a.period_assignment == {
        "a1": "train1",
        "a2": "train2",
        "a3": "train3:first_half",
        "a4": "train3:second_half",
    }
    assert params.data.hai.label_column == "attack"
    assert params.data.modbus.label_source == "attacker_address"
    assert params.fusion.site_b.exclude_latency


def test_time_scheme_requires_matching_assignment() -> None:
    raw = _raw()
    del raw["data"]["partition"]["site_a"]["period_assignment"]["a4"]
    with pytest.raises(ValidationError, match="exactly the Site A clients"):
        Params.model_validate(raw)

    raw = _raw()
    raw["data"]["partition"]["site_a"]["period_assignment"]["a4"] = "train3:first_half"
    with pytest.raises(ValidationError, match="different period"):
        Params.model_validate(raw)

    raw = _raw()
    raw["data"]["partition"]["site_a"]["period_assignment"]["a1"] = "test1"
    with pytest.raises(ValidationError, match="period must look like"):
        Params.model_validate(raw)


def test_other_schemes_do_not_need_assignment() -> None:
    raw = _raw()
    raw["data"]["partition"]["site_a"]["scheme"] = "iid"
    del raw["data"]["partition"]["site_a"]["period_assignment"]
    assert Params.model_validate(raw).data.partition.site_a.scheme == "iid"


def test_attacker_cannot_be_a_testbed_device() -> None:
    raw = _raw()
    raw["data"]["modbus"]["attacker_ip"] = "185.175.0.4"
    with pytest.raises(ValidationError, match="attacker_ip"):
        Params.model_validate(raw)


def test_invalid_ip_is_rejected() -> None:
    raw = _raw()
    raw["data"]["modbus"]["devices"]["ied1a"] = "185.175.0.400"
    with pytest.raises(ValidationError, match="IPv4"):
        Params.model_validate(raw)


def test_segments_must_divide_the_window() -> None:
    raw = _raw()
    raw["models"]["vae"]["segments_per_window"] = 7
    with pytest.raises(ValidationError, match="segments_per_window"):
        Params.model_validate(raw)


def test_site_b_clients_must_match_modbus_devices() -> None:
    raw = _raw()
    raw["data"]["partition"]["site_b"]["clients"]["b3"] = "ied4c"
    with pytest.raises(ValidationError, match="modbus devices"):
        Params.model_validate(raw)
