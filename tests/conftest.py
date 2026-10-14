"""Shared synthetic fixture data for the pipeline, evaluation and contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agmesh.common.config import Params, load_params
from agmesh.data import site_a, site_b
from agmesh.data.fixture import make_fixture

REPO_PARAMS = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"


@pytest.fixture(scope="session")
def params() -> Params:
    return load_params(REPO_PARAMS)


@pytest.fixture(scope="session")
def fixture_paths(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return make_fixture(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def site_a_run(
    params: Params, fixture_paths: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict[str, Any], Path]:
    out = tmp_path_factory.mktemp("parts") / "site_a"
    return site_a.partition(params, fixture_paths["hai_dir"], out), out


@pytest.fixture(scope="session")
def interim_dir(
    params: Params, fixture_paths: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    interim = tmp_path_factory.mktemp("interim")
    site_b.prepare(params, fixture_paths["modbus_dir"], interim)
    return interim


@pytest.fixture(scope="session")
def site_b_run(
    params: Params, interim_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[dict[str, Any], Path]:
    out = tmp_path_factory.mktemp("parts") / "site_b"
    return site_b.partition(params, interim_dir, out), out
