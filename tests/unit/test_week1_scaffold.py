r"""
tests/unit/test_week1_scaffold.py
Week 1 gate: 73 tests confirming all directories, __init__.py files,
Dockerfiles, config files, and critical imports are correct.
Run: pytest tests/unit/test_week1_scaffold.py -v
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DIRS = [
    ".github/workflows",
    "data/raw",
    "data/processed",
    "data/partitions/node_01",
    "data/partitions/node_02",
    "data/partitions/node_03",
    "data/partitions/node_04",
    "data/partitions/node_05",
    "edge/models",
    "edge/training",
    "edge/embedding",
    "federated",
    "agents/triage",
    "agents/investigator",
    "agents/security",
    "agents/dispatcher",
    "api",
    "evaluation",
    "infra/docker",
    "infra/k3s",
    "infra/kafka",
    "infra/certs",
    "infra/ollama",
    "knowledge_base/mitre_ics_raw",
    "knowledge_base/nist_800_82",
    "scripts",
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    "notebooks",
    "docs/architecture",
]


@pytest.mark.parametrize("p", REQUIRED_DIRS)
def test_directory_exists(p):
    assert (PROJECT_ROOT / p).is_dir(), f"Missing directory: {p}"


REQUIRED_INIT = [
    "edge/__init__.py",
    "edge/models/__init__.py",
    "edge/training/__init__.py",
    "edge/embedding/__init__.py",
    "federated/__init__.py",
    "agents/__init__.py",
    "agents/triage/__init__.py",
    "agents/investigator/__init__.py",
    "agents/security/__init__.py",
    "agents/dispatcher/__init__.py",
    "api/__init__.py",
    "evaluation/__init__.py",
]


@pytest.mark.parametrize("p", REQUIRED_INIT)
def test_init_file_exists(p):
    assert (PROJECT_ROOT / p).is_file(), f"Missing __init__.py: {p}"


REQUIRED_ROOT_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".env.example",
    ".gitignore",
    "params.yaml",
    "README.md",
]


@pytest.mark.parametrize("p", REQUIRED_ROOT_FILES)
def test_root_file_exists(p):
    f = PROJECT_ROOT / p
    assert f.is_file() and f.stat().st_size > 0, f"Missing or empty: {p}"


REQUIRED_DOCKERFILES = [
    "infra/docker/edge.Dockerfile",
    "infra/docker/fl_server.Dockerfile",
    "infra/docker/agents.Dockerfile",
    "infra/docker/api.Dockerfile",
]


@pytest.mark.parametrize("p", REQUIRED_DOCKERFILES)
def test_dockerfile_valid(p):
    f = PROJECT_ROOT / p
    assert f.is_file(), f"Missing: {p}"

    content = f.read_text(encoding="utf-8")

    assert (
        "FROM python:3.12" in content
        or "FROM python:3.12-slim" in content
    ), f"Must use python:3.12 base image: {p}"


CRITICAL_PACKAGES = [
    "torch",
    "opacus",
    "flwr",
    "langgraph",
    "fastapi",
    "mlflow",
    "loguru",
    "rich",
    "kafka",
    "nats",
    "qdrant_client",
    "sklearn",
    "pandas",
    "pyarrow",
]


@pytest.mark.parametrize("pkg", CRITICAL_PACKAGES)
def test_package_importable(pkg):
    try:
        importlib.import_module(pkg)
    except ImportError as exc:
        pytest.fail(f"Cannot import {pkg}: {exc}")


def test_numpy_below_2():
    import numpy as np

    assert int(np.__version__.split(".")[0]) < 2, (
        f"numpy {np.__version__} >= 2.0 breaks torch 2.3! "
        'Fix: pip install "numpy==1.26.4"'
    )


def test_params_yaml_valid():
    import yaml

    params = yaml.safe_load(
        (PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8")
    )

    required_keys = [
        "epsilon",
        "delta",
        "max_grad_norm",
        "fedprox_mu",
        "fl_rounds",
        "embedding_dim",
        "vae_latent_dim",
        "window_size",
        "target_auroc",
        "target_f1",
    ]

    for k in required_keys:
        assert k in params, f"params.yaml missing key: {k}"


def test_env_example_has_required_keys():
    content = (
        PROJECT_ROOT / ".env.example"
    ).read_text(encoding="utf-8")

    required_env_keys = [
        "NODE_ID",
        "DATA_TYPE",
        "EPSILON",
        "DELTA",
        "FLOWER_SERVER_HOST",
        "NATS_URL",
        "KAFKA_BOOTSTRAP",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "POSTGRES_DSN",
        "MLFLOW_TRACKING_URI",
        "JWT_SECRET",
        "OLLAMA_BASE_URL",
        "PHOENIX_ENDPOINT",
    ]

    for k in required_env_keys:
        assert k in content, f".env.example missing: {k}"


def test_kafka_compose_valid():
    import yaml

    doc = yaml.safe_load(
        (
            PROJECT_ROOT / "infra/kafka/docker-compose.yml"
        ).read_text(encoding="utf-8")
    )

    assert "services" in doc
    assert "kafka" in doc["services"]


def test_k3s_config_valid():
    import yaml

    docs = list(
        yaml.safe_load_all(
            (
                PROJECT_ROOT / "infra/k3s/cluster-config.yaml"
            ).read_text(encoding="utf-8")
        )
    )

    assert len(docs) >= 1