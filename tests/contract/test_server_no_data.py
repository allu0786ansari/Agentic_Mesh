"""The federated server holds no data: no data-loader imports, no file reads (Plan W6)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SERVER_SIDE = ("server", "strategies", "fed_eval", "registry", "protocol")
SRC = Path(__file__).resolve().parents[2] / "src" / "agmesh" / "federated"
FORBIDDEN_MODULES = (
    "agmesh.data",
    "agmesh.experiments",
    "pandas",
    "pyarrow",
    "agmesh.federated.client",
)
FORBIDDEN_CALLS = (
    "open",
    "read_parquet",
    "read_csv",
    "load_partition",
    "read_text",
    "read_bytes",
    "load",
)


def test_server_imports_no_data_loader() -> None:
    code = (
        "import sys\n"
        + "".join(f"import agmesh.federated.{m}\n" for m in SERVER_SIDE)
        + "print('\\n'.join(sorted(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    leaked = [m for m in out if m.startswith(FORBIDDEN_MODULES)]
    assert leaked == [], f"server-side modules import data code: {leaked}"


def test_server_never_reads_files() -> None:
    for name in SERVER_SIDE:
        tree = ast.parse((SRC / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                assert called not in FORBIDDEN_CALLS, f"{name}.py calls {called}()"
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(FORBIDDEN_MODULES), f"{name}.py: {node.module}"
