import subprocess
import sys


def test_flower_simulation_stub_boots() -> None:
    result = subprocess.run(
        [sys.executable, "federated/simulate.py", "--rounds", "1", "--stub"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Week 1 scaffold is operational" in result.stdout
