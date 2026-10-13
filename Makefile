# Agentic Mesh v3 — common commands (Implementation Plan v3.0, §5).
# Run from the repository root inside WSL2/Linux. All Python runs through uv.

UV ?= uv
PY := $(UV) run python

.PHONY: help setup setup-fl lock lint format typecheck test check \
        checksums verify-checksums inspect-data bench-llm bench-ray feasibility

help:
	@echo "setup             install core + dev dependencies (creates uv.lock on first run)"
	@echo "setup-fl          also install the federated-learning stack (flwr, ray, torch CPU)"
	@echo "lint / format     ruff lint and format check / auto-fix"
	@echo "typecheck         mypy (strict)"
	@echo "test              pytest"
	@echo "check             lint + typecheck + test (what CI runs)"
	@echo "checksums         record SHA-256 of data/raw into data/raw_checksums.json"
	@echo "verify-checksums  verify data/raw against data/raw_checksums.json"
	@echo "inspect-data      Week 1 spike: dataset inspection  -> results/feasibility/data_inspection.json"
	@echo "bench-llm         Week 1 spike: Ollama benchmark    -> results/feasibility/llm_bench.json"
	@echo "bench-ray         Week 1 spike: Flower/Ray benchmark -> results/feasibility/ray_bench.json"

setup:
	$(UV) sync --group dev

setup-fl:
	$(UV) sync --group dev --group fl

lock:
	$(UV) lock

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest

check: lint typecheck test

checksums:
	$(PY) scripts/verify_checksums.py --record

verify-checksums:
	$(PY) scripts/verify_checksums.py --verify

inspect-data:
	$(PY) scripts/inspect_data.py

bench-llm:
	$(PY) scripts/bench_llm.py

# Uses the fl group; run `make setup-fl` first. Never run at the same time as bench-llm (WR-4).
bench-ray:
	$(UV) run --group fl python scripts/bench_ray.py

feasibility: inspect-data bench-llm bench-ray
