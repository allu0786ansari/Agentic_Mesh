# Agentic Mesh v3 — common commands (Implementation Plan v3.1, §5).
# Run from the repository root inside WSL2/Linux. All Python runs through uv.

UV ?= uv
PY := $(UV) run python

.PHONY: help setup setup-fl lock lint format typecheck test check \
        checksums verify-checksums inspect-data bench-llm bench-ray feasibility \
        prepare partition check-data data fixture fixture-smoke check-results baselines scoring-selection ground-truth-draft \
        training-budget summarise shift-selection dp-sanity train x8 g2-pilot g2-tuning

help:
	@echo "setup             install core + dev + ml (torch CPU) dependencies"
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
	@echo "prepare           Week 2: parse Modbus captures       -> data/interim/modbus"
	@echo "partition         Week 2: Site A and Site B partitions -> data/partitions"
	@echo "check-data        Week 2: partition checks            -> results/data/partition_check.json"
	@echo "data              prepare + partition + check-data"
	@echo "fixture-smoke     run the whole data pipeline on the synthetic fixture (what CI runs)"
	@echo "check-results     validate every result artifact under results/X*/"
	@echo "baselines         X1 non-private baselines, all project seeds, validation -> results/X1/"
	@echo "training-budget   ADR 0004: label-free max_epochs check -> results/X1/training_budget.json"
	@echo "summarise         seed CIs + event-level CIs and paired tests -> results/X1/summary_val.json"
	@echo "shift-selection   ADR 0005: shift-robust scoring chosen on validation -> results/X1/shift_selection.json"
	@echo "dp-sanity         Week 5: single-client DP training at eps 1, 3, 10, inf -> results/privacy/"
	@echo "x8                Week 7: SecAgg+ systems experiment on Fed-A -> results/X8/"
	@echo "g2-pilot          Week 7: FedAvg and FedProx at eps 10, both sites, 50 rounds -> results/X1/"
	@echo "g2-tuning         Week 7: ADR 0008 DP lr x clip grid and FedProx mu grid -> results/federated/g2_*"
	@echo "train             Week 6: one federated run, e.g. make train SITE=site_a [ROUNDS=5 smoke] [EPS=3] [STRATEGY=fedavg] [BACKEND=flower]"
	@echo "scoring-selection ADR 0003: score aggregation chosen on validation -> results/X1/"
	@echo "ground-truth-draft  first-pass attribution mapping (reads test labels, logged)"

setup:
	$(UV) sync --group dev --group ml

setup-fl:
	$(UV) sync --group dev --group ml --group fl

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

prepare:
	$(PY) -m agmesh.data.pipeline prepare

partition:
	$(PY) -m agmesh.data.pipeline partition

check-data:
	$(PY) -m agmesh.data.pipeline check

data: prepare partition check-data

FIXTURE := fixtures/data

fixture:
	$(PY) -m agmesh.data.fixture --out $(FIXTURE)/raw

fixture-smoke: fixture
	$(PY) -m agmesh.data.pipeline prepare --modbus-dir $(FIXTURE)/raw/modbus2023 --interim-dir $(FIXTURE)/interim
	$(PY) -m agmesh.data.pipeline partition --hai-dir $(FIXTURE)/raw/hai/hai-fixture \
		--interim-dir $(FIXTURE)/interim --out $(FIXTURE)/partitions
	$(PY) -m agmesh.data.pipeline check --out $(FIXTURE)/partitions --no-expectations --fixture \
		--output $(FIXTURE)/partition_check.json

check-results:
	$(PY) -m agmesh.evaluation.checker

baselines:
	$(PY) -m agmesh.experiments.x1_baselines

scoring-selection:
	$(PY) -m agmesh.experiments.scoring_selection

ground-truth-draft:
	$(PY) -m agmesh.evaluation.ground_truth site_a
	$(PY) -m agmesh.evaluation.ground_truth site_b

training-budget:
	$(PY) -m agmesh.experiments.training_budget

summarise:
	$(PY) -m agmesh.experiments.summarise X1

shift-selection:
	$(PY) -m agmesh.experiments.shift_selection

dp-sanity:
	$(PY) -m agmesh.experiments.dp_sanity

SITE ?= site_a
train:
	$(PY) -m agmesh.federated.run --site $(SITE) \
		$(if $(ROUNDS),--rounds $(ROUNDS) --smoke,) $(if $(EPS),--epsilon $(EPS),) \
		$(if $(STRATEGY),--strategy $(STRATEGY),) $(if $(BACKEND),--backend $(BACKEND),) \
		$(if $(SEEDS),--seeds $(SEEDS),) $(if $(H1C),--h1c,)

x8:
	$(UV) run --group fl python -m agmesh.experiments.x8_secagg

g2-pilot:
	for site in site_b site_a; do for st in fedavg fedprox; do \
		$(PY) -m agmesh.federated.run --site $$site --strategy $$st --epsilon 10 || exit 1; \
	done; done

g2-tuning:
	for lr in 0.001 0.003; do for c in 0.5 1 2; do \
		$(PY) -m agmesh.federated.run --site site_a --strategy fedavg --epsilon 3 \
			--lr $$lr --clip $$c --tag g2_tuning || exit 1; \
	done; done
	for mu in 0.001 0.1; do \
		$(PY) -m agmesh.federated.run --site site_a --strategy fedprox --epsilon 10 \
			--mu $$mu --tag g2_mu || exit 1; \
	done
