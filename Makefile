# Agentic Mesh v3 — common commands (Implementation Plan v3.1, §5).
# Run from the repository root inside WSL2/Linux. All Python runs through uv.

UV ?= uv
PY := $(UV) run python

.PHONY: help setup setup-fl lock lint format typecheck test check \
        checksums verify-checksums inspect-data bench-llm bench-ray feasibility \
        prepare partition check-data data fixture fixture-smoke check-results baselines scoring-selection ground-truth-draft \
        training-budget summarise shift-selection dp-sanity train x8 g2-pilot g2-tuning \
        edge-model edge latency setup-kb kb kb-sources retrieval-queries retrieval-eval mesh llm-calibrate \
	certs stack-up stack-down approvals-api security-tests

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
	@echo "edge-model        Week 8: train and save the dev global model, e.g. make edge-model SITE=site_b"
	@echo "edge              Week 8: replay -> score -> evidence -> alerts, e.g. make edge SITE=site_b SPLIT=val"
	@echo "latency           Week 8: detection latency p50/p95 from the latest edge runs -> results/latency/"
	@echo "setup-kb          Week 9: install the knowledge-base group (Qdrant, MiniLM, pypdf)"
	@echo "kb-sources        Week 9: download NIST SP 800-82r3 (SHA-256 recorded on first use)"
	@echo "kb                Week 9: build the ATT&CK for ICS knowledge base -> data/kb/, results/kb/manifest.json"
	@echo "retrieval-queries Week 9: render the validation queries from the latest edge runs"
	@echo "retrieval-eval    Week 9: Recall@5 and MRR (the first reference run is the PT6 measurement)"
	@echo "mesh              Week 10: agent mesh over an edge run, e.g. make mesh SITE=site_b MODE=retrieval_only SPLIT=val"
	@echo "llm-calibrate     ADR 0013: LLM timeout by G0's rule on 30 real prompts, e.g. make llm-calibrate MODE=full"
	@echo "certs             Week 11: development PKI in certs/dev (3 CAs, server/client certs, JWT key)"
	@echo "stack-up          Week 11: one site's node APIs as separate mTLS processes, e.g. make stack-up SITE=site_b"
	@echo "stack-down        Week 11: stop the node API processes"
	@echo "approvals-api     Week 11: operator approval API for one mesh run, e.g. make approvals-api RUN=results/mesh/<run>"
	@echo "security-tests    Week 11: mTLS, JWT, egress and redaction negative tests"
	@echo "g2-pilot          Week 7: FedAvg and FedProx at eps 10, both sites, 50 rounds -> results/federated/g2_pilot/"
	@echo "g2-tuning         Week 7: ADR 0008 DP lr x clip grid and FedProx mu grid -> results/federated/g2_*"
	@echo "train             Week 6: one federated run, e.g. make train SITE=site_a [ROUNDS=5 smoke] [EPS=3] [STRATEGY=fedavg] [BACKEND=flower]"
	@echo "scoring-selection ADR 0003: score aggregation chosen on validation -> results/X1/"
	@echo "ground-truth-draft  first-pass attribution mapping (reads test labels, logged)"

setup:
	$(UV) sync --group dev --group ml

setup-fl:
	$(UV) sync --group dev --group ml --group fl

setup-kb:
	$(UV) sync --group dev --group ml --group fl --group kb

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

# Reproduces the ADR 0008 pilots with the settings in force before G2 (Site A DP lr 1e-3).
g2-pilot:
	for st in fedavg fedprox; do \
		$(PY) -m agmesh.federated.run --site site_b --strategy $$st --epsilon 10 --tag g2_pilot || exit 1; \
		$(PY) -m agmesh.federated.run --site site_a --strategy $$st --epsilon 10 --lr 0.001 --tag g2_pilot || exit 1; \
	done

g2-tuning:
	for lr in 0.001 0.003; do for c in 0.5 1 2; do \
		$(PY) -m agmesh.federated.run --site site_a --strategy fedavg --epsilon 3 \
			--lr $$lr --clip $$c --tag g2_tuning || exit 1; \
	done; done
	$(PY) -m agmesh.federated.run --site site_a --strategy fedavg --epsilon 3 \
		--lr 0.01 --clip 1 --tag g2_tuning
	for mu in 0.001 0.1; do \
		$(PY) -m agmesh.federated.run --site site_a --strategy fedprox --epsilon 10 \
			--mu $$mu --lr 0.001 --tag g2_mu || exit 1; \
	done

# Week 8 (ADR 0009): the development global model is FedProx at ε = 3, seed 11.
EDGE_MODEL = results/models/$(SITE)__fedprox__eps3/seed_11

edge-model:
	$(PY) -m agmesh.federated.run --site $(SITE) --strategy fedprox --epsilon 3 --seeds 11 \
		--tag week8_dev --save-model results/models

edge:
	$(PY) -m agmesh.edge.runtime --site $(SITE) --split $(or $(SPLIT),val) \
		--model $(or $(MODEL),$(EDGE_MODEL)) $(if $(SPEED),--speed $(SPEED),)

latency:
	$(PY) -m agmesh.evaluation.latency --segment detection --split $(or $(SPLIT),val)

# Week 9 (ADR 0010)
kb-sources:
	$(UV) run --group kb python -c "from pathlib import Path; from agmesh.kb.build import fetch_nist; print(fetch_nist(Path('data/kb')))"

kb:
	$(UV) run --group kb python -m agmesh.kb.build

retrieval-queries:
	$(UV) run --group kb python -m agmesh.evaluation.retrieval_eval --split val --queries-only

retrieval-eval:
	$(UV) run --group kb python -m agmesh.evaluation.retrieval_eval --split val \
		$(if $(VERSION),--version $(VERSION),)

# Week 10 (ADR 0012): MODE = full | retrieval_only | llm_no_retrieval | rule_only
mesh:
	$(UV) run --group kb python -m agmesh.mesh.runner --site $(SITE) --mode $(or $(MODE),full) \
		--split $(or $(SPLIT),val) --nodes $(or $(NODES),local) $(ARGS)

certs:
	$(PY) -m agmesh.security.ca --out certs/dev

stack-up:
	$(PY) -m agmesh.stack up --site $(SITE) $(ARGS)

stack-down:
	$(PY) -m agmesh.stack down

approvals-api:
	$(PY) -m agmesh.mesh.approvals_api --run $(RUN)

security-tests:
	$(UV) run pytest tests/security -q

llm-calibrate:
	$(UV) run --group kb python -m agmesh.mesh.llm_calibrate --mode $(MODE) $(ARGS)
