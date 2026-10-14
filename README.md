# Agentic Mesh v3

Federated, differentially private anomaly detection with knowledge-grounded agentic
attribution for industrial control systems — designed to run on a single 8 GB machine.

This repository is the **v3 redesign**, revised to v3.1 after a full analysis of
both datasets. It is being built phase by phase from the documents in `docs/`:

- [Proposal v3.1](docs/proposal_v3.md) — research questions, design, evaluation and success criteria.
- [Implementation Plan v3.1](docs/implementation_plan_v3.md) — 16 weeks, decision gates G0–G4, contracts.
- [Data card](docs/data_card.md) — measured facts for HAI 21.03 and CIC Modbus 2023; the same facts are checked by `configs/data_expectations.yaml`.

This repository starts fresh from v3. Earlier implementations (v1 and v2.1) were removed;
the design reasons for leaving them behind are in Proposal Appendix A.

## Progress

| Phase | Weeks | Status |
|---|---|---|
| 0 — Feasibility | 1 | **Done.** 21/21 data checks matched on the laptop; Gate G0 closed ([ADR 0001](docs/adr/0001-g0-decisions.md)): Llama 3.2 3B, L_LLM 24 s, 2 CPUs per Ray client |
| 1 — Data and evaluation first | 2–4 | **Weeks 2–4 done; Gate G1 in review.** Decisions accepted ([ADR 0002](docs/adr/0002-g1-proposals.md)), scoring chosen on validation ([ADR 0003](docs/adr/0003-scoring-selection.md)), training budget fixed label-free ([ADR 0004](docs/adr/0004-training-budget.md)), baselines on five seeds with CIs ([runbook](docs/runbooks/phase1_g1_seeds.md)), shift-robust local scoring (R3) chosen and event-level inference adopted ([ADR 0005](docs/adr/0005-shift-robust-scoring-and-inference.md), [runbook](docs/runbooks/phase1_g1_shift.md)), first-pass ground truth awaiting the author's reviews ([runbook](docs/runbooks/phase1_g1.md)). Publication plan: [research agenda](docs/research_agenda.md) |
| 2 — Privacy-preserving federated detection | 5–7 | Not started |
| 3 — Operations path and agentic attribution | 8–11 | Not started |
| 4 — Experiments and systems evidence | 12–14 | Not started |
| 5 — Packaging and reporting | 15–16 | Not started |

## Quick start (WSL2 Ubuntu or Linux)

```bash
make setup      # core + dev dependencies (Python 3.12 via uv)
make check      # ruff, mypy, pytest
make help       # all commands
```

The full Week 1 procedure, including WSL2 memory settings, data placement and the
benchmarks, is in [docs/runbooks/phase0_week1.md](docs/runbooks/phase0_week1.md).

## Layout

```
configs/params.yaml     single source of truth for every setting (validated on load)
configs/data_expectations.yaml  measured dataset facts, checked by scripts/inspect_data.py
configs/partition_expectations.yaml  expected partition counts, checked by make check-data
dvc.yaml                DVC stages: prepare (Modbus parsing) and partition
src/agmesh/             the package
  common/               config, provenance, artifacts, checksums, memory sampling
  feasibility/          Week 1 spikes: LLM benchmark, Ray benchmark, data inspection
  data/                 Week 2 pipeline: HAI and Modbus loading, labels, windows, partitions, fixture
  evaluation/           Week 3 harness: metrics, thresholds, calibration, stats, sealed test, checker
  models/               Week 4: model inputs, VAE and autoencoder (torch), PCA and Isolation Forest
  experiments/          experiment runners (X1 baselines so far)
  security/             egress allowlist (mTLS and JWT arrive in Week 11)
scripts/                command-line entry points
tests/unit/             unit tests (synthetic inputs only; CI never needs raw data)
docs/                   proposal, plan, threat model, data card, ADRs, runbooks
results/feasibility/    committed Week 1 measurement artifacts
```

## Working rules

Every reported number comes from a JSON artifact with provenance (git commit, data
and parameter hashes, machine), and each artifact states whether its data is
`runtime`, `fixture` or `synthetic`. Raw data is tracked by DVC, never by Git. See
Implementation Plan v3.1 §2 for the full list.
