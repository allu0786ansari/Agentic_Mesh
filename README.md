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
| 1 — Data and evaluation first | 2–4 | **Weeks 2–4 done; G1 passed for detection (`g1-frozen`); ground truth reviewed in two passes (protocol v1.1, ATT&CK for ICS v20.0), awaiting the `gt-frozen` tag.** Decisions accepted ([ADR 0002](docs/adr/0002-g1-proposals.md)), scoring chosen on validation ([ADR 0003](docs/adr/0003-scoring-selection.md)), training budget fixed label-free ([ADR 0004](docs/adr/0004-training-budget.md)), baselines on five seeds with CIs ([runbook](docs/runbooks/phase1_g1_seeds.md)), shift-robust local scoring (R3) chosen and event-level inference adopted ([ADR 0005](docs/adr/0005-shift-robust-scoring-and-inference.md), [runbook](docs/runbooks/phase1_g1_shift.md)), ground truth from the documented HAI timetable, independently verified in pass 2 ([mapping protocol](docs/mapping_protocol.md) §6). Publication plan: [research agenda](docs/research_agenda.md) |
| 2 — Privacy-preserving federated detection | 5–7 | **Weeks 5–7 done; Gate G2 passed** ([ADR 0008](docs/adr/0008-g2-decisions.md): SecAgg+ X8, pilots, DP lr tuned). Week 6: federations, both backends, stragglers, H1c ([ADR 0007](docs/adr/0007-federation-week6.md)). Week 5: RDP ledger per client, budget guard, DP normalisation histograms, DP-SGD; sanity check at ε 1/3/10/∞ ([ADR 0006](docs/adr/0006-dp-design-week5.md), [runbook](docs/runbooks/phase2_week5.md)). X1–X2 grid and Phase 3 next |
| 3 — Operations path and agentic attribution | 8–11 | **Week 8 built** ([ADR 0009](docs/adr/0009-edge-runtime-week8.md), [runbook](docs/runbooks/phase3_week8.md)): edge runtime with streaming scoring against the frozen global model, minimised evidence records (contract-tested), durable SQLite bus with leases and redelivery, session store, Site B fusion; detection latency p95 7 ms (A) and 1 ms (B) on validation. Ground truth frozen (`gt-frozen`). **Week 9 built** ([ADR 0010](docs/adr/0010-knowledge-base-retrieval-week9.md), [runbook](docs/runbooks/phase3_week9.md)): ATT&CK for ICS v20.0 knowledge base (104 techniques, 404 passages, embedded Qdrant), documentation-based evidence renderer, 48-query retrieval set, frozen rule baseline; PT6 measured on the laptop. Weeks 10–11 next |
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
