# Agentic Mesh v3

Federated, differentially private anomaly detection with knowledge-grounded agentic
attribution for industrial control systems — designed to run on a single 8 GB machine.

This branch (`phase3`) is the **v3 redesign**. It is being built phase by phase from
the documents in `docs/`:

- [Proposal v3.0](docs/proposal_v3.md) — research questions, design, evaluation and success criteria.
- [Implementation Plan v3.0](docs/implementation_plan_v3.md) — 16 weeks, decision gates G0–G4, contracts.

The v2.1 code is kept for reference only in [`legacy/v2/`](legacy/v2). It is excluded
from linting, type checking, tests and CI.

## Progress

| Phase | Weeks | Status |
|---|---|---|
| 0 — Feasibility | 1 | **Code done.** Waiting for the laptop runs ([runbook](docs/runbooks/phase0_week1.md)) and the Gate G0 review |
| 1 — Data and evaluation first | 2–4 | Not started |
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
src/agmesh/             the package
  common/               config, provenance, artifacts, checksums, memory sampling
  feasibility/          Week 1 spikes: LLM benchmark, Ray benchmark, data inspection
  security/             egress allowlist (mTLS and JWT arrive in Week 11)
scripts/                command-line entry points
tests/unit/             unit tests (synthetic inputs only; CI never needs raw data)
docs/                   proposal, plan, threat model, data card, ADRs, runbooks
results/feasibility/    committed Week 1 measurement artifacts
legacy/v2/              v2.1 code, reference only
```

## Working rules

Every reported number comes from a JSON artifact with provenance (git commit, data
and parameter hashes, machine), and each artifact states whether its data is
`runtime`, `fixture` or `synthetic`. Raw data is tracked by DVC, never by Git. See
Implementation Plan v3.0 §2 for the full list.
