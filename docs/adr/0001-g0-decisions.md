# ADR 0001 — Gate G0 decisions (feasibility)

**Status:** Proposed. Waiting for the Week 1 measurements from the development laptop.
**Date:** *(fill in when accepted)*
**Inputs:**
- `results/feasibility/data_inspection.json`
- `results/feasibility/llm_bench.json`
- `results/feasibility/ray_bench.json`
- `data/raw_checksums.json`

## Context

Gate G0 (Implementation Plan v3.0 §7) fixes the decisions that later weeks depend on.
Each decision below is taken from measured numbers, not assumptions. When a decision
changes a value, `configs/params.yaml` is updated in the same commit as this ADR.

## Decisions

| # | Decision | Options | Evidence to use | Chosen | Reason |
|---|---|---|---|---|---|
| 1 | HAI release | 21.03 (local) | `data_inspection.hai` (files readable, no LFS pointers) | | |
| 2 | Site A partition scheme | `regime` / `time` | `g0_hints.site_a_partition_scheme`, regime candidates | | |
| 3 | RQ2 fusion data | real device overlap / labelled synthetic composition | `g0_hints.rq2_fusion_data`, attack overlap seconds | | |
| 4 | Site B labels | attack logs / capture-level only | `g0_hints.site_b_attack_labels` | | |
| 5 | Site B attribution ground truth | dataset mapping / written mapping protocol / tactic level only | `g0_hints.site_b_attribution_ground_truth` | | |
| 6 | LLM model | Phi-3.5-mini / Llama 3.2 3B | schema-valid rate, grounded rate, p95 latency, memory | | |
| 7 | L_LLM (full-mode latency target) | measured | `suggested_l_llm_s` of the chosen model | | |
| 8 | Default attribution mode | `full` / `retrieval_only` | Is L_LLM acceptable? | | |
| 9 | Ray client resources | CPUs per client | `ray_bench` memory and round time | | |
| 10 | Memory budget, measured (W1) | — | peak PSS of each benchmark | | |

## Consequences

*(Fill in: what changes in params.yaml, which weeks are affected, and any scope cut
taken from the cut order in Proposal §12.)*
