# ADR 0001 — Gate G0 decisions (feasibility)

**Status:** Accepted. Gate G0 is closed.
**Date:** 13 October 2026 (decisions 1–5, from the data analysis); 14 October 2026
(decisions 6–10, from the laptop measurements)
**Inputs:**
- `results/feasibility/data_inspection.json` (laptop run with `--expectations configs/data_expectations.yaml`)
- `results/feasibility/llm_bench.json`
- `results/feasibility/ray_bench.json`
- `data/raw_checksums.json`

## Context

Gate G0 (Implementation Plan v3.1 §7) fixes the decisions that later weeks depend on.
Each decision is taken from measured numbers, not assumptions. When a decision changes
a value, `configs/params.yaml` is updated in the same commit as this ADR.

Decisions 1–5 were taken after a full analysis of the uploaded HAI 21.03 files and the
CIC Modbus 2023 captures. The same facts are encoded in `configs/data_expectations.yaml`;
the laptop inspection confirmed all 21 checks (`data_inspection.json`, run on commit
`46e1a03`).

**Measurement machine.** WSL2 (kernel 6.6, glibc 2.35) with 4 logical CPUs on 2 physical
cores, 5.9 GB of memory visible to Linux and 4 GB swap. Ollama 0.40.3; flwr 1.39.0,
ray 2.55.1, torch 2.14.1+cpu.

## Decisions

| # | Decision | Options | Evidence | Chosen | Reason |
|---|---|---|---|---|---|
| 1 | HAI release | 21.03 (local) | Files readable; 3 train and 5 test files; 921,603 and 402,005 rows | **21.03** | The only complete release available; later releases are LFS pointers |
| 2 | Site A partition scheme | `regime` / `time` | `g0_hints.site_a_partition_scheme` = `time`; top regime candidate `P1_PCV02Z` is a calibration offset; four-cluster silhouette 0.34 following file boundaries | **`time`** (a1 = `train1`, a2 = `train2`, a3/a4 = halves of `train3`) | No separable regimes; recording periods are balanced and differ by real drift (HAI-3) |
| 3 | RQ2 fusion data | real multi-device attack / single-device attack / labelled synthetic composition | `g0_hints.rq2_fusion_data` = `single_device_attack`; captures overlap in time; attacker seen on `ied1a` only | **Real single-device attack day plus a labelled synthetic multi-device scenario** | RQ2 becomes a trade-off: false-alarm reduction on benign data (H2a), single-device loss on the real attack (H2b), multi-device gain on synthetic data (H2c) (MB-3, MB-12) |
| 4 | Site B labels | attack logs / attacker address / capture-level only | `g0_hints.site_b_attack_labels` = `attacker_address`; no logs in the download | **Attacker address, per window, marked derived** | Capture-level labels would mark all 3,681 IED1A attack-day windows as attacks; only 51 contain attacker traffic (MB-2, MB-6) |
| 5 | Site B attribution ground truth | dataset mapping / written mapping protocol / case study | `g0_hints.site_b_attribution_ground_truth` = `needs_mapping_protocol_or_manual_check`; no MITRE mapping in the dataset | **Hand-mapped case study of the seven episodes; Site A tactic level is the primary RQ3 result** | Seven episodes from one attacker position cannot support an accuracy estimate (Proposal §10.4) |
| 6 | LLM model | Phi-3.5-mini / Llama 3.2 3B | Llama 3.2 3B: 30/30 calls succeeded, p50 11.1 s, p95 13.6 s, 7.2 tokens/s, schema-valid and grounded 100%, model 2.2 GB. Phi-3.5-mini: 29/30 (one 180 s timeout), p50 21.6 s, p95 70.9 s, 4.2 tokens/s, schema-valid and grounded 100%, model 3.1 GB | **Llama 3.2 3B** (`llama3.2:3b`) | Five times lower p95, no timeouts, and 0.9 GB less memory on a machine whose memory peaked at 5.9 of 5.9 GB during the Phi run. Agreement with the intended technique on the synthetic prompts (37% vs 45%) is indicative only; RQ3 measures attribution quality properly |
| 7 | L_LLM (full-mode latency target) | measured | `suggested_l_llm_s` = 21 s (1.5 × p95) | **LLM call timeout 21 s; L_LLM = 24 s** | The LLM call gets 1.5 × its measured p95; the rest of the path keeps the 3 s `retrieval_only` budget. Re-measured under the full operations load at G3 |
| 8 | Default attribution mode | `full` / `retrieval_only` | 24 s per incident is acceptable for analyst-facing attribution; detection and dispatch of low-impact alerts do not wait for it | **`full`** | The `retrieval_only` fallback stays in place for timeouts and invalid output (Proposal §8.4) |
| 9 | Ray client resources | CPUs per client | 2 CPUs: 2 concurrent clients, 3.8 s per steady round, peak PSS 1.8 GB. 1 CPU: 4 concurrent, 4.3 s per round, peak PSS 2.8 GB. Both completed all rounds | **`num_cpus: 2`** | Faster and 0.9 GB lighter on 2 physical cores; extra concurrency only adds contention |
| 10 | Memory budget, measured (W1) | — | Training stage: 1.8 GB PSS for 4 clients with a 2.5 M-parameter model (larger than the real Site A VAE). LLM: Ollama reports 2.2 GB for Llama 3.2 3B; system memory peaked near the 5.9 GB limit during the LLM runs | **Training ≤ 2.5 GB; operations must fit in 5.9 GB with the LLM at 2.2 GB** | Stage isolation (Proposal §9.4) stays mandatory. Ollama's RSS (5.4 GB) counts its memory-mapped model file and overstates its share; PSS was unreadable because Ollama runs as another user. X7 re-measures in Week 14 |

## Consequences

Decisions 1–10 change `configs/params.yaml` as follows:

- `data.hai`: `label_column: attack`, `process_label_columns`, `drop_train_constant_features: true`,
  `drop_features: [P4_HT_PS]`, `scaling: {method: standard, clip_sd: 6.0}`.
- `data.modbus`: device and host addresses, `attacker_ip: 185.175.0.7`,
  `label_source: attacker_address`, `idle_gap_s: 60`.
- `partition.site_a`: `scheme: time` with `period_assignment`.
- `models.vae`: `input_mode: segment_means`, `segments_per_window: 6` (confirmed on validation in Week 4).
- `fusion.site_b.exclude_latency: true`.
- `llm.model: llama3.2:3b`, with Phi-3.5-mini as `alternative_model`.
- `mesh.attribution.llm_timeout_s: 21` and the new `mesh.attribution.l_llm_s: 24`.
- `federated.client_resources.num_cpus: 2` (unchanged; now measured).

Affected weeks: 2 (partition, parsing, labels), 4 (model input, ground truth),
8 (evidence fields, fusion), 9 (renderer) and 13 (X3, X4). Performance targets PT3,
PT4 and PT5 were revised before any model results exist (Proposal §11.2). No scope cut
from the cut order was taken.
