# ADR 0001 — Gate G0 decisions (feasibility)

**Status:** Partly accepted. Data decisions 1–5 are accepted from the full data
analysis (Proposal v3.1, Appendix B). LLM and Ray decisions 6–10 wait for the Week 1
measurements on the development laptop.
**Date:** October 2026 (decisions 1–5)
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
the laptop inspection confirms them when all 21 checks match. A mismatch reopens the
affected decision.

## Decisions

| # | Decision | Options | Evidence | Chosen | Reason |
|---|---|---|---|---|---|
| 1 | HAI release | 21.03 (local) | Files readable; 3 train and 5 test files; 921,603 and 402,005 rows | **21.03** | The only complete release available; later releases are LFS pointers |
| 2 | Site A partition scheme | `regime` / `time` | `g0_hints.site_a_partition_scheme` = `time`; top regime candidate `P1_PCV02Z` is a calibration offset; four-cluster silhouette 0.34 following file boundaries | **`time`** (a1 = `train1`, a2 = `train2`, a3/a4 = halves of `train3`) | No separable regimes; recording periods are balanced and differ by real drift (HAI-3) |
| 3 | RQ2 fusion data | real multi-device attack / single-device attack / labelled synthetic composition | `g0_hints.rq2_fusion_data` = `single_device_attack`; captures overlap in time; attacker seen on `ied1a` only | **Real single-device attack day plus a labelled synthetic multi-device scenario** | RQ2 becomes a trade-off: false-alarm reduction on benign data (H2a), single-device loss on the real attack (H2b), multi-device gain on synthetic data (H2c) (MB-3, MB-12) |
| 4 | Site B labels | attack logs / attacker address / capture-level only | `g0_hints.site_b_attack_labels` = `attacker_address`; no logs in the download | **Attacker address, per window, marked derived** | Capture-level labels would mark all 3,681 IED1A attack-day windows as attacks; only 51 contain attacker traffic (MB-2, MB-6) |
| 5 | Site B attribution ground truth | dataset mapping / written mapping protocol / case study | `g0_hints.site_b_attribution_ground_truth` = `needs_mapping_protocol_or_manual_check`; no MITRE mapping in the dataset | **Hand-mapped case study of the seven episodes; Site A tactic level is the primary RQ3 result** | Seven episodes from one attacker position cannot support an accuracy estimate (Proposal §10.4) |
| 6 | LLM model | Phi-3.5-mini / Llama 3.2 3B | schema-valid rate, grounded rate, p95 latency, memory | *(pending laptop run)* | |
| 7 | L_LLM (full-mode latency target) | measured | `suggested_l_llm_s` of the chosen model | *(pending)* | |
| 8 | Default attribution mode | `full` / `retrieval_only` | Is L_LLM acceptable? | *(pending)* | |
| 9 | Ray client resources | CPUs per client | `ray_bench` memory and round time | *(pending)* | |
| 10 | Memory budget, measured (W1) | — | peak PSS of each benchmark | *(pending)* | |

## Consequences

Decisions 1–5 change `configs/params.yaml` as follows (already committed):

- `data.hai`: `label_column: attack`, `process_label_columns`, `drop_train_constant_features: true`,
  `drop_features: [P4_HT_PS]`, `scaling: {method: standard, clip_sd: 6.0}`.
- `data.modbus`: device and host addresses, `attacker_ip: 185.175.0.7`,
  `label_source: attacker_address`, `idle_gap_s: 60`.
- `partition.site_a`: `scheme: time` with `period_assignment`.
- `models.vae`: `input_mode: segment_means`, `segments_per_window: 6` (confirmed on validation in Week 4).
- `fusion.site_b.exclude_latency: true`.

Affected weeks: 2 (partition, parsing, labels), 4 (model input, ground truth),
8 (evidence fields, fusion), 9 (renderer) and 13 (X3, X4). Performance targets PT3,
PT4 and PT5 were revised before any model results exist (Proposal §11.2). No scope cut
from the cut order was taken.
