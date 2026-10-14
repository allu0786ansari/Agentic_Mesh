# Preregistration (Gate G1)

**Status:** Draft. Values marked **[decide]** need the author's choice (ADR 0002). Once
accepted, this file is committed with the ground-truth mappings and the commit is
tagged `g1-frozen`. After that, nothing here changes; a missed target is reported as a
finding.

## 1. Frozen evaluation configuration

| Item | Value |
|---|---|
| Parameters file | `configs/params.yaml` at the `g1-frozen` commit (its SHA-256 is recorded in every result artifact) |
| Partitions | `data/partitions/*/manifest.json` from `dvc.lock` at that commit |
| Labels | `docs/labeling_protocol.md` v1.0 |
| Ground truth | `evaluation/ground_truth/mapping_site_{a,b}.json` under `docs/mapping_protocol.md` |
| Site A model inputs | 56 features **[decide: minus `P1_PCV02Z`]**, six 10 s segment means per feature |
| Site B model inputs | 18 features minus `peers`, `new_peers` **[decide]**; `log1p`, standardised |
| Operating point | Threshold per node from its normal-validation windows at window FPR 1%; also reported at 0.5% and 5% |
| Event tolerance | 60 s after the event end |
| Seeds | 11, 23, 37, 41, 53 (five for headline configurations, three otherwise) |

## 2. Primary metrics

- Detection: AUROC (window level); event recall with the realised window FPR; range-based F1; median time to detect; false alarms per hour.
- Privacy: per-client (ε, δ) including the normalisation-statistics cost.
- Attribution: Site A technique top-1 and top-3 and tactic accuracy against the majority-class baseline; raw grounding rate; Site B per-episode case study.

## 3. Targets (Proposal §11.2)

| ID | Target | Week 4 validation evidence |
|---|---|---|
| PT1 | Federated DP model at ε ≤ 10: AUROC no more than 0.05 below centralised non-private, on each site | Central VAE 0.916 (A), central AE 1.000 on b1 (B) |
| PT2 | Federated DP model at ε ≤ 10: AUROC at or above the mean local-only AUROC, on each site | Local VAE 0.844 (A), local AE 0.999 on b1 (B) |
| PT3 | **[decide]** recommended: Site A event recall ≥ 0.70 on the 35 test events **and** realised test window FPR ≤ 2%, for the federated DP model | Central VAE: event recall 1.0 at FPR 0.03–0.05 (validation) |
| PT4 | Site B 2-of-3 fusion reduces false alarms per hour by ≥ 30% against the best single node at the same thresholds | Attack-day normal false alarms: 13–16 per hour (AE), mostly b1 and b3 |
| PT5 | Site A attribution: technique top-3 ≥ 0.70 **[decide: technique or tactic level]**, above the majority-class baseline; `full` ≥ `retrieval_only` with CI. Site B: JSON validity ≥ 95%; raw grounding rate reported | — |
| PT6 | Retrieval Recall@5 ≥ 0.85 on the held-out query set | — |
| PT7 | Latency p95: detection ≤ 1 s; investigation ≤ 2 s; incident-to-dispatch ≤ 3 s (`retrieval_only`) and ≤ 24 s (`full`, L_LLM from G0) | LLM call p95 13.6 s (G0) |

## 4. Statistical analysis

- Results as mean ± 95% bootstrap CI over seeds.
- Paired comparisons (same seeds) by paired bootstrap; the exact Wilcoxon signed-rank test is reported alongside (with five seeds it cannot reach p < 0.05 on its own).
- Holm correction across the comparisons within each RQ: RQ1 (PT1, PT2 per site), RQ2 (H2a, H2b), RQ3 (`full` against each ablation).

## 5. Test-set access

Test partitions are read only by `--final` runs after the `g4-frozen` tag, once per
frozen configuration and seed, and every read is logged. The one exception is the
Site A ground-truth labelling pass in `docs/mapping_protocol.md` §2, which is logged
under the configuration `ground_truth_site_a` before any attribution result exists.
