# ADR 0002 — Gate G1 proposals (evaluation frozen)

**Status:** Proposed. Each decision below needs the author's choice; nothing is frozen
and the `g1-frozen` tag does not exist yet.
**Date:** 14 October 2026
**Inputs:** `results/X1/*/seed_11.json` (Week 4 validation baselines, one seed, no
DP), `results/data/partition_check.json`, Proposal v3.1 §6, §10, §11.

## Context

Gate G1 (Implementation Plan §7) freezes the evaluation before any federated or
private result exists: the model inputs, the ground-truth mappings, the fusion
candidates, and the preregistered targets PT1–PT7. PT3 may be revised here using
validation baselines only, and never afterwards.

The Week 4 baselines were run on validation data only: thresholds from each node's
normal-validation windows at 1% window FPR, scores evaluated on the node's
attack-validation partition (15 HAI events on Site A, 4 Modbus episodes on b1).
`metrics` in each artifact is the mean over nodes; `per_client` holds each node.

## Validation baselines (seed 11, no DP)

**Site A (HAI), all 56 features**

| Config | AUROC | AUPRC | Event recall | Range F1 | Window FPR | FA/h |
|---|---|---|---|---|---|---|
| local VAE (segment means) | 0.844 | 0.656 | 1.000 | 0.081 | **0.637** | 7.0 |
| central VAE | 0.916 | 0.758 | 1.000 | 0.206 | 0.051 | 4.3 |
| local VAE (flat windows) | 0.832 | 0.664 | 1.000 | 0.159 | 0.505 | 4.7 |
| local PCA | 0.847 | 0.664 | 1.000 | 0.226 | 0.537 | 4.8 |
| central PCA | 0.890 | 0.767 | 0.938 | 0.361 | 0.015 | 1.6 |
| local Isolation Forest | 0.718 | 0.153 | 0.396 | 0.165 | 0.009 | 0.8 |
| central Isolation Forest | 0.789 | 0.275 | 0.250 | 0.205 | 0.008 | 0.6 |

**Site A without `P1_PCV02Z`** (decision 2)

| Config | AUROC | AUPRC | Event recall | Range F1 | Window FPR | FA/h |
|---|---|---|---|---|---|---|
| local VAE | 0.831 | 0.664 | 1.000 | 0.212 | 0.304 | 10.9 |
| central VAE | 0.914 | 0.750 | 1.000 | 0.353 | 0.030 | 1.9 |
| local VAE (flat) | 0.845 | 0.661 | 1.000 | 0.185 | 0.166 | 10.7 |
| local PCA | 0.887 | 0.694 | 0.938 | 0.352 | 0.053 | 6.4 |
| central PCA | 0.903 | 0.754 | 0.938 | 0.484 | 0.008 | 0.9 |

**Site B (Modbus), without `peers` and `new_peers`** (decision 1). AUROC and event
recall are defined on b1 only; FPR and FA/h are averaged over b1–b3.

| Config | AUROC (b1) | AUPRC | Event recall | Range F1 | Window FPR | FA/h |
|---|---|---|---|---|---|---|
| local AE | 0.999 | 0.994 | 1.000 | 0.217 | 0.059 | 16.1 |
| central AE | 1.000 | 1.000 | 1.000 | 0.198 | 0.052 | 12.7 |
| local PCA | 0.993 | 0.860 | 1.000 | 0.281 | 0.142 | 21.1 |
| central PCA | 0.993 | 0.860 | 1.000 | 0.194 | 0.019 | 4.2 |
| local Isolation Forest | 0.874 | 0.254 | 0.250 | 0.025 | 0.024 | 5.1 |
| new-peer rule | 1.000 | 1.000 | 1.000 | 0.196 | 0.000 | 0.0 |

High event recall at a high window FPR is not detection: when most windows alert,
every event is "found". Event recall must be read together with the FPR.

## Decisions

| # | Decision | Evidence | Options | Recommendation |
|---|---|---|---|---|
| 1 | **Site B identity features** (MB-15) | `new_peers` equals the label on every attack-day window and `peers` separates it perfectly. The new-peer rule scores AUROC 1.0 and FPR 0 with no model at all. Without the two features the autoencoder still reaches AUROC 0.999 on b1 from behaviour alone | (a) exclude from the model, report the rule as a baseline; (b) keep all 18 | **(a)**, currently implemented (`models.ae.exclude_features`). Keeps Site B a test of learned behaviour; `new_peer` stays in evidence records |
| 2 | **`P1_PCV02Z`** (new finding HAI-8) | It is a calibration offset that is almost constant within a recording period and steps between periods, so for three of the four clients the attack-validation normal windows sit a median 12–18 SD from the client's training mean on this one feature. Excluding it cuts local PCA FPR from 0.54 to 0.05 and local VAE FPR from 0.64 to 0.30, with similar AUROC | (a) exclude from model inputs and report its changes through the training-constant change rule, like `P1_PCV02D`; (b) keep | **(a)** (`models.vae.exclude_features: [P1_PCV02Z]`). It describes how a plant was configured, not how it behaves |
| 3 | **Threshold transfer on Site A** | Even without `P1_PCV02Z`, local VAEs exceed the 1% target on attack-validation normals (a1 0.50, a3 0.29, a4 0.41; a2 0.03). Each client's normal validation comes from its own recording period, while attack segments come from the test files' periods. Central models stay at 1–6% | (a) keep the policy (threshold from the node's own normal validation) and report realised FPR; (b) also use the normal windows of attack validation to set thresholds | **(a)**. Poor transfer of local-only models is the effect federation is meant to fix (RQ1, PT2); hiding it would weaken the comparison. Realised test FPR is always reported next to event recall |
| 4 | **Site A input mode** (HAI-6) | Segment means and flat windows give similar AUROC (0.844 vs 0.832; 0.831 vs 0.845 without `P1_PCV02Z`); flat has 10× the inputs, and DP noise grows with model size | segment means / flat | **Segment means** (current default) |
| 5 | **Attack-day drift on Site B** (new finding MB-16) | On the attack day, normal windows alert at 7–10% on b1 and b3 against 1% on benign validation, driven by TCP resets and latency. Only 1 of 12 HMI alerts falls inside an attack episode, so this is drift between days, not indirect attack evidence | (a) keep labels and report the false alarms; (b) relabel | **(a)**. Report it; it bounds what fusion can gain (RQ2) |
| 6 | **Site A attribution ground truth** (RQ3) | HAI attacks manipulate the process, so at tactic level nearly all of them are expected to fall under *Impair Process Control*; tactic accuracy would then be trivial (the majority class near 100%) | (a) map at technique level within process manipulation (setpoint change → T0836 Modify Parameter; actuator command → T0855 Unauthorized Command Message; sensor or view falsification → T0856/T0832), using the targeted process and the variable types that deviate during each event; (b) tactic level only and accept a degenerate result | **(a)**, following `docs/mapping_protocol.md`. It needs the author's two review passes, and it reads the test events' label columns and deviating variable names for ground truth only. The access is logged and no model choice uses it |
| 7 | **PT3 (Site A event recall)** | The central VAE reaches event recall 1.0 at window FPR 0.03–0.05; the best local model reaches 0.94–1.0 only at FPR 0.05–0.64 | keep "≥ 0.70 at window FPR ≤ 1%" / restate | **Restate:** event recall ≥ 0.70 *and* realised test window FPR ≤ 2% for the federated DP model; report both regardless |
| 8 | **Seeds and runtime** | One full X1 baseline pass takes about 3 minutes on the sandbox CPU | — | Five seeds for X1–X4 headline configurations, as planned |

## Consequences (if accepted as recommended)

- `configs/params.yaml`: `models.vae.exclude_features: [P1_PCV02Z]`; `models.ae.exclude_features` stays `[peers, new_peers]`.
- Proposal Appendix B gains HAI-8 and MB-16; §11.2 PT3 is restated; §10.4 Site A ground truth becomes technique-level within process manipulation.
- `evaluation/ground_truth/mapping_site_a.json` and `mapping_site_b.json` are built under `docs/mapping_protocol.md`, reviewed twice, and committed with `docs/preregistration.md`; then the commit is tagged `g1-frozen`.
