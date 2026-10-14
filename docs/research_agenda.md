# Research Agenda and Publication Plan

**Status:** v1, 14 October 2026, written at Gate G1. Claims below are *planned*; each
becomes a result only through the experiments named next to it, on the sealed test
set, after `g4-frozen`. Related work was checked on 14 October 2026 (§5); re-check it
before submission.

## 1. What makes these results publishable

Reviewers in ICS security and federated learning reject most often for four reasons:
test-set tuning, label leakage, point-adjusted or otherwise inflated metrics, and
novelty that is already published. The project is built against all four:

| Risk | Safeguard in this repository |
|---|---|
| Tuning on test data | Sealed test split behind `--final` and the `g4-frozen` tag; every access logged (`results/test_access.log`); thresholds from normal validation only |
| Choices made after seeing results | Decisions recorded as ADRs *before* the runs they govern; ADR 0003's grid was pushed (commit `9081602`) before it ran; preregistration frozen at G1 |
| Label leakage | `LabelFreeFrame` contract tests; label-proxy diagnostics (`make check-data`) |
| Inflated metrics | Event recall reported with realised FPR; range-based F1; no point adjustment (a test shows why); five seeds with bootstrap CIs |

## 2. Papers

### Paper A — Evaluation pitfalls in two public ICS datasets (short paper)

**Claim.** Two widely used ICS datasets contain traps that silently inflate or break
detection results, and both can be detected automatically.

1. *Derived labels become features.* In CIC Modbus 2023 the external-attack labels can
   only be derived from the attacker's address, and a traffic feature ("a peer never
   seen in training") then reproduces the label exactly: a one-line rule reaches AUROC
   1.0 with zero false positives (ADR 0002, decision 1). Capture-level labels, the other
   common shortcut, mark all 3,681 IED1A attack-day windows as attacks when only 51 are.
2. *Identity-blind detection is still possible.* Without identity features, an
   autoencoder still separates the attack from behaviour alone (AUROC 0.999 on
   validation), which is the honest version of the benchmark.
3. *Calibration offsets masquerade as drift.* In HAI 21.03, `P1_PCV02Z` steps between
   recording periods; per-period models then flag 54–64% of other periods' normal
   windows at a 1% target FPR. Excluding that one variable cuts this to 5–30%
   (HAI-8).
4. *Day-to-day drift on the attack day.* Normal Modbus traffic on the attack day alerts
   at 7–10% against 1% on benign days (MB-16).

**Evidence.** Week 2–4 artifacts (all on validation, already produced); test-split
confirmation from X1 after G4.
**Venue type.** ICS/CPS security workshop (for example CyberICPS at ESORICS, or
CPSIoTSec at ACM CCS).
**Novelty check.** No published analysis of the CIC Modbus 2023 label/identity proxy
was found (§5). Threshold miscalibration under drift is known (Shi et al., 2026); our
contribution there is the specific cause (a calibration offset) and the per-period
partition that exposes it.

### Paper B — Federated DP detection for an ICS fleet with recording-period drift (main paper)

**Claim.** When each plant's normal data comes from a different operating period,
local detectors do not transfer; federated training with DP-SGD recovers most of
the centralised model's transfer at a stated (ε, δ), and a **DP federated threshold**
transfers better than per-plant thresholds.

- **RQ1 / H1a, H1b** as preregistered (Proposal §2).
- **H1c (new, secondary, preregistered at G1).** A threshold computed from the
  clients' DP noisy normal-score histograms (pooled 99th percentile, its ε added to
  each client's budget) gives a lower realised window FPR on the test periods than
  per-node thresholds, at equal event recall, for the same federated model.
- Validation baseline motivating it: local VAE realised FPR 0.30 vs central 0.03 at
  a 1% target (ADR 0003).
- **Revised at ADR 0005 (validation; changes the paper's framing).** The local FPR
  came from operating-point shift between recordings. A label-free local robust
  adaptive threshold (R3, median + k·MAD over the last 30 minutes) brings the local
  VAE to FPR 0.021 and AUROC 0.922. That leaves central +0.026 AUROC
  [+0.004, +0.061] (event-level, Holm p = 0.020) and +0.005 FPR (not significant).
  Federation must now beat R3, not the broken local baseline. The claim has to rest
  on the AUROC/AUPRC gap and on H1c against R3. If the test set confirms the small
  headroom, "local adaptive scoring closes most of the federation gap under
  operating-point shift" is itself a publishable negative-result finding.

**Evidence.** X1 (utility over ε ∈ {1, 3, 10, ∞}), X2 (heterogeneity, FedAvg vs
FedProx), H1c arm added to X1; five seeds.
**Venue type.** ICS/CPS security or applied ML-for-security conference or journal.
**Novelty check.** FL with DP for industrial anomaly detection is crowded (§5), so the
novelty is not "FL + DP". It is (i) a realistic period-drift partition of a public
ICS dataset with event-level, sealed-test evaluation and persistent per-client
accounting; (ii) DP federated threshold calibration evaluated on that drift; (iii) a
fair local baseline: a preregistered, label-free shift-robust local scorer, and
event- and block-level inference instead of seed-level tests. The
closest threshold work (Laridi et al.) aggregates summary statistics without DP and
on non-ICS data.

### Paper C — Grounded, privacy-minimal LLM attribution of ICS incidents (main paper)

**Claim.** A local 3B-parameter model, constrained to choose among ATT&CK for ICS
techniques retrieved for minimised evidence records, attributes HAI incidents better
than retrieval alone or an unconstrained LLM, never emits an ungrounded technique,
and runs within 24 s per incident on a laptop CPU.

- **RQ3 / H3** with technique-level ground truth (ADR 0002 decision 6; mapping protocol).
- Four ablation modes (X4), majority-class baseline, raw grounding rate, latency (X5).
- Site B: seven-episode case study.
- **Ground truth from the dataset's documentation (mapping protocol v1.0).** All 50
  HAI 21.03 events match the attack timetable published with the dataset: 32 are
  setpoint-led (T0836) and 18 output-led (T0855). The majority and constant-answer
  baselines are 0.64.
  - The attacks hide setpoint changes in the HMI, so a data-only rule names the
    documented technique for only 18 of 50 events.
  - Results are stratified by SP-led versus CO-led events. Recognising a hidden
    setpoint change from how the controller's output tracks it is the test of
    *grounded* attribution, and a possible contribution in its own right.

**Venue type.** Security or applied-AI-for-security venue; a workshop version is
possible after X4.
**Novelty check.** LLMs explaining Modbus traffic (Saber et al., 2026, GPT-4o, no
ATT&CK mapping) and RAG for enterprise incident analysis (Cadet et al., 2026, 70B
models) exist. The combination here is new as far as we found: local small model,
schema-constrained to retrieved ICS techniques, evidence minimised for privacy,
ground truth frozen before results.

## 3. Threats to validity (to state in every paper)

| Threat | Mitigation | What remains |
|---|---|---|
| One attack day and one attacker position on Site B | Per-episode reporting, no Site B accuracy target, synthetic multi-device scenario labelled as synthetic | Site B results are indicative |
| Ground truth from the same signals the detector sees | The mapping uses full-resolution event data, timing and variable types; attribution sees only minimised records (top-5 feature names, coarse magnitudes, no ordering). The rule baseline is written from evidence records and frozen before test | A rule that recovers variable types from feature names may score well; report it |
| Testbed data | Two independent testbeds; results scoped to them | Generalisation to production plants is untested |
| Small seed count | Five seeds, paired bootstrap; Wilcoxon reported but cannot reject alone | Small effects will not be significant |
| Scoring and inputs chosen on 15 validation events | Small, pre-declared choices only (ADRs 0002–0003) | Some optimism in validation numbers; test numbers are the claim |

## 4. Order of work

1. **G1 freeze:** author reviews the mappings twice, accepts H1c, then tag `g1-frozen`.
2. **Phase 2 (Weeks 5–7):** DP-SGD with persistent accounting, federations, federated
   evaluation and the DP threshold (H1c).
3. **Phase 3 (Weeks 8–11):** evidence records, fusion, knowledge base, mesh.
4. **Phase 4 (Weeks 12–14):** X1–X8, then `g4-frozen` and the single test run per
   configuration and seed.
5. **Writing:** Paper A can be drafted after G4 confirms its findings on test; Papers B
   and C after Phase 4.

## 5. Related work consulted (14 October 2026)

- Laridi, Palmer, Tam — *Enhanced Federated Anomaly Detection Through Autoencoders Using Summary Statistics-Based Thresholding* (arXiv 2410.09284).
- Shi, Xiang, Zhang, Fu — *Where Does Detection Fail? A Residual-Evidence Diagnostic Workflow for Industrial Control Systems* (arXiv 2608.02821).
- Saber, Youssef, Kundur, Kong — *Large Language Models as Explainable Cyberattack Detectors for Energy Industrial Control Systems* (arXiv 2604.26079).
- Cadet et al. — *Retrieval-Augmented LLMs for Security Incident Analysis* (arXiv 2603.18196).
- Survey context: *Network Intrusion Datasets: A Survey, Limitations, and Recommendations* (arXiv 2502.06688); *Errors in the CICIDS2017 dataset…* (HAL hal-03775466).
- Public CIC Modbus 2023 project with an evaluation-integrity checklist but no label-proxy analysis: github.com/dmtkfs/ics-modbus-anomaly-detection.
