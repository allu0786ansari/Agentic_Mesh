# Preregistration (Gate G1)

**Status:** Frozen in two parts (decided 15 October 2026, before any Phase 2 result).

1. **Detection evaluation: frozen at the tag `g1-frozen`.** This covers
   §1 (except the ground-truth row), §2 detection and privacy metrics, §3 detection
   and privacy targets, §4 and §5. These are the decisions of ADRs 0002-0005. After
   the tag, none of them changes; a missed target is reported as a finding.
2. **Attribution ground truth: frozen at the tag `gt-frozen`.**
   - **Covers:**
     - the ground-truth and ATT&CK rows of §1;
     - the attribution metrics of §2 and target PT5;
     - `evaluation/ground_truth/` and `evaluation/attack/` under
       `docs/mapping_protocol.md` v1.1.
   - **Basis:** the verification pass 2 (15 October 2026, mapping protocol §6). The
     author delegated it to the assistant, so no human has verified the mapping; the
     papers disclose this.
   - **Required before Phase 3:** the tag must exist before any attribution output is
     produced (Week 8). The Phase 3 runners refuse to run without it.

Phase 2 (private federated detection) does not read the ground truth, so the split
does not let any attribution decision depend on a detection result, or the reverse.

## 1. Frozen evaluation configuration

| Item | Value |
|---|---|
| Parameters file | `configs/params.yaml` at the `g1-frozen` commit (its SHA-256 is recorded in every result artifact) |
| Partitions | `data/partitions/*/manifest.json` from `dvc.lock` at that commit |
| Labels | `docs/labeling_protocol.md` v1.0 |
| Ground truth | `evaluation/ground_truth/mapping_site_{a,b}.json` under `docs/mapping_protocol.md` v1.1 (two passes, `reviewed: true`), frozen at `gt-frozen` (see Status) |
| ATT&CK for ICS | v20.0, `mitre-attack/attack-stix-data` @ `c2494b2` (SHA-256 pinned in `src/agmesh/evaluation/attack_ics.py`); extract `evaluation/attack/ics_attack_20.0_techniques.json` |
| Site A model inputs | 56 features minus `P1_PCV02Z` (ADR 0002), six 10 s segment means per feature; score = mean squared error, no smoothing (ADR 0003), then the robust adaptive threshold R3 (ADR 0005) |
| Training budget | VAE `max_epochs` 200, AE 30, early stopping as configured (ADR 0004; non-private runs) |
| Site B model inputs | 18 features minus `peers`, `new_peers` (ADR 0002); `log1p`, standardised; score = mean of the three largest normalised feature errors, no smoothing (ADR 0003) |
| Operating point | Threshold per node from its normal-validation windows at window FPR 1%; also reported at 0.5% and 5% |
| Event tolerance | 60 s after the event end |
| Seeds | 11, 23, 37, 41, 53 (five for headline configurations, three otherwise) |

## 2. Primary metrics

- Detection: AUROC (window level); event recall with the realised window FPR; range-based F1; median time to detect; false alarms per hour.
- Privacy: per-client (ε, δ) including the normalisation-statistics cost.
- **Attribution:**
  - Site A technique top-1 and top-3 against the majority-class and constant-answer
    baselines (0.64), with predicted IDs first normalised to the pin by MITRE's
    `revoked-by` relationships (T0855 → T1692.001, T0856 → T1692.002). A parent
    technique does not match a sub-technique (mapping protocol §5).
  - Tactic accuracy is reported, but on Site A a constant tactic scores 1.0, so it
    is never used for a claim.
  - Raw grounding rate.
  - Site B per-episode case study.

## 3. Targets (Proposal §11.2)

| ID | Target | Week 4 validation evidence |
|---|---|---|
| PT1 | Federated DP model at ε ≤ 10: AUROC no more than 0.05 below centralised non-private, on each site | Central VAE 0.914 (A), central AE 1.000 on b1 (B); validation, G1 inputs |
| PT2 | Federated DP model at ε ≤ 10: AUROC at or above the mean local-only AUROC, on each site | Local VAE 0.829 (A), local AE 0.999 on b1 (B); validation, G1 inputs |
| PT3 | Site A event recall ≥ 0.70 on the 35 test events **and** realised test window FPR ≤ 2%, for the federated DP model (ADR 0002) | Central VAE: event recall 1.0 at FPR 0.03 (validation, G1 inputs) |
| PT4 | Site B 2-of-3 fusion reduces false alarms per hour by ≥ 30% against the best single node at the same thresholds | Attack-day normal false alarms: 13–16 per hour (AE), mostly b1 and b3 |
| PT5 | Site A attribution: technique top-3 ≥ 0.70 against the acceptable sets, above the majority-class baseline (0.64 under protocol v1.1); `full` ≥ `retrieval_only` with CI. Site B: JSON validity ≥ 95%; raw grounding rate reported | — |
| PT6 | Retrieval Recall@5 ≥ 0.85 on the held-out query set | — |
| PT7 | Latency p95: detection ≤ 1 s; investigation ≤ 2 s; incident-to-dispatch ≤ 3 s (`retrieval_only`) and ≤ 24 s (`full`, L_LLM from G0) | LLM call p95 13.6 s (G0) |

## 3a. Secondary hypothesis added at G1

**H1c.** For the federated DP model on Site A, a threshold computed from the clients'
DP noisy normal-validation score histograms (pooled 99th percentile; its ε is added to
each client's budget by basic composition) gives a lower realised test window FPR than
per-node thresholds, at event recall within 0.05. Tested in X1 at ε = 10, five seeds,
paired bootstrap. Motivation: validation local VAE FPR 0.30 against central 0.03
(ADR 0003); `docs/research_agenda.md`, Paper B.

*Update (ADR 0005, before any test access).* Site A now uses R3, a label-free local
adaptive threshold. It brings the local VAE validation FPR to 0.021. H1c is
therefore tested against two comparators: per-node fixed thresholds on R0 scores,
and R3. Both comparisons are reported, and Holm correction is applied across them.

## 3b. Secondary target added after PT6 (ADR 0011, 15 October 2026)

**PT6 result.** PT6 was measured once on validation queries (ADR 0010): Recall@5
0.333, so it was not met. That result stands.

**The change.** A post-hoc retrieval change, v2, was declared before it was run:

- techniques whose only tactic is Impact are removed from the candidates;
- the LLM is offered 10 candidates instead of 5.

**PT6-v2.** This is v2's Recall@10 on the **test** incidents (Site A's 35 events,
Site B's episodes 4–6; rendered queries only), measured once after `g4-frozen`.

- **Target:** ≥ 0.85.
- **Also reported:** Recall@5, MRR and v1 on the same queries.
- **Status:** validation numbers for v2 are exploratory. PT6-v2 does not replace
  PT6.

## 4. Statistical analysis

Updated at G1 by ADR 0005 (accepted 14 October 2026, before any test access).

- **Primary:** event-level inference. A two-stage stratified block bootstrap
  resamples attack events and 30-minute normal blocks (60 minutes on Site B) within
  each node, with B = 10 000. The same draws are used for every configuration
  (paired), and seeds are averaged inside each draw. Results are reported as the
  observed value with its 95% percentile CI; a comparison is reported as the
  difference, its CI, and a one-sided p-value.
- **Secondary:** mean ± 95% CI over seeds, which shows training variance, and the
  exact Wilcoxon signed-rank test over seeds. With five seeds it cannot reach
  p < 0.05.
- Holm correction across the comparisons within each RQ: RQ1 (PT1, PT2 per site),
  RQ2 (H2a, H2b), RQ3 (`full` against each ablation).
- **Sensitivity metric:** window FPR excluding the 15 minutes after each attack (TaPR
  ambiguous section), always reported next to plain FPR and never used for a claim.
- **Scoring:** Site A scores use the robust adaptive threshold R3 (ADR 0005) in
  every model and mode; Site B uses R0.

## 5. Test-set access

Test partitions are read only by `--final` runs after the `g4-frozen` tag, once per
frozen configuration and seed, and every read is logged. The one exception is the
Site A ground-truth labelling pass in `docs/mapping_protocol.md` §2, which is logged
under the configuration `ground_truth_site_a` before any attribution result exists.
