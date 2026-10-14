# ADR 0005 — Operating-point shift in the local baselines, and event-level inference

**Status:** Accepted by the author (14 October 2026). The protocol was committed
(fc19946) before any candidate was implemented or run; the result and the decision
are added below the line after the single validation run.
**Depends on:** ADR 0002 (G1 inputs), ADR 0003 (scoring), ADR 0004 (training budget).

## Why

**1. The local false-alarm rate comes from shift, not from training.** Five seeds,
validation (ADR 0004):

| Site A | AUROC | Window FPR (target 0.01) |
|---|---|---|
| Local VAE | 0.846 [0.842, 0.850] | 0.279 [0.232, 0.322] |
| Local PCA | 0.887 | 0.053 |
| Central VAE | 0.915 [0.910, 0.920] | 0.044 |

Training longer makes the local models worse: with patience 50 the FPR is 0.351
(ADR 0004 sensitivity). Two diagnostics, seed 11, validation labels only:

- **When the false alarms happen.** The local VAE's false-alarm rate is about the
  same before the first attack of a segment as after one (a1 0.37, a3 0.35, a4 0.46
  before the first attack). So the alarms are not recovery after attacks; the
  recording is different. PCA's false alarms, by contrast, cluster in the 15 minutes
  after an attack (a3: 0.34 in 0–5 min, 0.01 before the first attack). That is
  post-attack recovery.
- **What drives them.** Variables that are almost constant inside one client's
  training window but sit elsewhere in the July test recordings. Examples are the
  setpoints `P1_B4002`, `P1_B2004` and `P1_B4022`, turbine speed `P2_SIT01/02`, and
  vibration `P2_VXT03`/`P2_VYT03`. For a3, `P1_B4002` has reconstruction error 0.01
  in training and 3.69 in its false alarms. The non-linear VAE extrapolates badly
  outside its training support; linear PCA degrades more gracefully.

So the local baseline is unfair in a specific way. A single feature at a new
operating point can dominate the window score, and the threshold has no way to
adapt. If federation is compared against this baseline as it stands, a reviewer can
rightly say the gain is only against a broken detector. The paper needs the best
label-free local remedy as its baseline, chosen by a rule fixed in advance.

**2. Seed-level tests cannot carry the paper's claims.** With five seeds the exact
Wilcoxon test cannot go below p = 0.0625. PCA is deterministic, so its seed spread
is zero, and a seed bootstrap only measures VAE initialisation noise. It says
nothing about the uncertainty that matters: which attacks and which normal periods
happened to be recorded. Claims need inference over **events and time blocks**.

**Honesty note.** The candidates below were motivated by validation diagnostics.
The choice among them is made once, by the rule below, on validation. Confirmation
can only come from the sealed test set at G4.

---

## Part 1 — Shift-robust scoring (Site A)

### Candidates (4, each with fixed parameters; no grid)

All candidates act on the per-feature reconstruction errors or on the window score,
after the model. None retrains the model, changes inputs or partitions, or uses
labels. Each applies identically to every detector (VAE, PCA, Isolation Forest where
defined), in every mode (local, central, and later federated and DP).

| Id | Name | What it does | Parameters (fixed now) |
|---|---|---|---|
| R0 | current | ADR 0003 scoring: mean of per-feature errors; per-node threshold at 1% FPR on `val_normal` | — |
| R1 | bounded evidence | Each feature's window error is replaced by its quantile in the node's own **training** error distribution (empirical CDF, tied ranks averaged), and the score is the mean over features. A feature at a new operating point can add at most 1/55 of the score. This is the probability integral transform | none |
| R2 | causal error detrending | Each feature's window error minus the median of that feature's errors over the past history window, floored at 0. A persistent offset (a new operating point) is removed within minutes; a sudden change (an attack) is kept | history H = 30 min, guard gap g = 2 min (the most recent 2 min are excluded from the history), minimum history 5 min |
| R3 | robust adaptive threshold | The score is unchanged; the threshold at time t is median + k·MAD of the node's scores over the past history. k is calibrated per node so the rule gives 1% FPR on `val_normal`. This is the classic dynamic-threshold baseline (Hundman et al., KDD 2018), made robust with median and MAD | H = 30 min, g = 2 min, minimum history 5 min |

**Why these parameters.** HAI 21.03 attacks last 17–422 s (ADR 0003); the longest
validation attack is 7.0 min. With H = 30 min, an attack fills under a quarter of the
history, so the median does not move. The 2-minute gap keeps an attack's own onset
out of its baseline. The 5-minute minimum keeps the median meaningful.

**Warm-up (R2, R3).** Validation-attack segments start 0.2–3.9 h before their first
attack, and there is no data before a segment. For the first 5 minutes of every
segment (validation-normal, validation-attack, and test alike), R2 and R3 **abstain**:

- An abstained window raises no alarm and gets the lowest score.
- An abstained attack window counts as missed.
- FPR is computed over **scored** windows only, so abstention cannot lower it for
  free.
- The share of abstained windows is reported. It is about 5% of Site A
  validation-attack windows (17% in the shortest, 0.5-hour segment) and under 1% of
  validation-normal windows.

In continuous deployment there is no warm-up after the first 5 minutes; the
evaluation is pessimistic here.

### Selection rule (validation labels, as in ADR 0003)

Inputs: Site A, five seeds (11, 23, 37, 41, 53), the neural model (VAE), local and
central, ADR 0004 budget. For each candidate, take the mean over seeds and over the
two modes of the macro window FPR, event recall and AUPRC.

1. **Guards.** A candidate is eligible only if its event recall ≥ R0's − 0.05 and its
   AUPRC ≥ R0's − 0.02. With about four validation events per client, losing one
   event costs about 0.06 in macro recall, so the guard means no event may be lost
   on average.
2. **Criterion.** Among eligible candidates, the lowest window FPR.
3. **Ties.** Within 0.01 FPR, the simpler candidate wins, in the order R0, R1, R3,
   R2.

The rule picks one candidate, and Site A uses it from then on for every model and
mode. All four candidates stay in the artifact as a sensitivity table, including
for PCA and Isolation Forest, which are reported but not used in the choice.

### Site B

Site B is outside the selection. Each node's validation-attack partition is one
0.4-hour segment that starts with the attack, so R2 and R3 can never warm up. AUPRC
is already about 1.000 for every model, so validation cannot rank candidates
(ADR 0003 robustness check). Site B keeps R0. R1, which needs no warm-up, is
reported as a sensitivity only.

### Consequences for later experiments

- If R3 is chosen, it is a local adaptive threshold and therefore a direct
  competitor of H1c, the DP federated threshold. H1c is then tested against R3 as
  well as against fixed per-node thresholds.
- If R1 or R2 is chosen, H1c histograms are built from the transformed scores.
- The federated models use the same scoring. Federation then has to beat the best
  local remedy, not the broken one.

---

## Part 2 — Event-level inference (both sites; replaces seed-level tests as primary)

**Resampling units.**

- **Attack events.** Each event takes its attack windows with it. On Site B the
  unit is the episode on its node.
- **Normal blocks.** Normal windows are grouped into non-overlapping 30-minute
  blocks per node (60 minutes on Site B, where the normal periods run about 30 h).
  Blocks keep the within-block autocorrelation, so the intervals are not too
  narrow.

**Procedure.** Two-stage stratified block bootstrap, B = 10 000:

1. Resample events, and normal blocks within each node, with replacement.
2. On each resample, recompute each metric from the stored per-window scores and
   alarms, for both configurations in a comparison, using the same resample
   indices (paired).
3. For each configuration and resample, average the metric over the five seeds.
   The seeds are then averaged out, not used as replicates.
4. Report the mean difference, its 95% percentile CI, and the one-sided p-value
   (share of resamples with difference ≤ 0, with +1 correction).
5. Apply Holm correction within each research question, as in preregistration §4.

**Metrics.**

| Metric | Resampled over |
|---|---|
| Event recall, time to detect | events |
| Window FPR, false alarms per hour | normal blocks |
| AUROC, AUPRC, range F1 | both |

Thresholds stay as calibrated; the bootstrap does not re-fit them.

**What stays secondary.** Mean ± 95% CI over seeds, which shows training variance,
and the exact Wilcoxon test over seeds are still reported in the summary, but no
claim rests on them.

**Storage.** Each X1 artifact also saves its per-window scores, alarms, labels,
event ids and window times as `seed_<n>.scores.npz` beside the JSON (float32,
compressed; about 3–4 MB for all of X1). The checker verifies that the file exists
and that its hash matches the one recorded in the artifact.

**Declared sensitivity metric.** FPR excluding the 15 minutes after each attack
ends. This is the "ambiguous section" idea of TaPR (Hwang et al.), the metric
designed by the HAI authors: physical recovery after an attack is not a clean
normal period. It is always reported next to plain FPR and is never used for
selection or for a primary claim.

---

## What this ADR does not do

- No architecture or capacity search, no retraining per candidate, no new input
  features, and no change to partitions or windows.
- No test-set access.
- No change to the training budget, which ADR 0004 already fixed.
- No change to the H1c hypothesis text, only to what it is compared against
  (above).

## Implementation and run order (after acceptance)

1. **Code.** `models/shift.py` holds R1–R3: training-ECDF transform, causal
   rolling-median detrending, and the median/MAD adaptive threshold, with abstention
   on warm-up. `evaluation/inference.py` holds the two-stage block bootstrap and
   Holm correction. Unit tests include causality (no future window changes a
   score), warm-up abstention, and the bootstrap's coverage on synthetic data with
   a known difference.
2. **X1 artifacts** gain the score files and record which shift candidate was used.
3. **Run once:** `make shift-selection` writes `results/X1/shift_selection.json`
   (Site A, five seeds).
4. Write the decision below the line, set `models.scoring.site_a.shift` in
   `configs/params.yaml`, then re-run `make baselines` and `make summarise`, which
   now produces event-level CIs.
5. Update preregistration §4: event-level bootstrap is primary; seed-level tests
   are secondary.

---

## Results and decision

**Run.** One run, 14 October 2026, from commit 7f9ea93 (clean tree); Site A, seeds
11, 23, 37, 41, 53, ADR 0004 budget, validation only. Output:
`results/X1/shift_selection.json`.

**Implementation details fixed while coding, before the run.**

- Warm-up is the 2-minute gap plus the 5-minute minimum history. R2 and R3 therefore
  abstain for the first **7 minutes** of each segment, not 5.
- The MAD in R3 is floored at 0.001 times the node's median training score, so a
  perfectly flat history cannot divide by zero.
- Abstention measured: **7.2%** of validation-attack windows.
- Time to detect is resampled as the mean, and range F1 is not resampled (see
  `evaluation/inference.py`).

**Neural model (VAE), mean over seeds and the two modes; this is the rule's input:**

| Candidate | AUPRC | Event recall | Window FPR | FPR excl. recovery | Eligible |
|---|---|---|---|---|---|
| R0 current | 0.707 | 0.994 | 0.162 | 0.157 | yes (reference) |
| R1 bounded evidence | 0.472 | 0.950 | 0.074 | 0.070 | **no**: AUPRC guard |
| R2 causal detrending | 0.759 | 0.975 | 0.021 | 0.009 | yes |
| R3 adaptive threshold | 0.767 | 0.969 | **0.019** | 0.010 | yes |

**Decision: R3, the robust adaptive threshold** (`models.scoring.site_a.shift: r3`).
It has the lowest FPR among the eligible candidates. R2 is within the 0.01 tie
margin, and R3 comes before R2 in the simplicity order, so the tie-break agrees.
Site B stays at R0.

**By mode** (mean over five seeds):

| Model, mode | AUROC R0 → R3 | AUPRC R0 → R3 | Event recall R0 → R3 | Window FPR R0 → R3 | False alarms/h R0 → R3 |
|---|---|---|---|---|---|
| VAE local | 0.846 → **0.922** | 0.667 → 0.741 | 1.000 → 0.938 | 0.279 → **0.021** | 11.2 → 1.3 |
| VAE central | 0.915 → **0.948** | 0.748 → 0.792 | 0.988 → 1.000 | 0.044 → 0.016 | 3.1 → 1.0 |
| PCA local | 0.887 → 0.935 | 0.694 → 0.760 | 0.938 → 0.938 | 0.053 → 0.013 | 6.4 → 1.2 |
| PCA central | 0.903 → 0.924 | 0.754 → 0.758 | 0.938 → 0.938 | 0.008 → 0.013 | 0.9 → 1.4 |
| Isolation Forest local | 0.723 → 0.809 | 0.162 → 0.237 | 0.337 → 0.558 | 0.005 → 0.017 | 0.7 → 1.1 |

**What it means.**

1. **The broken local baseline is fixed.** The local VAE's false-alarm rate falls
   from 0.279 to 0.021, near the 0.01 target. Its AUROC rises by 0.076. The cost is
   one missed validation event in local mode on average (recall 1.000 → 0.938). The
   paper must report that.
2. **The federation headroom shrinks a lot.** Central minus local VAE goes from
   +0.069 to about +0.026 AUROC, and from 0.235 to about 0.005 in FPR. Most of the
   earlier "federation gain" was operating-point shift, which a label-free local
   adaptive threshold removes. RQ1 and H1c must now be argued against R3, not R0.
   This is the honest framing, and it makes the federated paper's claim harder but
   more credible.
3. **The VAE-versus-PCA question remains open.** Under R3, PCA beats the VAE locally
   (0.935 against 0.922 AUROC), and the VAE beats PCA centrally (0.948 against
   0.924). The event-level tests will show whether either gap is real.
4. **R1 fails its guard.** Bounding each feature's evidence also bounds a real
   single-variable attack, so AUPRC drops by 0.24.
5. **Recovery accounts for about half of R2's and R3's remaining false alarms.**
   Excluding the 15 minutes after each attack, the FPR is 0.009–0.010. This is a
   sensitivity check only.

The confirmatory test of all of this is the sealed test set at G4.
