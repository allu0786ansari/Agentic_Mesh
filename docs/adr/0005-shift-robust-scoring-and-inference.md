# ADR 0005 — Operating-point shift in the local baselines, and event-level inference

**Status:** Proposed (14 October 2026). This protocol is committed before any
candidate is implemented or run. Once the author accepts it, it is implemented, run
once on validation, and the result and decision are added below the line.
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

*(Filled in after the run.)*
