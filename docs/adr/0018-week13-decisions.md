# ADR 0018 — Week 13 decisions: model acceptance, noisy-OR, presenting X4

**Status:** Accepted, 20 October 2026, on the author's instruction to proceed with
the recommendation.

**When it was written:** after the Week 13 validation results (ADR 0017 §9) and
before any test-split result. No test data has been read.

**Amends:** ADR 0017 §1, the acceptance rule for retrained models. Nothing else
in ADR 0017 changes.

**Builds on:**

- the Plan, Week 14 task 2 (the reproduction tolerance);
- the preregistration (PT4, PT5);
- the Proposal §10.3 (calibration policy).

## 1. Acceptance of the retrained headline models

### The finding

ADR 0017 §9.1 and §9.7 report the following.

- **The mismatch:** three Site A seeds (23, 37 and 41) do not reproduce their X1
  artifacts to 1e-9. The largest AUROC difference is 2.5e-5. Seed 41's false
  alarms per hour differ by 0.054.
- **The reruns are reproducible.** A cloud retrain of seed 23 equals the laptop
  rerun exactly, on different hardware. The three X1 originals are the runs that
  cannot be reproduced. Their cause is not identified.
- **Privacy is unchanged.** The privacy accounting of all three is identical to
  X1, ε for ε.

### The decision

- **Acceptance rule.** A retrained model is accepted when it meets the Plan's own
  reproduction tolerance: "ε exactly; AUROC ± 0.005 for the same seed". ε exactly
  means identical privacy accounting.
  - The 1e-9 comparison of ADR 0017 §1 stays. It is run and reported, but it no
    longer excludes a seed.
  - `x3_models.accept` implements the rule; `x3_models verify` prints both checks.
  - Under it, all ten models are accepted.
- **Which models count.** From Week 13 on, including the Week 14 test runs, the
  headline models are the retrained models in `results/models/x3/`. They are the
  reproducible ones.
- **The Week 12 analysis is not rewritten.** The X1 originals stay in
  `results/X1/`, and `analysis_val.json` is not changed.
- **Sensitivity check.** `x3_models sensitivity` reruns the X1 and X2 analyses with
  the three reruns substituted. It writes `results/X1/rerun_sensitivity_val.json`.
  - 25 of 34 tests are identical.
  - In the other 9, the largest changes are 6e-7 in an effect estimate, 1.5e-5 in a
    CI bound and 2e-4 in a p-value.
  - No conclusion in ADR 0016 §8 changes.
- **So it can be diagnosed next time.** The provenance's `environment` now records
  the installed versions of the packages that can change numerical results:
  numpy, scipy, torch, opacus, scikit-learn, pandas and flwr. The lock file's hash
  says what should be installed; these say what was.

### Why this is not tuning on results

- **Direction.** The rule was chosen without regard to which way the differences
  go. They are of both signs, AUROC −1.4e-5 to +2.5e-5, and far below any effect
  that is reported.
- **Source of the tolerance.** It is the project's own reproduction standard,
  written in the Plan before Week 12. It is not a new number.
- **Alternatives rejected:**
  - excluding three of five Site A seeds would remove 60% of the seeds from the
    Site A robustness suite and from the Site A test runs;
  - re-running X1 would replace committed Week 12 artifacts after their analysis
    was read.

### Consequence

- **Robustness suite:** rerun with all five Site A seeds.
- **X3:** Site B only; it was already exact on every seed, so its results are
  unchanged.

## 2. Noisy-OR at the matched τ

### The finding

ADR 0017 §9.2 reports the following.

- **No benign probability above zero.** On all five seeds, every benign Site B
  alert has isotonic-calibrated probability 0.
- **Why:** the calibrator, fitted on validation as Proposal §10.3 requires, places
  every attack window above every benign alert score.
- **What the rule then gives:** the matched-τ rule of ADR 0017 §2 finds no
  qualifying value, so τ = ∞. Noisy-OR never alarms.

### The decision

- **The rule is kept.** Noisy-OR at the matched τ is reported as degenerate: zero
  alarms, on validation and on test. τ is fixed per seed on validation, so it is ∞
  on test too.
- **No other τ is set**, on validation or on test.

### Why

- **Any other τ would be chosen after seeing this result.**
- **Probability-based recall on validation would be in-sample.** The calibrator
  is fitted on the validation attack windows, so any validation recall for a
  detector that uses calibrated probabilities is in-sample.
  - At τ just above 0, noisy-OR would look perfect on the validation attack day
    for that reason alone.
- **PT4 is unaffected.** It is defined on 2-of-3, which uses thresholds, not
  probabilities.
- **In the paper:** the finding is reported as a limitation of probability-level
  fusion in this setting. Isotonic calibration at a 1% FPR threshold maps benign
  alerts to 0. The same mechanism makes every benign mesh incident "low" severity.

## 3. Presenting X4

### The finding

ADR 0017 §9.3 reports the following, on the 15 validation events.

- **No mode exceeds the majority-class baseline** (0.733).
- **`llm_no_retrieval` gives one constant answer.** It answers T0836 on every
  event, so its score is that baseline exactly.
- **PT5's top-3 bar is missed.** `full` top-3 is 0.533, below PT5's 0.70.

### The decision

- **PT5 is evaluated on test exactly as preregistered.**
  - Top-3 ≥ 0.70 against the acceptable sets.
  - Above the majority-class baseline, 0.64 under protocol v1.1. The baseline on
    the 35 test events is reported beside it.
  - `full` ≥ `retrieval_only` with a CI.
  - Site B: JSON validity ≥ 95%.
- **The attribution pipeline is not changed before the test run:** the knowledge
  base, retrieval v2, k, the prompt, the model and the LLM settings stay as they
  were at G3 (ADR 0015).
  - Changing it on 15 validation events, after reading their accuracy, would fit
    the pipeline to them.
  - The validation numbers are development evidence and are reported as such.
- **A fixed diagnostic is added, descriptive only.** Every X4 mode reports its
  answer diversity: answers given, distinct techniques, the modal technique and
  its share. A mode with one answer for every event is flagged `constant`.
  - The text then describes that mode's accuracy as a constant-answer baseline,
    not as attribution.
  - The diagnostic changes no metric and no target.
- **Tactic accuracy** stays reported and unused for any claim. A constant answer
  scores 1.0 on Site A.

## 4. What runs next

Week 14 goes ahead as planned:

1. X7 resources;
2. clean-checkout reproduction, at the same tolerance as §1;
3. `g4-frozen`;
4. the single `--final` test run per configuration and seed.

The X3, X4 and robustness code used for test is the code at `g4-frozen`, with the
rules above.
