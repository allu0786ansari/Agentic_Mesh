# ADR 0020 — Gate G4: the test results, and what happened on the way

**Status:** Accepted, 22 October 2026.

**Records:** the single test run made after `g4-frozen` (`96a6782`), under ADR 0019.

**Where the numbers live.** Every number below comes from a committed artifact. The
summary is `results/test/summary_test.{json,md}`, built by
`experiments/final_summary.py` from the test artifacts alone.

## 1. Integrity of the run

* **No code changed after the tag.** Every commit after the tag, up to the last test
  run (`61317fe`), touches only `results/`.
* **Access log** (`results/test_access.log`):
  * one entry per configuration, seed and client for every training run: 250
    artifacts, X1 705 entries, X2 190;
  * X3 per seed and client (15);
  * the two edge replays (7);
  * the eight mesh runs;
  * X4 and PT6-v2.

  The tagged commit is in every entry.
* **Completion tier** (`results/test/completion_check.json`): every criterion
  passes except CC7, which is waived (ADR 0019 §2a).

**Deviations, all disclosed:**

1. **Out-of-band read.** The test read made while drafting ADR 0019, disclosed in
   its §0.
2. **Two mesh attempts failed and were rerun.** Site A `full` and
   `llm_no_retrieval` each failed on a first attempt, which logged its access and
   wrote no run. Each was run again.
3. **X4 was scored twice.**
   * **First:** at 15:22 UTC on 20 October, X4 was scored when only Site A
     `rule_only` and `retrieval_only` had test runs.
   * **Why it could not be redone at once:** the scorer refuses to overwrite a test
     artifact.
   * **Second:** with the author's approval, the partial artifact was kept as
     `attribution_test__partial_20261020T1522Z.json`, and X4 was scored once more
     on all eight finished runs. That read the Site A test labels a second time.
   * **No choice arose:** the runs were fixed and logged, the scorer is
     deterministic, and the two modes scored the first time are identical in both
     artifacts. That was checked.

## 2. Two corrections to earlier records

1. **The validation comparison had a bug.** In each final run's artifact,
   `validation.comparison` marked all 40 non-private (ε = ∞) runs as "not within
   tolerance".
   * **The cause:** the live ε (`float('inf')`) was compared with the stored `"inf"`.
   * **The fix:** `compare_validation` now compares the stored form, and the
     completion tier and the summary recompute it from the artifacts.
   * **Corrected result:** **249 of 250** final runs reproduce their validation
     artifact within the Plan's tolerance. 192 of them are exact.
   * **The one miss:** Site A `central__vae__epsinf` seed 41, |ΔAUROC| 0.0056.
2. **Training is not always bit-exact.** ADR 0017 §9.7 and ADR 0019 §2a called CPU
   training deterministic. The final runs show it is not always so.
   * **Where it varies:** 58 of the 250 retrained runs are not exact. All 58 are
     Site A runs through the federated pipeline (22 of its configurations). The
     median |ΔAUROC| is about 1e-5, and the largest is 0.0056.
   * **What is exact:** every Site B run and every non-federated baseline.
   * **Why the claims are unaffected:** each test artifact also holds the same
     model's validation metrics. The model evaluated on test is therefore the one
     whose validation behaviour is recorded beside it.

## 3. Results (test split; the claims)

| Target | Test result | Verdict |
|---|---|---|
| **PT1** Site A: FedProx ε 3 against centralised non-private, margin 0.05 | −0.064 [−0.100, −0.037] AUROC (0.865 vs 0.929) | **not met** |
| **PT2** Site A: against mean local-only non-private | −0.061 [−0.101, −0.030] (vs 0.926) | **not met** |
| **PT1, PT2** Site B | AUROC 1.000 for every model; Holm p 0.0004 | met, at the ceiling |
| **PT3** Site A: event recall ≥ 0.70 and window FPR ≤ 2% | recall 0.676 [0.523, 0.819], node mean; pooled over the 35 events 0.674 (23–24 per seed; 25 needed). FPR 0.0083 | **not met** (recall) |
| **PT4** Site B: 2-of-3 reduces FA/h by ≥ 30% against the best single node | 0.614 [0.477, 0.733], one-sided p 0.0001 | met |
| **PT5** Site A `full`: top-3 ≥ 0.70 and above the majority baseline 0.64; `full` ≥ `retrieval_only`; Site B JSON validity ≥ 95% | top-3 0.400 [0.229, 0.571]; majority 0.629 on these events; `full` − `retrieval_only` +0.114 [−0.029, +0.257]; validity 100% | **not met** |
| **PT6-v2** v2 Recall@10 ≥ 0.85, rendered test queries | 0.889 on 27 queries (v1: 0.630; Recall@5 0.519; MRR 0.293). 11 Site A incidents produced no query | met |
| **PT7** latency p95 | detection 2.3 / 0.6 ms; investigation 235 / 146 ms; dispatch `retrieval_only` 182 / 327 ms; `full` 16.0 / 17.0 s (A / B) | met |

### Secondary hypotheses

**H1c** (Site A, ε 10): the pooled DP threshold against per-node thresholds.

| Comparator | Window FPR | Holm p | Recall within 0.05 |
|---|---|---|---|
| per-node R0 | 0.006 against 0.031 | 0.0002 | not shown (−0.033 [−0.089, 0.000]) |
| R3 | 0.006 against 0.008 | 0.19 | not shown |

H1c as preregistered (lower FPR at equal recall) is **not shown**. The FPR reduction
against R0 is.

**H1b** (FedProx against FedAvg, homogeneous and heterogeneous, both sites): no
difference on any comparison (Holm p = 1.0). **Not supported.** Heterogeneity raises
Site A AUROC by about 0.004, as on validation.

**H2b** (Site B test episodes 4–6):

* **b1:** detects all three, with a median time to detect of 10 s.
* **2-of-3:** 0.667, a recall loss of 0.333 [0.000, 1.000] against b1.
* **b2 and b3** observe no attacker traffic, yet they alarm inside some episodes.
  The chance of that, from each node's own false-alarm rate:

  | Episode (length) | Chance for b2 | b2 observed | Chance for b3 | b3 observed |
  |---|---|---|---|---|
  | 4 (20 min) | 0.68 | 5/5 seeds | 0.56 | 4/5 |
  | 5 (4.5 min) | 0.26 | 5/5 | 0.19 | 1/5 |

  * Episode 4 is consistent with chance.
  * b2 on episode 5 is not explained by chance alone. All seeds share the same
    data, so 5/5 is close to one observation, and no mechanism is claimed.
  * 2-of-3's recall rests on these coincidences.

### Descriptive notes

* **PT4's comparator.** PT4's best single node was chosen per seed on validation,
  as declared. Against the node with the fewest test false alarms (b3, 2.27/h),
  the 2-of-3 reduction is about 0.44.
* **PT6-v2's denominator.** If the 11 incidents without a query counted as misses,
  Recall@10 would be 0.632.
* **X4 answers:**
  * Site A: `llm_no_retrieval` answers T0836 for every event (flagged constant).
  * Site B: `retrieval_only` answers T0814 for every incident (flagged constant).
  * `full` gives 4 distinct answers on Site A, with the modal T0836 at 54%.
* **Noisy-OR** never alarms on test, as declared (ADR 0018 §2).
* **Figures:** `results/figures/test/` (privacy–utility, heterogeneity, privacy
  table), labelled *test*.

## 4. What this means for the papers

These are findings and are reported as such:

* **Paper B.**
  * Under period drift, the federated DP model does not reach the centralised or
    local non-private models on Site A.
    * At ε 3 it is 0.064 AUROC short.
    * At ε 10 it is 0.048 short, with a CI of [−0.078, −0.024] that also crosses
      the 0.05 margin.
  * It does beat local-only training through the same DP pipeline: +0.024 [0.005,
    0.046] at ε 3.
  * The pooled DP threshold lowers false positives against R0 but not against R3.
* **Paper C.**
  * On the 35 test events, no attribution mode reaches the 0.70 top-3 target.
  * No mode beats the majority-class baseline.
  * `full` (top-3 0.400) does not exceed the constant-answer `llm_no_retrieval`
    (0.429).
  * The gain of `full` over `retrieval_only`, +0.114, is not significant.
  * Grounding and JSON validity are 100%.
* **The fusion trade-off (RQ2)** holds on test: the false-alarm reduction is
  significant, at a recall cost.
