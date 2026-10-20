# ADR 0019 — Week 14: resources, reproduction, Gate G4 and the single test run

**Status:** Accepted, 20 October 2026. Written before `g4-frozen` and before any
test-split result.

**Implements:** Plan Week 14 tasks 1–4; preregistration §3, §3a, §3b and §5;
CC2, CC6, CC7 and CC8.

**Builds on:**

- ADRs 0013–0018: the attribution pipeline frozen at G3, the run matrix, the X3–X6
  protocol, and the Week 13 decisions.

**Scope:**

- **What it fixes:** how every test-split number is produced, in what order, and
  how each access to test data is checked.
- **Analyses:** they are the ADR 0016 §4 and ADR 0017 analyses, run on `split =
  test`, unchanged except where this ADR says so.

## 0. Disclosure: a read of test data before this ADR

While drafting this protocol in the cloud on 20 October 2026, a structure check
read the test partitions directly with pandas. That bypassed `load_partition`'s
gate, by mistake.

| | |
|---|---|
| **Columns read** | `window_start`, `label` and `episode` (Site B, b1–b3); `time`, `attack` and `run_id` (Site A, a1–a4). No feature column. |
| **What was seen** | row counts; attack-label counts (Site B: 39 windows on b1, none on b2 or b3); episode IDs 4–6; the test date (1 Feb 2023); Site A attack-row and segment counts |
| **Not involved** | any model, score, threshold or metric |
| **Logged** | after the fact, in `results/test_access.log` under `protocol_inspection_week14` (seven entries, `out_of_band: true`) |

* **Why it cannot have influenced anything:** nothing in this ADR or in any
  configuration was chosen from those counts.
  * The test events and episodes were already fixed by the frozen mapping and the
    partition manifest.
  * The episode times and the attack-day date are recorded in
    `data/partitions/site_b/manifest.json`.
* **From here on:** structure comes from the manifest and the frozen mapping only.

## 1. X7 — resources (CC8)

`evaluation/resources.py` samples memory every 0.5 s.

* **Per process:** RSS, and PSS from `/proc/<pid>/smaps_rollup`.
* **System:** used memory (total minus available) and swap, from `/proc/meminfo`
  inside WSL2.

**The two stages** (WR-4: they never run together):

| Stage | Command | What is sampled |
|---|---|---|
| **training** | Site A FedProx ε 3, seed 11, in-process backend, as the grid runs it (`resources run --stage training -- …`) | the run's process tree |
| **operations** | in order: the Site A validation edge replay; `make stack-up`, which starts the seven node APIs over mTLS and the approval API; one Site A `full` mesh run over mTLS, with Ollama (Llama 3.2 3B Q4, `num_ctx` 2048) | one watcher (`resources watch --stage operations`) samples every process of those roles throughout. Phoenix is not run, since traces are files (ADR 0014). |

* **Output:** `results/X7/resources.json`, with per stage:
  * the peak PSS per role;
  * the peak PSS of the whole stage;
  * the peak system used memory and the peak swap.
* **Pass:**
  * no stage reports an out-of-memory error;
  * the peak system used memory is ≤ 6.0 GB, the §3.3 budget inside WSL2.

  Swap is reported, not budgeted.
* **Plan §3.3:** the "Measured (W14)" column is filled from this artifact.

## 2. Clean-checkout reproduction (CC7)

### Deviation from the Plan

There is no DVC remote, so `dvc pull` cannot run. In its place, the raw data is
copied from the working checkout and checked against `data/raw_checksums.json`
(`make verify-checksums`). The partitions are then **rebuilt** from it, not copied.

### Procedure

1. Fresh `git clone` into a new directory.
2. `uv sync --locked` with every dependency group (dev, ml, fl, kb, figures).
3. Copy `data/raw/`.
4. `make reproduce`.

### What `make reproduce` checks

`make reproduce` writes `results/reproduce/reproduction.json`.

1. **Checksums:** the raw-data checksums match.
2. **Partitions:** `make data` rebuilds them. Every partition file's SHA-256 equals
   the committed manifest's.
3. **Headline models:** FedProx ε 3, seed 11, is retrained on both sites. It meets
   the Plan's tolerance against the committed X1 artifact: identical privacy
   accounting (ε exactly), and AUROC within ±0.005. The exact differences are
   reported.
4. **One non-private baseline per site:** `central__vae` (A) and `central__ae` (B),
   seed 11, under the same tolerance.

**Pass:** all four hold. **Failure:** reported as is, and G4 waits.

## 3. Gate G4

### What is frozen

The tag `g4-frozen` is made on the laptop after §1 and §2 pass, on a clean tree,
and pushed. It freezes:

* the code;
* `configs/` (parameters, the run matrix, `configs/mesh/llm_timeouts.lock.json`,
  retrieval v2, the stack);
* `uv.lock`;
* the knowledge base build;
* the ground truth (`gt-frozen` is unchanged);
* the models: the headline models in `results/models/x3/` (ADR 0018 §1) and the
  deployed edge models in `results/models/site_*__fedprox__eps3/seed_11`.

### Which models are used where

* **X1 and X2 retrain** every configuration (§4). No model of those configurations
  was saved, and training is deterministic (ADR 0017 §9.7).
* **X3** uses the five headline Site B models.
* **X4, PT6-v2 and PT7 use the deployed edge models.** That is the operational
  pipeline frozen at G3: it fed every X4 and X5 validation run and the G3 decision.

  This clarifies ADR 0018 §1. Its "headline models" are the detection-experiment
  models, X3 and the robustness suite. The deployed edge models are not replaced:
  a different Site A model would produce different incidents from the ones whose
  attribution was developed.

### Preconditions for every `--final` access

`sealed.authorise` enforces three things.

* **The tag exists.**
* **The code is the tagged code:** `HEAD` is the tagged commit, or a descendant
  whose differences from it are only under `results/`.
* **The tree is clean outside `results/`.** Appending to `results/test_access.log`
  does not count as dirty.

**A change to code after the tag** needs a new ADR and a new tag. Runs already
made stay as they are and are reported.

## 4. The test runs

All of them run **on the laptop**, so that `results/test_access.log` has one
writer. Every run reads its test partitions through `load_partition(final=…)`,
and every client logs **one entry per run**: a client that evaluates more than
once in a run reads its partition once.

| # | Run | What reads test | Config IDs in the log |
|---|---|---|---|
| 1 | **X1 baselines:** `x1_baselines --final`, every site, model and seed (14 configurations × 5 seeds) | each node's own test partition | `X1/<config>` |
| 2 | **X3:** `x3_fusion --final`, five Site B seeds | b1–b3 test | `X3/site_b__fusion` |
| 3 | **Edge:** `edge.runtime --final` with the deployed models | each node's test stream | `edge__site_a`, `edge__site_b` |
| 4 | **Mesh**, on each test edge run, in the four modes: `rule_only`, `retrieval_only`, `full` over mTLS, and `llm_no_retrieval` | nothing: it reads the edge run's alerts and evidence; the tag is still required, and each run logs once | `mesh__<site>__<mode>` |
| 5 | **X4:** `x4_export`, then `x4_score --split test --final` | the test labels, for the event spans, as in validation | `X4/attribution` |
| 6 | **PT6-v2:** `retrieval_eval --split test --final`, rendered queries from the test edge runs; v2 and v1 | nothing new: the evidence comes from #3, and the spans from the mapping and the manifest | `retrieval__test` |
| 7 | **The X1/X2 matrix:** `grid --final`, all 180 runs in the frozen matrix | each client's test partition, once per run | `X1/<config>`, `X2/<config>` |

* **Order:** 1 → 2 → 3 → 4 → 5 → 6 → 7.
  * The short runs that carry PT1–PT6 go first.
  * Within 7, the configurations the preregistered tests use go first: the
    headline, ε 10 and H1c, the central and local DP baselines, and FedAvg ε 3.
* **Training and operations never run together:** #4 needs Ollama; #1, #2 and #7
  need it stopped.

### What each `--final` training run does

It retrains exactly as validation did: same code, seed and data. It then evaluates
twice in the same process:

* **validation, as before:** its metrics are recorded and compared with the
  committed validation artifact under ADR 0018 §1. A mismatch is flagged, kept and
  reported; the run is not repeated.
* **test**, at the validation thresholds:
  * the per-node 1% FPR threshold on normal validation;
  * R3 or R0 as configured;
  * for H1c, the pooled DP threshold. It is released from the validation
    histograms in that run, so no new privacy cost arises for test.

Test artifacts are written to `results/test/<experiment>/<config>/seed_<n>.json`,
with `split = test` and the score file beside them.

### Retries

* **A run that completed** (its test artifact exists) is never run again.
* **A run that crashed before writing its artifact** may run again.
  * Each attempt is logged by the grid with its exit code.
  * A repeated access-log entry for the same configuration, seed and client is
    accepted by the checker only when the earlier attempt left no artifact.

## 5. The test analyses (computed in the cloud from committed artifacts)

* **X1 and X2:** the ADR 0016 §4 tests on `results/test`.
  * Families and Holm as declared: RQ1 is PT1 and PT2 per site; H1b; H1c with two
    comparators.
  * The event-level two-stage block bootstrap, B = 10 000; 30 min normal blocks on
    Site A and 60 min on Site B; seeds averaged inside each draw.
  * IID is unpaired, as in ADR 0016.
  * The figures are regenerated from the test artifacts and labelled *test*.
* **PT3:** the headline model on Site A, five seeds.
  * Event recall on the 35 test events, and the realised test window FPR, each as
    the seed mean with its event-level bootstrap CI.
  * PT3 is met when the seed means meet both bounds: recall ≥ 0.70 and FPR ≤ 0.02.
    The CIs are reported beside them.
* **X3 (PT4, H2b).**
  * **Benign test time:** the span all three nodes cover, minus each test episode's
    span (union over nodes) widened by the 60 s tolerance.
  * **False alarms:** an alarm that meets no widened episode span.
  * **Blocks:** 60 min over benign test time.
  * The detectors are those of ADR 0017 §2, at the validation thresholds and the
    validation τ (∞ on every seed; ADR 0018 §2).
  * **PT4:** the 2-of-3 reduction against the best single node, chosen per seed on
    **validation** as in ADR 0017. The one-sided test is H1: reduction > 0.30.
  * **H2b:** recall per test episode (4–6) and time to detect; b1 is the
    reference.
  * **H2c:** validation only (synthetic, ADR 0017 §3).
* **X4 (PT5).** The ADR 0017 §4 and ADR 0018 §3 rules, on the 35 test events, the
  test incidents and the four modes.
  * **PT5** is evaluated on `full`, the default mode fixed at G3 (ADR 0015).
    * Top-3 must be ≥ 0.70 and above the preregistered majority-class baseline
      0.64. The baseline on the 35 events is reported beside it.
    * `full` ≥ `retrieval_only` is tested by a paired event bootstrap of top-3. The
      clause is met when the difference is ≥ 0, and its CI is reported.
    * The other modes are the ablations, reported beside it.
  * **Site B:** the case study on test episodes 4–6.
  * **JSON validity ≥ 95%:** over the LLM calls of every incident that matches any
    of the seven episodes (0–3 from the validation runs, 4–6 from the test runs),
    per LLM mode. Raw grounding is reported the same way.
* **PT6-v2:** v2 Recall@10 ≥ 0.85 on the test rendered queries; Recall@5, MRR and
  v1 are reported on the same queries (preregistration §3b). A missed incident has
  no query and is counted.
* **PT7 on test:** p50 and p95 per segment from the test edge and mesh runs,
  reported beside the validation X5. The targets are those of PT7.

Every number is reported, met or missed. A missed target is a finding.

## 6. Checks (`checker --tier completion`; CC1–CC8)

| Check | What it requires |
|---|---|
| **CC2** | every test artifact has an access-log entry per client, at a commit that is the tag or a results-only descendant; no configuration, seed and client has two entries unless the earlier attempt left no artifact; every final run's validation comparison is recorded |
| **CC3** | `data_source = runtime` and full provenance on every artifact used |
| **CC4** | the privacy rule on every test artifact |
| **CC6** | the frozen matrix and baselines complete on test, five seeds, CIs reported |
| **CC7** | `reproduction.json` passes |
| **CC8** | `resources.json` passes |

CC1 and CC5 are the existing contract tests.

## 7. Compute and the cut order

Laptop estimates, from the measured validation times:

| Run | Time |
|---|---|
| the X1/X2 matrix | 22.5 h |
| baselines | ≈ 0.5 h |
| X3 | ≈ 10 min |
| edge | ≈ 10 min |
| the mesh in four modes | ≈ 4–7 h, depending on the number of test incidents (the LLM modes dominate) |
| X7 and reproduction | ≈ 1 h |

* **If time runs short**, the Proposal §12 cut order applies, decided before the
  run:
  1. X8 is not re-run on test; it is a systems experiment.
  2. Then the systems-heterogeneity arm of X2.
  3. Then the `llm_no_retrieval` ablation.
* **Nothing else is cut.** A configuration that is cut is reported as not run, never
  as a result.
