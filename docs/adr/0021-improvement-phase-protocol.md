# ADR 0021 — Improvement phase: development data, fresh final data, targets and trials

**Status:** Accepted, 22 October 2026, on the author's instruction. It is written
before any improvement trial and before any of the final data is read.

**Context.** The G4 test run (ADR 0020) met PT4, PT6-v2 and PT7, and met PT1 and PT2
on Site B. It missed PT1 and PT2 on Site A, PT3 and PT5. The author chose to improve
the components that missed, then evaluate once more before the paper.

The HAI 21.03 and CIC Modbus test splits have been opened. Approaches tuned against
them would be judged on data already seen. This phase therefore develops on data
we may look at, and keeps fresh data for one final run.

## 1. Data roles

| Site | Development (may be read freely) | Final (sealed until `g5-frozen`) |
|---|---|---|
| A | HAI 21.03, all splits including the former test split; HAI 22.04, train and test (58 test attacks, 100 h) | **HAI 23.05**: 4 training files, 2 test files with 52 attacks (79 h), and its label files |
| B | CIC Modbus 2023 external-attacker scenario (all of it) and **compromised-IED** scenario (attacker IED1B, node b2) | **compromised-SCADA** scenario (attacker: the SCADA HMI, node b3) |

**Recorded in `configs/datasets/phase5.yaml`:**

* the roles;
* the expected SHA-256 of every HAI file: the Git LFS object IDs of
  `icsdataset/hai@2a814ce`, so a download is verified without opening it;
* the sealed paths. `sealed.assert_not_sealed` refuses any read under them without
  a `FinalAccess` after the `g5-frozen` tag, under ADR 0019 §3's commit rules, and
  logs each read.

**The CIC captures.** They have no published checksums, so their checksums are
recorded on download (`python -m agmesh.data.phase5 record`) and committed.

**Disclosure.** Before this ADR, while confirming the datasets:

* one HAI 22.04 summary (`summary(test1.csv).txt`) was read, which lists that file's
  attack times;
* the attack counts of the four 22.04 test files were read;
* the technical-details PDF's version table was read. It gives, per release, the
  number of attacks and the durations, including 23.05: 52 attacks, 79 h.

No HAI 23.05 data, label file or attack description was read, and no
compromised-SCADA file was obtained.

## 2. Targets for the final run (fixed now)

The preregistered targets stay where they measure what they should.

### RQ1 and detection (Site A, HAI 23.05)

* **T1 (PT1):** the federated DP model at ε ≤ 10 is no more than 0.05 AUROC below
  centralised non-private training: non-inferiority, one-sided. Which configuration
  is the headline is declared before G5.
* **T2 (PT2):** that model's AUROC is at or above the mean of local-only
  non-private training.
* **T3 (PT3):** event recall ≥ 0.70 on the 23.05 test events, and realised window
  FPR ≤ 2%.

### RQ2 (Site B, compromised-SCADA)

* **T4 (PT4):** the fused detector reduces false alarms per hour by ≥ 30% against
  the best single node (chosen on development), at the same per-node thresholds.
* **T4b (new):** the fused detector's event recall is no more than 0.10 below the
  best single node's. Fusion must not buy its false-alarm reduction by losing
  single-device attacks.

### RQ3 (attribution)

The metric changes. Top-3 is gameable: a fixed list of the three common HAI
techniques scores close to 1. The primary metric is now top-1.

* **T5a:** top-1 ≥ 0.70 on the HAI 23.05 test events, against acceptable sets under
  mapping protocol v1.1, human-verified (§5).
* **T5b:** top-1 is above the majority-class baseline on the same events, by a
  one-sided paired event bootstrap, p < 0.05.
* **T5c:** top-1 ≥ 0.50 in each stratum (setpoint-led and output-led). A constant
  answerer cannot meet it.
* **T5d:** JSON validity ≥ 95%, and grounding after validation 100%.

Reported, not targets: top-3 beside the best fixed top-3 list from development, and
tactic accuracy. On Site B the compromised scenarios have attack logs, so
attribution accuracy is reported there descriptively.

### Kept as they are

* **T6 (PT6-v2):** Recall@10 ≥ 0.85 on the rendered final queries. Missed incidents
  are reported both ways.
* **T7 (PT7):** the latency targets, with any new model.
* **T8 (CC8):** peak system memory ≤ 6.0 GB with any new model (X7, re-measured).

### Analysis

As in ADR 0016 §4 and ADR 0017:

* the event-level bootstrap, B = 10 000, five seeds;
* Holm within each RQ;
* every target reported, met or missed.

## 3. How trials are run

### Trial areas and candidates

Each area lists its candidates, in the order they are tried.

| Area | Candidates |
|---|---|
| **Attribution** | A1 typed evidence (variable types, first deviation, setpoint–value–output relations); A2 typed-rule prior → retrieval → LLM with abstention; A3 a 7B local model at 4-bit if within T7/T8; A4 evidence-aware re-ranking |
| **Detection and DP** | D1 close the training-pipeline gap (0.033 on 21.03); D2 temporal detectors (TCN or LSTM autoencoder, USAD, graph deviation network); D3 cheaper DP (adaptive clipping, a DP adapter on a non-private pretrained base, batch and steps) |
| **Fusion** | F1 benign-only conformal p-values combined across nodes; F2 2-of-3 or any single node with very strong evidence |

### Where they are evaluated

| Area | Development data |
|---|---|
| **Site A** | HAI 21.03, with models trained on train and the former test split used as the development test; **and** HAI 22.04, trained on its train files and tested on its test files |
| **Attribution** | the 50 HAI 21.03 events and the 58 HAI 22.04 events: 108 events |
| **Site B** | external and compromised-IED |

### The rules for every trial

* **The log:** every trial is logged in `results/dev/trials.jsonl` (`experiments/trials.py`).
  An entry holds the area, hypothesis, configuration, commit, development data,
  metrics and decision.
* **Reporting:** every trial is reported in the paper, including those discarded.
* **Selection:** per area, the trial that best meets that area's targets on **both**
  development sets (Site A: 21.03 and 22.04), judged by the worst-case margin to the
  target. On a tie, the simpler one wins.
* **Budget:** at most 8 trials per area.
* **Stopping:** an area stops early when its targets are met on both development sets
  with a margin of 0.02.
* **What then freezes:** whatever is best when the budget or the time runs out.
* **No peeking:** nothing in the final data is opened to choose anything.

## 4. Gate G5 and the final run

### The freeze

* **The tag:** after the trials, the configurations are frozen and tagged
  `g5-frozen`, under the same rules as G4.
* **What is frozen:** the code, the configs, the lock, the chosen models' settings,
  the headline configuration and the ground truth.

### The final run

The final run is one run per configuration and seed, with every access logged.

* **Site A:**
  * train on the four 23.05 training files, one client per file, with a normal
    validation split under the same scheme;
  * set thresholds on normal validation;
  * evaluate once on the 23.05 test files.
* **Site B:**
  * train on the existing benign captures;
  * set thresholds on normal validation;
  * evaluate once on the compromised-SCADA captures.
* **Attribution:** the frozen pipeline runs on the final incidents in `full` and the
  ablation modes.

### Reporting

The paper reports both evaluations:

* **G4:** HAI 21.03 and CIC external, ADR 0020. It is the first sealed evaluation,
  and it stands.
* **G5:** HAI 23.05 and CIC compromised-SCADA.

## 5. Ground truth

* **HAI 22.04 and 23.05:** mapping protocol v1.1 is applied to the attack
  descriptions in the technical-details PDF (pp. 30–35).
  * **For 23.05, how it is built:** from the documentation and the label files only
    (attack times and attacked points), by the ground-truth tool, through a logged
    access before G5. This is the same procedure as for 21.03.
  * **No model or result is involved.**
* **Human verification (required before G5):** the author verifies every mapping
  (21.03, 22.04, 23.05) against the documentation and records the reviewer, the date
  and any disagreements. This closes the limitation disclosed in ADR 0020.
* **CIC compromised scenarios:** the attack types in their attack logs are mapped to
  ATT&CK for ICS under the same protocol.

## 6. Schedule (four weeks)

| Week | Work |
|---|---|
| 1 | download and verification; HAI 22.04 partitions; ground truth for 22.04 and 23.05; the author's verification; the trial tool; first attribution trials |
| 2 | attribution trials (A1–A4) |
| 3 | detection and DP trials (D1–D3) and fusion trials (F1–F2) |
| 4 | freeze (G5), the final run, the analysis, ADR 0022 |
