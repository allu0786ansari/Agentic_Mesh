**PROJECT PROPOSAL**

# Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

*Federated, differentially private anomaly detection with knowledge-grounded agentic attribution for industrial control systems*

| | |
|---|---|
| **Author** | Allaudin Ansari |
| **Document type** | Research and engineering proposal |
| **Version** | 3.1 — Data-informed revision of v3.0 (supersedes v2.1) |
| **Date** | October 2026 |
| **Companion document** | Implementation Plan v3.1 |
| **Domain** | ICS security · federated learning · differential privacy · agentic AI |

---

## Revision Note

**v3.1 (data-informed revision).** Before Week 2, both datasets were analysed in full: HAI 21.03 and the CIC Modbus 2023 captures obtained for this project (one external-attacker day plus nine benign days per device). The analysis produced 21 findings, HAI-1 to HAI-7 and MB-1 to MB-14. Appendix B maps each one to the decision it changed. The changes are:

1. **Site A is partitioned by recording period, not operating regime** (§4.1). HAI shows one weak regime (turbine demand) and a calibration offset that follows file boundaries. Clients are `train1`, `train2` and the two halves of `train3`.
2. **HAI facts are fixed** (§6.1): 79 columns, 57 vary in training, 56 are modelled; the label is `attack`; there are 50 attack events in the test files. Validation and test are split by event segment (§6.2).
3. **The Site A model input is six 10-second segment means per feature** (336 inputs), not a flattened 3,360-value window, to keep DP-SGD noise manageable (§6.4).
4. **Site B labels come from the attacker's address at window level** (§6.1). The attack day available has one external attacker that reached only IED1A (51 of 3,681 10-second windows). Training uses benign days 22–31 January; the attack day is used only for attack validation and test (§6.2).
5. **Site B features are defined around what the attacks changed** and from each node's own side of the traffic (§6.6). Latency stays a single-device feature and is **excluded from fusion**, because the two secure IEDs stall together on normal days (§8.2).
6. **RQ2 becomes a fusion trade-off question** (§2): k-of-n fusion should cut false alarms but cannot help against an attack seen by one device. A labelled synthetic multi-device scenario is a secondary experiment.
7. **RQ3 is evaluated at tactic level on HAI's 50 events, plus a case study of the seven Modbus episodes** (§10.4). The Site B technique-accuracy target is removed (PT5).
8. **Week 2 parses captures with `dpkt`** instead of exporting fields with `tshark`; a memory-mapped reader handles the 50–400 MB files on 8 GB.

Everything else in v3.0 stands.

**v3.0.** Version 3.0 is a redesign, not an edit. A structured review of v2.1 and its implementation plan identified **68 issues, 16 of them critical**. Appendix A lists every issue, how it was resolved, and where the resolution appears in this proposal and in the Implementation Plan.

The critical issues fell into five groups:

- a federation design that could not work as specified;
- privacy accounting errors;
- an evaluation design that could not produce defensible numbers;
- an attribution pipeline that compared incompatible vector spaces;
- claims about cross-node correlation, compliance and latency that the chosen datasets and hardware could not support.

**Principal changes in v3.0**

1. **Horizontal partitioning.** HAI is split horizontally into a four-node *process fleet* in which every node has the same feature schema. The v2.1 subsystem split was vertical partitioning, which FedAvg and FedProx cannot aggregate.
2. **Federable network model.** Network nodes use a federable autoencoder. Isolation Forest becomes a local baseline only.
3. **Two independent federations.** Fed-A has 4 clients and Fed-B has 3. Evaluation is federated, so the server never holds data.
4. **Corrected privacy accounting.** Each client has one total budget, accounting persists across rounds and is never reset, and the privacy unit and threat model are explicit.
5. **Minimised evidence records.** Nodes share structured evidence records instead of latent embeddings. The Attribution Agent renders that evidence as text and retrieves in the knowledge-base embedding space.
6. **Bounded LLM role.** The LLM makes one validated, grounded attribution call per incident. Triage and investigation are deterministic. High-impact actions require human approval.
7. **Evaluation-first methodology.** The test set is sealed, thresholds come from validation data only, metrics are event-level, every configuration runs on five seeds, baselines and ablations are scheduled, and targets are preregistered.
8. **Three tiers of success criteria.** Completion criteria, performance targets and engineering acceptance are kept separate.
9. **Scoped correlation claims.** Cross-node correlation is claimed only within the substation site, where devices share a capture period.

**Retained from v2.1**

- The research thesis and the three-module structure.
- FedAvg and FedProx through Flower; DP-SGD through Opacus.
- The four-agent LangGraph mesh and the MITRE ATT&CK for ICS knowledge base.
- A small model served locally through Ollama (Phi-3.5-mini in v2.1; Llama 3.2 3B chosen at Gate G0 in v3.1).
- Embedded Qdrant, SQLite, Arize Phoenix, DVC and MLflow.
- The 8 GB single-machine scope, with Docker Compose for reproducibility.

## Planning Assumptions

| ID | Assumption | Sections affected if it changes |
|---|---|---|
| PA-1 | 16-week schedule from a clean restart; one developer working about 35–40 hours per week | Implementation Plan, all phases |
| PA-2 | The deliverable is a university major-project report, structured so it can also become a workshop or conference paper | §2, §10, §11 |
| PA-3 | An 8 GB RAM laptop running Windows 11 with WSL2 (Ubuntu 24.04). Native Linux also works; the WSL-specific steps are then skipped | §9.4 |
| PA-4 | The four-agent LangGraph mesh is retained, but the LLM is used for one attribution and explanation step per incident | §8 |

---

## Executive Summary

Industrial operators need anomaly detection that learns from many installations. The telemetry it would learn from is usually sensitive, commercially or for safety reasons, and often cannot be pooled. Detectors trained at a single installation see too little behaviour to generalise. And an alert alone does not tell an operator what is happening or what to do.

This project builds and evaluates a **decentralised agentic mesh** that combines three components:

1. Local anomaly detectors at edge nodes.
2. Federated training with differential privacy (DP-SGD), so nodes learn a shared model without exchanging raw telemetry.
3. A small multi-agent layer that groups alerts into incidents, gathers minimised evidence from nodes at the same site, and attributes each incident to MITRE ATT&CK for ICS techniques with a locally hosted language model. The model's output is grounded in retrieved knowledge-base entries and validated against them.

The system is evaluated on two public ICS datasets, modelled as two sites:

- **Site A — process fleet.** Four installations of the same process type, built from the HAI dataset.
- **Site B — substation.** Three Modbus devices, built from the CIC Modbus 2023 dataset.

The evaluation answers three research questions:

- **RQ1:** how much detection quality federated DP training costs, compared with centralised and local-only training.
- **RQ2:** what fusing evidence across nodes at one site gains and loses: fewer false alarms, against missed single-device attacks.
- **RQ3:** whether retrieval-grounded LLM attribution beats retrieval-only and LLM-only baselines at tactic level.

The methodology is evaluation-first. The test set is sealed; thresholds are selected on validation data; metrics are event-level and avoid inflated point-adjusted scores; every configuration runs on five seeds with confidence intervals; and targets are preregistered before any federated results exist.

The full system runs on one 8 GB machine as cooperating local processes, and is packaged with Docker Compose so that others can reproduce the results.

---

## 1. Problem Statement

### 1.1 Learning across installations without pooling data

The most valuable industrial AI use cases depend on operational data that organisations are reluctant or unable to share: predictive maintenance, intrusion detection, and process-integrity monitoring. Process telemetry can reveal production volumes, recipes, equipment condition and security posture. Pooling it in a central data lake creates contractual and data-residency obstacles, and concentrates a high-value target for attackers.

Federated learning lets installations train a shared model without moving raw data. It does not stop model updates from leaking information about the training data, however; that is why this project adds differential privacy to the updates.

### 1.2 Isolated detectors and unexplained alerts

A detector trained at one installation learns only that installation's normal behaviour, so it adapts poorly to new operating regimes. Separately, a raw anomaly score is not actionable: operators need to know what deviated, what attacker behaviour it most resembles, and what response is appropriate. Today that interpretation is done manually and slowly.

### 1.3 Objective

The objective is to design and empirically evaluate a system that:

- (a) trains shared anomaly detectors across nodes without moving raw telemetry, with a stated, measured differential-privacy guarantee;
- (b) combines evidence from several nodes observing the same site;
- (c) turns incidents into grounded, validated MITRE ATT&CK for ICS attributions with safe, policy-gated response recommendations;

all on a single 8 GB development machine.

---

## 2. Research Questions, Hypotheses and Contributions

### 2.1 Research questions and hypotheses

| RQ | Question | Hypothesis (falsifiable) | Experiments |
|---|---|---|---|
| **RQ1** | What detection quality does federated training with DP-SGD retain, compared with centralised and local-only training, on non-IID industrial partitions? | **H1a:** at ε ≤ 10, the federated DP model's AUROC is within 0.05 of centralised non-private training and above the mean local-only model on each site. **H1b:** under unequal local work and client dropout, FedProx reaches the target validation loss in fewer rounds than FedAvg; with full, equal participation the two are statistically indistinguishable. | X1, X2 |
| **RQ2** | What does fusing calibrated evidence from several nodes at the same site gain and lose compared with the best single node? | **H2a:** on benign Site B data, 2-of-3 fusion lowers false alarms per hour relative to the best single node at the same per-node threshold, with a 95% CI excluding zero. **H2b:** on the real external-attack day, which only IED1A observes, noisy-OR fusion keeps the single-node event recall while 2-of-3 fusion loses it; the loss is reported, not hidden. **H2c (secondary, synthetic):** on a labelled synthetic scenario in which an attack is visible on two or more devices, fusion raises event recall at matched FPR. | X3 |
| **RQ3** | Does retrieval-grounded attribution with a local small language model improve MITRE ATT&CK for ICS attribution over simpler alternatives? | **H3:** on HAI's attack events, retrieval plus LLM has higher tactic accuracy than retrieval-only and LLM-without-retrieval baselines, and every emitted technique ID is grounded in the retrieved candidates. The seven Modbus episodes are reported as a qualitative case study with validity and grounding rates. | X4 |

### 2.2 Contributions

- **C1 — Evaluation protocol.** A reproducible protocol for privacy-preserving federated ICS anomaly detection: sealed test data, validation-only thresholds, event-level and range-based metrics, ε sweeps, and multi-seed confidence intervals.
- **C2 — Privacy–utility characterisation.** Empirical results for FedAvg and FedProx with DP-SGD on two ICS datasets under controlled heterogeneity.
- **C3 — Evidence fusion.** A minimised, structured evidence record and a within-site fusion method, with its false-alarm benefit and its single-device blind spot both measured.
- **C4 — Grounded attribution.** An attribution pipeline whose outputs are constrained and validated, with policy-gated dispatch, evaluated against rule-based, retrieval-only and ungrounded-LLM ablations.
- **Artifact.** An open, reproducible codebase in which every reported number comes from a provenance-stamped artifact.

---

## 3. Scope

| Category | In scope |
|---|---|
| **Research core** (carries claims) | Federated DP anomaly detection (Modules 1–2); within-site evidence fusion; grounded attribution (Module 3); all experiments X1–X4 |
| **Engineering support** (acceptance-tested; no research claims) | Session durability and idempotency; mTLS and JWT between processes; tracing; resource benchmark; SecAgg+ systems experiment; Docker Compose packaging; CI |
| **Out of scope** (future work) | Malicious or Byzantine clients (poisoning defences); attacks spanning multiple sites or organisations; live plant deployment; formal DP for evidence records; other industries such as finance and healthcare; multi-node cluster deployment |

---

## 4. System Overview

### 4.1 Sites and nodes

| Node | Site | Represents | Data | Model | Federation |
|---|---|---|---|---|---|
| a1, a2, a3, a4 | **Site A — process fleet** | Four installations of the same process type | HAI 21.03, the 56 modelled features. a1 = `train1`, a2 = `train2`, a3 and a4 = the first and second halves of `train3` | VAE | Fed-A (4 clients) |
| b1 | **Site B — substation** | IED1A (secure IED, 185.175.0.4) | CIC Modbus 2023, per-device 10 s window features from the node's own side of the traffic (shared schema) | Dense autoencoder | Fed-B (3 clients) |
| b2 | Site B | IED1B (insecure IED, 185.175.0.5) | As b1 | Dense autoencoder | Fed-B |
| b3 | Site B | SCADA HMI (185.175.0.3) | As b1; the HMI is a Modbus client, so its features differ in distribution from the IEDs (the non-IID part of Fed-B) | Dense autoencoder | Fed-B |

**Why horizontal partitioning.** FedAvg and FedProx average parameters, so every client must have the same model shape and the same feature schema. Version 2.1 gave each HAI node a different subsystem. That made the data vertically partitioned and left the VAEs with different input sizes, so they could not be averaged.

In v3.0, every Site A node sees all features. v3.0 planned to make the nodes non-IID by giving each one a different operating regime. The v3.1 data analysis found that HAI 21.03 does not support this: the only real regime is turbine power demand (`P4_ST_PS` at 0 or 50), clustering six-hour blocks on setpoints gives weak groups (silhouette 0.34 for four clusters) that follow file boundaries, and the largest between-block difference is a calibration offset (`P1_PCV02Z`) that separates `train1` from the other files. Site A is therefore **partitioned by recording period** (HAI-3). The four clients are nearly equal (60–66.5 hours each) and differ through real drift between periods: 10 features have a two-sample KS statistic above 0.2 between `train1` and `train3`, and the turbine speed sensors shift by about 1.35 standard deviations. This resembles a fleet of plants calibrated and operated slightly differently. An IID variant (shuffled six-hour blocks) is kept for experiment X2.

**Why two separate sites.** HAI and CIC Modbus 2023 were recorded on different testbeds at different times, so nothing links an event in one to an event in the other. The sites therefore never share parameters, and the project makes no cross-site correlation claims.

Cross-node evidence fusion (RQ2) is evaluated on Site B. Its three captures overlap in time on every recorded day (verified in the data analysis). On the attack day, however, the attacker reached only IED1A, so a real attack gives fusion a single device to work with; §2 frames RQ2 around that trade-off. On Site A, nodes are separate installations recorded at different times, so investigation there gathers additional evidence from the alerting node only.

### 4.2 Three research modules

| Module | Focus | Primary output |
|---|---|---|
| **1 — Edge detection** | Local scoring of telemetry windows; calibration; per-feature deviation | Calibrated anomaly probability and a minimised evidence record |
| **2 — Privacy-preserving federated learning** | FedAvg and FedProx with DP-SGD; federated evaluation; SecAgg+ systems experiment | Shared detector per site with a reported (ε, δ) per client |
| **3 — Fusion and grounded attribution** | Incident grouping, within-site fusion, bounded investigation, retrieval-grounded attribution, policy-gated dispatch | Validated technique and tactic attribution with mitigations and a full trace |

---

## 5. Threat Model and Privacy Claims

### 5.1 Assets and adversaries

**Assets**

- Raw telemetry at each node.
- Each client's model updates.
- Evidence records.
- Prompts and knowledge-base content.
- Dispatched actions.
- Service credentials.

| ID | Adversary | Capability assumed |
|---|---|---|
| ADV-1 | Honest-but-curious aggregation server | Follows the protocol; inspects everything it receives |
| ADV-2 | Curious peer client | Follows the protocol; inspects the global models it receives |
| ADV-3 | Network attacker between services | Can observe, replay or inject traffic on local service links |
| ADV-4 | Compromised or faulty agent process | Can issue node queries using any credentials it holds |
| ADV-5 | Exfiltration through external services | Investigation content sent to a third-party LLM API |

**Out of scope:** clients that poison updates; compromised node hosts; side channels; physical access.

### 5.2 Mechanisms

| Mechanism | Defends against | Guarantee or effect |
|---|---|---|
| DP-SGD (Opacus) at every client | ADV-1, ADV-2 | (ε, δ)-DP with respect to one training window of that client's data, over the full sequence of released updates |
| Noisy normalisation statistics | ADV-1 | Gaussian mechanism; its ε is added to the client's total budget |
| SecAgg+ (systems experiment, §7.4) | ADV-1 | The server sees only the aggregate of the updates, not any individual update |
| Federated evaluation | ADV-1 | The server receives only aggregate metrics, never test data |
| mTLS between all processes | ADV-3 | Encrypted, mutually authenticated links; plaintext connections are rejected |
| Per-request JWT (audience, scope, expiry, one-time ID) | ADV-3, ADV-4 | Each node query is limited to one session, one site and one time window, and cannot be replayed |
| LLM egress allowlist | ADV-5 | The mesh refuses to start if the LLM endpoint is not loopback or explicitly allowlisted |
| Evidence minimisation (§5.4) | ADV-1 to ADV-4 | No raw values or latent vectors ever leave a node |
| Policy-gated dispatch (§8.5) | ADV-4 | High-impact actions require human approval |

### 5.3 What leaves a node

- **During training:** clipped, noised model updates, plus one set of noisy normalisation statistics.
- **During federated evaluation:** aggregate metric counts and binned score histograms.
- **During operation:** alerts and evidence records (§6.6).

Nothing else leaves a node. Schema contract tests on every outbound message enforce this (Completion criterion CC5).

### 5.4 Evidence minimisation and residual risk

Evidence records contain only:

- identifiers and time bounds;
- the anomaly score and calibrated probability;
- the *names* of the top-k deviating features, with direction and a coarse magnitude bucket;
- bucketed protocol counters (Site B only).

They contain no raw values, reconstructions or latent vectors. These records are **minimised but not formally differentially private**. That residual risk is stated in §13, and applying DP to evidence records is listed as future work.

### 5.5 How privacy and compliance are described

The proposal makes only claims the design supports:

- The DP guarantee is stated as (ε, δ) per client, with a defined privacy unit (§7.3), and ε is reported for every configuration.
- No regulatory-compliance claim is made. The accurate statement is that keeping raw telemetry on the node *reduces* data-transfer exposure and supports compliance work. It does not replace legal review.
- Industrial telemetry is often not personal data, so the motivation is framed around confidentiality and data residency rather than GDPR or HIPAA.
- No cost or fine figures are quoted.

---

## 6. Module 1 — Edge Detection

### 6.1 Datasets

Both datasets were analysed in full before Week 2. `configs/data_expectations.yaml` records the measured facts, and `scripts/inspect_data.py` checks any local copy against them (21 checks).

**HAI 21.03**
- Release pinned: HAI 21.03, 3 training files (921,603 rows) and 5 test files (402,005 rows), sampled at 1 Hz. SHA-256 checksums are recorded in the data manifest and the data card.
- 79 sensor and actuator columns across processes P1–P4. **57 vary in training** and 22 are constant. One of the 57, `P4_HT_PS`, is an exact copy of `P4_ST_PS` and is dropped, leaving **56 modelled features** (HAI-2).
- Training-constant variables that change in test (for example `P1_PCV02D`, `P2_Emerg`, `P2_OnOff`) are not model inputs. A separate label-free rule reports "a training-constant variable changed" as evidence.
- **Label:** the `attack` column (HAI-1). The per-process columns `attack_P1`, `attack_P2` and `attack_P3` are never model inputs; they record which subsystems an event targeted and feed the tactic-level ground truth (§10.4).
- The test files contain **50 attack events** (8,947 attack rows; median duration 162 s, shortest 17 s). By target: P1 23, P2 11, P1+P2 8, P1+P3 3, P2+P3 3, P3 2. The smallest gap between events is 24 minutes.

**CIC Modbus 2023**
- The captures available are one **external-attacker day** (1 February 2023, about 10 hours per device) and **nine benign days** (22–31 January) for IED1A, IED1B and the SCADA HMI. Benign data ends about 11 hours before the attack day starts.
- Captures are parsed directly with `dpkt` and a memory-mapped pcap/pcapng reader, which keeps memory flat on the 50–400 MB files. `tshark` is not needed.
- **Labels come from the attacker's address** (185.175.0.7), window by window, and are marked as derived (MB-2). The public attack logs are not part of the obtained data. A window is an attack window if the node sees any IPv4 packet from or to the attacker in it. Capture-level labels are never used.
- On the attack day the attacker sent 538,648 packets to **IED1A only**, in **seven episodes**: four short reconnaissance probes, a 20-minute slow read scan, a 4-minute flood of 65,535 coil writes to one address, and a final burst of writes to two addresses (MB-6). That gives **51 attack windows out of 3,681** (1.4%) on IED1A. IED1B and the HMI saw no attacker traffic; every one of their attack-day windows is normal (MB-11).
- The obtained data has no compromised-IED or compromised-HMI attacker positions, and the documentation does not map attacks to MITRE techniques. Both are handled in §2 and §10.4.

Synthetic anomalies are used only for robustness tests and for the secondary RQ2 scenario (H2c). They are generated as explicit, labelled perturbations (spike, drift, stuck-at, replay, cross-device write burst) and always reported as synthetic.

### 6.2 Splits, windows and labels

**Split before windowing.** Data is split into train, validation and test by time first, with a one-window gap at each boundary. Windows are generated afterwards, so no window crosses a split.

**Site A (HAI)**

| Partition | Content |
|---|---|
| Training | The first 85% of each client's recording period |
| Normal validation | The last 15% of each client's recording period |
| Attack validation | About 15 of the 50 events, as whole **event segments** (boundaries halfway between neighbouring events), stratified by process combination and spread across the four clients (HAI-5) |
| Test | The remaining 35 event segments; sealed (§10.1) |

Test files are assigned to clients segment by segment so that each client receives attack segments for evaluation.

**Site B (Modbus)**

| Partition | Content |
|---|---|
| Training | Benign days 22–31 January, excluding the last 15% of each node's benign time |
| Normal validation | The last 15% of each node's benign time |
| Attack validation | The first attacker episodes on the attack day (episodes 0–3), with the normal attack-day windows around them |
| Test | The rest of the attack day (episodes 4–6 and the surrounding normal windows); sealed |

**Idle periods are dropped before windowing** (MB-7): any stretch longer than 60 s with no HMI polling (for example the 16-hour gap on IED1A on 22 January) is cut and recorded in the partition manifest.

**Windows**

| Site | Length | Training stride | Inference stride |
|---|---|---|---|
| A | 60 s | 60 s (non-overlapping; see privacy unit, §7.3) | 10 s |
| B | 10 s | 10 s | 5 s |

**Labels**
- Site A: a window is labelled *attack* if it overlaps any row with `attack = 1`. An *attack event* is a maximal contiguous labelled interval.
- Site B: a window is labelled *attack* on a node if that node sees attacker traffic in it. An *attack event* is an attacker episode (a run of attacker activity with gaps under 60 s).
- Events are never split across partitions or across clients.

### 6.3 Normalisation

A global model needs consistently scaled inputs. Before training, each client releases the per-feature count, sum and sum of squares with Gaussian noise added. The server aggregates them into global mean and variance. The privacy cost of this step (ε = 0.1 by default) is added to each client's budget by basic composition.

Features are standardised with the global mean and standard deviation and clipped to ±6 standard deviations (HAI-4). Min–max scaling is not used: test data goes outside the training range, and zero-variance features would divide by zero.

### 6.4 Models

| Site | Model | Notes |
|---|---|---|
| A | Variational autoencoder over **segment means**: each 60 s window is summarised as six 10 s means per feature, 6 × 56 = 336 inputs (HAI-6) | Hidden layers 256 → 128, latent size 16. Only Opacus-compatible layers (no batch normalisation). Anomaly score = mean reconstruction error over the inputs. A flattened 60 × 56 window (3,360 inputs) is compared on validation in Week 4 as an alternative |
| B | Dense autoencoder over per-node window features | Hidden layers 64 → 32. Same scoring approach |
| A, B | PCA reconstruction error; Isolation Forest | **Local baselines only**; not federated (Isolation Forest has no gradients to aggregate) |

The input reduction matters for privacy. Each Site A client has about 3,000–3,400 training windows, fewer than the inputs of a flattened window, and DP-SGD noise is added to every parameter. With batch 256 and 50 rounds of one local epoch, the RDP accountant gives σ ≈ 8.4 at ε = 1, 3.2 at ε = 3 and 1.3 at ε = 10 (HAI-7). Site B clients have about 60,000 training windows each, so the same budgets need only σ ≈ 2.1, 1.0 and 0.63.

### 6.5 Scoring, calibration and thresholds

- Each window produces a scalar score and a per-feature error vector. The per-feature errors are what make the evidence explainable.
- Scores are calibrated to probabilities with isotonic regression, fitted on that node's validation data.
- Thresholds are set per node on **normal validation windows**, targeting a window false-positive rate of **1%**. Results are also reported at 0.5% and 5%.
- Thresholds are never chosen on test data, and never set at a percentile that fixes the reported FPR by construction.

### 6.6 Evidence record

```json
{
  "evidence_id": "uuid",
  "site_id": "site_b",
  "node_id": "b1",
  "window_start": "ISO-8601",
  "window_end": "ISO-8601",
  "model_version": "sha256:…",
  "score": 0.0,
  "threshold": 0.0,
  "calibrated_prob": 0.0,
  "top_features": [{"name": "feature_name", "direction": "up|down", "magnitude": "low|medium|high"}],
  "protocol_summary": {"read_ops": "0|1-10|11-100|>100", "write_ops": "…", "diagnostic_ops": "…",
                       "exception_responses": "…", "request_rate": "low|normal|high",
                       "new_peer": false, "unseen_address": false, "oversized_quantity": false,
                       "malformed_frames": "0|1-10|>10", "slow_responses": "0|1-10|>10"},
  "provenance": {"run_id": "…", "data_source": "runtime"}
}
```

- `top_features` holds at most 5 entries.
- Magnitude buckets are defined against the node's normal validation error distribution: low = p90–p99, medium = p99–p99.9, high = above p99.9.
- `protocol_summary` is present for Site B only. Its fields follow what the attacks on the attack day actually changed (MB-4, MB-9):
  - `new_peer`: a peer address never seen by this node in training;
  - `unseen_address`: a register or coil address never read or written in training;
  - `oversized_quantity`: a requested quantity above the training maximum (the attacker asked for 65,535 registers);
  - `malformed_frames`: stacked frames or MBAP length mismatches;
  - `slow_responses`: responses slower than this node's training p99.9.
- Address and peer sets are learned per node, because each IED has its own point list.
- Site B features are computed **from the node's own side of the traffic** (MB-14): requests sent and received per function code, peers, addresses, frame anomalies and latency. The HMI is a client of three IEDs; each IED is a server to one HMI. The schema is shared, so Fed-B can average, but the distributions differ.
- Latency features are robust (window median and count of slow responses) with per-node baselines (MB-10). The secure IEDs (IED1A and the unmonitored IED4C) stall for tens of seconds most days; the insecure IED1B never does.

---

## 7. Module 2 — Privacy-Preserving Federated Learning

### 7.1 Federations

There are two independent federations, and no parameters are shared between them:

- **Fed-A:** nodes a1–a4, VAE. Minimum 3 of 4 clients per round.
- **Fed-B:** nodes b1–b3, autoencoder. Minimum 2 of 3 clients per round.

### 7.2 Aggregation strategies

FedAvg and FedProx are compared under identical seeds, client order, round limits, timeouts and evaluation partitions. FedProx is swept over μ ∈ {0.001, 0.01, 0.1}.

Comparing the two strategies is a **hypothesis** (H1b), not a success criterion. It is tested under two conditions:

- **Homogeneous:** full participation and equal local epochs.
- **Systems-heterogeneous:** local epochs drawn from {1, 2, 3}, plus a 20% chance per client per round of dropping out.

### 7.3 Differential privacy

- **Privacy unit:** one non-overlapping 60 s (Site A) or 10 s (Site B) training window of one client. Because training windows do not overlap, each raw reading belongs to exactly one training unit.
- **Mechanism:** DP-SGD through Opacus, with per-sample gradient clipping (default max norm 1.0) and Poisson sampling.
- **Calibration:** the noise multiplier is calibrated *once, up front*, for the planned total number of steps: rounds × local epochs × steps per epoch, including any warm-up. The RDP accountant is used.
- **Persistent accounting:** each client's accountant state is saved to disk and reloaded every round, so the privacy budget is never silently reset when the federation framework recreates client objects.
- **Budget:** one total target ε per client per training run. The sweep is {1, 3, 10}, with non-private training (ε = ∞) as a reference. δ = min(10⁻⁵, 1 / (10 × N_c)), where N_c is the client's number of training windows.
- **Exhaustion:** a budget guard refuses further local training once the next step would exceed the target ε. Accounting is never reset.
- **Reporting:** every artifact records ε, δ, noise multiplier, clipping norm and steps for each client, including the normalisation-statistics cost.

### 7.4 Secure aggregation (systems experiment)

SecAgg+ is integrated on Fed-A through the API of the pinned Flower version and evaluated as a **systems experiment (X8)** that measures three things:

1. Correctness: the aggregate matches the plaintext aggregate within quantisation tolerance.
2. Computation and communication overhead.
3. Tolerance to client dropout.

The main experiment matrix runs without SecAgg+, because it changes the aggregate only by quantisation error, which X8 measures. With 3–4 clients, SecAgg+ offers limited anonymity; this is stated as a limitation.

### 7.5 Federated evaluation

- Clients evaluate the global model on their own validation and test partitions and return aggregate metrics.
- Per-client AUROC is exact. Pooled AUROC is computed from per-client binned score histograms (1,000 fixed bins on calibrated probability); the binning error is reported.
- The server never loads evaluation data.
- A *centralised oracle* configuration exists only as a labelled baseline.

### 7.6 Stragglers and stopping

- **Stragglers:** a per-round timeout applies. A round proceeds once the minimum number of clients has reported. Late or missing clients are logged. A node is marked `degraded` in the registry after three consecutive missed rounds.
- **Stopping:** training stops at 50 rounds, or when federated validation loss has improved by less than 0.5% (relative) for five consecutive rounds.
- **Convergence is measured against an external target:** the number of rounds until federated validation loss falls within 2% of the centralised non-private baseline's validation loss. This avoids the circular "within 2% of final loss" measure from v2.1.

### 7.7 Retraining policy

Continuous retraining is not claimed. Retraining happens periodically. Each period is a new training run with its own data and its own budget, defined per deployment period, and cumulative privacy loss across periods is reported.

---

## 8. Module 3 — Evidence Fusion and Grounded Attribution

### 8.1 Overview

Alerts from edge nodes go to a LangGraph state machine with four agents. Only the Attribution Agent calls the language model, and it makes exactly one call per incident. Every session is stored durably, can be replayed without side effects, and always ends in a terminal state.

### 8.2 Agents

| Agent | Implementation | Responsibility |
|---|---|---|
| **Triage** | Deterministic | Deduplicates alerts and groups them into incidents per site within Δt (default 60 s). Computes the fused incident score (Site B: noisy-OR of calibrated probabilities; a k-of-n variant with k = 2 is evaluated as an alternative). Latency-only anomalies are **excluded from fusion** and kept as single-device context, because the secure IEDs stall together on normal days (MB-13). Assigns severity. Opens a session with an idempotency key |
| **Investigator** | Deterministic | Requests evidence from nodes **at the same site only**, up to 3 hops. On Site A, it requests extra evidence (adjacent windows, subsystem breakdown) from the alerting node only. Every hop is persisted before the graph advances |
| **Attribution** (renamed from "Security") | Retrieval plus one constrained LLM call | Renders evidence as text, retrieves candidate techniques, and asks the LLM to choose among those candidates and explain the choice. Validates the output, falls back when needed, and attaches mitigations |
| **Dispatcher** | Policy table | Converts the attribution into actions. Executes low-impact actions automatically; queues high-impact actions for human approval. Dispatch is idempotent per session and action type |

**Ablation modes:**
- `full` (default).
- `retrieval_only`: no LLM; the top-1 retrieved technique is used.
- `llm_no_retrieval`: the LLM chooses from the full ICS technique list with no retrieved context.
- `rule_only`: a hand-written rule table, frozen before testing.

### 8.3 Sessions, hops and failure handling

- **Session key:** the idempotency key is SHA-256 of the site ID plus the sorted IDs of the alerts that opened the incident. Replaying an alert therefore reuses the existing session and can never dispatch twice.
- **Terminal states:** `COMPLETED`, `COMPLETED_DEGRADED`, `ABSTAINED`, `FAILED`.
- **Per-hop limits:** connect timeout 0.2 s and read timeout 0.5 s. At most one retry, and only for transient failures (connection error, timeout, HTTP 503), with exponential backoff and jitter.
- **Investigation deadline:** 3 s overall. When it expires, the session continues with the evidence collected so far and is marked `COMPLETED_DEGRADED`.
- **Circuit breaker (per node):** opens after 2 consecutive failures; allows a half-open probe after 30 s. State transitions are emitted as metric events.

### 8.4 Knowledge base and attribution

**Knowledge base**
- The MITRE ATT&CK for ICS STIX bundle is parsed into one chunk per technique: name, description, detection guidance, tactics, and mitigation relationships.
- NIST SP 800-82 Rev. 3 is chunked as *supplementary context*. Those chunks can be retrieved as context but are never offered as technique candidates.
- Chunks are embedded with a single sentence-transformer model (default `all-MiniLM-L6-v2`, 384 dimensions) and stored in embedded Qdrant collections: `mitre_ics`, `nist_80082` and `incident_memory`.

**Attribution flow**
1. **Render:** a deterministic template turns the evidence records into a short text description, such as "write operations >100 from new peer to IED1A; request rate high; exception responses 11–100". This text is embedded with the *same* encoder as the knowledge base, which removes v2.1's mismatch between 128- and 384-dimension vectors.
2. **Retrieve:** the top-5 ICS techniques are retrieved by cosine similarity.
3. **Decide:** one LLM call (Llama 3.2 3B via Ollama, chosen at G0; temperature 0, context window 2,048 tokens) returns JSON. The JSON is constrained by a schema whose `technique_id` field may only take the IDs of the retrieved candidates.
4. **Validate:** the output must parse against the schema, the ID must be one of the candidates, and confidence must be at least 0.5. If any check fails, or the call times out, the system falls back to the top-1 retrieved technique and records `attribution_mode = retrieval_fallback`.
5. **Mitigations** come from the STIX mitigation relationships for the chosen technique. They are looked up deterministically, never generated by the LLM.

### 8.5 Action policy

| Action | Impact | Execution |
|---|---|---|
| Operator alert with attribution and trace link | Low | Automatic |
| Increase monitoring (extend evidence retention, raise log verbosity) | Low | Automatic |
| Change a node's detection threshold | High (could suppress detection) | Requires approval |
| Isolate a node or block a peer | High | Requires approval |

High-impact actions are recorded as `pending_approval` and are executed only through an authenticated approval endpoint. Every action record links to its session, evidence and trace.

### 8.6 Latency budget

| Segment | Definition | Target (p95) |
|---|---|---|
| Detection | Window end (replay time) → alert published | ≤ 1 s |
| Investigation | Session opened → evidence complete | ≤ 2 s |
| Incident-to-dispatch, `retrieval_only` mode | Session opened → action recorded | ≤ 3 s |
| Incident-to-dispatch, `full` mode | Session opened → action recorded | ≤ L_LLM = 24 s, set at Gate G0 from measured LLM performance (p95 13.6 s per call) |

The 3-second target from v2.1 is kept for the LLM-free path. The LLM path gets a measured, honest target, because a 3B-parameter model generating structured output on a laptop CPU cannot meet 3 seconds. Gate G0 measured Llama 3.2 3B at p95 13.6 s per call on the development laptop, giving an LLM timeout of 21 s and L_LLM = 24 s.

---

## 9. Architecture

### 9.1 Layers

| Layer | Components | Responsibility |
|---|---|---|
| Edge | Telemetry replay, edge runtime (frozen global model, calibrators, thresholds), per-node evidence store | Scoring, evidence records, alerts |
| Federated learning | Flower simulation (Ray backend), FedAvg and FedProx, Opacus, federated evaluation, SecAgg+ (X8), MLflow | Private training and evaluation |
| Agentic mesh | LangGraph graph with Triage, Investigator, Attribution and Dispatcher; Ollama | Incidents, investigation, attribution, dispatch |
| Shared services | SQLite (registry, sessions, actions, bus), embedded Qdrant, node APIs (FastAPI over mTLS) | Durable state, retrieval, evidence access |
| Observability and provenance | Arize Phoenix traces, JSONL metric events, artifact manifests, DVC | Auditability and reproducibility |

### 9.2 Process topology and messaging

The system runs as several cooperating local processes, not as one process. This is what makes the mTLS, JWT and failure tests meaningful.

| Process | Instances | Notes |
|---|---|---|
| `edge-runtime` | 1 per site | Replays and scores telemetry; writes evidence; publishes alerts |
| `node-api` | 1 per node (7) | Lightweight FastAPI serving that node's evidence store over mTLS, with JWT checks |
| `mesh` | 1 | LangGraph agents; bus consumer; approval endpoint |
| `ollama` | 1 | Local LLM server on loopback |
| `phoenix` | 0–1 | Started during mesh runs for trace capture |

**Messaging**
- The bus is a **SQLite-backed outbox and inbox**. Messages are leased, acknowledged and retried; consumers are idempotent. Together these give at-least-once delivery across processes and survive restarts.
- An in-memory bus implementation is used for unit tests.
- Topics: `alerts.{site}`, `actions.{site}`, `approvals`.
- A NATS or Kafka adapter is the production path (§13).

### 9.3 Data flows

**Training flow**
- T1 — Prepare and partition data from a pinned manifest (DVC stage).
- T2 — Clients release noisy normalisation statistics; the server aggregates them.
- T3 — In each round, clients train locally with DP-SGD (accountant persisted) and return clipped, noised updates.
- T4 — The server aggregates the updates with FedAvg or FedProx and logs round statistics to MLflow.
- T5 — Federated evaluation runs on validation data; the stopping rule is checked.
- T6 — The final model is frozen. Each node fits its calibrator and threshold on its own validation data.
- T7 — An artifact with full provenance is written.

**Operations flow**
- O1 — Replay streams each node's sealed test telemetry. This happens only in final runs; development runs use validation data.
- O2 — The edge runtime scores windows with the frozen model.
- O3 — Windows above threshold produce an evidence record, stored locally, and an alert published to the bus.
- O4 — Triage groups alerts into incidents, computes the fused score and opens an idempotent session.
- O5 — The Investigator gathers same-site evidence over mTLS, using one JWT per request (at most 3 hops).
- O6 — Attribution renders, retrieves, makes its one LLM call and validates the result.
- O7 — The Dispatcher applies the action policy; high-impact actions wait for approval.
- O8 — The session reaches a terminal state. The trace is stored in Phoenix and metric events are emitted.

### 9.4 Environment and resource budget

- **Environment:** WSL2 running Ubuntu 24.04, with `.wslconfig` set to `memory=6GB` and `swap=4GB`. Docker Engine runs inside WSL2, not Docker Desktop, to avoid a second virtual machine.
- **Stage isolation rule:** training and operations never run at the same time.

| Stage | Resident components | Estimated peak (to be replaced by measured values in Weeks 1 and 14) |
|---|---|---|
| Training | Python, torch CPU, Ray with 2 concurrent client actors, MLflow local backend | 2.5–3.0 GB |
| Operations | Edge runtime ~0.8 GB; 7 node APIs ~0.5 GB; mesh with MiniLM and Qdrant ~0.8 GB; Ollama Phi-3.5-mini Q4 with 2,048-token context ~2.5 GB; Phoenix ~0.4 GB | 4.5–5.0 GB |
| Windows host outside WSL | Operating system and desktop | ~2 GB |

### 9.5 Observability and provenance

- **Metric events:** every event is a JSONL record with a schema version, `run_id`, `session_id` where applicable, and `data_source`, which is one of `runtime`, `fixture` or `synthetic`.
- **Traces:** Phoenix traces carry `session_id`, and one trace is exported per session for archiving.
- **Result artifacts:** each records the git SHA, SHA-256 hashes of the dataset manifest, checkpoint and parameter file, the lock-file hash, the seed and the split.
- **Fixture data cannot pass evaluation gates.** Reports built from fixture or synthetic data carry a visible watermark, and the success checker rejects them.

### 9.6 Technology stack

Exact versions are pinned in a lock file during Week 1; compatibility checks are listed in the Implementation Plan §4.

| Component | Role |
|---|---|
| Python 3.12 (3.11 fallback), uv | Runtime and locked dependencies |
| PyTorch (CPU), Opacus | VAE and autoencoder; DP-SGD with RDP accounting |
| scikit-learn | PCA and Isolation Forest baselines; isotonic calibration; metrics |
| Flower (simulation with Ray backend) | FedAvg, FedProx, SecAgg+, federated evaluation |
| pandas, pyarrow, dpkt | Data preparation; Modbus/TCP capture parsing with a memory-mapped pcap/pcapng reader |
| LangGraph | Agent state machine |
| Ollama with Llama 3.2 3B (chosen at G0; Phi-3.5-mini as alternative) | Local LLM with schema-constrained JSON output |
| sentence-transformers, qdrant-client (embedded) | Knowledge-base embeddings and retrieval |
| SQLite | Registry, sessions, hops, actions, bus |
| FastAPI, uvicorn, httpx | Node APIs and the mesh client over mTLS |
| cryptography, PyJWT | Development CA and certificates; session tokens |
| Arize Phoenix with OpenInference instrumentation | Agent traces |
| MLflow (local backend, UI on demand), DVC | Experiment tracking; data and model versioning |
| matplotlib | Figures generated only from artifacts |
| Ruff, mypy, pytest, gitleaks, GitHub Actions | Quality gates and CI |
| Docker Engine and Compose | Reproducible runtime |

---

## 10. Evaluation Framework

### 10.1 Splits and the sealed test set

- Validation data is used for all tuning, calibration, threshold selection and gate decisions.
- Test data can be read only through an evaluation entry point run with `--final`. Every access is logged with the configuration hash.
- Final configurations are frozen at Gate G4, and each is evaluated on test data exactly once per seed.

### 10.2 Detection metrics

**Threshold-free**
- AUROC and AUPRC at window level.

**At the operating point** (window FPR ≤ 1% on validation)
- *Event recall:* an event is detected if any alert window overlaps the interval from its start to 60 s after its end.
- Window precision and window FPR.
- Range-based precision, recall and F1 (Tatbul et al., 2018).
- Median and mean time-to-detect.
- False alarms per hour.

Point-adjusted F1 is not used, because it is known to inflate time-series anomaly detection results (Kim et al., 2022).

### 10.3 Threshold and calibration policy

- Thresholds are set per node, on that node's normal validation windows, at the target FPR.
- Calibration uses isotonic regression on validation data.
- The Site B fusion Δt and k are chosen on validation data.
- No parameter is chosen using test data.

### 10.4 Attribution and retrieval metrics

**Ground truth**
- A mapping from attack to tactic set (and, where the evidence supports it, a technique set) is built under a written protocol (`docs/mapping_protocol.md`) and frozen at Gate G1, before any attribution results exist. Two review passes are made, and disagreements are recorded.
- **Site A (primary, quantitative):** the 50 HAI events are mapped at **tactic level**, using the targeted processes (`attack_P1`–`attack_P3`) and the variables that deviate in each event. HAI attacks do not map cleanly to individual techniques. Test events only are scored; validation events tune nothing in the mapping.
- **Site B (case study):** the seven Modbus episodes are mapped by hand from their observed behaviour (for example, reconnaissance reads, a register scan with oversized quantities, a coil-write flood). The dataset documentation gives no MITRE mapping and seven episodes from one attacker position are too few for an accuracy estimate, so Site B reports validity rate, grounding rate and the chosen technique per episode, not accuracy.

**Attribution metrics**
- Tactic accuracy (primary, Site A).
- Technique top-1 and top-3 accuracy, where a technique set exists (secondary).
- Abstention rate.
- **Grounding rate:** the share of raw LLM outputs whose ID is among the retrieved candidates. It is reported before validation; after validation it is 100% by construction.
- A majority-class baseline is always reported alongside.

**Retrieval metrics**
- Recall@5 and MRR on a held-out query set. Queries are rendered from validation incidents, plus paraphrases written independently of the knowledge-base chunks, so the queries do not share text with their targets.

**No external judges.** No metric depends on an external LLM judge; RAGAS is not used.

### 10.5 Baselines and ablations

| Baseline or ablation | Purpose |
|---|---|
| Local-only, non-private (each node trains alone) | Is federation worth it? (H1a) |
| Centralised, non-private (oracle) | Upper reference for utility (H1a) |
| Centralised at ε = 10 | Separates the cost of DP from the cost of federation |
| FedAvg vs FedProx (μ grid) | H1b |
| IID vs non-IID partition | Effect of heterogeneity |
| ε ∈ {1, 3, 10, ∞} | Privacy–utility curve |
| PCA reconstruction; Isolation Forest (local) | Classical reference points |
| Best single node vs noisy-OR vs 2-of-3 (Site B, real and synthetic) | H2a–H2c |
| `rule_only` / `retrieval_only` / `llm_no_retrieval` / `full` | H3 |
| SecAgg+ on vs off (Fed-A) | Correctness and overhead (X8) |

### 10.6 Experiment matrix and statistics

| ID | Experiment | Answers |
|---|---|---|
| X1 | Detection utility: local-only, centralised, FedAvg and FedProx across the ε sweep, both sites | RQ1, H1a |
| X2 | Heterogeneity: IID vs non-IID; homogeneous vs systems-heterogeneous; FedAvg vs FedProx with μ grid | RQ1, H1b |
| X3 | Fusion on Site B: best single node vs noisy-OR vs 2-of-3, measuring false alarms per hour on benign validation and test time (H2a), event recall on the real attack day (H2b), and event recall on the labelled synthetic multi-device scenario (H2c, reported as synthetic) | RQ2 |
| X4 | Attribution ablation, four modes: tactic accuracy on HAI test events; Site B episode case study | RQ3 |
| X5 | Latency profile, end to end, in both modes | §8.6 |
| X6 | Fault and chaos tests: node failure before, during and after a hop; process kill; slow node; duplicate alert replay | EA1 |
| X7 | Resource benchmark on 8 GB: peak memory per stage | CC8 |
| X8 | SecAgg+ systems experiment | §7.4 |

**Statistics**
- Five seeds per configuration (three for non-headline configurations if compute requires; see the Implementation Plan).
- Results reported as mean ± 95% bootstrap CI.
- Paired comparisons across seeds use paired bootstrap or a Wilcoxon signed-rank test.
- When several comparisons are made, this is stated and Holm correction is applied.

### 10.7 Systems metrics

- Latency by segment (p50 and p95).
- Session terminal rate and duplicate-dispatch count.
- Circuit-breaker transitions.
- Peak memory per stage.
- Trace recoverability rate.

---

## 11. Success Criteria

Success is defined in three tiers. **Completion criteria** must all pass for the project to be complete. **Performance targets** are preregistered at Gate G1 and reported honestly whether or not they are met; a missed target is a finding, not a failure of the project. **Engineering acceptance** criteria cover the supporting systems.

### 11.1 Completion criteria (all must pass)

| ID | Criterion |
|---|---|
| CC1 | **No label leakage.** Scoring code never receives label columns; a type-level guard and a test enforce this |
| CC2 | Thresholds, calibration and fusion parameters come from validation data only; test access is logged and occurs once per frozen configuration and seed |
| CC3 | Every reported number comes from an artifact with full provenance and `data_source = runtime` |
| CC4 | **Correct privacy accounting.** Per-client ε (including normalisation statistics and warm-up) is ≤ the configured target; the accountant persists across rounds and restarts; budget exhaustion stops participation; nothing resets |
| CC5 | Only allowlisted message types leave a node (model updates, aggregate metrics, alerts, evidence records); contract tests pass |
| CC6 | Experiments X1–X4 are complete with the specified seeds and CIs reported |
| CC7 | A clean checkout reproduces the smoke pipeline in CI on the public fixture; with the DVC remote, it reproduces headline metrics within the stated tolerance |
| CC8 | The 8 GB resource benchmark is recorded, with no out-of-memory failures in any stage |

### 11.2 Performance targets (preregistered; reported either way)

| ID | Target |
|---|---|
| PT1 | Federated DP model at ε ≤ 10: AUROC no more than 0.05 below centralised non-private, on each site |
| PT2 | Federated DP model at ε ≤ 10: AUROC at or above the mean local-only AUROC, on each site |
| PT3 | Site A event recall ≥ 0.70 on the 35 test events at window FPR ≤ 1%. Provisional: may be revised at G1 using validation-only baseline results on the 15 validation events, never afterwards. Site B event recall on the test episodes is reported, with no target, because there are only three |
| PT4 | Site B fusion: 2-of-3 fusion reduces false alarms per hour by at least 30% relative to the best single node at the same per-node threshold (H2a). The single-device recall loss on the real attack day (H2b) is reported with no target |
| PT5 | Site A attribution: tactic accuracy ≥ 0.70 on HAI test events and above the majority-class baseline; `full` mode ≥ `retrieval_only`, reported with CI. Site B: JSON validity ≥ 95% and raw grounding rate reported over the seven episodes; no accuracy target |
| PT6 | Retrieval Recall@5 ≥ 0.85 on the held-out query set |
| PT7 | Latency p95: detection ≤ 1 s; investigation ≤ 2 s; incident-to-dispatch ≤ 3 s in `retrieval_only` mode and ≤ L_LLM in `full` mode |

### 11.3 Engineering acceptance

| ID | Criterion |
|---|---|
| EA1 | Every session reaches a terminal state under the X6 fault scenarios; a replayed alert never dispatches twice |
| EA2 | mTLS is enforced across processes and plaintext is rejected; JWTs that are expired, have the wrong audience, are out of scope or are replayed are rejected; a non-allowlisted LLM endpoint is refused; the secret scan is clean |
| EA3 | Each session's trace can be recovered from Phoenix by `session_id`, and each action links to its trace |
| EA4 | No high-impact action executes without approval in any test |
| EA5 | CI passes: Ruff, mypy, pytest (≥ 80% coverage on core packages), gitleaks, fixture smoke reproduction |
| EA6 | Docker Compose runs the smoke pipeline end to end |

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation | Decision point |
|---|---|---|---|---|
| DP federated utility far below centralised | Medium | High | Tune clipping norm, model size and local epochs on validation; report the full privacy–utility curve honestly | G2 |
| Site B device captures do not overlap in time | Resolved | — | Verified in the data analysis: the three captures overlap on every day | G0 (closed) |
| Only one Site B attacker position (external) is available; the attack reaches one device | Certain | High (RQ2) | RQ2 reframed as a fusion trade-off (H2a, H2b); labelled synthetic multi-device scenario as H2c; obtaining the compromised-IED and compromised-HMI days remains optional and would be added as real data | G0 |
| Few Site B attack events (seven episodes, 51 windows) | Certain | Medium | Site B detection reported per episode without a target; Site A carries the quantitative detection and attribution results | G1 |
| No published MITRE mapping for CIC Modbus attacks | Certain | Medium | Site B attribution becomes a case study; Site A tactic level is the primary result; mapping protocol with two review passes | G0, G1 |
| Site A DP noise too high at ε = 1 (σ ≈ 8.4) | High | Medium | Segment-mean input (336 inputs); batch 256; report the full ε curve | G1, G2 |
| Normal latency stalls on the secure IEDs cause false alarms | High | Medium | Robust per-node latency features; latency excluded from fusion | W2, W8 |
| LLM too slow, or JSON validity rate too low | High | Medium | Schema-constrained decoding; smaller or alternative model; `retrieval_only` as the default mode with the LLM for explanation only | G0, G3 |
| Out of memory on 8 GB | Medium | High | Stage isolation; Ray concurrency limits; 2,048-token context; swap; measured budget | W1, W14 |
| Flower or Ray unstable on WSL2 | Low–Medium | Medium | Pinned versions; run clients sequentially as a fallback | W1 |
| SecAgg+ integration problems | Medium | Low | Kept as a separate systems experiment; documented limitation | G2 |
| Modbus label derivation errors | Medium | High | Labels from the attacker's address; written protocol; automated check that the label counts match `configs/data_expectations.yaml` (51 attack windows on IED1A, none elsewhere); manual spot-check of every episode boundary | W2 |
| HAI operating regimes not separable | Occurred | Medium | Time partition adopted at G0 (HAI-3) | G0 (closed) |
| Experiment compute exceeds the schedule | Medium | Medium | Per-run time measured in Week 6; three seeds for non-headline configurations; overnight runs | G2 |
| Schedule slip | Medium | High | Decision gates, a buffer in Week 14, and a predefined cut order: X8 SecAgg+ → systems-heterogeneity arm of X2 → `llm_no_retrieval` ablation → Docker full profile (smoke profile is always kept) | Every gate |

---

## 13. Limitations and Future Work

**Limitations**

- Evidence is fused only *within* a site; attack campaigns spanning sites are not tested, because no public dataset links them.
- Site B has one real attacker position, one attack day and seven episodes. Its detection and attribution results are indicative, and the multi-device fusion benefit is shown only on synthetic data.
- Site B labels are derived from the attacker's address, not from the dataset's attack logs.
- Evidence records are minimised but carry no formal DP guarantee.
- No defence against poisoning by malicious clients.
- With 3–4 clients per federation, secure aggregation provides limited anonymity.
- Both datasets come from testbeds, not production plants.
- LLM explanations are grounded and validated, but their faithfulness is not formally guaranteed.
- Everything runs on one machine, so network effects are simulated.

**Future work**

- DP for evidence records and federated evaluation.
- Robust aggregation against poisoning.
- Cross-site correlation, if a linked dataset becomes available.
- A production profile replacing local components: NATS or Kafka for the SQLite bus, PostgreSQL for SQLite, a Qdrant server for embedded Qdrant, Flower's deployment runtime for simulation, OpenTelemetry and Prometheus for JSONL events, and Kubernetes for Docker Compose. The local and production profiles would share message schemas and evaluation contracts.
- Evaluation in other industrial domains.

---

## 14. Conclusion

Version 3.0 keeps the ambition of the original proposal — private learning across industrial nodes, combined with explainable, knowledge-grounded attribution — and makes every part of it testable.

- Federations are built on horizontally partitioned data that can actually be aggregated.
- The privacy guarantee has a defined unit, a single persistent budget and an explicit threat model.
- Nodes share minimised evidence rather than invertible embeddings.
- Attribution is grounded in the knowledge base and validated, and runs in one LLM call.
- The evaluation is designed before the models, with sealed test data, event-level metrics, baselines, ablations, multiple seeds and preregistered targets.

The result is a project whose claims, positive or negative, can be defended and reproduced on an 8 GB machine.

---

## References

- Abadi, M. et al. (2016). Deep Learning with Differential Privacy. *ACM CCS*.
- Bell, J. et al. (2020). Secure Single-Server Aggregation with (Poly)Logarithmic Overhead. *ACM CCS*.
- Beutel, D. et al. (2020). Flower: A Friendly Federated Learning Research Framework. *arXiv:2007.14390*.
- Bonawitz, K. et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. *ACM CCS*.
- Canadian Institute for Cybersecurity. CIC Modbus Dataset 2023. University of New Brunswick.
- Kim, S. et al. (2022). Towards a Rigorous Evaluation of Time-Series Anomaly Detection. *AAAI*.
- Li, T. et al. (2020). Federated Optimization in Heterogeneous Networks (FedProx). *MLSys*.
- McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*.
- Mironov, I. (2017). Rényi Differential Privacy. *IEEE CSF*.
- MITRE. ATT&CK for Industrial Control Systems.
- NIST SP 800-82 Rev. 3 (2023). Guide to Operational Technology (OT) Security.
- Shin, H.-K. et al. HAI: HIL-based Augmented ICS Security Dataset. *USENIX CSET*.
- Tatbul, N. et al. (2018). Precision and Recall for Time Series. *NeurIPS*.
- Yousefpour, A. et al. (2021). Opacus: User-Friendly Differential Privacy Library in PyTorch. *arXiv:2109.12298*.

---

## Appendix A — Issue-to-Resolution Traceability

Severity: **C** = critical, **M** = major, **m** = minor. "Plan" refers to weeks (W) and gates (G) in Implementation Plan v3.0.

### A.1 Research framing and scope

| ID | Sev | Issue in v2.1 | Resolution in v3.0 | Where |
|---|---|---|---|---|
| R1 | C | Cross-node attack claim untestable across HAI and Modbus | Two independent sites; fusion only within Site B; overlap verified at G0 with synthetic fallback | §4.1, §12; Plan W1/G0, W8, W13 |
| R2 | C | No research questions or hypotheses | RQ1–RQ3 with falsifiable hypotheses | §2 |
| R3 | M | Scope too broad | Research core separated from engineering support; cut order defined | §3, §12 |
| R4 | M | Compliance and impact overclaims | Claims removed; precise DP statement | §1, §5.5 |
| R5 | M | Continuous learning conflicts with fixed ε | Periodic retraining with a budget per period | §7.7 |
| R6 | m | Industry table untested | Moved to future work | §3, §13 |
| R7 | m | Module/Phase naming inconsistent | Single three-module structure | §4.2 |

### A.2 Data and partitioning

| ID | Sev | Issue in v2.1 | Resolution in v3.0 | Where |
|---|---|---|---|---|
| D1 | C | Subsystem split is vertical; VAEs cannot be averaged | Horizontal split with a shared schema (by recording period since v3.1; see HAI-3) | §4.1; Plan W2 |
| D2 | C | No validation split | Temporal train/val/test with gaps; segment-level attack validation | §6.2; Plan W2 |
| D3 | M | Window leakage; boundary labels undefined | Split before windowing; explicit labelling rule | §6.2; Plan W2 |
| D4 | M | Per-node normalisation | Noisy federated normalisation statistics | §6.3; Plan W6 |
| D5 | M | Modbus label derivation unspecified | Written labelling protocol plus checks | §6.1; Plan W2 |
| D6 | M | Subjective MITRE ground truth | Frozen mapping protocol; tactic level for Site A | §10.4; Plan W4/G1 |
| D7 | m | HAI release not pinned | Pinned release and checksums | §6.1; Plan W1 |
| D8 | m | "Faker" anomaly injection | Explicit labelled perturbations, robustness and fallback only | §6.1; Plan W13 |

### A.3 Detection models and evidence

| ID | Sev | Issue in v2.1 | Resolution in v3.0 | Where |
|---|---|---|---|---|
| M1 | C | Isolation Forest nodes cannot federate | Federable autoencoder for Site B; Isolation Forest as baseline | §6.4; Plan W4 |
| M2 | C | Embeddings from incomparable spaces | Structured evidence records replace embeddings | §6.6; Plan W8 |
| M3 | C | Embeddings invertible; not private | Minimised evidence; residual risk stated | §5.4; Plan W8 |
| M4 | M | 95th-percentile threshold fixes FPR | FPR-targeted thresholds on validation | §6.5, §10.3; Plan W3 |
| M5 | M | No detection baselines | PCA, Isolation Forest, local-only, centralised | §10.5; Plan W4 |
| M6 | m | Aggregate score only | Per-feature error vector | §6.5; Plan W4 |
| M7 | m | Input noise on Isolation Forest is not DP | Removed | §6.4 |

### A.4 Federated learning

| ID | Sev | Issue in v2.1 | Resolution in v3.0 | Where |
|---|---|---|---|---|
| F1 | C | Only 3 real clients; straggler test impossible | Fed-A 4 clients (min 3); Fed-B 3 clients (min 2) | §7.1; Plan W6 |
| F2 | C | Server evaluates on centralised test data | Federated evaluation; oracle only as baseline | §7.5; Plan W6 |
| F3 | M | Simulation clients are Ray processes, not coroutines | Concurrency limits; measured memory | §9.4; Plan W1, W6 |
| F4 | M | SecAgg wiring and API mismatch | SecAgg+ systems experiment through the pinned API | §7.4; Plan W7 |
| F5 | M | FedProx > FedAvg as pass/fail | Hypothesis H1b with seeds and heterogeneity arms | §2, §7.2; Plan W12 |
| F6 | M | Circular convergence target | Rounds-to-target against centralised baseline | §7.6; Plan W6 |
| F7 | M | "Global beats local" untested; vague knowledge sharing | Local-only baseline; vague mechanism dropped | §10.5; Plan W4 |
| F8 | m | Warm-up not accounted | Included in budget calibration | §7.3; Plan W5 |

### A.5 Privacy

| ID | Sev | Issue in v2.1 | Resolution in v3.0 | Where |
|---|---|---|---|---|
| P1 | C | Per-round vs total ε inconsistent | One total ε per client; up-front calibration | §7.3; Plan W5 |
| P2 | C | Accounting reset breaks DP | Budget guard; no reset; persisted accountant | §7.3; Plan W5, W7 |
| P3 | C | No threat model | Adversaries ADV-1 to ADV-5; mechanism table | §5; Plan W1 |
| P4 | M | Privacy unit undefined; overlapping windows | Non-overlapping training windows define the unit | §7.3; Plan W2, W5 |
| P5 | M | δ not tied to dataset size | δ = min(10⁻⁵, 1/(10·N_c)) | §7.3; Plan W5 |
| P6 | M | ε = 10 justified by appeal to industry | ε sweep and privacy–utility curve | §10.5; Plan W12 |

### A.6 Agentic layer

| ID | Sev | Issue in v2.1 | Resolution in v3.0 | Where |
|---|---|---|---|---|
| A1 | C | 128-dim vs 384-dim retrieval mismatch | Evidence rendered as text; same encoder as knowledge base | §8.4; Plan W9 |
| A2 | C | Latency targets ignore CPU LLM speed | Segmented budget; L_LLM measured at G0; 3 s kept for LLM-free path | §8.6; Plan W1/G0, W13 |
| A3 | M | FL evaluation used as alert trigger | Edge runtime raises alerts | §9.3; Plan W8 |
| A4 | M | LLM used where rules suffice; no LLM-free baseline | Deterministic Triage and Investigator; ablation modes | §8.2; Plan W10, W13 |
| A5 | M | LLM output not validated | Schema-constrained, ID-validated output with fallback | §8.4; Plan W10 |
| A6 | M | Autonomous isolation unsafe | Approval for high-impact actions | §8.5; Plan W10 |
| A7 | M | Manual review of attribution | Automated scoring against frozen ground truth | §10.4; Plan W4, W13 |
| A8 | M | Queries to unrelated-dataset nodes | Same-site investigation only | §8.2; Plan W10 |
| A9 | M | No durable sessions or idempotency | Session table, persisted hops, idempotency key | §8.3; Plan W8, W10 |
| A10 | m | Arbitrary knowledge-base size target | Retrieval quality on held-out queries | §10.4; Plan W9 |

### A.7 Infrastructure, security and resources

| ID | Sev | Issue in v2.1 | Resolution in v3.0 | Where |
|---|---|---|---|---|
| I1 | M | asyncio.Queue described as at-least-once | SQLite outbox and inbox with leases and acknowledgements | §9.2; Plan W8 |
| I2 | M | In-process mTLS and JWT meaningless | Separate processes per node and mesh | §9.2; Plan W11 |
| I3 | M | No RAM budget | Budget table; stage isolation; X7 benchmark | §9.4; Plan W1, W14 |
| I4 | M | OS assumption unclear; Ray on Windows | WSL2 configuration; Docker Engine in WSL2 | §9.4; Plan W1 |
| I5 | M | RAGAS defaults to an external LLM | Deterministic retrieval metrics; no external judge | §10.4; Plan W9 |
| I6 | m | python-jose; replay not prevented | PyJWT with aud, exp, scope and one-time `jti` | §5.2; Plan W11 |
| I7 | m | Toolchain drift | Ruff and mypy only | §9.6; Plan W1 |
| I8 | m | Docker never scheduled | Docker Compose in Week 15 | Plan W15 |

### A.8 Evaluation and success criteria

| ID | Sev | Issue in v2.1 | Resolution in v3.0 | Where |
|---|---|---|---|---|
| E1 | C | Criterion 2 contradictory | Split into H1a and H1b | §2, §11 |
| E2 | C | Evaluation built last | Harness and baselines built in Weeks 3–4 | Plan W3, W4 |
| E3 | M | Window-level, point-adjust inflation | Event-level, range-based, time-to-detect | §10.2; Plan W3 |
| E4 | M | Detection depends on the agent pipeline | Detection and investigation metrics separated | §10, §11 |
| E5 | M | Fixed targets without baselines | Relative targets preregistered at G1 | §11.2; Plan G1 |
| E6 | M | Retrieval metric mismatch between documents | Recall@5, MRR and grounding rate | §10.4 |
| E7 | M | Ablations and seeds unscheduled | X1–X8 matrix with 5 seeds | §10.6; Plan W12–W14 |
| E8 | M | Research and engineering criteria mixed | Three tiers | §11 |
| E9 | M | Fixture data can pass gates | `data_source` provenance; checker rejects fixtures | §9.5; Plan W3 |

### A.9 Plan structure

| ID | Sev | Issue in v2.1 | Resolution in v3.0 | Where |
|---|---|---|---|---|
| S1 | M | Infrastructure front-loaded | Just-in-time infrastructure | Plan §2 |
| S2 | M | Unrealistic hard weekly gates | Decision gates G0–G4 with fallbacks | Plan §7 |
| S3 | M | No buffer or writing time | Buffer in W14; writing in W16 | Plan W14, W16 |
| S4 | m | Inconsistent counts and targets | Single source of truth in `params.yaml` | Plan §6 |
| S5 | m | Results produced in notebooks | Scripts write artifacts; notebooks for exploration only | Plan §2 |

*— End of Proposal —*

**Allaudin Ansari** · October 2026

---

## Appendix B — Data Findings to Decisions (v3.1)

Findings come from the full analysis of HAI 21.03 and of the CIC Modbus 2023 attack and benign captures. Each one is checked on the laptop by `scripts/inspect_data.py --expectations configs/data_expectations.yaml`.

| ID | Finding | Decision | Where |
|---|---|---|---|
| HAI-1 | `attack` is the overall label; `attack_P1`–`attack_P3` mark targeted processes | `attack` is the label; process columns feed tactic ground truth only | §6.1, §10.4; `data.hai.label_column` |
| HAI-2 | 22 of 79 features are constant in training; `P4_HT_PS` duplicates `P4_ST_PS` | Model 56 features; separate rule for changes in training-constant variables | §6.1; `drop_train_constant_features`, `drop_features` |
| HAI-3 | Only one weak regime; the main split follows file boundaries | Partition Site A by recording period; IID variant for X2 | §4.1; `partition.site_a.scheme: time` |
| HAI-4 | Test values exceed training ranges; some features have zero variance | Standard scaling with ±6 SD clipping | §6.3; `data.hai.scaling` |
| HAI-5 | 50 events, smallest gap 24 min | Validation/test split by event segment, about 15/35 | §6.2 |
| HAI-6 | A flattened window has more inputs than a client has training windows | Six 10 s segment means per feature (336 inputs) | §6.4; `models.vae.input_mode` |
| HAI-7 | About 3,000–3,400 windows per client; σ ≈ 8.4 at ε = 1 | Keep the ε sweep; batch 256; report the curve | §6.4, §7.3 |
| MB-1 | Attack logs and other attacker positions are not in the obtained data | Labels from attacker address; optional extra data recorded as a risk | §6.1, §12 |
| MB-2 | Capture-level labels would mark all 3,681 IED1A attack-day windows as attacks; only 51 contain attacker traffic | Window-level labels, marked derived | §6.1; `data.modbus.label_source` |
| MB-3 | Only IED1A saw the external attacker | RQ2 reframed as a trade-off | §2 |
| MB-4 | The attacks changed function-code mix, addresses, quantities, peers and frames | Feature schema built around those | §6.6 |
| MB-5 | Attack-day quiet hours are not proven normal for IED1A | Train on benign days only | §6.2 |
| MB-6 | 51 of 3,681 attack-day windows on IED1A have attacker IPv4 traffic, in seven episodes | IED1A carries Site B detection; per-episode reporting | §6.1, §11.2 |
| MB-7 | Long idle gaps with no polling (up to 16 h) | Drop idle periods over 60 s before windowing | §6.2; `data.modbus.idle_gap_s` |
| MB-8 | Benign data ends 11 h before the attack day | Benign days for training and normal validation; attack day for attack validation and test | §6.2 |
| MB-9 | Unseen address, new peer and oversized quantity separate every episode | Learn address and peer sets per node | §6.6 |
| MB-10 | Secure IEDs stall for tens of seconds daily; the insecure one never does | Robust per-node latency features | §6.6 |
| MB-11 | IED1B and HMI saw no attacker traffic | All their attack-day windows are normal | §6.1 |
| MB-12 | Normal write bursts rarely coincide across devices | 2-of-3 fusion for false-alarm reduction (H2a) | §2, §10.6 |
| MB-13 | The secure IEDs stall together on normal days | Latency excluded from fusion | §8.2; `fusion.site_b.exclude_latency` |
| MB-14 | The HMI is a client; the IEDs are servers | Features from each node's own side; one schema | §4.1, §6.6 |

The window figure in MB-6 counts IPv4 traffic only. An earlier count of 80 windows also included 29 windows containing only ARP frames from the attacker, which carry no Modbus content.
