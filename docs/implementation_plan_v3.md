**IMPLEMENTATION PLAN**

# Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

*Week-by-week engineering guide — 16 weeks*

| | |
|---|---|
| **Author** | Allaudin Ansari |
| **Version** | 3.1 — Data-informed revision of v3.0 (supersedes v2.1) |
| **Date** | October 2026 |
| **Companion document** | Project Proposal v3.1 (cited below as "Proposal §x") |
| **Language** | Python 3.12 (3.11 fallback) |
| **Target machine** | 8 GB RAM; Windows 11 with WSL2 (Ubuntu 24.04), or native Linux |

---

## Revision Note

**v3.1.** The plan follows the data analysis recorded in Proposal Appendix B. Weeks 1, 2, 4, 8, 9 and 13 change; the order of work and the gates do not.
- **Week 1:** data inspection now checks a local copy against `configs/data_expectations.yaml` (21 checks) and reports the Gate G0 data hints.
- **Week 2:** Site A uses the time partition; Modbus captures are parsed by a built-in memory-mapped reader (no `tshark`); labels come from the attacker's address; Site B trains on benign days and evaluates on the attack day; idle periods are cut.
- **Week 4:** the VAE takes six segment means per feature; Site A ground truth is tactic-level for the 50 HAI events; Site B mapping covers the seven episodes as a case study.
- **Week 8:** the evidence record gains the attack-driven protocol fields; latency is excluded from fusion.
- **Week 9:** the evidence-to-text renderer covers the new fields.
- **Week 13:** X3 measures the fusion trade-off (H2a–H2c); X4 scores Site A tactics and reports the Site B case study.

**v3.0.** This plan was rebuilt from scratch to implement Proposal v3.0. Three things changed compared with v2.1.

**Order of work**
- Evaluation and baselines are built *before* the federated models.
- Infrastructure is added only in the week that first needs it.
- Weekly metric targets are replaced by decision gates that have defined fallbacks.

**Design**
- Horizontal partitions and two federations.
- A single persistent privacy budget per client.
- Evidence records instead of embeddings.
- A grounded attribution step that makes one LLM call.
- Cooperating local processes with a SQLite-backed bus.

**Traceability**
- Each week ends with a **Resolves** line listing the flaw IDs from Proposal Appendix A that it closes.

---

## 1. How to Read This Plan

Every week has five parts:

- **Objective** — what must work by the end of the week.
- **Tasks** — the ordered implementation steps.
- **Deliverables** — the files, modules or artifacts produced.
- **Validation** — commands or tests that confirm the week is complete.
- **Resolves** — flaw IDs from Proposal Appendix A.

Commands use `make` targets defined in the `Makefile` (§5). File paths are relative to the repository root. Code snippets show structure only.

---

## 2. Working Rules

These rules apply in every week. They are what prevent the problems that affected v2.1.

| Rule | Statement | Prevents |
|---|---|---|
| **WR-1 Evaluation first** | The metric library, threshold policy, leakage guard and result-artifact schema exist before any model is trained | E2 |
| **WR-2 Sealed test set** | Test data is reachable only through `--final` entry points, every access is logged, and final runs happen only after Gate G4 freezes configurations | D2, CC2 |
| **WR-3 Every number from an artifact** | Reported numbers come from scripts that write JSON artifacts with provenance. Notebooks are for exploration only. Figures are generated from artifacts by `make figures` | S5, E9 |
| **WR-4 Stage isolation** | Training and operations never run at the same time on the 8 GB machine | I3 |
| **WR-5 Just-in-time infrastructure** | Each service (Qdrant, Phoenix, MLflow UI, mTLS) is added in the week that first uses it | S1 |
| **WR-6 Gates, not hard weekly metrics** | Progress is judged at Gates G0–G4 (§7); metric thresholds are preregistered at G1 | S2 |
| **WR-7 One toolchain from Week 1** | Ruff (lint and format), mypy, pytest and gitleaks run in CI on every push | I7 |
| **WR-8 Single source of truth** | Counts, windows, targets and hyperparameters live only in `configs/params.yaml`; documents and code read from it | S4 |

---

## 3. Environment and Resources

### 3.1 WSL2 configuration (skip on native Linux)

Create `%UserProfile%\.wslconfig` on the Windows host:

```ini
[wsl2]
memory=6GB
processors=4
swap=4GB
```

Then:
- Run `wsl --shutdown` and reopen Ubuntu 24.04.
- Install Docker **Engine** inside WSL2. Do not use Docker Desktop, which runs a second virtual machine and uses extra memory.
- Install Ollama inside WSL2.
- Keep the repository on the Linux filesystem (`~/agentic-mesh`), not under `/mnt/c`, for I/O speed.

### 3.2 Ollama settings

| Setting | Value | Reason |
|---|---|---|
| `OLLAMA_NUM_PARALLEL` | 1 | One request at a time keeps memory predictable |
| `OLLAMA_MAX_LOADED_MODELS` | 1 | Prevents two models being resident at once |
| `num_ctx` | 2048 | Limits KV-cache memory |
| `temperature` | 0 | Deterministic attribution |
| `keep_alive` | 30m during runs | Avoids reload latency between incidents |
| Bind address | 127.0.0.1 | Loopback only, matching the egress allowlist |

### 3.3 Memory budget

The "Measured" column is filled in during Week 1 (spikes) and Week 14 (X7 benchmark). Training and operations never run together (WR-4).

| Stage | Components | Estimate | Measured (W1) | Measured (W14) |
|---|---|---|---|---|
| Training | Python, torch CPU, Ray with 2 client actors, MLflow local backend | 2.5–3.0 GB | 1.8 GB PSS (Ray benchmark, 4 clients, 2 CPUs each; MLflow not included) | |
| Operations | Edge runtime, 7 node APIs, mesh with MiniLM and Qdrant, Ollama with Llama 3.2 3B Q4, Phoenix | 4.5–5.0 GB | LLM alone: 2.2 GB model; system memory peaked near the limit during the benchmark | |
| Budget inside WSL2 | | ≤ 6.0 GB (swap 4 GB) | 5.9 GB visible, 4 GB swap | |

---

## 4. Dependency Policy

Exact versions are pinned in Week 1 with `uv lock`, producing `uv.lock`. `requirements.lock` is exported from it for Docker. No version numbers appear in this plan, because the lock file is the single source of truth.

| Component | Purpose | Compatibility check in Week 1 |
|---|---|---|
| torch (CPU wheel), numpy | Models | The torch build supports the locked numpy major version |
| opacus | DP-SGD and RDP accountant | Supports the locked torch; accountant state can be saved and restored |
| flwr[simulation] (Ray) | Federations | Ray wheels exist for Python 3.12 on Linux; check the recommended simulation API and the **SecAgg+ integration path** for this version |
| scikit-learn, pandas, pyarrow | Baselines, calibration, data | — |
| (none: built-in reader) | Modbus/TCP capture parsing: `feasibility/captures.py` walks pcap and pcapng records and `data/modbus.py` decodes IPv4, TCP and MBAP headers directly | All 37 captures (40 M packets) parse in about 4 minutes with under 0.5 GB of memory |
| langgraph | Agent state machine | Graph compiles; checkpointing is not required (state lives in SQLite) |
| sentence-transformers | `all-MiniLM-L6-v2`, 384 dimensions | Model downloads and encodes on CPU |
| qdrant-client | Embedded mode (`path=`) | Local mode creates a collection and runs search |
| fastapi, uvicorn, httpx | Node APIs and mesh client over mTLS | uvicorn accepts `ssl_cert_reqs` for client-certificate checks |
| cryptography, PyJWT | Development CA, certificates, tokens | — |
| arize-phoenix, openinference instrumentation for LangChain | Traces | A local trace appears |
| mlflow, dvc | Tracking and versioning | Local backend works without an always-on server |
| matplotlib | Figures from artifacts | — |
| pydantic v2 | Schemas and contracts | — |
| Dev: ruff, mypy, pytest, pytest-asyncio, pytest-cov, hypothesis, gitleaks (CI action) | Quality gates | — |

**Removed compared with v2.1:** python-jose (replaced by PyJWT), RAGAS (replaced by deterministic retrieval metrics), black and isort (replaced by Ruff).

---

## 5. Repository Structure

```
agentic-mesh/
├── Makefile                      # setup, data, baselines, train, kb, mesh, experiments, figures, check, reproduce-smoke
├── pyproject.toml / uv.lock / requirements.lock
├── dvc.yaml                      # prepare → partition → baselines → federated → evaluate
├── configs/
│   ├── params.yaml               # single source of truth (WR-8)
│   ├── experiments/              # X1–X8 configuration grids
│   └── policy/actions.yaml       # action → impact → execution
├── data/                         # DVC-tracked; never in Git
│   ├── raw/{hai,cic_modbus_2023}/
│   ├── interim/                  # parsed Modbus request/response tables, cleaned HAI tables
│   └── partitions/{site_a/{a1..a4},site_b/{b1..b3}}/{train,val_normal,val_attack,test}.parquet + manifest.json
├── fixtures/                     # small synthetic public fixture for CI (schema-identical)
├── src/agmesh/
│   ├── common/                   # config loader, pydantic schemas, provenance, hashing, events, LabelFreeFrame
│   ├── data/                     # hai.py, modbus.py, site_a.py, site_b.py, windows.py, events.py, manifest.py,
│   │                             # checks.py, access.py (load_partition), fixture.py, pipeline.py
│   ├── models/                   # vae.py, autoencoder.py, baselines.py, scoring.py
│   ├── privacy/                  # calibrate.py, accountant_store.py, budget_guard.py, noisy_stats.py
│   ├── federated/                # client.py, server.py, strategies.py, fed_eval.py, stragglers.py, secagg.py, run.py
│   ├── edge/                     # replay.py, runtime.py, evidence.py, evidence_store.py
│   ├── bus/                      # base.py, memory.py, sqlite_bus.py
│   ├── nodeapi/                  # app.py (FastAPI), deps.py (JWT + scope checks)
│   ├── kb/                       # stix_parser.py, chunking.py, build.py, retriever.py, render.py
│   ├── mesh/                     # state.py, graph.py, triage.py, fusion.py, investigator.py, breaker.py,
│   │                             # attribution.py, llm.py, dispatcher.py, sessions.py, runner.py, approvals.py
│   ├── security/                 # ca.py, tls.py, jwt.py, egress.py, redaction.py
│   └── observability/            # tracing.py, metrics.py
│   ├── evaluation/               # metrics.py, events.py, thresholds.py, calibration.py, stats.py, diagnostics.py,
│   │                             # sealed.py, results.py, checker.py; later harness_*.py, retrieval_eval.py,
│   │                             # latency.py, resources.py, figures.py
├── evaluation/ground_truth/      # mapping_site_a.json, mapping_site_b.json (frozen at G1)
├── experiments/                  # x1_utility.py … x8_secagg.py (thin runners over src/)
├── results/                      # artifacts (DVC-tracked), test-access log
├── tests/{unit,contract,integration,security,chaos}/
├── docker/                       # compose.yaml (profiles: smoke, full), Dockerfiles
├── docs/
│   ├── proposal_v3.md, implementation_plan_v3.md
│   ├── threat_model.md, data_card.md, labeling_protocol.md, mapping_protocol.md, preregistration.md
│   └── adr/                      # architecture decision records
└── scripts/                      # download_datasets.sh, verify_checksums.py, bench_llm.py, bench_ray.py, make_certs.py
```

---

## 6. Core Contracts

These schemas are defined as pydantic models in `src/agmesh/common/schemas.py`. Contract tests in `tests/contract/` enforce them.

### 6.1 `configs/params.yaml` (outline)

```yaml
project:
  seeds: [11, 23, 37, 41, 53]

data:
  hai:
    release: "21.03"
    sample_rate_hz: 1
    label_column: attack
    process_label_columns: [attack_P1, attack_P2, attack_P3]
    drop_train_constant_features: true
    drop_features: [P4_HT_PS]
    scaling: {method: standard, clip_sd: 6.0}
  modbus:
    devices: {ied1a: 185.175.0.4, ied1b: 185.175.0.5, scada-hmi: 185.175.0.3}
    attacker_ip: 185.175.0.7
    label_source: attacker_address
    idle_gap_s: 60
  windows:
    site_a: {length_s: 60, train_stride_s: 60, infer_stride_s: 10}
    site_b: {length_s: 10, train_stride_s: 10, infer_stride_s: 5}
  splits: {normal_val_fraction: 0.15, gap_windows: 1, attack_val_fraction: 0.30}
  partition:
    site_a:
      clients: [a1, a2, a3, a4]
      block_hours: 6
      scheme: time               # regime | iid | time  (G0 decision 2)
      period_assignment: {a1: train1, a2: train2, a3: "train3:first_half", a4: "train3:second_half"}
    site_b: {clients: {b1: ied1a, b2: ied1b, b3: scada-hmi}}

models:
  vae: {hidden: [256, 128], latent: 16, lr: 1.0e-3, batch: 256, input_mode: segment_means, segments_per_window: 6}
  ae:  {hidden: [64, 32], lr: 1.0e-3, batch: 256}

federated:
  strategy: fedprox            # fedavg | fedprox
  proximal_mu: 0.01            # grid: [0.001, 0.01, 0.1]
  max_rounds: 50
  local_epochs: 1              # heterogeneous arm: sampled from [1, 2, 3]
  dropout_prob: 0.0            # heterogeneous arm: 0.2
  min_fit_clients: {site_a: 3, site_b: 2}
  round_timeout_s: 120
  stopping: {patience: 5, min_rel_delta: 0.005}
  rounds_to_target_tolerance: 0.02
  client_resources: {num_cpus: 2}

privacy:
  target_epsilon: 10.0         # sweep: [1, 3, 10, .inf]
  delta: "min(1e-5, 1/(10*n_train_windows))"
  max_grad_norm: 1.0
  accountant: rdp
  normalisation_stats_epsilon: 0.1

evaluation:
  target_window_fpr: 0.01      # also report 0.005, 0.05
  event_tolerance_s: 60
  calibration: isotonic
  pooled_auroc_bins: 1000
  bootstrap_resamples: 2000

fusion:
  site_b: {delta_t_s: 60, method: noisy_or, k_of_n: 2, exclude_latency: true}

mesh:
  max_hops: 3
  connect_timeout_s: 0.2
  read_timeout_s: 0.5
  max_retries: 1
  investigation_deadline_s: 3.0
  breaker: {failure_threshold: 2, open_seconds: 30}
  attribution: {mode: full, top_k: 5, min_confidence: 0.5, llm_timeout_s: 21, l_llm_s: 24}   # G0 decision 7

llm:
  base_url: "http://127.0.0.1:11434"
  model: "llama3.2:3b"                        # G0 decision 6
  alternative_model: "phi3.5:3.8b-mini-instruct-q4_K_M"
  num_ctx: 2048
  temperature: 0

security:
  jwt: {issuer: mesh, ttl_s: 120}
  egress_allowlist: ["127.0.0.1", "::1", "localhost"]
```

### 6.2 Partition manifest (`manifest.json`)

The manifest records:
- `site_id`, `node_id` and the source file SHA-256 hashes;
- the HAI release or Modbus capture IDs;
- the partition scheme and seed;
- time ranges for each split and row and window counts;
- attack event IDs per split;
- schema version, feature list hash, and code git SHA.

### 6.3 Evidence record and alert

The evidence record follows Proposal §6.6 exactly. An alert carries:

```
alert_id, evidence_id, site_id, node_id, window_end, calibrated_prob, run_id, data_source
```

Contract test: outbound models allow **only** whitelisted fields. String fields are checked against enumerations or the feature-name list. Numeric fields are limited to `score`, `threshold` and `calibrated_prob`. Any raw sensor value or vector fails the test.

### 6.4 SQLite schema (`storage.db`)

```sql
CREATE TABLE node_registry (node_id TEXT PRIMARY KEY, site_id TEXT NOT NULL, model_type TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active', missed_rounds INTEGER NOT NULL DEFAULT 0);

CREATE TABLE sessions (session_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
  site_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('OPEN','INVESTIGATING','ATTRIBUTING','DISPATCHING',
         'COMPLETED','COMPLETED_DEGRADED','ABSTAINED','FAILED')),
  opened_at TEXT NOT NULL, closed_at TEXT, hop_count INTEGER NOT NULL DEFAULT 0,
  attribution_mode TEXT, trace_id TEXT, run_id TEXT NOT NULL);

CREATE TABLE hops (session_id TEXT NOT NULL, hop_index INTEGER NOT NULL, node_id TEXT NOT NULL,
  outcome TEXT NOT NULL, latency_ms REAL, started_at TEXT NOT NULL,
  PRIMARY KEY (session_id, hop_index));

CREATE TABLE actions (action_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, action_type TEXT NOT NULL,
  impact TEXT NOT NULL CHECK (impact IN ('low','high')),
  status TEXT NOT NULL CHECK (status IN ('dispatched','pending_approval','approved','rejected')),
  technique_id TEXT, tactic TEXT, confidence REAL, payload TEXT NOT NULL, trace_id TEXT,
  created_at TEXT NOT NULL, UNIQUE (session_id, action_type));

CREATE TABLE bus_messages (msg_id TEXT PRIMARY KEY, topic TEXT NOT NULL, payload TEXT NOT NULL,
  created_at TEXT NOT NULL, available_at TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
  lease_owner TEXT, lease_until TEXT, acked_at TEXT);
```

- The bus uses SQLite **WAL mode**.
- Consumers claim messages with a lease; a message is acknowledged only after its handler commits.
- Expired leases make the message available again (at-least-once delivery).
- Handlers are idempotent through `sessions.idempotency_key` and the `actions` UNIQUE constraint.

### 6.5 Result artifact (`results/<experiment>/<config_id>/seed_<n>.json`)

```json
{
  "schema_version": "1.0",
  "experiment": "X1",
  "config_id": "site_a__fedprox_mu0.01__eps10",
  "seed": 11,
  "split": "test",
  "data_source": "runtime",
  "provenance": {
    "git_sha": "…", "params_sha256": "…", "manifest_sha256": "…",
    "checkpoint_sha256": "…", "lockfile_sha256": "…",
    "python": "3.12.x", "started_at": "…", "finished_at": "…"
  },
  "privacy": {"a1": {"epsilon": 9.81, "delta": 1.0e-6, "noise_multiplier": 1.07,
                     "max_grad_norm": 1.0, "steps": 4200, "includes_stats_epsilon": true}},
  "federation": {"rounds": 37, "rounds_to_target": 21, "late_clients": 3, "dropped_clients": 1},
  "metrics": {"auroc": 0.0, "auprc": 0.0, "event_recall": 0.0, "range_f1": 0.0,
              "window_fpr": 0.0, "ttd_median_s": 0.0, "false_alarms_per_hour": 0.0,
              "pooled_auroc_bin_error": 0.0}
}
```

The checker (`evaluation/checker.py`) reads **only** these artifacts. It rejects `data_source != "runtime"`, missing provenance fields, and test-split artifacts that have no matching entry in `results/test_access.log`.

### 6.6 Bus topics

| Topic | Producer | Consumer | Payload |
|---|---|---|---|
| `alerts.site_a`, `alerts.site_b` | edge-runtime | mesh (Triage) | Alert |
| `actions.site_a`, `actions.site_b` | Dispatcher | operator log / node handler | Action |
| `approvals` | approval API | Dispatcher | Approval decision |

---

## 7. Decision Gates

| Gate | When | Inputs | Decisions | Fallback if inputs are poor |
|---|---|---|---|---|
| **G0 — Feasibility** | End of W1 | LLM benchmark, Ray memory spike, dataset inspection (with the 21 expectation checks) | HAI release; Site A partition scheme; RQ2 fusion data; Site B label source; Site B attribution ground truth; LLM model (Phi-3.5-mini or Llama 3.2 3B); L_LLM latency target; Ray concurrency. Data decisions 1–5 were taken from the v3.1 analysis and are confirmed by the laptop inspection | Expectation mismatch → resolve the download before Week 2. LLM too slow → `retrieval_only` becomes the default mode |
| **G1 — Evaluation frozen** | End of W4 | Harness, validation-only baselines, labelling and mapping protocols | Freeze the evaluation config, ground-truth mappings and fusion candidates; preregister PT1–PT7 in `docs/preregistration.md` | PT3 may be revised here, using validation baselines only, and never afterwards |
| **G2 — Federation sound** | End of W7 | Pilot federated runs on validation; privacy tests; per-run time | Headline ε candidates; μ grid; seeds per configuration (5, or 3 for non-headline); whether X8 stays in scope | Utility far below baseline → tune clipping norm, local epochs and model width on validation; report the curve honestly |
| **G3 — Mesh complete** | End of W11 | Integration, security and latency tests | Default attribution mode; final L_LLM | Latency or validity poor → `retrieval_only` default, with the LLM producing explanations after dispatch |
| **G4 — Results frozen** | End of W14 | All X1–X8 artifacts | Freeze configurations; run final test evaluations; after this, only writing and packaging | Cut order from Proposal §12 |

---

## 8. Week-by-Week Plan

### Phase 0 — Feasibility (Week 1)

---

#### Week 1 — Environment, feasibility spikes, data acquisition

**Objective:** a locked, reproducible environment with CI; both datasets downloaded and checksummed; the three feasibility questions answered with measurements; Gate G0 passed.

**Tasks**

1. **Environment.** Configure WSL2 (§3.1). Create the repository skeleton (§5) using a `src/` layout. Run `uv init` and add dependencies (§4). Run `uv lock` and export `requirements.lock`. Add a `Makefile` with `setup`, `lint`, `typecheck` and `test` targets.
2. **CI.** Add `.github/workflows/ci.yml` running Ruff, mypy, pytest and gitleaks on every push.
3. **Data.** `scripts/download_datasets.sh` fetches HAI (candidate release) and CIC Modbus 2023. `scripts/verify_checksums.py` records SHA-256 hashes. Run `dvc init`, track `data/raw/`, and document the DVC remote in `docs/data_card.md` (default: an external drive or the cloud storage of your choice).
4. **Spike: LLM.** `scripts/bench_llm.py` sends 30 representative attribution prompts (rendered-evidence text plus 5 candidate techniques, JSON schema output) to Phi-3.5-mini and to Llama 3.2 3B. It records tokens per second, p50 and p95 latency, JSON-schema validity rate, and resident memory (`ollama ps`, `/proc` RSS).
5. **Spike: Flower and Ray.** `scripts/bench_ray.py` runs 4 dummy torch clients for 3 rounds with `client_resources={"num_cpus": 2}`, and records peak memory and time per round.
6. **Spike: data inspection.**
   - HAI: feature list, sampling rate, number of attack events per test file, and candidate regime variables (setpoints, load).
   - Modbus: list device captures, check whether IED1A, IED1B and SCADA HMI captures **overlap in time**, and count attacker packets and 10 s windows per device.
   - Check whether the dataset documentation maps attacks to MITRE techniques.
   - Compare the local copy with `configs/data_expectations.yaml` (`--expectations`); all 21 checks should match. A mismatch means a different release or an incomplete download, and is resolved before Week 2.
7. **Documentation.** Draft `docs/threat_model.md` from Proposal §5.

**Deliverables:** locked environment; green CI; raw data with checksums under DVC; `results/feasibility/{llm_bench,ray_bench,data_inspection}.json`; draft threat model; filled "Measured (W1)" column in §3.3.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Environment consistent | `uv sync --locked && make lint typecheck test` | All pass |
| CI green | GitHub Actions | Green |
| Checksums recorded | `python scripts/verify_checksums.py --verify` | All files match |
| Feasibility artifacts | `ls results/feasibility/` | 3 JSON files |
| **Gate G0** | Review against §7 | Decisions written to `docs/adr/0001-g0-decisions.md` and `params.yaml` |

**Resolves:** D7, F3 (measured), I3 (baseline), I4, I7, P3 (draft), A2 (measured), R1 (overlap verified).

---

### Phase 1 — Data and Evaluation First (Weeks 2–4)

---

#### Week 2 — Data pipeline, labelling and partitions

**Objective:** both sites partitioned exactly as specified in Proposal §4.1 and §6.2, with manifests, the labelling protocol and a public synthetic fixture.

**Tasks**

1. **HAI loading** (`data/hai.py`): parse timestamps, handle missing values (forward-fill limited to 5 s, otherwise drop the row), take `attack` as the label, keep `attack_P1`–`attack_P3` aside for ground truth, and build the feature list from `params.yaml` (drop the 22 training-constant features and `P4_HT_PS`, leaving 56). Add the label-free rule that flags changes in training-constant variables.
2. **Splits** (`data/site_a.py`, `data/site_b.py`):
   - Site A: the last 15% of each client's recording period becomes normal validation, with a one-window gap.
   - Cut the HAI test files into one *event segment* per event, with boundaries halfway between neighbouring events. Assign about 15 of the 50 segments to attack validation, stratified by process combination (P1, P2, P1+P2, P1+P3, P2+P3, P3) and spread across the four clients; the rest are test.
   - Site B: benign days 22–31 January are training plus normal validation (last 15% of each node's benign time). The attack day is attack validation (episodes 0–3) and test (episodes 4–6), each with its surrounding normal windows.
   - **Split before windowing.**
3. **Site A partition** (`data/site_a.py`):
   - `scheme: time` (default, G0 decision 2): a1 = `train1`, a2 = `train2`, a3 and a4 = the two halves of `train3`, as set by `period_assignment`.
   - `scheme: iid` for X2: shuffled `block_hours` blocks, assigned in equal numbers with a fixed seed.
   - `scheme: regime` is kept for completeness but is not used; HAI 21.03 has no separable regimes (Proposal §4.1).
   - Assign test event segments to clients round-robin within each stratum, so each client gets attack segments.
4. **Modbus parsing and labels** (`data/modbus.py`, `data/site_b.py`):
   - Parse each capture through the memory-mapped reader in `feasibility/captures.py`, decoding headers directly: time, endpoints, ports, TCP flags, MBAP header (transaction ID, length), function code, address, quantity, exception flag, and stacked frames. Pair requests with responses to get latency.
   - Cut idle periods (no HMI polling for more than `idle_gap_s`) and record them in the manifest.
   - Learn per-node sets from training data only: peers, addresses read or written, maximum quantity per function code, latency p99.9.
   - Aggregate into 10 s windows with one schema, computed from the node's own side: requests sent and received per function code; writes per address and whether the address was seen in training; distinct and new peers; new TCP connections and resets; stacked frames and length mismatches; quantities above the training maximum; latency median and slow-response count.
   - Label each window on each node from the attacker's address (`label_source: attacker_address`), following `docs/labeling_protocol.md`. The protocol states that labels are derived, defines window membership (any IPv4 packet to or from the attacker), defines episodes (attacker activity split at gaps above `episode_gap_s` = 125 s), and requires the automatic check that IED1A has 51 attack windows (10 s windows on the epoch grid) in seven episodes and IED1B and the HMI have none. Spot-check every episode boundary by hand.
5. **Windowing** (`data/windows.py`): training windows are non-overlapping (this defines the privacy unit); inference windows use the configured stride. Site A windows are also reduced to six 10 s segment means per feature when `input_mode: segment_means`. A window is labelled attack if it overlaps any attack sample.
6. **Manifests and pipeline:** write manifests (§6.2). Add DVC stages `prepare` and `partition`.
7. **Fixture:** `python -m agmesh.data.fixture` (`make fixture`) generates a tiny synthetic dataset with the same schemas (HAI-like CSVs and small Modbus pcaps with an attacker address), so CI runs without raw data.

**Deliverables:** partitions and manifests for 7 nodes; `results/data/partition_check.json`; `docs/labeling_protocol.md`; fixture; DVC stages (`dvc.yaml`, `dvc.lock`).

**Validation**

| Check | Command | Expected |
|---|---|---|
| No window crosses a split; training windows never overlap | `pytest tests/unit/data/test_windows_events.py tests/unit/data/test_site_a.py` | Pass (includes a property test) |
| No event split across partitions or clients | `pytest tests/unit/data` | Pass |
| Manifests deterministic | `pytest tests/unit/data -k deterministic`; on real data, `make partition` twice and compare `manifest.json` hashes | Identical |
| Integrity and label checks on real data | `make check-data` (`configs/partition_expectations.yaml`) | All checks pass: 56 features, 50 events (15/35), 7 episodes (4/3), IED1A 51 attack windows, IED1B and HMI 0 |
| Process label columns never features | `make check-data` (`site_a.no_label_column_is_a_feature`) | Pass |
| Fixture pipeline in CI | CI job `fixture-smoke` (`make fixture-smoke`) | Pass |

**Resolves:** D1, D2, D3, D5, P4 (unit defined), R1.

---

#### Week 3 — Evaluation harness, built before any model

**Objective:** a tested metric library, threshold and calibration policy, leakage guard, artifact writer, sealed-test mechanism and checker skeleton — all working on the fixture.

**Tasks**

1. **Metrics** (`agmesh/evaluation/metrics.py`, `agmesh/evaluation/events.py`; numpy only, so no new dependency):
   - AUROC and AUPRC.
   - Event recall with tolerance; range-based precision, recall and F1.
   - Time-to-detect; window FPR; false alarms per hour.
   - Pooled AUROC from histograms, with the binning error.
2. **Thresholds** (`agmesh/evaluation/thresholds.py`): an FPR-targeted threshold computed from normal validation scores.
3. **Calibration** (`agmesh/evaluation/calibration.py`): isotonic regression by pool-adjacent-violators, and the reliability curve with expected calibration error.
4. **Statistics** (`agmesh/evaluation/stats.py`): bootstrap CIs, paired bootstrap, exact Wilcoxon signed-rank, Holm correction. With five seeds the smallest possible two-sided Wilcoxon p-value is 0.0625, so the paired bootstrap is the primary paired test and Wilcoxon is reported alongside it.
5. **Leakage guard** (`agmesh/common/frames.py`, `agmesh/data/access.py`): a read-only `LabelFreeFrame` that refuses label columns, returned by `load_partition` with the labels in a separate object; it is the only type scoring functions accept. A contract test fails if any partition exposes a label column or a feature that copies the label. `agmesh/evaluation/diagnostics.py` also reports *label proxies*: features computed from traffic that reproduce the label by construction (Site B `new_peers`; see Proposal Appendix B, MB-15).
6. **Sealed test** (`agmesh/evaluation/sealed.py`): `load_partition(..., "test")` needs a `FinalAccess`, created only by `--final` entry points; it requires the `g4-frozen` tag and appends every access to `results/test_access.log`.
7. **Result artifacts** (`agmesh/evaluation/results.py`): `write_result` validates and writes the §6.5 schema with provenance; metric events use `common/artifacts.write_event`, including `data_source`.
8. **Checker skeleton** (`agmesh/evaluation/checker.py`, `make check-results`): schema, `data_source = runtime`, provenance including the lock-file hash, a clean working tree, required detection metrics, and a matching access-log entry for every test-split result.
9. **Tests:** hand-computed examples, plus a test demonstrating that point-adjusted F1 inflates scores (documents why it is excluded).

**Deliverables:** the evaluation package with at least 90% test coverage; checker skeleton.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Metric unit tests | `pytest tests/unit/evaluation -q --cov=agmesh.evaluation` | Pass; coverage ≥ 90% |
| Leakage guard | `pytest tests/contract/test_label_free.py` | Pass |
| Checker rejects fixtures | `pytest tests/unit/evaluation/test_sealed_results_checker.py -k checker` | Fixture artifact rejected |
| Test access logged | `pytest tests/unit/evaluation/test_sealed_results_checker.py -k sealed` | Access without `--final` raises; without the tag raises; with both, logged |
| Label proxies reported | `make check-data` | Notes `new_peers` (identical to the label) on b1 |

**Resolves:** E2, E3, E4, E9, M4, S5; groundwork for CC1–CC3.

---

#### Week 4 — Models, non-private baselines, ground truth, preregistration

**Objective:** VAE and autoencoder implemented; local-only and centralised non-private baselines measured on **validation**; ground-truth mappings frozen; targets preregistered; Gate G1 passed.

**Tasks**

1. **Models:**
   - `models/vae.py`: a VAE over six 10 s segment means per feature (6 × 56 = 336 inputs; hidden 256 → 128, latent 16), Opacus-compatible (no batch normalisation). The flattened 3,360-input variant is trained once on validation as a comparison; the choice is recorded at G1 (`models.vae.input_mode`).
   - `models/autoencoder.py`: dense autoencoder (64 → 32).
   - `models/inputs.py`: model inputs from `load_partition` (label-free): Site A windows → segment means, Site B `log1p` and scaling; `exclude_features` per site; per-feature error vectors.
   - `models/nets.py`: training with early stopping on normal-validation loss, seeded initialisation, reconstruction scoring (window score plus per-input errors).
2. **Baselines** (`models/baselines.py`): PCA reconstruction error, Isolation Forest, and on Site B the new-peer rule (MB-15).
3. **Baseline runs:** `make baselines` (`agmesh.experiments.x1_baselines`) trains local-only and centralised non-private models for both sites on one seed, sets each node's threshold on its normal validation and evaluates on its attack validation (`agmesh.evaluation.detection`). About 3 minutes on a laptop CPU. Results and the G1 decisions they raise are in ADR 0002.
4. **Ground truth:** write `docs/mapping_protocol.md`. Build `evaluation/ground_truth/mapping_site_a.json` (each of the 50 HAI events → tactic set, from its targeted processes and deviating variables; technique set where the evidence supports one) and `mapping_site_b.json` (each of the seven Modbus episodes → technique and tactic, for the case study). Do two review passes on separate days, and log any disagreements.
5. **Preregistration** (`docs/preregistration.md`): the evaluation config hash, PT1–PT7 (PT3 may be revised using validation baselines only), primary metrics, and statistical tests.

**Deliverables:** model modules with tests; validation baseline artifacts; frozen mappings; preregistration.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Model unit tests | `pytest tests/unit/models -q` | Pass; per-feature errors have the correct shape |
| Opacus compatibility | `pytest tests/unit/models -k opacus` | Only `Linear`/`ReLU` modules (Opacus's own validator runs in Week 5, when Opacus is installed) |
| Baselines on validation | `make baselines && make check-results` | 14 artifacts in `results/X1/`; all accepted |
| **Gate G1** | Review against §7 | Mappings and preregistration committed and tagged `g1-frozen` |

**Resolves:** M1, M5, M6, M7, D6, A7, E5, F7 (baseline available).

---

### Phase 2 — Privacy-Preserving Federated Detection (Weeks 5–7)

---

#### Week 5 — Differential privacy and persistent accounting

**Objective:** local DP-SGD training with up-front calibration, a persistent accountant, a budget guard, and a correct δ rule — all proven by tests.

**Tasks**

1. **Calibration** (`privacy/calibrate.py`): compute the total steps (rounds × local epochs × steps per epoch, plus warm-up) and obtain the noise multiplier for the target (ε, δ) from Opacus's RDP accountant. Resolve δ from the `params.yaml` formula.
2. **Persistent accountant** (`privacy/accountant_store.py`): save and load the RDP accountant history per client (JSON under `results/privacy/<run_id>/<node>.json`).
3. **Budget guard** (`privacy/budget_guard.py`): before each step or epoch, project ε and refuse to train if it would exceed the target. Raise `BudgetExhausted`, which the client handles by declining `fit`. There is no reset path anywhere in the code.
4. **Noisy normalisation statistics** (`privacy/noisy_stats.py`): Gaussian mechanism on count, sum and sum of squares, with its ε recorded and added by basic composition.
5. **DP training wrapper:** Opacus `make_private` with the calibrated noise multiplier and Poisson sampling.
6. **Sanity check:** single-client DP training at ε ∈ {1, 3, 10} on validation.

**Deliverables:** privacy package; DP training wrapper; sanity artifacts.

**Validation**

| Check | Command | Expected |
|---|---|---|
| ε increases monotonically and persists across restart | `pytest tests/unit/privacy/test_accountant_store.py` | Pass |
| Guard blocks overspend | `pytest tests/unit/privacy/test_budget_guard.py` | `BudgetExhausted` raised; no reset reachable |
| Final ε ≤ target, including statistics | `pytest tests/unit/privacy/test_total_budget.py` | Pass |
| δ rule | `pytest tests/unit/privacy/test_delta.py` | δ < 1/N_c |

**Resolves:** P1, P2, P4, P5, F8.

---

#### Week 6 — Federations, federated evaluation, stragglers

**Objective:** Fed-A (4 clients) and Fed-B (3 clients) train with FedAvg or FedProx under DP, evaluate federatively, handle stragglers, and write complete artifacts.

**Tasks**

1. **Client** (`federated/client.py`): loads only its own partition. `fit` runs DP training through the budget guard, with the accountant persisted; `evaluate` returns aggregate counts and binned histograms.
2. **Server** (`federated/server.py`): has **no data loader imports**. Uses `strategies.py` (FedAvg, FedProx) and `fed_eval.py` (aggregates metrics, pooled AUROC).
3. **Normalisation round:** a federated round for noisy statistics, run before training.
4. **Stopping and convergence:** the patience rule; rounds-to-target computed against the centralised baseline's validation loss from Week 4.
5. **Stragglers** (`federated/stragglers.py`):
   - Round timeout and minimum client counts (Fed-A 3/4, Fed-B 2/3).
   - Dropout injection (`dropout_prob`) and heterogeneous local epochs for X2.
   - Mark a node `degraded` in `node_registry` after 3 consecutive misses.
6. **Runner** (`federated/run.py`): config → run → artifact. Uses the simulation API recommended for the pinned Flower version, with `client_resources` from G0. MLflow uses the local backend.
7. **Determinism:** fixed seeds for client selection, data order and noise sampling.

**Deliverables:** working federations for both sites; straggler handling; federated evaluation; per-run timing recorded.

**Validation**

| Check | Command | Expected |
|---|---|---|
| 5-round smoke test, both sites | `make train SITE=site_a ROUNDS=5 && make train SITE=site_b ROUNDS=5` | Artifacts written; loss decreases |
| Server holds no data | `pytest tests/contract/test_server_no_data.py` | Pass (import and filesystem-access check) |
| Straggler tolerance | `pytest tests/integration/test_stragglers.py` | Rounds complete with 3/4 (Fed-A) and 2/3 (Fed-B) clients |
| Federated evaluation vs centralised on fixture | `pytest tests/integration/test_fed_eval.py` | Per-client metrics identical; pooled AUROC within the reported bin error |
| Determinism | Two runs with the same seed | Metrics identical within tolerance |

**Resolves:** F1, F2, F3, F6, D4.

---

#### Week 7 — SecAgg+ systems experiment, federated pilot, Gate G2

**Objective:** SecAgg+ verified on Fed-A as a separate experiment; headline configuration piloted on validation; compute budget confirmed; Gate G2 passed.

**Tasks**

1. **SecAgg+** (`federated/secagg.py`): integrate through the pinned Flower version's SecAgg+ mechanism.
2. **X8** (`experiments/x8_secagg.py`): measures correctness against the plaintext aggregate (quantisation error), overhead (time per round, bytes) and dropout tolerance.
3. **Pilot:** FedAvg and FedProx at ε = 10, one seed, validation only, both sites.
4. **Compute plan:** estimate total CPU hours for X1 and X2 from the measured per-run time. If it exceeds about 60 hours, drop non-headline configurations to 3 seeds.

**Deliverables:** X8 artifact; pilot artifacts; `docs/adr/0002-g2-decisions.md`.

**Validation**

| Check | Command | Expected |
|---|---|---|
| SecAgg+ correctness | `pytest tests/integration/test_secagg.py` | Aggregate within quantisation tolerance |
| X8 artifact | `ls results/X8/` | Present |
| No privacy reset | `pytest tests/integration/test_budget_across_rounds.py` | ε continues across rounds; client stops at budget |
| **Gate G2** | Review against §7 | Decisions recorded |

**Resolves:** F4; P2 (verified end to end).

---

### Phase 3 — Operations Path and Agentic Attribution (Weeks 8–11)

---

#### Week 8 — Edge runtime, evidence records, SQLite bus, fusion

**Objective:** streaming detection with the frozen global model produces minimised evidence and durable alerts; Site B fusion is implemented as a pure, tested function.

**Tasks**

1. **Replay** (`edge/replay.py`): streams a node's partition in real time or at an accelerated rate, using validation data during development.
2. **Edge runtime** (`edge/runtime.py`): loads the frozen model with the node's calibrator and threshold, scores windows, and on threshold crossing builds an evidence record. Stores it in the per-node evidence store (`edge/evidence_store.py`, SQLite) and publishes an alert.
3. **Evidence builder** (`edge/evidence.py`): top-5 features by error, direction, magnitude buckets (p90/p99/p99.9 from validation), and the Modbus `protocol_summary` (Proposal §6.6), including `new_peer`, `unseen_address`, `oversized_quantity`, `malformed_frames` and `slow_responses`. Site A also emits a flag when a training-constant variable changes. Never includes raw values, addresses or peer IPs.
4. **Bus** (`bus/sqlite_bus.py`): WAL mode, leases, acknowledgements, retry on lease expiry. `bus/memory.py` for unit tests.
5. **Fusion** (`mesh/fusion.py`): noisy-OR and k-of-n over calibrated probabilities within Δt. When `exclude_latency` is set, an alert whose top features are latency-only does not count towards fusion and is attached as single-device context. Tune Δt and k on Site B validation data.
6. **Sessions** (`mesh/sessions.py`): the schema from §6.4 and idempotency-key creation.

**Deliverables:** edge runtime; evidence records; durable bus; fusion module; session store.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Evidence minimisation | `pytest tests/contract/test_evidence_schema.py` | Only allowlisted fields; property test finds no raw values |
| Bus at-least-once | `pytest tests/integration/test_bus_redelivery.py` | Consumer killed before ack → message redelivered; processed once thanks to idempotency |
| Replay → alert | `make edge SITE=site_b SPLIT=val` | Alerts in bus; evidence in node stores |
| Fusion | `pytest tests/unit/mesh/test_fusion.py` | Pass, including a test that simultaneous latency-only alerts on two IEDs do not fuse |
| Detection latency | `python -m evaluation.latency --segment detection --split val` | p95 recorded (target ≤ 1 s) |

**Resolves:** A3, M2, M3, I1, A9 (session store), R1.

---

#### Week 9 — Knowledge base, evidence-to-text, retrieval evaluation

**Objective:** the MITRE ATT&CK for ICS knowledge base is indexed in the same embedding space as the rendered evidence, and retrieval quality is measured on held-out queries.

**Tasks**

1. **STIX parsing** (`kb/stix_parser.py`): load the ICS domain bundle (pinned version with checksum) and extract techniques, tactics and mitigation relationships.
2. **Chunking** (`kb/chunking.py`): one chunk per technique (name, description, detection). NIST SP 800-82 Rev. 3 sections are chunked as supplementary context.
3. **Build** (`kb/build.py`): embed with `all-MiniLM-L6-v2` into embedded Qdrant collections `mitre_ics`, `nist_80082` and `incident_memory` (all 384-dimensional, cosine).
4. **Renderer** (`kb/render.py`): a deterministic template turning evidence records into text, covering every `protocol_summary` field (for example "coil writes >100 to an address unseen in training from a new peer; requested quantity above the training maximum") and, for Site A, the targeted process and deviating variable names.
5. **Retriever** (`kb/retriever.py`): top-k from `mitre_ics`, with NIST context retrieved separately.
6. **Retrieval evaluation set** (`evaluation/retrieval_eval.py`): queries rendered from **validation** incidents (the 15 Site A validation events and Site B episodes 0–3), plus 30 paraphrases written without looking at the knowledge-base chunks. Compute Recall@5 and MRR.
7. **Rule baseline:** write the rule table (`configs/policy/rule_baseline.yaml`), which maps evidence patterns to techniques. Freeze it before any test run.

**Deliverables:** built knowledge base (DVC-tracked); renderer; retriever; retrieval artifact; rule baseline.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Knowledge base built | `make kb` | Every ICS technique in the bundle is present (count matches the parser) |
| Same embedding space | `pytest tests/contract/test_embedding_dims.py` | Query and knowledge-base dimensions both 384 |
| Retrieval quality | `python -m evaluation.retrieval_eval --split val` | Recall@5 and MRR recorded (target Recall@5 ≥ 0.85) |
| No query/target overlap | `pytest tests/unit/test_retrieval_set.py` | Paraphrase queries share no 8-gram with target chunks |

**Resolves:** A1, A10, I5, E6.

---

#### Week 10 — Agent mesh

**Objective:** the full LangGraph mesh — Triage, Investigator, Attribution and Dispatcher — runs end to end against fake node APIs, with all four ablation modes.

**Tasks**

1. **Graph** (`mesh/state.py`, `mesh/graph.py`): state = session ID, site, incident, evidence list, hop count, candidates, attribution, actions, status. Every node persists to SQLite before returning.
2. **Triage** (`mesh/triage.py`): consumes alerts, groups them into incidents (site, Δt), computes the fused score, assigns severity, and opens an idempotent session.
3. **Investigator** (`mesh/investigator.py`, `mesh/breaker.py`):
   - Site B: query other same-site nodes for evidence in the incident window.
   - Site A: query the alerting node for adjacent-window evidence.
   - At most 3 hops; timeouts and a single retry for transient errors with backoff and jitter; per-node circuit breaker; 3 s overall deadline; `COMPLETED_DEGRADED` when it expires.
4. **Attribution** (`mesh/attribution.py`, `mesh/llm.py`):
   - Render, then retrieve the top 5.
   - One Ollama call with a JSON schema whose `technique_id` is an enumeration of the candidate IDs.
   - Validate: parses, ID among candidates, confidence ≥ 0.5.
   - Otherwise fall back to `retrieval_fallback`.
   - Mitigations come from STIX relationships.
   - Modes: `full`, `retrieval_only`, `llm_no_retrieval`, `rule_only`.
5. **Dispatcher** (`mesh/dispatcher.py`, `mesh/approvals.py`): action policy from `configs/policy/actions.yaml`. Low-impact actions are dispatched; high-impact actions are set to `pending_approval`. An authenticated approval endpoint handles decisions. Writes go through the outbox; dispatch is idempotent on (session, action type).
6. **Runner** (`mesh/runner.py`): bus consumer loop; the graph runs in a worker thread.
7. **Test double:** a fake node API that can fail before, during or after a hop, and can be slow.

**Deliverables:** complete mesh; ablation modes; integration tests.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Full graph on a synthetic incident | `pytest tests/integration/test_graph_e2e.py` | Terminal status; 4 agent steps persisted |
| Hop bound | Property test | `hop_count ≤ 3` always |
| Grounding | `pytest tests/unit/mesh/test_attribution_validation.py` | An out-of-candidate ID always triggers fallback |
| Idempotent replay | `pytest tests/integration/test_replay_idempotent.py` | Same alert twice → one session, one dispatch |
| Approval gate | `pytest tests/integration/test_approvals.py` | High-impact action never executes without approval |
| All four modes run | `make mesh MODE={full,retrieval_only,llm_no_retrieval,rule_only} SPLIT=val` | Sessions complete in each mode |

**Resolves:** A4, A5, A6, A8, A9.

---

#### Week 11 — Process boundaries, mTLS, JWT, tracing, Gate G3

**Objective:** the operations stack runs as separate processes with enforced mTLS and scoped JWTs; traces are correlated by session; the security negative tests pass; Gate G3 is passed.

**Tasks**

1. **Certificates** (`security/ca.py`, `scripts/make_certs.py`): a development CA and one certificate per process (mesh, each node API, the approval API).
2. **mTLS** (`security/tls.py`): uvicorn with client-certificate verification; the plaintext port is disabled; the httpx client uses its own certificate and verifies against the CA.
3. **JWT** (`security/jwt.py`, `nodeapi/deps.py`): PyJWT. One token per request with:
   - `iss = mesh`, `aud = node_id`, `sub = session_id`;
   - `scope = evidence:read`, plus `site` and `window_start`/`window_end`;
   - `exp` ≤ 120 s and a unique `jti`.

   The node API checks every claim and keeps a replay cache of `jti` values until they expire.
4. **Egress check** (`security/egress.py`): the mesh refuses to start if the LLM base URL is not on the allowlist.
5. **Redaction** (`security/redaction.py`): tokens, keys and authorisation headers are removed from logs.
6. **Tracing** (`observability/tracing.py`): Phoenix with OpenInference instrumentation; `session_id` and `run_id` as span attributes; one trace exported per session to `results/traces/`.
7. **Local stack** (`make stack`): starts the node APIs, edge runtime and mesh as separate processes, with Phoenix on demand.

**Deliverables:** secured multi-process stack; security test suite; trace export.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Security negative tests | `pytest tests/security -q` | Rejects plaintext, a wrong client certificate, expired, wrong-audience, out-of-scope and replayed JWTs, and a non-allowlisted LLM URL |
| Redaction | `pytest tests/security/test_redaction.py` | No secrets in logs |
| Trace recoverable | `python -m evaluation.checker --traces` | Every session ID has a trace |
| Multi-process end to end | `make stack && make mesh SPLIT=val` | Sessions complete across process boundaries |
| **Gate G3** | Review latency and validity against §7 | Default mode and L_LLM recorded in `docs/adr/0003-g3-decisions.md` |

**Resolves:** I2, I6; supports EA2 and EA3.

---

### Phase 4 — Experiments and Systems Evidence (Weeks 12–14)

---

#### Week 12 — X1 utility and X2 heterogeneity

**Objective:** RQ1 answered with complete, multi-seed artifacts.

**Tasks**

1. Run X1 across both sites, sequentially, with overnight batches. Configurations: local-only and centralised (ε = ∞ and the headline ε), plus FedAvg and FedProx across ε ∈ {1, 3, 10, ∞}.
2. Run X2:
   - IID vs non-IID (Site A).
   - Homogeneous vs systems-heterogeneous (local epochs from {1, 2, 3}, dropout 0.2).
   - FedAvg vs FedProx with μ ∈ {0.001, 0.01, 0.1}.
3. Tuning runs use validation data only. Final test evaluations of these configurations happen after G4 (W14) with `--final`.
4. Generate the privacy–utility curve and the convergence figures from artifacts (`make figures`).

**Deliverables:** X1 and X2 validation artifacts; figures; per-client privacy tables.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Matrix complete | `python -m evaluation.checker --experiments X1,X2 --split val` | Every configuration and seed present |
| Privacy compliance | Checker privacy rule | Every client ε ≤ target |
| Statistics | `python -m evaluation.stats --experiment X2` | Paired CIs for FedProx vs FedAvg in both arms |

**Resolves:** E7, F5, P6; supports PT1 and PT2.

---

#### Week 13 — X3 fusion, X4 attribution, X5 latency, X6 chaos

**Objective:** RQ2 and RQ3 answered; latency profiled; fault behaviour proven.

**Tasks**

1. **X3 fusion (Site B):** best single node vs noisy-OR vs 2-of-3 at the same per-node thresholds. Report (a) false alarms per hour on benign validation time (H2a), (b) event recall per episode on the real attack day (H2b), and (c) event recall and time-to-detect on the labelled synthetic multi-device scenario (H2c, marked synthetic), all with CIs. Build the synthetic scenario in `experiments/x3_synthetic.py` by injecting a labelled write burst into two devices' benign validation windows at the same time.
2. **X4 attribution:** the four modes on validation incidents. Site A: tactic accuracy, technique top-1 and top-3 where a technique set exists, abstention, raw grounding rate and the majority-class baseline. Site B: per-episode chosen technique, JSON validity and raw grounding rate (case study, no accuracy).
3. **X5 latency:** end to end in `full` and `retrieval_only` modes, reporting p50 and p95 per segment (detection, investigation, incident-to-dispatch).
4. **X6 chaos** (`tests/chaos/`):
   - Node fails before, during and after a hop.
   - A node API process is killed mid-session.
   - A slow node (above the read timeout).
   - The mesh is killed mid-session and restarted (the session resumes from SQLite).
   - Duplicate alert replay.
5. **Robustness:** an injected-perturbation suite (spike, drift, stuck-at, replay) labelled synthetic, used only for detection robustness tables.

**Deliverables:** X3, X4, X5 and X6 artifacts; chaos report.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Chaos scenarios | `pytest tests/chaos -q` | Every session terminal; zero duplicate dispatches; breaker transitions logged |
| Attribution artifacts | `python -m evaluation.checker --experiments X4` | All modes present; grounding rate after validation = 100% |
| Latency report | `python -m evaluation.latency --report` | p95 per segment recorded |

**Resolves:** R1, A2, A4 (ablation), A7, D8, E7; supports EA1 and PT4–PT7.

---

#### Week 14 — X7 resources, reproduction, final test runs, Gate G4 (includes buffer)

**Objective:** 8 GB evidence recorded; clean-checkout reproduction proven; configurations frozen; sealed test evaluations run once.

**Tasks**

1. **X7** (`evaluation/resources.py`): sample peak memory per process and stage (training, operations), plus WSL2 total memory, during representative runs. Fill in the "Measured (W14)" column in §3.3.
2. **Clean-checkout reproduction:** fresh clone → `uv sync --locked` → `dvc pull` → `make reproduce`. Headline validation metrics must match within the stated tolerance (ε exactly; AUROC ± 0.005 for the same seed).
3. **Gate G4:** freeze configurations (tag `g4-frozen`), then run `--final` test evaluations for every X1–X4 configuration and seed. Every access is logged.
4. **Buffer:** reruns or fixes. If time runs short, apply the cut order from Proposal §12.

**Deliverables:** X7 artifact; reproduction log; final test artifacts; `g4-frozen` tag.

**Validation**

| Check | Command | Expected |
|---|---|---|
| No out-of-memory errors | X7 artifact | Peak below budget in every stage |
| Reproduction | `make reproduce` on a fresh clone | Metrics within tolerance |
| Final test access | `results/test_access.log` | Exactly one entry per configuration and seed |
| Checker, research tiers | `python -m evaluation.checker --tier completion` | CC1–CC8 pass |

**Resolves:** I3, S3; supports CC7 and CC8.

---

### Phase 5 — Packaging and Reporting (Weeks 15–16)

---

#### Week 15 — Docker Compose, final CI, figures

**Objective:** the smoke pipeline runs under Docker Compose; CI is fully green; every figure is regenerated from artifacts.

**Tasks**

1. Write `docker/compose.yaml` with these services: `edge-runtime`, `node-api` (7 instances through the compose scale setting or separate service entries), `mesh`, `ollama`, and `phoenix` (optional).
   - The **smoke profile** uses the fixture and `retrieval_only` mode, so it needs no LLM.
   - The **full profile** uses real data and the LLM.
   - Memory limits per service follow §3.3.
2. Finalise CI: Ruff, mypy, pytest with coverage ≥ 80% on `src/agmesh`, gitleaks, the fixture reproduction job, and a Docker smoke-build job.
3. `make figures` regenerates every report figure from artifacts. Each figure's caption records the artifact hashes it came from.

**Deliverables:** Compose files; green CI; final figures.

**Validation**

| Check | Command | Expected |
|---|---|---|
| Compose smoke run | `docker compose --profile smoke up --abort-on-container-exit` | Smoke checker passes |
| CI | GitHub Actions | All jobs green |
| Figures traceable | `pytest tests/unit/test_figures_provenance.py` | Every figure lists its source artifacts |
| Engineering tier | `python -m evaluation.checker --tier engineering` | EA1–EA6 pass |

**Resolves:** I8; supports EA5 and EA6.

---

#### Week 16 — Report, documentation, handoff, release

**Objective:** a complete written report (in a paper-ready structure), final documentation and a tagged release.

**Tasks**

1. Write the report: introduction; research questions; threat model; method (Modules 1–3); evaluation protocol; results for X1–X8, with negative results reported; limitations; future work. Every number is cited by artifact.
2. Finalise `docs/threat_model.md`, `data_card.md`, `labeling_protocol.md`, `mapping_protocol.md`, and the ADRs.
3. Write `README.md`: overview, environment setup, `make setup`, `make reproduce-smoke`, the results table generated from artifacts, and the architecture diagram.
4. Write `HANDOFF.md`: how to run the smoke pipeline, train a federation, run an investigation, approve an action, add a node, known limitations, and the production path (§12).
5. Run the full checker one final time and tag `v3.0.0`.

**Deliverables:** final report; documentation set; README; HANDOFF; `v3.0.0` release.

**Validation**

| Check | Command | Expected |
|---|---|---|
| All tiers | `python -m evaluation.checker --all` | Completion and engineering tiers pass; performance targets reported (met or not) |
| Release | `git tag` | `v3.0.0` |

**Resolves:** S3; R3 (scope documented).

---

## 9. Experiment Runbook

The run counts below assume 5 seeds. Per-run time is measured in Week 6; if the X1+X2 total exceeds about 60 CPU-hours, non-headline configurations drop to 3 seeds (decided at G2).

| ID | Configurations | Runs (5 seeds) | Data | When |
|---|---|---|---|---|
| X1 | Per site: local-only × {∞, headline ε}; centralised × {∞, headline ε}; FedAvg and FedProx × {1, 3, 10, ∞} | 2 sites × 12 configs × 5 = 120 | Validation (W12); test (W14) | W12, W14 |
| X2 | Site A: IID vs non-IID × FedAvg/FedProx (4); heterogeneous arm: FedAvg + FedProx × 3 μ (4); headline ε | 8 × 5 = 40 | Validation; test | W12, W14 |
| X3 | Site B: single best, noisy-OR, 2-of-3 — benign false alarms, real attack day, synthetic multi-device scenario | 3 × 5 = 15 (+ synthetic) | Validation; test | W13, W14 |
| X4 | Four attribution modes: Site A tactic level (35 test events); Site B case study (seven episodes) | 4 × 5 = 20 | Validation; test | W13, W14 |
| X5 | Latency in `full` and `retrieval_only` modes, ≥ 100 incidents each | 2 | Validation | W13 |
| X6 | Six chaos scenarios × 20 sessions each | 6 | Fixture and validation | W13 |
| X7 | Resource benchmark: training and operations stages | 2 | Validation | W14 |
| X8 | SecAgg+ on vs off; dropout 0 / 1 client | 4 × 3 = 12 | Validation | W7 |

---

## 10. Testing Strategy

| Layer | Location | Purpose | Examples |
|---|---|---|---|
| Unit | `tests/unit/` | Pure logic | Metrics, thresholds, splits, windows, accountant, guard, fusion, breaker, attribution validation |
| Contract | `tests/contract/` | Enforce invariants across modules | Label-free scoring; evidence schema; server holds no data; embedding dimensions; outbound message allowlist |
| Integration | `tests/integration/` | Components working together | Federations with stragglers; federated evaluation; bus redelivery; full graph; replay idempotency; approvals; SecAgg+ |
| Security | `tests/security/` | Negative tests (EA2) | mTLS, JWT claims and replay, egress allowlist, redaction |
| Chaos | `tests/chaos/` | Fault behaviour (EA1) | Process kills, timeouts, slow nodes, duplicate alerts, mesh restart |
| Property | Hypothesis, across layers | Invariants under random input | `hop_count ≤ 3`; no raw values in evidence; ε monotonic |

CI runs the unit, contract, security and fixture-based integration tests on every push. Chaos tests and real-data integration tests run locally and are recorded as artifacts.

---

## 11. Deliverables Checklist

| Deliverable | Week |
|---|---|
| Locked environment, CI green, raw data checksummed under DVC | 1 |
| Feasibility artifacts (LLM, Ray, data) and G0 decisions | 1 |
| Partitions and manifests for 7 nodes; labelling protocol; public fixture | 2 |
| Evaluation package (≥ 90% coverage); leakage guard; sealed test; checker skeleton | 3 |
| VAE and autoencoder; PCA and Isolation Forest baselines; validation baseline artifacts | 4 |
| Frozen ground-truth mappings; preregistration; G1 tag | 4 |
| DP calibration, persistent accountant, budget guard, noisy statistics | 5 |
| Fed-A and Fed-B with federated evaluation and straggler handling | 6 |
| SecAgg+ systems experiment (X8); pilot; G2 decisions | 7 |
| Edge runtime, evidence records, SQLite bus, fusion, session store | 8 |
| MITRE ICS knowledge base, renderer, retriever, retrieval evaluation, rule baseline | 9 |
| Four-agent LangGraph mesh with ablation modes and the approval gate | 10 |
| Multi-process stack with mTLS, JWT, egress check, redaction, traces; G3 decisions | 11 |
| X1 and X2 artifacts and figures | 12 |
| X3–X6 artifacts; chaos report | 13 |
| X7 resource benchmark; clean-checkout reproduction; final test runs; G4 tag | 14 |
| Docker Compose (smoke and full profiles); final CI; figures from artifacts | 15 |
| Report, documentation set, README, HANDOFF, `v3.0.0` | 16 |

---

## 12. Production Path (future work)

The local and production profiles share schemas and evaluation contracts. Only the adapters change.

| Local component (this plan) | Production replacement | Interface kept stable |
|---|---|---|
| SQLite bus (outbox/inbox) | NATS JetStream or Kafka | `bus/base.py` publish, subscribe, ack |
| SQLite registry and sessions | PostgreSQL | Repository classes in `mesh/sessions.py` |
| Embedded Qdrant | Qdrant server or managed service | `kb/retriever.py` |
| Flower simulation | Flower deployment runtime on real nodes | Client and server apps unchanged |
| JSONL metrics and Phoenix | OpenTelemetry, Prometheus, Phoenix | `observability/metrics.py` event schema |
| Development CA | Organisational PKI or cert-manager | `security/tls.py` |
| Docker Compose | Kubernetes | Container images unchanged |

*— End of Implementation Plan —*

**Allaudin Ansari** · October 2026
