**IMPLEMENTATION PLAN**

**Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics**

*Week-by-Week Engineering Guide — 16 Weeks*

**Author:** Allaudin Ansari

**Language:** Python 3.12

**Version:** 2.1 (Local Development Scope — 8GB RAM Single-Machine Build)

**Date:** August 2026

---

**Note on this revision:** Version 1.0 specified a week-by-week guide built around a real multi-node K3s cluster, Kafka, NATS JetStream, PostgreSQL, and a full Prometheus/Grafana/OpenTelemetry observability stack — infrastructure that alone exceeds 8GB RAM before any model is loaded. Version 2.1 keeps the core algorithmic direction unchanged while aligning the implementation with the updated three-module research framing: Module 1 (federated edge detection), Module 2 (privacy-preserving federated learning), and Module 3 (knowledge-grounded agentic attribution). The revised roadmap replaces only the infrastructure components discussed and confirmed for this build: Flower simulation mode instead of K3s/Docker for local prototyping, an asyncio-based telemetry replay and message bus instead of Kafka/NATS, embedded Qdrant instead of a Qdrant service, SQLite instead of PostgreSQL, structured logging + Arize Phoenix instead of Prometheus/Grafana/OpenTelemetry, Helm dropped, Phi-3.5-mini (3.8B) instead of Phi-4 (14B), and the HAI + CIC Modbus 2023 datasets instead of SWAT + CICIDS. The implementation also now explicitly supports a Docker-based reproducibility layer so the final project is executable, portable, and reviewable beyond a single development machine.

**Table of Contents**

**Section 1 How to Read This Document**

**Section 2 Project Directory Structure**

**Section 3 Python 3.12 Compatibility & Package Versions**

**Section 4 Environment Setup Reference**

**Phase 1 Foundation (Weeks 1–3)**

**Phase 2 Local Intelligence (Weeks 4–6)**

**Phase 3 Federated Learning (Weeks 7–9)**

**Phase 4 Agentic Mesh (Weeks 10–13)**

**Phase 5 Evaluation & Documentation (Weeks 14–16)**

**Appendix A Full Deliverables Checklist**

**Appendix B Package Version Reference**

# Section 1 How to Read This Document

This document is the week-by-week engineering implementation guide for the Decentralized Agentic Mesh project, translated into the updated three-module research architecture. The system is still designed for a single 8GB RAM development machine during early prototyping, but it now also includes a Docker-based execution path for reproducibility, benchmarking, and presentation. The plan is explicit about the three research modules and the evidence required to support a publication-grade contribution. Each week follows the same format:

- **Objective** — what must be working by end of week
- **Key Tasks** — the ordered set of implementation actions for the week
- **Deliverables** — specific files, modules, or artifacts produced
- **Validation Checkpoint** — the command or test that confirms the week is complete

The document assumes a single developer working approximately 6–8 hours per day, 5 days per week. All commands target a Linux or WSL2 environment. Python 3.12 is the sole runtime. The project supports two execution modes: a lightweight local simulation for rapid development on an 8GB machine and a Docker-based reproducibility layer for reliable execution, benchmarking, and presentation. Docker is recommended for the final executable pipeline, while the local simulation remains useful for debugging and incremental iteration.

Where a file path is shown, it is relative to the project root. Where a code snippet is shown, it illustrates structure only.

# Section 2 Project Directory Structure

```
agentic-mesh/
├── .github/
│   └── workflows/
│       └── ci.yml                    # lint, test, build on every push
│
├── docker/
│   ├── docker-compose.yml            # reproducible runtime for all services
│   ├── Dockerfile.edge
│   ├── Dockerfile.fl
│   ├── Dockerfile.api
│   └── Dockerfile.agent
│
├── docs/
│   ├── proposal_v2_1.md
│   ├── implementation_plan_v2_1.md   # this document
│   └── architecture/                 # diagrams (Mermaid exports)
│
├── data/
│   ├── raw/                          # downloaded HAI and CIC Modbus 2023
│   ├── processed/                    # cleaned, windowed feature files (.parquet)
│   └── partitions/                   # per-node Non-IID splits
│       ├── node_01/ ... node_05/
│
├── edge/                             # code that runs on each simulated node
│   ├── __init__.py
│   ├── config.py                     # node_id, data_type, epsilon, delta
│   ├── telemetry_replay.py           # asyncio generator replaying dataset rows
│   ├── models/
│   │   ├── vae.py
│   │   ├── isolation_forest.py
│   │   └── model_selector.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── dp_trainer.py             # Opacus DP-SGD wrapper
│   ├── embedding/
│   │   └── insight_embedding.py
│   ├── bus.py                        # asyncio.Queue publisher/subscriber
│   └── main.py
│
├── federated/                        # Flower FL server, strategies, simulation
│   ├── server.py
│   ├── strategy.py                   # FedProx strategy
│   ├── secagg.py
│   ├── convergence.py
│   ├── client.py
│   ├── simulate.py                   # Flower simulation entrypoint (5 clients)
│   └── mlflow_callback.py
│
├── agents/                           # LangGraph agentic mesh
│   ├── graph.py
│   ├── state.py
│   ├── auth.py                       # JWT issuance + validation
│   ├── tracing.py                    # Arize Phoenix / OTel setup
│   ├── metrics.py                    # structured JSON/CSV metric logging
│   ├── triage/triage_agent.py
│   ├── investigator/
│   │   ├── investigator_agent.py
│   │   └── circuit_breaker.py
│   ├── security/
│   │   ├── security_agent.py
│   │   └── kb_builder.py
│   └── dispatcher/action_dispatcher.py
│
├── knowledge_base/
│   ├── mitre_ics_raw/
│   └── nist_800_82/
│
├── api/                              # FastAPI microservices (in-process)
│   ├── node_api.py
│   ├── fl_api.py
│   └── agent_api.py
│
├── evaluation/
│   ├── harness.py
│   ├── ragas_eval.py
│   └── chaos_test.py
│
├── storage/
│   └── sqlite_schema.py              # SQLite schema + connection helper
│
├── qdrant_data/                      # embedded Qdrant local storage (gitignored)
│
├── scripts/
│   ├── setup_env.sh
│   ├── download_datasets.sh          # HAI + CIC Modbus 2023 download with checksum
│   ├── partition_data.py
│   ├── build_kb.py
│   └── run_smoke_test.sh
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_prototyping.ipynb
│   └── 03_fl_convergence_analysis.ipynb
│
├── dvc.yaml
├── params.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

**Removed from v1.0:** `infra/docker/`, `infra/k3s/`, `infra/helm/`, `infra/kafka/`, `infra/certs/` (mTLS remains code-level, not a cert-manager directory). **Added:** `edge/telemetry_replay.py`, `edge/bus.py`, `storage/sqlite_schema.py`, `qdrant_data/`.

# Section 3 Python 3.12 Compatibility & Package Versions

Pin every version in `requirements.txt`. Packages removed from v1.0 are listed at the end of this section with their replacements.

| **Package** | **Version** | **Layer** | **Note** |
|---|---|---|---|
| torch | 2.3.1 | Edge / FL | CPU build. Native DP-SGD support via Opacus. |
| opacus | 1.4.1 | Edge / FL | DP-SGD. Requires torch >= 2.0. |
| scikit-learn | 1.5.1 | Edge | Isolation Forest. |
| numpy | 1.26.4 | All | Do not use 2.x — breaks torch 2.3. |
| pandas | 2.2.2 | All | |
| pyarrow | 16.1.0 | Data | Parquet I/O. |
| flwr | 1.9.0 | FL | Includes FedProx strategy and simulation runtime. |
| langchain | 0.2.11 | Agents | |
| langgraph | 0.1.19 | Agents | |
| langchain-community | 0.2.10 | Agents | Ollama integration included. |
| sentence-transformers | 3.0.1 | Agents / KB | |
| **qdrant-client** | 1.10.1 | Agents / Edge | **Used in embedded/local mode — no server required.** |
| fastapi | 0.111.1 | API | Run via `uvicorn` locally, no container needed. |
| uvicorn | 0.30.1 | API | |
| pydantic | 2.7.4 | All | |
| python-jose | 3.3.0 | Agents | JWT. |
| cryptography | 42.0.8 | Security | Code-level mTLS cert generation. |
| mlflow | 2.14.1 | MLOps | Local file-backed tracking server. |
| dvc | 3.51.2 | MLOps | |
| ragas | 0.1.14 | Evaluation | |
| pytest / pytest-asyncio | 8.2.2 / 0.23.7 | Dev | |
| python-dotenv | 1.0.1 | All | |
| loguru | 0.7.2 | All | Structured logging — replaces Prometheus instrumentation. |
| rich | 13.7.1 | Dev | |
| matplotlib | 3.9.0 | Evaluation | Post-hoc metric charting — replaces Grafana. |

**Removed from v1.0 (no longer needed):** `kafka-python`, `nats-py` (replaced by `edge/telemetry_replay.py` and `edge/bus.py`, both stdlib `asyncio`), `psycopg2-binary`, `pgvector` (replaced by stdlib `sqlite3`), `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc`, `prometheus-client` (replaced by `loguru` + `matplotlib`; Arize Phoenix instrumentation is retained via its own lightweight local client).

The single most important compatibility constraint remains numpy: PyTorch 2.3.x requires numpy < 2.0. Pin explicitly.

# Section 4 Environment Setup Reference

**Required host tools**

- Python 3.12 — verify: `python3.12 --version`
- Ollama — install via: `curl -fsSL https://ollama.com/install.sh | sh`
- Git and DVC — install dvc via pip alongside git
- MLflow tracking server — runs as a local process (`mlflow server --backend-store-uri sqlite:///mlflow.db ...`)

Docker is strongly recommended for the final executable project. A local simulation remains valid for development, but the reproducible benchmark and demo path should run through Docker Compose with dedicated service containers for the Flower runtime, Qdrant, MLflow, Phoenix, and API/agent workers.

**Project bootstrap sequence** (also encoded in `scripts/setup_env.sh`)

```
git clone https://github.com/<user>/agentic-mesh.git
cd agentic-mesh
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
dvc pull   # if a remote DVC store is configured
```

**Key environment variables (.env)**

| **Variable** | **Example Value** | **Used By** |
|---|---|---|
| NODE_ID | node_01 | Simulated edge node identity |
| DATA_TYPE | timeseries | Model selector: timeseries or tabular |
| EPSILON | 2.0 | DP-SGD per-round privacy budget |
| DELTA | 1e-5 | DP-SGD delta parameter |
| FLOWER_NUM_CLIENTS | 5 | Flower simulation client count |
| **QDRANT_PATH** | **./qdrant_data** | **Embedded Qdrant local storage path (replaces QDRANT_HOST/PORT)** |
| **SQLITE_PATH** | **./agmesh.db** | **SQLite database file (replaces POSTGRES_DSN)** |
| MLFLOW_TRACKING_URI | http://localhost:5000 | MLflow experiment server |
| JWT_SECRET | \<32-char random string\> | JWT signing key |
| OLLAMA_BASE_URL | http://localhost:11434 | Phi-3.5-mini inference endpoint |
| **OLLAMA_MODEL** | **phi3.5:3.8b-mini-instruct-q4_K_M** | **Primary agent SLM (replaces Phi-4 modelfile)** |
| PHOENIX_ENDPOINT | http://localhost:6006 | Arize Phoenix tracing |

# Phase 1 Foundation (Weeks 1–3)

Phase 1 builds the skeleton every later phase depends on and establishes the execution environment for the three-module research pipeline. No core ML code is written yet. By end of Week 3, the Flower simulation environment, asyncio messaging, embedded Qdrant, SQLite, Arize Phoenix, and Docker-based project runtime are all verified working, and both datasets are downloaded and partitioned. No Phase 2 work begins until every Week 3 checkpoint passes.

**Module alignment:**
- Module 1 foundation: telemetry replay and node registry
- Module 2 foundation: FL simulation scaffold and tracking
- Module 3 foundation: Qdrant + tracing + agent memory infrastructure

---

**Week 1 — Repository scaffold, Flower simulation environment, asyncio messaging**

**Objective:** Git repository initialised with the full directory structure; Python 3.12 environment installs cleanly with no dependency conflicts; the Flower simulation environment runs with five stub coroutine clients; the asyncio-based telemetry replay generator and internal message bus are verified end-to-end.

**Key Tasks:**
- Initialise git, create the full directory tree from Section 2, add `__init__.py` to every package, create `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.env.example`, `README.md`.
- Create and activate the Python 3.12 virtual environment; install all packages; verify critical imports (`torch`, `opacus`, `flwr`, `langgraph`, `qdrant_client`) and pin versions; run `pip check`.
- Write `edge/telemetry_replay.py` — an `asyncio` generator that reads a Parquet partition row-by-row and yields it at a configurable simulated rate, standing in for a live Kafka producer.
- Write `edge/bus.py` — a thin wrapper around `asyncio.Queue` providing `publish(topic, message)` / `subscribe(topic)` semantics used for all inter-agent and node-to-mesh messaging, standing in for NATS JetStream.
- Write `federated/simulate.py` scaffold using `flwr.simulation.start_simulation()` with five stub `client_fn` clients that do nothing yet but confirm the simulation runtime boots.
- Write `scripts/verify_messaging.py` — sends 10 test messages through `edge/bus.py`, reads them back, asserts round-trip success.

**Deliverables:** Full repo scaffold committed; verified `.venv` with pinned dependencies; working `telemetry_replay.py` and `bus.py`; Flower simulation boots with 5 stub clients.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| Package install clean | `pip check` | No broken requirements |
| Messaging round-trip | `python scripts/verify_messaging.py` | "All messaging checks passed" |
| Flower simulation boots | `python federated/simulate.py --rounds 1 --stub` | 5/5 stub clients respond |

---

**Week 2 — Dataset acquisition, Non-IID partitioning, SQLite node registry**

**Objective:** HAI and CIC Modbus 2023 datasets are downloaded, cleaned, and partitioned into five Non-IID node allocations as Parquet files. SQLite is initialised with the node metadata registry populated.

**Key Tasks:**
- Apply for/download HAI (open access) and CIC Modbus Dataset 2023 (citation-only download from UNB); run checksum verification via `scripts/download_datasets.sh --verify-only`; document dataset sizes and class distributions in `notebooks/01_data_exploration.ipynb`.
- Build the cleaning pipeline in `scripts/partition_data.py`: for HAI, parse timestamps, forward-fill sparse missing sensor values, min-max normalise fitted on the normal partition only, apply a sliding window (60s window, stride 10) for VAE input; for CIC Modbus, load and concatenate pcap-derived feature CSVs, drop high-NaN columns, clip outliers, normalise, and **derive attack-window labels from the accompanying attack logs/scenario documentation** since the dataset does not ship a per-packet label column.
- Partition into five simulated nodes: Node 01–03 assigned distinct HAI process/sensor subsystems (time-series, VAE); Node 04–05 assigned distinct CIC Modbus attack-scenario subsets (tabular/network flow, Isolation Forest). Write each node's `train.parquet`, `test.parquet`, `meta.json` to `data/partitions/node_0X/`.
- Implement `storage/sqlite_schema.py`: creates `node_registry`, `alert_records`, and `fl_rounds` tables (same schema as the original PostgreSQL design, translated to SQLite types — `TEXT` for VARCHAR/UUID, `REAL` for FLOAT, no native `VECTOR` type since embeddings live in Qdrant, not SQLite).
- Write `scripts/seed_registry.py` to populate `node_registry` from each partition's `meta.json`. Initialise DVC and track `data/raw/`, `data/processed/`, `data/partitions/`. Create `params.yaml` with all hyperparameters.

**Deliverables:** Five partitioned node datasets with metadata; SQLite schema created and seeded; DVC tracking active; `params.yaml` committed.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| Partitions exist | `ls data/partitions/node_0*/` | train/test/meta per node |
| Node registry populated | `sqlite3 agmesh.db "SELECT node_id, model_type FROM node_registry"` | 5 rows returned |
| DVC tracking active | `dvc status` | No untracked data changes |
| Parquet readable | quick pandas read of `node_01/train.parquet` | Shape printed without error |

---

**Week 3 — Embedded Qdrant, Arize Phoenix, MLflow, structured logging**

**Objective:** Embedded Qdrant is initialised with its collection schemas, Arize Phoenix is accepting local traces, MLflow is tracking experiments, and the structured JSON/CSV logging convention (replacing Prometheus) is in place and verified.

**Key Tasks:**
- Initialise embedded Qdrant via `QdrantClient(path="./qdrant_data")` in a setup script (`scripts/init_qdrant.py`); create the three collections — `insight_embeddings` (128-dim, Cosine), `mitre_kb` (384-dim, Cosine), `agent_memory` (384-dim, Cosine) — with the same payload schemas as the original design.
- Run Arize Phoenix locally (`docker run` is optional here — Phoenix also runs as a plain local Python process via `phoenix.launch_app()` if avoiding Docker entirely is preferred); create `agents/tracing.py` exposing `get_tracer(name)`, verify a test span appears in the Phoenix UI.
- Start the local MLflow tracking server; create `federated/mlflow_callback.py` stub; log a test metric and verify it appears in the MLflow UI.
- Create `agents/metrics.py` — replaces the Prometheus instrumentation module. Defines lightweight functions that append structured JSON lines to `logs/metrics.jsonl` for: anomaly scores, embedding publish events, epsilon consumption, FL round counters/loss, agent investigation latency, hop counts, and circuit breaker activations. Write a companion `evaluation/plot_metrics.py` that reads `logs/metrics.jsonl` and renders matplotlib charts on demand — this is what stands in for Grafana dashboards throughout the project.
- Write `tests/integration/test_infra.py`: checks embedded Qdrant, SQLite, MLflow, and the asyncio message bus are all reachable/functional (no Kafka/NATS/Postgres checks needed). Create `.github/workflows/ci.yml` for lint + unit tests on push.

**Deliverables:** Embedded Qdrant with 3 collections; working Phoenix tracing; MLflow tracking a test run; `agents/metrics.py` and `plot_metrics.py` in place; infra integration tests passing.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| Qdrant collections created | `python scripts/init_qdrant.py` | 3 collections listed |
| Phoenix trace visible | http://localhost:6006 | Test span appears |
| MLflow metric logged | http://localhost:5000 | Test metric visible |
| Structured logging works | `python -c "from agents.metrics import log_event; log_event('test', {})"` then check `logs/metrics.jsonl` | Line appended |
| Infra integration tests | `pytest tests/integration/test_infra.py -v` | All pass |

# Phase 2 Local Intelligence (Weeks 4–6)

Phase 2 builds the ML core on each simulated edge node and corresponds to Module 1 of the research architecture: privacy-preserving edge detection without central data transfer. By end of Week 6, both the VAE and Isolation Forest are trained on their respective node partitions, producing Insight Embeddings published via the asyncio message bus, with a local evaluation harness confirming detection quality. No federated aggregation code is written yet.

**Module alignment:** Module 1 — local anomaly detection and node-level embeddings

---

**Week 4 — VAE architecture, local training loop, Opacus DP-SGD**

**Objective:** The VAE is implemented, trains on the three HAI-assigned node partitions with DP-SGD, and produces a reconstruction-error anomaly score per input window.

**Key Tasks:**
- Implement `edge/models/vae.py`: encoder (Linear → ReLU → Linear → ReLU → mu/log_var heads) → reparameterisation → decoder, with combined reconstruction + KL-divergence loss. Input dimension derived from each node's `meta.json` (window_size × feature_count).
- Implement the standard training loop in `edge/training/trainer.py` (DataLoader, Adam, checkpointing) and wrap it with Opacus in `edge/training/dp_trainer.py` using `PrivacyEngine.make_private_with_epsilon(...)`; log consumed epsilon after training.
- Implement `edge/config.py` reading hyperparameters from `params.yaml` and `.env` into a single `Config` dataclass used across all edge modules.
- Build the HAI windowed dataset loader in `edge/data_loader.py`, feeding from `edge/telemetry_replay.py` for the streaming/live-simulation path.
- Run training for Node 01–03; verify checkpoints save and MLflow logs per-epoch training loss; write unit tests for the VAE forward pass and the DP trainer's epsilon output.

**Deliverables:** `edge/models/vae.py`, `edge/training/dp_trainer.py`, trained checkpoints for nodes 01–03, passing unit tests.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| VAE unit test | `pytest tests/unit/test_vae.py -v` | Pass |
| DP trainer unit test | `pytest tests/unit/test_dp_trainer.py -v` | Pass — epsilon returned |
| Training run completes | `NODE_ID=node_01 python edge/main.py --mode train` | Checkpoint saved, epsilon logged |
| MLflow run visible | http://localhost:5000 | Train loss logged per epoch |

---

**Week 5 — Isolation Forest, model selector, embedded Qdrant integration**

**Objective:** Isolation Forest is implemented for the CIC Modbus nodes (04–05); the model selector routes each node to the correct model type from SQLite metadata; embedded Qdrant is wired into the edge layer for embedding storage.

**Key Tasks:**
- Implement `edge/models/isolation_forest.py` wrapping `sklearn.ensemble.IsolationForest`, fit on normal-only data, with input-level differential privacy (Gaussian noise scaled to sensitivity/epsilon applied to feature columns before fitting, since IsoForest has no gradients to clip via Opacus).
- Implement `edge/models/model_selector.py`: queries SQLite `node_registry` for `data_type`, returns the correct instantiated model and trainer — this remains the single decision point all other code calls into.
- Build the tabular data loader for CIC Modbus features in `edge/data_loader.py`; run Isolation Forest training for nodes 04–05; verify attack-row anomaly scores are shifted higher than normal-row scores.
- Implement `edge/embedding/insight_embedding.py`'s `QdrantEmbeddingStore` class against the **embedded** Qdrant client (`store()` / `search()` methods, same interface as the original server-based design).

**Deliverables:** Isolation Forest module with unit tests, model selector with unit tests, embedded Qdrant embedding store, trained checkpoints for nodes 04–05.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| IsoForest unit test | `pytest tests/unit/test_isolation_forest.py -v` | Pass |
| Model selector unit test | `pytest tests/unit/test_model_selector.py -v` | Pass |
| Qdrant store unit test | `pytest tests/unit/test_qdrant_store.py -v` | Pass |
| Node 04 training completes | `NODE_ID=node_04 python edge/main.py --mode train` | Model saved, scores logged |

---

**Week 6 — Insight Embedding pipeline, asyncio bus publisher, local evaluation harness**

**Objective:** All five nodes produce Insight Embeddings from local model inference, publish them via the asyncio message bus, store them in embedded Qdrant, and pass a local evaluation harness measuring AUROC/F1 against labeled test data.

**Key Tasks:**
- Finalise `InsightEmbeddingGenerator`: fixed 128-dim, L2-normalised embeddings for both VAE (concatenated mu/log_var projected via a trained linear head) and IsoForest (anomaly score + top-128 PCA components) nodes, so embeddings from different node types remain directly comparable.
- Implement the publisher side of `edge/bus.py` (`EmbeddingPublisher`, replacing the NATS JetStream publisher) with retry-with-backoff on failure, publishing to topic `embeddings.{node_id}`.
- Wire `edge/main.py --mode infer`: loads checkpoints, runs inference on the test partition in batches, thresholds at the 95th percentile of the normal score distribution, generates and publishes Insight Embeddings for anomalous windows, logs events via `agents/metrics.py`.
- Implement `evaluation/harness.py`'s `LocalEvalHarness`: computes AUROC, F1, and FPR at a target operating point per node; target AUROC >= 0.80 at this local-only stage.
- Build the in-process FastAPI node query endpoint (`api/node_api.py`): `GET /embeddings`, `GET /health`, `POST /infer`, protected with a static dev JWT token for now (full JWT issuance arrives in Week 10).

**Deliverables:** Working end-to-end local inference → embedding → publish → store pipeline for all 5 nodes; local evaluation harness with results table; node API endpoint live.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| Inference pipeline runs | `NODE_ID=node_01 python edge/main.py --mode infer` | Embeddings published and stored in Qdrant |
| Local eval harness | `python -m evaluation.harness` | AUROC table printed, all nodes >= 0.75 |
| Node API health check | `curl http://localhost:8000/health` | JSON with status ok |
| All unit tests passing | `pytest tests/unit/ -v` | All pass |

# Phase 3 Federated Learning (Weeks 7–9)

This phase implements Module 2 of the research architecture: privacy-preserving federated learning across distributed, heterogeneous edge nodes. The focus is to show that the global model remains stable under non-IID industrial partitions while privacy is preserved through DP-SGD and secure aggregation. This is the main technical section for the paper's empirical contribution.

**Module alignment:** Module 2 — FedProx, DP-SGD, SecAgg, convergence analysis, privacy accounting

Phase 3 wires all five simulated nodes into a federated training loop via Flower's simulation runtime. By end of Week 9, the global model outperforms any individual local model, FedProx is demonstrably superior to FedAvg on the Non-IID partition, and every round is logged and reproducible.

---

**Week 7 — Flower simulation client, FedProx strategy, Secure Aggregation**

**Objective:** The Flower simulation runs FedProx with all five simulated clients participating, and Secure Aggregation ensures the server only observes the summed gradients.

**Key Tasks:**
- Implement `federated/client.py`'s `FlowerClient(fl.client.NumPyClient)`: `get_parameters`, `fit` (runs `dp_trainer.train()` for `local_epochs`), `evaluate` — each simulated client runs its own DP-SGD independently within the shared simulation process.
- Configure `federated/strategy.py` with Flower's built-in `FedProx` strategy (`fraction_fit=1.0`, `min_fit_clients=3`, `proximal_mu` from `params.yaml`, custom `aggregate_fit` hook logging straggler counts via `agents/metrics.py`).
- Wire `federated/secagg.py` using Flower's built-in `SecAggPlusStrategy` wrapping the FedProx strategy — this operates identically in simulation mode as it would across real machines.
- Implement `federated/simulate.py` using `fl.simulation.start_simulation()` with a `client_fn` returning a `FlowerClient` per simulated node — this **is** the production entrypoint for this build, not just a test harness.
- Run a 5-round simulation; verify global loss decreases, MLflow logs each round, and SecAgg aggregation is confirmed active in logs; write a unit test for `FlowerClient.get_parameters()`.

**Deliverables:** Working FedProx + SecAgg simulation across 5 clients; MLflow logging per-round metrics.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| FL simulation runs | `python federated/simulate.py --rounds 5` | 5 rounds complete, global loss logged |
| SecAgg active | grep simulation logs for SecAgg confirmation | Present every round |
| MLflow rounds logged | http://localhost:5000 | 5 runs with global_loss metric |
| FL client unit test | `pytest tests/unit/test_fl_client.py -v` | Pass |

---

**Week 8 — Convergence criteria, straggler handling, full MLflow integration**

**Objective:** The federated loop stops automatically on convergence, handles simulated dropped nodes gracefully, and logs a complete reproducible experiment including model weights as artifacts.

**Key Tasks:**
- Implement `federated/convergence.py`'s `ConvergenceChecker` (patience-based stopping on rolling validation loss delta); integrate into the `aggregate_fit` hook to terminate the simulation on convergence or `max_rounds`, whichever comes first.
- Configure straggler handling via `min_fit_clients` and a per-round timeout in `ServerConfig`; log missing-node events; mark a node `degraded` in SQLite `node_registry` after 3 consecutive missed rounds. Write an integration test simulating one client timing out.
- Upgrade `federated/mlflow_callback.py` for full experiment logging: one run per training session, per-round metrics as steps, final hyperparameters/weights/AUROC/F1 as summary artifacts, per-node epsilon totals as metrics.
- Implement the server-side `evaluate_fn` in `federated/server.py`: evaluates the global VAE model against the combined HAI test partition. Document clearly that the Isolation Forest nodes (04–05) contribute only Insight Embeddings to shared Qdrant memory rather than gradients, since federated gradient aggregation applies to the VAE nodes only — their improvement path is Qdrant-based knowledge sharing in Phase 4.
- Run a full 50-round simulation; plot the convergence curve in `notebooks/03_fl_convergence_analysis.ipynb`.

**Deliverables:** Convergence checker, straggler handling with SQLite status updates, full MLflow experiment logging, 50-round simulation results.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| Convergence unit test | `pytest tests/unit/test_convergence.py -v` | Pass |
| Straggler integration test | `pytest tests/integration/test_straggler.py -v` | Pass — round completes with 4/5 clients |
| Full 50-round simulation | `python federated/simulate.py --rounds 50` | Convergence triggered, all metrics in MLflow |
| Global AUROC | MLflow run summary | >= 0.88 (target 0.92 after Week 9 tuning) |

---

**Week 9 — DVC pipeline, FedAvg vs FedProx comparison, hyperparameter tuning**

**Objective:** The DVC pipeline encodes the full training workflow reproducibly; a controlled experiment proves FedProx outperforms FedAvg on the Non-IID partition; the global model reaches AUROC >= 0.92.

**Key Tasks:**
- Define `dvc.yaml` stages: `partition_data` → `train_local_models` → `federated_train` → `evaluate`, each with explicit `deps`/`outs`. Verify with `dvc repro` and `dvc dag`.
- Add a `--strategy` flag to `federated/simulate.py` (`fedprox` / `fedavg`); run both as separate MLflow experiments on identical hyperparameters except `proximal_mu`; compare convergence curves and final AUROC/variance in `notebooks/03_fl_convergence_analysis.ipynb`; save the comparison chart to `docs/architecture/`.
- Tune hyperparameters if AUROC < 0.92: local epochs per round, `proximal_mu` sweep (0.001 / 0.01 / 0.1), `max_grad_norm` — re-running via `dvc repro` after each `params.yaml` change so only affected stages recompute.
- Register the final global model in the MLflow model registry (`GlobalAnomalyDetector v1`); implement `federated/model_loader.py` for loading it by version.

**Deliverables:** Working DVC pipeline; FedProx vs FedAvg comparison chart and table; tuned global model at AUROC >= 0.92; model registered in MLflow.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| DVC pipeline runs | `dvc repro` | All stages complete, `evaluation/results.json` produced |
| FedProx beats FedAvg | Notebook comparison chart | FedProx AUROC > FedAvg AUROC on Non-IID partition |
| Global AUROC target | `evaluation/results.json` | global_auroc >= 0.92 |
| Model registered | http://localhost:5000/#/models | GlobalAnomalyDetector v1 |

# Phase 4 Agentic Mesh (Weeks 10–13)

This phase implements Module 3: knowledge-grounded agentic attribution. The multi-agent reasoning layer is designed to transform raw anomalies into structured root-cause hypotheses grounded in MITRE ATT&CK for ICS and cross-node evidence. This is where the project moves from detection to explainability, which is crucial for publication quality.

**Module alignment:** Module 3 — Triage, Investigator, Security, Dispatcher, MITRE retrieval, evidence fusion, action dispatch

Phase 4 builds the reasoning layer on top of the federated model. By end of Week 13, the full LangGraph state machine runs end-to-end — Triage flags an anomaly, Investigator queries nodes, Security classifies against MITRE ATT&CK, Dispatcher sends an action — traced in Arize Phoenix.

---

**Week 10 — Ollama + Phi-3.5-mini, LangGraph scaffold, JWT auth, code-level mTLS**

**Objective:** Phi-3.5-mini is serving locally via Ollama and responding to structured prompts; the LangGraph state machine skeleton compiles with stub agents; JWT session tokens are issued and validated.

**Key Tasks:**
- Pull and verify Phi-3.5-mini via Ollama (`ollama pull phi3.5:3.8b-mini-instruct-q4_K_M`); create a custom Modelfile with low temperature and a structured-JSON system prompt; verify JSON-formatted responses via `langchain_community.llms.Ollama`. Confirm resident RAM footprint is ~2.2–2.5GB via `ollama ps`.
- Define `agents/state.py`'s `AgentState` TypedDict (session_id, alert, investigation_evidence, hop_count, mitre_match, remediation, status) and build the `agents/graph.py` `StateGraph` with stub Triage/Investigator/Security/Dispatcher nodes and conditional routing edges.
- Implement `agents/auth.py`'s `JWTAuthManager` (create/validate short-TTL session tokens); write unit tests including TTL expiry.
- Implement code-level mTLS: generate self-signed dev certs via the `cryptography` library (no cert-manager/K3s dependency), configure `uvicorn` with `ssl_keyfile`/`ssl_certfile` for the node API, configure `httpx` clients with `cert=`/`verify=`.
- Run the graph end-to-end with stub agents on a synthetic alert; confirm a root span with child spans appears in Phoenix.

**Deliverables:** Working local Phi-3.5-mini inference; compiled LangGraph skeleton; JWT auth module; dev mTLS certs.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| Phi-3.5-mini responding | `ollama run agmesh-phi3.5 'Return JSON: {ok: true}'` | Valid JSON returned |
| LangGraph compiles | `python -c "from agents.graph import app; print(app)"` | No error |
| Stub graph runs end-to-end | Invoke with synthetic alert | status: complete |
| JWT unit test | `pytest tests/unit/test_auth.py -v` | Pass |
| Phoenix traces visible | http://localhost:6006 | Root span + child spans per node |

---

**Week 11 — Triage Agent, Investigator Agent, circuit breaker**

**Objective:** The Triage Agent uses Phi-3.5-mini to assess alert severity and decide escalation; the Investigator Agent queries up to three nodes for corroborating evidence with full circuit breaker protection.

**Key Tasks:**
- Implement `agents/triage/triage_agent.py`: embedded-Qdrant memory check against similar past embeddings, an LLM assessment prompt requesting structured JSON (`escalate`, `severity_level`, `reasoning`), state update, SQLite `alert_records` logging, Phoenix span.
- Implement `agents/investigator/circuit_breaker.py`'s three-state `CircuitBreaker` (CLOSED / OPEN / HALF_OPEN) and a shared `CircuitBreakerRegistry` keyed by node_id; write state-transition unit tests.
- Implement `agents/investigator/investigator_agent.py`: LLM-guided target node selection, bounded 3-hop querying via the node API (through the circuit breaker, 500ms timeout per hop), evidence accumulation, hop-count metric logging.
- Write an integration test mocking the LLM and node API responses, asserting correct state transitions and hop-count bounds.
- Profile end-to-end Triage + Investigator latency with real Phi-3.5-mini against the live in-process node API; target combined latency < 1.5 seconds.

**Deliverables:** Working Triage and Investigator agents; circuit breaker with tests; latency profile within target.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| Circuit breaker unit test | `pytest tests/unit/test_circuit_breaker.py -v` | Pass |
| Triage + Investigator integration | `pytest tests/integration/test_triage_investigator.py -v` | Pass |
| Hop count bounded | Assertion in integration test | hop_count <= 3 always |
| Latency budget | 10 real timed runs | Mean < 1.5s |

---

**Week 12 — MITRE ATT&CK knowledge base, Security Agent, Action Dispatcher**

**Objective:** The MITRE ATT&CK for ICS knowledge base is indexed in embedded Qdrant; the Security Agent classifies evidence with a confidence-scored technique match; the Action Dispatcher publishes a structured remediation via the asyncio bus.

**Key Tasks:**
- Implement `agents/security/kb_builder.py`: parse the MITRE ATT&CK for ICS STIX bundle and NIST SP 800-82 text, chunk descriptions, embed via Sentence Transformers, ingest into the embedded `mitre_kb` Qdrant collection with technique ID/tactic/platform payload fields. Run `scripts/build_kb.py` and verify ~1500–2000 points indexed.
- Implement `agents/security/security_agent.py`: mean-pool evidence embeddings, retrieve top-5 KB matches from embedded Qdrant, LLM classification prompt returning structured JSON (`technique_id`, `tactic`, `confidence`, `reasoning`, `recommended_mitigations`), confidence threshold fallback to `UNKNOWN`, SQLite update, Phoenix span.
- Implement `agents/dispatcher/action_dispatcher.py`: build the structured action payload, map tactic → action type, **publish via `edge/bus.py`'s asyncio.Queue action topic** (replacing the NATS JetStream publish step), update SQLite `alert_records`, log latency metric, Phoenix span.
- Write a full-graph integration test (all four agents, mocked LLM/node API/bus) and then a real end-to-end run using a real HAI or CIC Modbus attack-window embedding; confirm total latency < 3 seconds and 4 Phoenix spans per session.

**Deliverables:** Indexed MITRE KB; working Security Agent and Action Dispatcher; full graph integration test passing; real end-to-end run under 3 seconds.

**Validation Checkpoint:**

| Check | Command / Action | Expected Result |
|---|---|---|
| MITRE KB indexed | embedded Qdrant `mitre_kb` collection count | >= 1500 points |
| Full graph integration test | `pytest tests/integration/test_full_graph.py -v` | All assertions pass |
| Real Phi-3.5-mini end-to-end | Manual run, inspect state | technique_id returned, reasoning coherent |
| Latency < 3 seconds | 5 real timed runs | Mean < 3.0s |
| Phoenix 4 spans per session | http://localhost:6006 | triage, investigate, classify, dispatch spans visible |

---

**Week 13 — Full mesh integration, Phoenix trace validation, end-to-end smoke test**

**Objective:** All phases are wired together in a single end-to-end smoke test running without manual intervention — inference on the test set triggers FL evaluation, which triggers Triage through to Dispatch, all within one Python process and traced in Phoenix.

**Key Tasks:**
- Wire `federated/server.py`'s `evaluate_fn` as the trigger: after each round, if drift or an anomaly threshold is exceeded, publish an alert onto the asyncio bus topic `alerts.global`.
- Implement `agents/graph_runner.py`: subscribes to `alerts.global` via `edge/bus.py`, deserialises the alert, invokes the LangGraph app (in a thread pool executor since LangGraph is sync and the subscriber loop is async), logs session completion.
- Since there is no K3s/Helm deployment step in this build, run the agent mesh and FL server as concurrent local processes (or as async tasks within one orchestrating script, `scripts/run_local_stack.py`) rather than deploying to a cluster.
- Run 5 complete investigation sessions via injected alerts; verify in Phoenix that each session shows the expected 4-span structure with no ERROR status; export one complete trace as `docs/architecture/sample_trace.json`.
- Write `scripts/run_smoke_test.sh`: runs a short FL simulation with an injected attack event, waits for agent dispatch, asserts an alert was dispatched and logged in SQLite, asserts Phoenix recorded the expected spans.

**Deliverables:** FL → agent-mesh trigger wired end-to-end; `graph_runner.py`; `run_local_stack.py`; passing smoke test script.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| FL → Agent trigger works | `python federated/simulate.py --rounds 3 --inject-attack` | Alert dispatched, logged in SQLite |
| Phoenix traces complete | http://localhost:6006 | 4-span trace per session, no ERROR status |
| Full smoke test | `bash scripts/run_smoke_test.sh` | "=== Smoke Test PASSED ===" |

# Phase 5 Evaluation & Documentation (Weeks 14–16)

The final phase is focused on validation, ablations, and publication packaging. It targets the empirical story required for a research paper: baselines, privacy accounting, ablation studies, latency analysis, and reproducibility. This section also includes the Docker packaging required for a clean demo and final execution path.

**Module alignment:** Cross-module evaluation, ablation study, reproducibility, final documentation, Docker runtime packaging

Phase 5 proves the system works. By end of Week 16, every metric in the proposal's evaluation framework is measured and recorded, the chaos test passes, matplotlib-based metric charts are complete, and the final evaluation report is written.

---

**Week 14 — Custom evaluation harness, HAI + CIC Modbus attack simulation**

**Objective:** The custom evaluation harness runs all labeled attack windows from both datasets through the complete system and produces a structured results file: detection rate, false positive rate, root-cause accuracy, and mean investigation time.

**Key Tasks:**
- Implement `evaluation/harness.py`'s `EvalHarness`: `load_attack_windows()` (HAI + CIC Modbus test partitions, enriched with MITRE technique ground truth), `load_normal_windows()`, `run_window()` (feeds a window through local model → embedding → simulated FL global score → Triage-to-Dispatch if above threshold → records outcome), `compute_metrics()` (detection rate, FPR, root-cause accuracy, mean/p95 latency).
- Build the ground-truth attack → MITRE technique mapping file (`evaluation/attack_ground_truth.json`) for both HAI's labeled attack scenarios and CIC Modbus's attack-log-derived windows (see Week 2 labeling note).
- Run the full evaluation (`python -m evaluation.harness --output evaluation/results.json`); if LLM inference latency makes this slow on 8GB RAM, run in smaller sequential batches rather than parallel processes, since concurrent Ollama instances would exceed the RAM budget.
- Analyse results against targets (detection_rate >= 0.90, FPR <= 0.05, root_cause_accuracy >= 0.80, mean_latency <= 3.0s) and close any gaps (threshold tuning, KB chunking, prompt adjustments, hop reduction).

**Deliverables:** `evaluation/results.json` with all target metrics measured; ground-truth mapping file; gap-closing iteration if needed.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| Harness runs without error | `python -m evaluation.harness --output evaluation/results.json` | results.json produced |
| Detection rate target | parse results.json | >= 0.90 |
| FPR target | parse results.json | <= 0.05 |
| Mean latency target | parse results.json | <= 3.0 seconds |

---

**Week 15 — Chaos testing, RAGAS retrieval evaluation, matplotlib metric dashboards**

**Objective:** Chaos testing confirms zero hung investigations on simulated node failure; RAGAS confirms embedded Qdrant retrieval quality; matplotlib charts (replacing Grafana) present all key system metrics.

**Key Tasks:**
- Implement `evaluation/chaos_test.py`'s `ChaosTest`: since there is no real K3s pod to kill, simulate node failure by having the node's in-process API handler raise a timeout/connection error on command (a configurable fault-injection flag), run concurrent investigations, kill a target mid-investigation, and assert all sessions reach a terminal SQLite status within a timeout. Cover the same four scenarios as the original design (pre-investigation failure, mid-hop failure, dual failure, recovery after timeout).
- Implement `evaluation/ragas_eval.py`: build a 50-pair query/ground-truth-context evaluation set against embedded Qdrant, compute RAGAS context recall and context precision (targets >= 0.85 / >= 0.80).
- Build `evaluation/plot_metrics.py` output: four matplotlib figures replacing the four original Grafana dashboards — (1) FL health: loss per round, FedProx vs FedAvg AUROC, epsilon consumed, straggler rate; (2) Edge node status: anomaly score distributions, embeddings published over time; (3) Agent mesh performance: latency percentiles, hop-count histogram, circuit breaker activations, MITRE technique frequency; (4) System overview: total alerts, detection rate, mean time to dispatch, epsilon budget status. Save all as PNGs in `docs/architecture/`.
- Re-run the FedProx vs FedAvg comparison with final tuned hyperparameters and record final numbers; verify total epsilon <= 10.0 across all nodes.

**Deliverables:** Passing chaos test across all four scenarios; RAGAS results meeting targets; four matplotlib dashboard exports; finalised FL metrics.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| Chaos test all scenarios | `python -m evaluation.chaos_test` | "All chaos scenarios passed" |
| RAGAS context recall | parse `evaluation/ragas_results.json` | >= 0.85 |
| All 4 metric charts generated | `ls docs/architecture/*.png` | 4 dashboard PNGs present |
| Epsilon budget compliant | `evaluation/results.json` | All nodes epsilon_total <= 10.0 |

---

**Week 16 — Final evaluation report, architecture diagrams, cleanup, handoff**

**Objective:** Every success criterion from the proposal is measured and documented, all code is clean and tested, and the repository is ready for review or continuation by another engineer.

**Key Tasks:**
- Consolidate all results into `evaluation/final_report_data.json` (FL metrics, agent metrics, privacy metrics, retrieval metrics); write `evaluation/check_success_criteria.py` asserting every proposal Section 10 target and printing overall PASS/FAIL.
- Produce four architecture diagrams in `docs/architecture/` (Mermaid or draw.io, exported PNG/SVG): system layers, data flow (11-step sequence), LangGraph state machine, FedProx vs FedAvg convergence.
- Run full cleanup: `pytest tests/ -v` (all pass), `black . && isort .`, `mypy` across all packages (zero errors), docstring review, and a rewritten `README.md` covering overview, prerequisites, quick start (`bash scripts/setup_env.sh` then `bash scripts/run_smoke_test.sh`), the system-layers diagram, and the results table.
- Finalise `.github/workflows/ci.yml` (lint, unit tests, coverage) — no `deploy.yml`/Helm step needed for this build, since there is no cluster to deploy to.
- Final `dvc repro`, final `check_success_criteria.py` run, git tag `v2.1.0`, and a `HANDOFF.md` covering how to run the smoke test, trigger a training round, trigger an investigation manually, add a new simulated node, and documenting known limitations plus the future production-scaling path (K3s/Kafka/NATS/PostgreSQL/Prometheus swap-back for real deployment).

**Deliverables:** Final evaluation report data and pass/fail check; 4 architecture diagrams; clean, fully tested, linted codebase; finalised README and HANDOFF.md; tagged release.

**Validation Checkpoint:**

| Check | Command | Expected Result |
|---|---|---|
| All success criteria | `python evaluation/check_success_criteria.py` | Overall PASSED |
| All tests pass | `pytest tests/ -v` | All pass, coverage >= 80% |
| Linting clean | `black --check . && isort --check . && mypy edge/` | Zero errors |
| DVC reproducible | `dvc repro --force` | All stages complete without error |
| GitHub Actions green | GitHub Actions tab | CI green |
| Git tag created | `git tag` | v2.1.0 visible |

# Appendix A Full Deliverables Checklist

| **Deliverable** | **Phase** | **Week** |
|---|---|---|
| Full directory structure created and committed | 1 | 1 |
| Flower simulation environment boots with 5 stub clients | 1 | 1 |
| Asyncio telemetry replay + message bus verified | 1 | 1 |
| HAI and CIC Modbus 2023 downloaded and checksummed | 1 | 2 |
| Five Non-IID node partitions as Parquet files | 1 | 2 |
| SQLite node_registry populated with 5 nodes | 1 | 2 |
| DVC tracking initialised for data and models | 1 | 2 |
| Embedded Qdrant, Phoenix, MLflow, structured logging all running | 1 | 3 |
| Infrastructure integration tests passing | 1 | 3 |
| VAE architecture implemented with unit tests | 2 | 4 |
| Opacus DP-SGD integrated, epsilon tracked | 2 | 4 |
| VAE trained on node_01, node_02, node_03 | 2 | 4 |
| Isolation Forest implemented with unit tests | 2 | 5 |
| Model selector routes by node metadata | 2 | 5 |
| Embedded Qdrant collections created with correct schemas | 2 | 5 |
| Insight Embedding generator producing 128-dim L2-normalised vectors | 2 | 6 |
| Asyncio bus publisher working with retry logic | 2 | 6 |
| Node FastAPI endpoint live with JWT protection | 2 | 6 |
| Local evaluation harness: AUROC >= 0.75 per node | 2 | 6 |
| Flower FedProx strategy implemented (simulation mode) | 3 | 7 |
| Secure Aggregation active and verified in logs | 3 | 7 |
| 5-round simulation completes with MLflow logging | 3 | 7 |
| Convergence checker stopping training correctly | 3 | 8 |
| Straggler timeout with async fallback working | 3 | 8 |
| 50-round simulation: global AUROC >= 0.88 | 3 | 8 |
| DVC pipeline encoding full training workflow | 3 | 9 |
| FedProx outperforms FedAvg in controlled experiment | 3 | 9 |
| Global model AUROC >= 0.92 after tuning | 3 | 9 |
| GlobalAnomalyDetector v1 registered in MLflow | 3 | 9 |
| Phi-3.5-mini serving locally via Ollama, responding to prompts | 4 | 10 |
| LangGraph state machine compiled with 4 agent nodes | 4 | 10 |
| JWT session token issuance and validation working | 4 | 10 |
| Code-level mTLS dev certificates issued and services configured | 4 | 10 |
| Triage Agent using Phi-3.5-mini with correct JSON output | 4 | 11 |
| Circuit breaker: 3-state machine with correct transitions | 4 | 11 |
| Investigator Agent: 3-hop limit enforced | 4 | 11 |
| MITRE ATT&CK KB indexed in embedded Qdrant (>= 1500 points) | 4 | 12 |
| Security Agent returning technique_id with confidence | 4 | 12 |
| Action Dispatcher publishing via asyncio bus with ack | 4 | 12 |
| Full graph integration test passing | 4 | 12 |
| FL → bus → Agent trigger wired end-to-end | 4 | 13 |
| Local orchestrating script runs full stack concurrently | 4 | 13 |
| Phoenix 4-span trace per session, no ERRORs | 4 | 13 |
| Smoke test script passes end-to-end | 4 | 13 |
| Evaluation harness: detection_rate >= 0.90 | 5 | 14 |
| Evaluation harness: FPR <= 0.05 | 5 | 14 |
| Evaluation harness: root_cause_accuracy >= 0.80 | 5 | 14 |
| Evaluation harness: mean_latency <= 3.0s | 5 | 14 |
| Chaos test: zero hung investigations across all 4 scenarios | 5 | 15 |
| RAGAS context_recall >= 0.85 | 5 | 15 |
| All 4 matplotlib metric charts generated | 5 | 15 |
| All epsilon_total <= 10.0 per node | 5 | 15 |
| All success criteria: check_success_criteria.py PASSED | 5 | 16 |
| All tests passing, coverage >= 80% | 5 | 16 |
| Linting clean: black, isort, mypy all zero errors | 5 | 16 |
| DVC full pipeline reproducible from scratch | 5 | 16 |
| GitHub Actions CI green | 5 | 16 |
| Git tag v2.1.0 created and pushed | 5 | 16 |
| HANDOFF.md written and committed | 5 | 16 |

# Appendix B Package Version Reference (requirements.txt)

```
# Core ML
torch==2.3.1
opacus==1.4.1
scikit-learn==1.5.1
numpy==1.26.4
pandas==2.2.2
pyarrow==16.1.0
joblib==1.4.2

# Federated Learning
flwr==1.9.0

# Agents and LLM
langchain==0.2.11
langchain-core==0.2.24
langgraph==0.1.19
langchain-community==0.2.10
sentence-transformers==3.0.1

# Vector DB (embedded mode — no server)
qdrant-client==1.10.1

# API
fastapi==0.111.1
uvicorn==0.30.1
pydantic==2.7.4
httpx==0.27.0

# Auth and Security
python-jose==3.3.0
cryptography==42.0.8

# MLOps
mlflow==2.14.1
dvc==3.51.2

# Evaluation
ragas==0.1.14
matplotlib==3.9.0

# Utilities
python-dotenv==1.0.1
loguru==0.7.2
rich==13.7.1
pyyaml==6.0.1

# --- requirements-dev.txt (additional) ---
pytest==8.2.2
pytest-asyncio==0.23.7
pytest-cov==5.0.0
black==24.4.2
isort==5.13.2
mypy==1.10.0
```

**Note:** `sqlite3` is part of the Python 3.12 standard library and requires no separate package entry.

*— End of Implementation Plan —*

**Allaudin Ansari**

August 2026
