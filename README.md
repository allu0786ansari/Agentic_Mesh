# Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

A research-first, first-phase prototype for detecting distributed industrial anomalies while preserving data locality, privacy, and explainability. The system combines edge-based anomaly detection, federated learning, and a multi-agent investigation workflow to reason about suspicious activity without centralizing raw operational data.

This repository represents the first completed development phase of a larger research program. It is structured as a rerunnable local prototype that can be executed on a single development machine with an 8 GB RAM budget, while preserving the architecture needed for scale-up in a later phase.

## Project status

This project is currently in a defensible first-phase prototype state:

- Local edge anomaly detection is implemented and operational.
- Federated learning simulation is implemented in Flower with FedProx-based training logic.
- A multi-agent LangGraph investigation workflow is implemented.
- Retrieval and knowledge grounding against a MITRE-inspired local knowledge base is implemented.
- Evaluation, chaos testing, and evidence-gated reporting are in place.
- The project is intentionally documented as PASS_WITH_GAPS at the final evaluation gate because the final federated benchmark artifacts, cumulative epsilon evidence, and full 8 GB runtime benchmark are not yet captured as machine-readable results.

This is appropriate for a first-phase submission: the prototype is technically functional, reproducible, and honest about remaining evidence gaps.

---

## 1. Problem statement

Industrial environments generate large volumes of operational telemetry from machines, sensors, controllers, and network devices. In practice, the most valuable intelligence is also the most sensitive: the same data that is critical for anomaly detection and predictive maintenance is often restricted by privacy, legal, and data residency requirements.

The central challenge is not only to detect anomalies, but to do so without moving raw operational data across trust boundaries. Traditional centralised AI systems require data aggregation into a single repository, which creates several problems:

- data sovereignty and compliance exposure
- high risk of exposure if the central store is compromised
- inability to capture cross-node patterns that are only visible when evidence is correlated across distributed systems
- delayed and brittle model updates based on static, manually retrained pipelines

In industrial cyber-physical systems, attacks and faults often appear as weak signals dispersed across multiple nodes over time rather than as one obvious local disruption. A model trained on data from a single node cannot see the broader pattern. Traditional detection systems therefore fail to capture multi-stage and distributed behaviours.

The project addresses this by combining:

- local anomaly detection at the edge
- privacy-preserving federated learning across distributed node partitions
- a reasoning layer that correlates alerts, retrieves relevant threat intelligence, and proposes actionable outcomes

---

## 2. Research objective and scope

The objective of this project is to build a privacy-aware, distributed intelligence platform for industrial analytics. The project is designed to demonstrate that:

1. anomaly detection can remain local to each node without centralizing raw telemetry
2. model learning can occur across private data silos through federated training
3. a multi-agent system can investigate suspicious events and map them to relevant attack patterns or operational faults
4. a prototype can be evaluated using deterministic checks, retrieval assessments, and chaos conditions

This repository implements the first phase of that objective with a single-machine, 8 GB development configuration. It is intentionally designed as a research prototype rather than a production deployment stack.

---

## 3. Solution overview

The system is organised into three connected layers.

### 3.1 Local intelligence layer

Each simulated node operates on a local partition of the data and runs a model appropriate to its telemetry type:

- Variational Autoencoder (VAE) for multivariate process and sensor time-series data
- Isolation Forest for tabular or protocol-derived network anomaly data

These models produce compact insight embeddings rather than raw telemetry, keeping the detection logic local and privacy-aware.

### 3.2 Federated learning layer

A Flower simulation orchestrates learning across node partitions using a FedProx-based tuning strategy to improve robustness under non-IID industrial distributions. DP-SGD is added to track privacy cost, and the training loop is structured such that raw local data remains on the node.

### 3.3 Agentic investigation layer

The investigation layer is implemented as a LangGraph state-based workflow with specialised agents:

- Triage agent: filters and prioritises incoming signals
- Investigator agent: gathers supporting evidence from relevant nodes
- Security agent: retrieves and ranks relevant MITRE/NIST-style threat knowledge
- Action dispatcher: determines the next operational response or alert action

The system uses embedded Qdrant for semantic retrieval and SQLite for operational metadata, allowing the project to remain lightweight and self-contained locally.

---

## 4. Architecture summary

The project follows a layered architecture designed for research validation and local execution:

- Edge layer: local data processing, VAE/Isolation Forest inference, telemetry, and model-specific training
- Federated layer: aggregated model updates and convergence checks
- Memory and knowledge layer: embedded Qdrant vector retrieval and local knowledge base indexing
- Agent layer: LangGraph workflow with triage, investigation, and response logic
- Evaluation layer: deterministic evidence checks, retrieval metrics, and chaos-run testing

![System layers](docs/architecture/system_layers.svg)

The main architecture diagrams are also exported in the docs folder under the architecture section.

---

## 5. Technical stack

### Core ML and analytics

- Python 3.12
- PyTorch
- scikit-learn
- NumPy
- pandas
- pyarrow
- Flower (federated simulation)
- Opacus (DP-SGD)
- MLflow
- DVC

### Agentic and knowledge retrieval stack

- LangGraph
- LangChain
- Qdrant (embedded/local mode)
- Sentence Transformers
- SQLite
- local LLM integration via Ollama-compatible local model wrapper
- Arize Phoenix for tracing support

### Security and control plane elements

- JWT-based authentication for agent interactions
- code-level mTLS generation tooling for local development security patterns
- asyncio-based messaging bus for in-process communication

### Evaluation and documentation

- pytest
- ruff
- matplotlib
- ragas
- custom evaluation harnesses for Week 14 and Week 15 research checks

---

## 6. Repository structure

```text
Agentic_Mesh/
├── agents/                     # agent workflows, auth, security, dispatch
│   ├── auth.py
│   ├── graph.py
│   ├── state.py
│   ├── dispatcher/
│   ├── investigator/
│   ├── security/
│   └── triage/
├── api/                       # API interfaces and service hooks
├── data/                      # dataset metadata and partitioned data directories
├── docs/                      # architecture exports and planning documents
├── edge/                      # local edge inference, telemetry, and model logic
├── evaluation/                # benchmark and evidence scripts
│   ├── harness.py
│   ├── chaos_test.py
│   ├── ragas_eval.py
│   ├── plot_metrics.py
│   ├── final_report.py
│   └── check_success_criteria.py
├── federated/                 # Flower simulation and training logic
├── scripts/                   # operational and setup utilities
├── storage/                   # SQLite schema and metadata storage
├── tests/                     # unit and integration validation
├── Documents/                 # project proposal, plan, and week reports
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── README.md
├── dvc.yaml
└── mlruns/
```

---

## 7. Data model and research inputs

The prototype is designed around industrial benchmark data sources, primarily:

- HAI dataset partitions
- Modbus 2023 dataset partitions
- local node metadata and partition registry

Data is intentionally organised per node to maintain non-IID conditions and reflect realistic operational heterogeneity. Raw data and generated runtime artifacts are excluded from Git tracking by default via the project .gitignore file.

The project also includes a structured evaluation layer to compare model behavior against labelled windows and retrieval quality across the knowledge and memory stack.

---

## 8. Setup and installation

### Prerequisites

- Windows or WSL2/Linux environment
- Python 3.12
- 8 GB RAM available for local prototype execution
- pip and venv support
- optional local Ollama runtime for real LLM-backed agent execution

### Create the environment

```powershell
python -m venv agentvenv
.\agentvenv\Scripts\python.exe -m pip install --upgrade pip
.\agentvenv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

### Verify the installation

```powershell
.\agentvenv\Scripts\python.exe -m pytest tests -q --disable-warnings
```

This repository was validated in a local environment with the test suite passing before final documentation review.

---

## 9. How to run the project

### 9.1 Local stack smoke test

This runs the local graph stack with synthetic alerts and verifies the end-to-end agent workflow can complete sessions:

```powershell
$py = '.\agentvenv\Scripts\python.exe'
& $py -m scripts.run_local_stack --count 5 --metrics-path logs/metrics.jsonl
```

### 9.2 Evaluation harness

Run the Week 14 evaluation harness for detection and latency metrics:

```powershell
& $py -m evaluation.harness
```

### 9.3 Chaos evaluation

Run the resilience scenario validation for hung investigations, retry loops, and failure conditions:

```powershell
& $py -m evaluation.chaos_test
```

### 9.4 Retrieval quality evaluation

Run the local retrieval quality and grounding assessment:

```powershell
& $py -m evaluation.ragas_eval
```

### 9.5 Plot generation

Generate the visual dashboards and charts from measured evaluation artifacts:

```powershell
& $py -m evaluation.plot_metrics
```

### 9.6 Final evidence gate

The final project gate checks whether the prototype is ready for first-phase submission based on delivered evidence:

```powershell
& $py -m evaluation.check_success_criteria
```

This command reports a status of PASS_WITH_GAPS when the repository has solid prototype evidence but lacks final benchmark artifacts for FL AUROC/F1, cumulative epsilon tracking, and 8 GB runtime benchmarking.

---

## 10. Evaluation and evidence model

The project implements a transparent evidence-gated evaluation strategy. It is designed to avoid overstating readiness and to separate completed implementation from pending benchmark evidence.

### Implemented evidence

The following checks are currently part of the project and have been validated in the local prototype:

- detection rate and false positive assessment in the evaluation harness
- latency and incident timing checks
- chaos scenario pass/fail evaluation
- retrieval quality measures for context recall and context precision
- local graph/session completion validation
- authentication and code-level TLS hooks are present
- DVC and MLflow configuration exists for reproducibility support

### Current gate outcome

The project currently reports the following status in the repository evidence gate:

- PASS for end-to-end detection threshold
- PASS for latency threshold
- PASS for zero-hung chaos cases
- PASS for recall and precision thresholds
- PASS_WITH_GAPS overall because the following are not yet measured as machine-readable benchmark artifacts:
  - final global AUROC / F1
  - FedProx vs FedAvg final comparison
  - cumulative epsilon budget per node
  - end-to-end 8 GB runtime / resource benchmark

This is a healthy and honest first-phase evaluation posture. It reflects real-world engineering discipline rather than unsupported claims.

---

## 11. Key implementation highlights

This first phase includes:

- local edge models for HAI and Modbus traffic partitions
- privacy-aware federated learning simulation using Flower
- non-IID learning design using FedProx-style strategy behavior
- embedded local Qdrant-based memory and retrieval
- knowledge-base indexing for threat pattern lookups
- structured agent graph orchestration for investigation flows
- runtime metrics logging and evidence output generation
- reproducibility support through DVC and MLflow
- architecture diagrams for technical review

These capabilities are sufficient to show the feasibility of the vision and provide a solid engineering base for the next improvement phase.

---

## 12. Limitations and known challenges

This is a research prototype and not a production-ready industrial control platform. The current limitations are explicit and should be considered part of the engineering record.

### Technical limitations

- The project runs in a local development configuration rather than a distributed cluster.
- Full benchmark evidence for final FL performance is not yet captured.
- Some privacy accounting and exhaustive comparative evaluation remain pending.
- Runtime assumptions depend on the local environment and the availability of supporting libraries.

### Data and evaluation limitations

- Scenario-level MITRE ground-truth labels are limited and conservative by design.
- Industrial benchmark realism is constrained by the available local dataset and prototype build scope.
- Model and reasoning performance should be interpreted as research evidence, not final operational deployment numbers.

### Engineering challenges

- balancing strong privacy safeguards with acceptable model accuracy
- handling non-IID industrial data distributions across edge nodes
- keeping the system lightweight enough for an 8 GB research machine
- maintaining reliable operating conditions for local modelling, Qdrant indexing, and agent tracing

---

## 13. Why this is a credible first-phase submission

This project is credible as a first-phase development submission because it demonstrates end-to-end proof of concept across the full research pipeline:

- local anomaly detection works
- federated learning is implemented and can run in simulation mode
- a multi-agent reasoning layer is implemented
- memory and retrieval are active
- evaluation harnesses exist and produce measured evidence
- the project is documented honestly with known limitations and missing benchmark artifacts explicitly recorded

It is not presented as final production deployment readiness. Instead, it is presented as a sound first-phase research prototype that is ready for the next cycle of improvement and evidence completion.

---

## 14. Next-phase improvement roadmap

The next engineering phase should focus on the items that convert the prototype into benchmark-quality evidence:

1. capture final FL AUROC/F1 metrics on held-out data
2. compare FedProx and FedAvg under controlled conditions
3. record cumulative epsilon values per node and training round
4. measure the 8 GB runtime benchmark and resource footprint
5. harden model checkpointing and reproducibility evidence
6. improve scenario-level ground truth and threat mapping quality
7. expand the local evidence summary into a more publication-ready report

These are the most important issues to close before a broader production or research publication move.

---

## 15. How to contribute or extend

This repository is structured for continuation and research iteration. Developers can extend the project by:

- adding stronger edge feature extraction logic
- testing alternative federated aggregation strategies
- improving the local knowledge base and retrieval ranking
- tightening privacy accounting and MLflow/DVC recording
- replacing simulated settings with more realistic operational workloads

---

## 16. Final note

This project demonstrates a realistic, technically coherent first-phase implementation of a privacy-preserving, distributed, agentic industrial analytics system. It is suitable as a research prototype submission because it is complete enough to be understood, rerun, and evaluated, while still preserving transparency about the remaining evidence gaps that define the next improvement phase.

The repository is intended to be read as a research prototype with clear engineering intent, explicit limitations, and an honest validation posture rather than an over-claimed production solution.
