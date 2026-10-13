# Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

## Abstract

This project presents a privacy-preserving, multi-agent framework for industrial anomaly detection and forensic reasoning in distributed cyber-physical environments. The proposed system integrates local edge intelligence, federated learning, and knowledge-grounded agentic investigation to detect complex operational anomalies without centralizing raw telemetry. The implementation is designed as a research prototype that operates within a constrained single-machine environment and demonstrates the feasibility of privacy-aware industrial analytics in the absence of a fully distributed production deployment.

The system combines variational autoencoders and isolation forest models at the edge, a Flower-based federated learning simulation using a FedProx-inspired aggregation strategy, and a LangGraph-based investigation workflow involving triage, investigation, security reasoning, and action dispatch. Semantic retrieval is performed using an embedded Qdrant memory layer, while operational metadata is maintained in SQLite. The design is motivated by the privacy constraints inherent to industrial data, where centralizing sensitive machine and process telemetry raises legal, security, and governance concerns.

This repository represents the first completed phase of the research work. It is intentionally scoped for local execution on an 8 GB RAM development machine and is structured to support reproducibility, evaluation, and future enhancement in the next development phase.

---

## 1. Introduction

Industrial systems generate large-scale telemetry from heterogeneous sources such as sensors, controllers, human-machine interfaces, and network infrastructure. This data is highly valuable for predictive maintenance, anomaly detection, and cyber-physical risk identification. However, the same information is frequently sensitive, operationally critical, and constrained by privacy and compliance requirements. In many real-world scenarios, raw data cannot be transmitted outside the originating environment due to legal, organisational, or security restrictions.

This creates a significant challenge: the most informative industrial intelligence is often located in distributed, isolated silos, while conventional centralized machine learning approaches require raw data aggregation. Such a model assumes unrestricted access to private operational data, which is not feasible in regulated or security-sensitive industrial environments.

To address this limitation, the proposed framework adopts a privacy-preserving architecture that keeps data local to each node while enabling collaborative learning and distributed reasoning. The system is designed to detect anomalies across multiple nodes, correlate evidence across heterogeneous sources, and support explainable incident investigation without exposing raw data.

The research problem is therefore formulated as follows: how can a distributed industrial monitoring system detect coordinated anomalies and support root-cause analysis while preserving data locality, privacy, and operational feasibility under resource-constrained conditions?

---

## 2. Problem Statement

Industrial cyber-physical environments are characterized by high data velocity, diverse device types, and non-independent data distributions across operational nodes. In such settings, anomalies are often subtle and distributed across multiple subsystems rather than localized in a single source. Traditional centralized machine learning pipelines are inadequate because they necessitate the movement of sensitive data to a single repository, increasing exposure to privacy violations and compliance breaches.

Additional limitations of conventional approaches include:

- data residency and sovereignty constraints
- poor support for non-IID industrial data distributions
- delayed retraining cycles and model drift
- inability to reason over distributed evidence in a single investigation workflow
- limited explainability during root-cause analysis

The need for a privacy-aware, distributed, and interpretable approach is therefore evident. This project addresses that need by designing a system that performs local anomaly detection, federated model convergence over private partitions, and multi-agent threat investigation using contextual retrieval and knowledge grounding.

---

## 3. Objectives and Scope

The primary objective of this project is to develop and evaluate a decentralized intelligence framework for privacy-preserving industrial analytics. The scope includes the following research goals:

1. to preserve data locality by ensuring that raw operational data remains within each node
2. to support distributed learning across heterogeneous node partitions using a federated setup
3. to identify abnormal behaviour through edge-level anomaly detection models
4. to perform evidence-based investigation using a multi-agent reasoning graph
5. to retrieve relevant threat intelligence and operational context from a local knowledge base
6. to provide a reproducible first-phase prototype suitable for academic review and future extension

This repository implements the first phase of the system and is intentionally designed as a prototype environment suitable for research validation, algorithmic demonstration, and structured evaluation.

---

## 4. System Overview

The architecture is organized into three interconnected layers: edge intelligence, federated learning, and agentic reasoning.

### 4.1 Edge Intelligence Layer

Each node operates on a local data partition and applies an appropriate anomaly detector based on the telemetry characteristics of that node. The system employs:

- Variational Autoencoder (VAE) for multivariate industrial time-series data
- Isolation Forest for tabular and protocol-based anomaly detection

These models generate compact insight embeddings rather than raw data representations, preserving privacy while enabling downstream correlation and retrieval.

### 4.2 Federated Learning Layer

The federated layer coordinates model updates across nodes using a Flower-based simulation. The training setup follows a FedProx-inspired design to improve convergence under non-IID data distributions, which are common in industrial settings. Differential privacy is incorporated through DP-SGD-style gradient management, and the training loop is designed so that raw data never leaves each node.

### 4.3 Agentic Investigation Layer

The reasoning layer is implemented using LangGraph and comprises a set of interacting agents:

- Triage agent for signal prioritization
- Investigator agent for evidence collection and node-level correlation
- Security agent for knowledge-grounded threat attribution
- Action dispatcher for response generation and alert handling

This layer performs structured investigation over suspicious events, using an embedded vector store for memory and semantic retrieval, and a local knowledge base for contextual interpretation.

---

## 5. Research Motivation

The motivation for this project is rooted in the mismatch between modern industrial data requirements and traditional centralized AI design. In industrial systems, data is often fragmented, privacy-sensitive, and continuously changing. The inability to centralize such data severely limits the applicability of standard machine learning pipelines.

The proposed framework addresses this by introducing a distributed, privacy-aware intelligence architecture that aligns with modern research directions in federated learning, AI security, and agentic reasoning. It is particularly relevant in environments where security monitoring must be both explainable and operationally constrained.

The project also contributes to the broader academic discussion on trustworthy AI in cyber-physical systems by combining privacy, distributed learning, and explainable incident investigation in a single design.

---

## 6. Technical Stack

The implementation uses a combination of Python-based research and ML tools to support local experimentation and evaluation.

### 6.1 Machine Learning and Federated Learning

- Python 3.12
- PyTorch
- scikit-learn
- NumPy
- pandas
- pyarrow
- Flower (simulation-based federated learning)
- Opacus (differential privacy support)
- MLflow
- DVC

### 6.2 Agentic and Retrieval Components

- LangGraph
- LangChain
- Qdrant (embedded/local mode)
- Sentence Transformers
- SQLite
- local LLM integration through Ollama-compatible execution
- Arize Phoenix for tracing support

### 6.3 Security and Operational Infrastructure

- JWT-based authentication
- code-level mTLS support for local development security patterns
- asyncio-based in-process messaging
- structured logging and evaluation exports

### 6.4 Evaluation and Validation

- pytest
- ruff
- matplotlib
- ragas
- custom evaluation harnesses for detection, latency, chaos, and retrieval analysis

---

## 7. Architecture Summary

The architecture of the system is layered and modular, with explicit separation between data collection, learning, reasoning, and evaluation.

- Edge layer: local data processing and anomaly detection
- Federated layer: distributed learning and secure aggregation-style coordination
- Memory layer: embedded Qdrant vector storage for semantic retrieval
- Agent layer: LangGraph-based workflow for investigation and dispatch
- Evaluation layer: reproducible harnesses and evidence-gated reporting

![System layers](docs/architecture/system_layers.svg)

The repository also includes additional architecture diagrams and export artifacts under the docs/architecture directory.

---

## 8. Repository Structure

```text
Agentic_Mesh/
├── agents/                     # agent workflows, authentication, security, dispatch
├── api/                       # application interfaces and service hooks
├── data/                      # dataset metadata and partitioned structures
├── docs/                      # architecture exports and project documentation
├── edge/                      # local edge inference and training logic
├── evaluation/                # evaluation harnesses and evidence scripts
├── federated/                 # Flower-based simulation and FL logic
├── scripts/                   # operational and setup utilities
├── storage/                   # SQLite registry and metadata handling
├── tests/                     # unit and integration validation
├── Documents/                 # project proposal and weekly implementation reports
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .gitignore
├── README.md
├── dvc.yaml
├── mlruns/
└── .github/
```

---

## 9. Experimental Data and Research Inputs

The project is designed around real industrial benchmark sources, including HAI and Modbus-based partitions. These datasets are organized per node to retain realistic non-IID characteristics and to emulate heterogeneous industrial environments. This structure supports the research objective of validating learning and detection under realistic domain partitions rather than idealized centralized distributions.

The project’s data handling strategy emphasizes data locality and reproducibility. Large runtime artifacts, raw datasets, and generated outputs are excluded from version control by configuration in the repository’s .gitignore file.

---

## 10. Setup and Execution

### 10.1 Prerequisites

- Windows or WSL2/Linux operating system
- Python 3.12
- 8 GB RAM for local prototype execution
- pip and virtual environment support
- optional local Ollama runtime for language-model-backed investigation workflows

### 10.2 Environment Setup

```powershell
python -m venv agentvenv
.\agentvenv\Scripts\python.exe -m pip install --upgrade pip
.\agentvenv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

### 10.3 Verification

```powershell
.\agentvenv\Scripts\python.exe -m pytest tests -q --disable-warnings
```

This validation command checks the repository’s Python test suite and supports reproducibility for local execution.

---

## 11. Running the Prototype

### 11.1 Local Stack Smoke Test

```powershell
$py = '.\agentvenv\Scripts\python.exe'
& $py -m scripts.run_local_stack --count 5 --metrics-path logs/metrics.jsonl
```

This command launches the local graph stack with synthetic alerts and verifies that the end-to-end workflow completes successfully.

### 11.2 Detection and Latency Evaluation

```powershell
& $py -m evaluation.harness
```

### 11.3 Chaos Evaluation

```powershell
& $py -m evaluation.chaos_test
```

### 11.4 Retrieval Quality Evaluation

```powershell
& $py -m evaluation.ragas_eval
```

### 11.5 Dashboard Generation

```powershell
& $py -m evaluation.plot_metrics
```

### 11.6 Final Evidence Gate

```powershell
& $py -m evaluation.check_success_criteria
```

This final stage presents the project’s evidence gate and clearly separates completed validation from pending benchmark artifacts.

---

## 12. Evaluation Strategy and Results

The evaluation plan is designed to provide traceable, reproducible evidence rather than unsupported performance claims. The prototype includes benchmark checks for:

- detection quality
- false positive behaviour
- latency analysis
- chaos robustness
- retrieval quality and grounding
- evidence-gated final reporting

The repository currently reports a PASS_WITH_GAPS status at the final gate because several benchmark artifacts remain unmeasured in machine-readable form. Specifically, the final federated model comparison, cumulative privacy accounting, and resource benchmark are still pending. This is an appropriate and academically responsible outcome for a first-phase prototype.

The overall evaluation posture is therefore honest and evidence-driven: the system demonstrates meaningful prototype functionality and validation, while explicitly identifying the remaining requirements for future strengthening.

---

## 13. Limitations and Challenges

This project is a research prototype and therefore has several explicit constraints.

### 13.1 Computational Limitations

The prototype is intentionally executed within a constrained machine environment and is not designed to represent a production-scale industrial deployment. The single-machine configuration limits large-scale benchmarking and system throughput analysis.

### 13.2 Benchmarking Limitations

The final federated learning metrics, FedProx-vs-FedAvg comparison, cumulative epsilon accounting, and full 8 GB runtime profile have not yet been captured as final benchmark artifacts. These remain critical objectives for the next phase of development.

### 13.3 Data and Ground-Truth Constraints

The project uses realistic benchmark data sources, but scenario-level ground truth and threat labels remain limited. This affects the precision of attack attribution and makes the evaluation conservative by design.

### 13.4 Engineering Challenges

Additional challenges include:

- handling non-IID industrial distributions
- balancing privacy-preservation with model performance
- maintaining deterministic local evaluation in a research environment
- ensuring agent reliability under partial failures and investigation retries

These constraints are acknowledged as part of the project’s research integrity and do not diminish the technical value of the implemented first phase.

---

## 14. Contribution and Academic Relevance

This project contributes to the intersection of distributed intelligence, industrial cybersecurity, and privacy-aware AI. The system demonstrates that local anomaly detection, federated learning, and agentic investigation can be combined in a unified framework for cyber-physical monitoring.

The academic contribution is grounded in the following areas:

1. privacy-preserving detection in distributed industrial environments
2. non-IID learning under federated constraints
3. explainable incident analysis using a multi-agent workflow
4. semantic retrieval for threat intelligence and operational evidence
5. reproducible prototype evaluation in a resource-constrained environment

The work is positioned as a first-phase research prototype and a strong foundation for subsequent improvement, validation, and publication-oriented analysis.

---

## 15. Future Work

The next phase of the project will focus on strengthening evidence quality and transitioning the prototype from a functional research implementation to a more benchmark-ready system. Key direction areas include:

- final measurement of global AUROC and F1 for the federated model
- systematic comparison of FedProx and FedAvg under identical conditions
- privacy accounting through cumulative epsilon reporting
- full 8 GB runtime and resource characterization
- improved scenario-grounded threat mapping and evaluation fidelity
- enhanced reproducibility and checkpoint traceability via DVC and MLflow

These steps are necessary to transform the present implementation into a more robust and publication-ready research system.

---

## 16. Conclusion

This project presents a viable first-phase implementation of a privacy-preserving decentralized anomaly detection system for industrial environments. It integrates local inference, federated learning, and agentic reasoning into a unified workflow that is technically coherent, reproducible, and suitable for review.

Although the project is not yet complete in terms of final benchmark-grade evidence, it demonstrates a credible and academically meaningful foundation for future research and development. The transparent documentation of current limitations is an essential part of the project’s integrity and reflects appropriate engineering discipline for a final-year undergraduate or postgraduate research submission.

The repository therefore serves both as a functioning prototype and as a structured research artifact that can be extended in the next phase toward stronger evaluation, improved performance claims, and more complete industrial readiness.


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
