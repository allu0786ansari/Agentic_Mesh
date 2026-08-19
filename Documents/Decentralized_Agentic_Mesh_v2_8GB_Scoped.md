**PROJECT PROPOSAL**

**Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics**

*A Federated, Multi-Agent AI System for Distributed Anomaly Detection*

**Author:** Allaudin Ansari

**Document Type:** Industrial Research & Engineering Proposal

**Version:** 2.1 (Local Development Scope — 8GB RAM Single-Machine Build)

**Date:** August 2026

**Domain:** Industrial IoT / Fintech / Cybersecurity / Federated AI

---

**Note on this revision:** Version 2.0 specified a production-grade infrastructure stack (K3s cluster, Kafka, NATS JetStream, PostgreSQL, Prometheus/Grafana/OpenTelemetry, Phi-4 14B) designed around real deployment constraints. Version 2.1 preserves every algorithmic and architectural contribution of that design — FedProx, DP-SGD with Opacus, the four-agent LangGraph mesh, MITRE ATT&CK RAG — while replacing infrastructure components that require multi-node clusters or persistent daemon services with functionally equivalent single-process substitutes that run within an 8GB RAM development machine. Where a swap was made, the original production-grade choice is noted as a future scaling path.

**Table of Contents**

**Executive Summary**

**1. Problem Statement**

1.1 The Privacy Paradox in Industrial AI

1.2 The Static Model Failure

1.3 The Objective

**2. Solution Overview**

2.1 Phase A — Local Intelligence

2.2 Phase B — The Learning Engine

2.3 Phase C — The Reasoning Layer

**3. Gap Analysis Against Existing Approaches**

**4. Technology Stack**

4.1 AI and Machine Learning Core

4.2 Agentic Orchestration

4.3 Infrastructure and Deployment

4.4 MLOps and Observability

4.5 Evaluation and Ground Truth

**5. System Architecture**

5.1 Architecture Layers

5.2 Data Flow

5.3 Security and Privacy Design Decisions

**6. Implementation Plan**

**7. Evaluation Framework**

**8. Market Relevance and Business Impact**

**9. Risk Register**

**10. Success Criteria**

**11. Conclusion**

---

# Executive Summary

Modern industrial environments — spanning manufacturing, financial services, energy utilities, and healthcare — generate vast quantities of sensitive operational data. The critical challenge is that the data necessary to train accurate AI models cannot legally or safely leave the systems where it originates. At the same time, isolated local models are too narrow to detect sophisticated, multi-stage threats that span across multiple systems and nodes.

This proposal describes the design, architecture, and implementation plan for a Decentralized Agentic Mesh: a distributed AI system that performs privacy-preserving anomaly detection and autonomous root-cause analysis across a network of private edge nodes. The system combines three advanced disciplines — Federated Learning for privacy-safe model training, Differential Privacy for mathematical privacy guarantees, and a Multi-Agent Reasoning Layer for autonomous investigation — into a single integrated platform.

The result is a system that detects complex distributed threats without ever accessing raw data, satisfies GDPR and HIPAA compliance requirements by design, operates at the edge for sub-second response times, and continuously learns from new operational patterns without manual retraining. This version of the proposal implements the full architecture as a **single-process simulation** — six federated clients, four reasoning agents, and all shared services run within one Python environment on an 8GB RAM development machine — with a clearly documented path to multi-node/containerized deployment as future work.

# 1. Problem Statement

## 1.1 The Privacy Paradox in Industrial AI

The most valuable AI use cases in industry — fraud detection, predictive maintenance, intrusion detection, and clinical decision support — all depend on sensitive data. Financial transaction records, factory sensor logs, medical records, and industrial control system telemetry are categories of data that are tightly regulated under frameworks such as GDPR, HIPAA, and the NIS2 Directive.

The traditional approach to training AI models requires centralising all data in one location — typically a cloud data lake — where a model can be trained over the combined dataset. This approach creates two immediate and severe problems. First, moving sensitive data across network boundaries introduces legal exposure under data residency laws. Second, aggregating data from multiple organisations or business units into a single location creates an extremely high-value target for attackers. A single breach of a centralised data lake can expose the operational data of thousands of machines or millions of customers simultaneously.

The result is what practitioners call the privacy paradox: the data that would produce the most accurate AI models is precisely the data that cannot be moved to train them.

## 1.2 The Static Model Failure

Even where data centralisation is legally permissible, the standard approach of training a single static model produces systems that are fundamentally inadequate for detecting modern industrial threats. A coordinated multi-stage cyber attack — such as an Advanced Persistent Threat (APT) campaign — does not produce a single large anomaly at one location. Instead, it produces a sequence of small, individually unremarkable events distributed across many systems over hours or days. A traditional model trained on one node's data has no visibility into what is happening on other nodes and cannot connect distributed observations into a coherent threat picture.

Furthermore, traditional models are retrained on a monthly or quarterly schedule by a data scientist running a manual batch process. In between retraining cycles, the model's accuracy degrades as operational conditions evolve — a phenomenon called model drift. By the time a new threat pattern is incorporated into the model, it may already have been exploited.

## 1.3 The Objective

The objective of this project is to build a system that simultaneously resolves both problems. The system must be capable of detecting complex, distributed anomaly patterns across multiple private data silos without ever accessing raw data from any node, while using a network of autonomous reasoning agents to investigate findings, cross-reference threat intelligence, and propose corrective actions in real time.

# 2. Solution Overview

The proposed system is organised into three research modules. This is not a generic “AI stack”; it is a tightly scoped scientific architecture designed to solve a real problem in privacy-preserving industrial analytics: how to detect distributed anomalies, learn across private edge data, and explain the likely root cause without ever centralising raw telemetry.

| **Module** | **Research Focus** | **Primary Contribution** |
|---|---|---|
| Module 1 — Federated Edge Detection | Local anomaly detection on private node partitions | Detects abnormal industrial behaviour without centralising raw data |
| Module 2 — Privacy-Preserving Federated Learning | FedProx + DP-SGD + SecAgg | Learns a shared model under non-IID and privacy-constrained conditions |
| Module 3 — Knowledge-Grounded Agentic Attribution | LangGraph investigation + MITRE retrieval + causal reasoning | Translates anomalies into explainable forensic conclusions and remediation guidance |

This three-module structure also provides the foundation for a stronger research narrative and a publication-grade evaluation plan. The project is intentionally positioned not as a generic “agentic orchestration demo,” but as a privacy-aware, distributed intelligence framework for industrial cyber-physical systems. The novelty lies in coupling federated learning and differential privacy with a reasoning layer that explains which attack pattern is most consistent with cross-node evidence.

## 2.1 Module 1 — Federated Edge Detection

Six simulated edge nodes are instantiated as **Flower simulation clients** — lightweight virtual participants running as Python coroutines within a single process, rather than as separate Docker containers under a Kubernetes cluster. This is the standard approach used in federated learning research to benchmark algorithms like FedProx without provisioning real multi-node infrastructure; it produces algorithmically identical results to a true multi-machine deployment while fitting comfortably within 8GB RAM. Each simulated node operates an independent local anomaly detection model trained exclusively on that node's data partition. No raw data ever leaves the node's logical boundary within the simulation.

The node allocation deliberately preserves both physical-process heterogeneity and network-device heterogeneity: Nodes 01–03 represent distinct HAI process/sensor partitions and use VAEs; Node 04 represents IED1A traffic; Node 05 represents IED1B traffic; and Node 06 represents SCADA HMI traffic. Nodes 04–06 use Isolation Forest over extracted Modbus network-flow windows. This six-way allocation avoids combining unrelated device distributions and provides a cleaner non-IID experimental design.

Two model types are deployed based on the nature of each node's data:

- Variational Autoencoder (VAE): Used for nodes producing multivariate time-series data, such as sensor and process telemetry from industrial control systems. VAEs learn the latent distribution of normal operational behaviour and flag deviations by reconstruction error.

- Isolation Forest: Used for nodes producing tabular/network flow data, such as protocol-level traffic snapshots. Isolation Forest identifies outliers by measuring how quickly a data point can be isolated in a random partitioning tree structure.

A node metadata registry (see Section 4.2) maintains the data type profile of each node and governs which model architecture is instantiated at each location. The output of each local model is not a raw alert — it is an Insight Embedding: a compressed mathematical representation of the anomaly in a fixed-dimensional latent space. Insight Embeddings contain no Personally Identifiable Information and no raw operational values. They encode only the pattern and severity of the deviation detected.

Model selection quality is measured using AUROC and F1-score against labeled holdout events from the ground truth dataset, allowing objective comparison of VAE and Isolation Forest performance per node type.

## 2.2 Phase B — The Learning Engine (Federated Learning with Privacy Guarantees)

Phase B implements the federated training loop using the Flower framework's simulation runtime. Rather than transmitting data, each simulated node computes gradient updates — representing what the local model has learned — and transmits only these updates to a central Flower server process (also run locally).

Three critical enhancements are made over a standard federated approach to ensure correctness and security:

- FedProx instead of FedAvg: Standard Federated Averaging (FedAvg) assumes that data distributions across nodes are similar (IID). Industrial environments are the opposite: a process-sensor node's data distribution is completely different from a network-flow node's. FedAvg degrades severely under these Non-IID conditions. FedProx adds a proximal regularisation term to the local objective function, which limits how far local model weights can deviate from the global model during each round. This produces a stable global model even with highly heterogeneous node data.

- Differential Privacy via DP-SGD: Before any gradient update is transmitted, Differentially Private Stochastic Gradient Descent (DP-SGD) is applied. This algorithm clips each gradient vector to a maximum L2 norm and adds calibrated Gaussian noise. The cumulative privacy cost is tracked as an epsilon budget across all training rounds. A total epsilon budget of 10.0 is set for full training — a threshold consistent with production federated learning deployments by Google and Apple for non-medical industrial data, and well within the range accepted by GDPR technical safeguard guidance. For healthcare applications, this budget would be tightened to epsilon of 1.0 or below, with a corresponding tuning of gradient clipping and noise multiplier to preserve model accuracy. This provides a formal, mathematical guarantee that it is computationally infeasible to reconstruct any individual data record from the transmitted gradients.

- Secure Aggregation (SecAgg): Flower's built-in Secure Aggregation protocol ensures the server can only observe the sum of all gradients, never any individual node's update in plaintext. This prevents even a compromised aggregation server from inferring node-level information. This mechanism is unchanged by running in simulation mode — SecAgg operates identically whether clients are separate machines or coroutines within one process.

The federated loop runs until one of two stopping conditions is met: a maximum round budget defined in the experiment configuration, or a convergence criterion based on the change in global validation loss falling below a defined threshold across three consecutive rounds. Simulated nodes that are configured to be slow or unresponsive during a round are handled by a stragglers timeout with graceful async fallback — the global model update proceeds with the gradients received, and the missing node is flagged for retry in the next round.

Model versioning is managed by DVC and MLflow, which record every training round's hyperparameters, gradient statistics, epsilon budget consumption, and model weights. This creates a complete audit trail and enables rollback to any prior checkpoint.

## 2.3 Phase C — The Reasoning Layer (Agentic Mesh)

Phase C implements the autonomous investigation and response capability using LangGraph, which supports stateful, cyclic multi-agent workflows. Unlike linear agent pipelines, LangGraph allows agents to iterate, delegate back, and revise conclusions — which is necessary for the recursive nature of threat investigation.

The Agentic Mesh consists of four specialised agents:

- Triage Agent: Continuously monitors the output of the Federated Model server. When the global model flags a potential anomaly pattern, the Triage Agent evaluates its severity score against a configurable threshold and routes it to the Investigator Agent if warranted. It maintains a short-term event buffer in embedded Qdrant to correlate multiple weak signals into a composite alert.

- Investigator Agent: Queries specific edge nodes for additional Insight Embeddings related to the suspected anomaly window. The investigation is bounded by a maximum of three query hops and a 500ms timeout per hop to ensure the process does not introduce latency that renders findings irrelevant. A circuit breaker pattern prevents the agent from hanging indefinitely on an unresponsive node — after two failed attempts, the node is marked temporarily unavailable and the investigation proceeds with available evidence.

- Security Agent: Receives the Investigator's compiled evidence set and cross-references it against a local Knowledge Base. This knowledge base is constructed by chunking MITRE ATT&CK for ICS technique descriptions and NIST SP 800-82 guidelines, embedding each chunk using Sentence Transformers, and storing the resulting vectors in an embedded Qdrant collection with structured payload fields for technique ID, tactic category, and applicable platform. At query time, the evidence package is embedded and used to retrieve the closest matching technique entries by cosine similarity. The agent identifies the closest matching threat pattern and generates a structured remediation recommendation with a confidence score.

- Action Dispatcher: Translates the Security Agent's recommendation into a structured action payload — which may include isolating a node, triggering an alert to an operator, or adjusting a local model's anomaly threshold — and dispatches it internally via an **asyncio.Queue-based message bus** (with the dispatch step also representable as a direct LangGraph state transition to the target agent node). This delivers the same at-least-once, ordered handoff guarantee needed for a single-machine simulation without running a standalone broker process.

Inter-agent authentication (JWT) and transport encryption (mTLS) remain part of the architecture as documented in Section 5.3, since these are lightweight, in-process/library-level mechanisms rather than standalone services, and do not materially affect the RAM budget. Agent reasoning traces are recorded end-to-end by Arize Phoenix, providing full observability into every decision step, tool call, and conclusion reached during an investigation cycle.

Shared agent memory is maintained in an **embedded Qdrant instance** (`QdrantClient(path="./qdrant_data")`, running in-process with no separate server container), allowing the Investigator Agent to retrieve semantically similar past anomaly events for comparative reasoning. Structured metadata — node identifiers, timestamps, model versions, and alert classifications — is stored in **SQLite**, which provides the same relational query capability PostgreSQL offered for this project's scale, with zero daemon/service overhead.

# 3. Gap Analysis Against Existing Approaches

The following table contrasts the capabilities of the traditional centralised AI approach currently standard in industrial environments against the proposed Decentralised Agentic Mesh. Each row identifies a specific capability dimension and explains why the current industry standard falls short.

| **Capability Dimension** | **Current Industry Standard** | **This System** |
|---|---|---|
| Data handling | All data moved to a central cloud data lake. High egress costs, high breach risk, legally blocked in regulated industries. | Federated Learning: raw data never moves. Only gradient updates are shared, with DP-SGD noise and SecAgg cryptographic protection. |
| Privacy guarantee | Anonymisation or pseudonymisation applied before transfer. Easily reversed with auxiliary data. Not a formal guarantee. | Mathematical epsilon-delta Differential Privacy applied per training round. Formally provable. GDPR Article 89 and HIPAA technical safeguard compliant. |
| FL algorithm | Not applicable in centralised systems. When used, FedAvg is the default. | FedProx with proximal regularisation for stable convergence on Non-IID industrial data distributions. |
| Anomaly detection scope | Single-node models detect local anomalies only. No cross-node correlation. | Global federated model incorporates patterns from all nodes. Detects coordinated, multi-stage threats invisible to any individual node. |
| Model learning cycle | Batch retraining by a data scientist on a monthly or quarterly schedule. | Continuous federated rounds triggered by operational events. Near real-time adaptation to new threat patterns and equipment degradation signatures. |
| Incident response | Model flags an anomaly. A human analyst logs in, reads a dashboard, and manually investigates. Response time: hours to days. | Agentic Mesh autonomously investigates, cross-references threat intelligence, and dispatches a structured remediation within sub-second to low-second timeframes. |
| Fault tolerance | Monolithic model: if the central server or data pipeline fails, the entire system halts. | Modular mesh: each edge node continues local detection independently. If one agent fails, the rest continue. Circuit breaker prevents cascade failure. |
| Observability | Dashboard visualisations of model outputs. No visibility into model reasoning. | Full reasoning chain tracing via Arize Phoenix. Every agent tool call, evidence evaluation, and decision is logged and auditable. |
| Compliance posture | Legal teams must audit data transfer agreements, cross-border data flows, and cloud vendor DPAs for every jurisdiction. | Data residency compliance is architecturally inherent. No data transfer agreements required. Audit trail is built into the federated training log. |

### Research novelty and publication framing
This project is designed to be publication-ready rather than only demonstrative. The novelty is not the use of federated learning or LLMs in isolation — both are established research areas — but the combination of three scientific components into a single privacy-aware industrial threat intelligence system:

1. Federated, Non-IID learning for edge industrial telemetry using FedProx and DP-SGD.
2. Cross-node anomaly evidence fusion to detect multi-stage attack patterns that are invisible to any single node.
3. Knowledge-grounded agentic attribution that maps distributed evidence to MITRE ATT&CK for ICS techniques and returns actionable remediation recommendations.

This creates a strong research thesis: a privacy-preserving distributed system that is both accurate and explainable under strict edge-data constraints. To support publication quality, the system must be evaluated with ablation studies comparing the full pipeline against the following baselines: local-only models, centralised model training, FedAvg instead of FedProx, no-DP and no-SecAgg variants, and no-agent/heuristic attribution baselines. These ablations are necessary to prove the contribution and are explicitly included in the evaluation plan.

### Dockerised reproducibility and execution model
The system is intentionally designed for reproducibility and review. While the research architecture can be run in a lightweight local simulation mode for development, the final project should be packaged using Docker and Docker Compose so that all services — the Flower simulation, Qdrant, MLflow, Phoenix, and the API/agent services — can be started consistently on any compatible machine. This is essential for a major project and for a publication-quality codebase because it ensures that other researchers, supervisors, and examiners can execute the pipeline without manually installing a complex dependency stack or debugging environment drift.

Docker is therefore not a cosmetic choice; it is the deployment and reproducibility layer that makes the system executable, portable, and reviewable. The local 8GB machine build remains useful for rapid development and validation, but the final research pipeline should be containerised for repeatable execution, benchmarking, and demonstration.

# 4. Technology Stack

The following tables define the complete set of technologies to be used in this project, organised by functional layer. Tool selection favours frameworks that preserve the algorithmic fidelity of a production deployment while running entirely within an 8GB RAM single-machine development environment.

**4.1 AI and Machine Learning Core**

| **Tool** | **Role** | **Justification** |
|---|---|---|
| PyTorch 2.x | Neural network architecture for VAE models | Industry standard deep learning framework. Native DP-SGD support via Opacus library. Production-ready. |
| Scikit-learn | Isolation Forest implementation | Mature, well-tested. Consistent API with PyTorch pipeline for model registry. |
| Opacus (Meta AI) | DP-SGD implementation | Purpose-built Differential Privacy library for PyTorch. Tracks epsilon budget automatically per training step. Supports Renyi DP accounting for tighter bounds. |
| Flower (flwr) | Federated Learning framework, run in **simulation mode** | De facto industry standard for federated ML. Simulation mode runs all clients as coroutines/lightweight processes on one machine — the standard way FL algorithms are benchmarked in research settings. Supports FedProx strategy natively. Same code path scales to a real multi-node deployment later with only a configuration change. |
| **Phi-3.5-mini (Microsoft, 3.8B)** | **Primary Small Language Model for agents** | **INT4-quantised via Ollama, ~2.2–2.5GB resident RAM. Strong instruction-following and structured/JSON tool-calling performance for its size — the property the agent mesh actually depends on, more than raw parameter count. Replaces Phi-4 14B, which requires 8GB+ RAM on its own and is infeasible alongside the rest of the stack on this machine.** |
| Llama 3.2 (Meta, 3B) | Open-licence fallback SLM option | Alternative to Phi-3.5-mini for deployments with strict open-source licensing requirements. Comparable capability profile at similar scale. |
| Sentence Transformers | Embedding generation for agent memory and KB | Fast semantic embedding for Insight Embedding storage, MITRE ATT&CK knowledge base indexing, and retrieval in Qdrant. |

**4.2 Agentic Orchestration**

| **Tool** | **Role** | **Justification** |
|---|---|---|
| LangGraph | Multi-agent state machine orchestration | Supports cyclic graphs, conditional branching, and stateful agent loops. Essential for recursive investigation workflows that cannot be expressed as linear chains. |
| LangChain | Tool abstractions and LLM interface | Provides standardised tool calling, prompt templates, and LLM provider abstraction used by LangGraph agents. |
| Ollama | Local SLM serving runtime | Serves Phi-3.5-mini and fallback SLMs locally via a REST API. Handles model loading, INT4/INT8 quantisation, and CPU routing. Eliminates per-token cloud API costs and keeps model inference fully on-premise. |
| **Qdrant (embedded mode)** | **Vector database for agent memory and knowledge base** | **Runs in-process via `QdrantClient(path="./qdrant_data")` — identical API and search quality to the server deployment, with zero container/daemon overhead. Used for both Insight Embedding memory and the MITRE ATT&CK knowledge base collection.** |
| **SQLite** | **Relational metadata storage** | **Stores node registry, alert records, model versions, and structured investigation outputs. Replaces PostgreSQL + pgvector — at this project's data volume, a standalone database daemon adds ~200–400MB of RAM overhead with no functional benefit, since vector search is already handled by Qdrant.** |
| **`asyncio.Queue` message bus / LangGraph state passing** | **Message broker for agent and edge communication** | **Provides ordered, at-least-once delivery of messages between the Triage, Investigator, Security, and Dispatcher agents within a single process. Replaces NATS JetStream, whose value (cross-machine pub/sub, persistent streams) is not needed when all agents run on one machine.** |
| **Python generator / `asyncio.Queue` telemetry replay** | **Event ingestion pipeline for edge nodes** | **Replays the ground-truth dataset row-by-row as a simulated live telemetry stream, feeding the preprocessing pipeline exactly as Kafka would but without a JVM-based broker process running in the background. Replaces Apache Kafka.** |
| FastAPI | REST microservice layer for each agent | High-performance async Python web framework. Used to expose each agent's interface and the FL server's aggregation endpoint locally. |

**4.3 Infrastructure and Deployment**

| **Tool** | **Role** | **Justification** |
|---|---|---|
| **Flower simulation runtime** | **Client/node orchestration for edge nodes** | **Replaces Docker + K3s for this project's local simulation. Six virtual clients run as coroutines within one Python process, which is algorithmically equivalent to six containerised nodes for the purpose of evaluating FedProx and DP-SGD, without the ~1.5–3GB of container/orchestration overhead a real K3s cluster would add on top of everything else running locally.** |
| **Docker + Docker Compose** | **Executable, portable research runtime** | **Used for the reproducibility layer, not as the scientific contribution itself. Containerises the Flower server, Qdrant, MLflow, Phoenix, and API agents for consistent execution across machines. The local simulation remains available for rapid prototyping, but the final pipeline is meant to be run in containers for review and benchmarking.** |
| mTLS (via cert-manager) | Transport security between agents | Mutual TLS with automatically rotated certificates ensures all inter-agent and node-to-FL-server communication is encrypted and mutually authenticated. Implemented at the library/code level for this build rather than via a cluster-managed certificate authority. |
| JWT (via python-jose) | Agent request authorisation | Short-TTL JWT tokens issued per investigation session prevent replay attacks and unauthorised node queries. |
| GitHub Actions | CI/CD pipeline | Automated testing and linting on push to main branch. Runs on GitHub's hosted runners, so it consumes no local RAM or compute during development. |
| ~~Helm~~ | ~~Kubernetes package management~~ | **Removed. With no K3s cluster to orchestrate, there is nothing for Helm to manage in this build.** |

**4.4 MLOps and Observability**

| **Tool** | **Role** | **Justification** |
|---|---|---|
| MLflow | Experiment tracking and model registry | Records every federated round: hyperparameters, gradient norms, epsilon budget consumption, validation metrics, and model artifacts. Local file-based backend — negligible RAM footprint. Enables reproducibility and rollback. |
| DVC (Data Version Control) | Dataset and model versioning | Versions training datasets, evaluation sets, and model checkpoints alongside code in Git. Ensures full experiment reproducibility. |
| Arize Phoenix | LLM and agent reasoning traceability | Open-source, locally-run observability platform for LLM applications. Records every agent tool call, prompt, completion, and chain-of-thought step. Enables post-hoc audit of investigation decisions. Retained as the one observability tool with direct research/evaluation value. |
| **Structured JSON/CSV logging + post-hoc matplotlib reporting** | **System and FL metrics visualisation** | **Replaces Prometheus + Grafana + OpenTelemetry. Round timing, gradient norms, agent latency, and queue depth are written to structured log files during runs, then charted with matplotlib for the evaluation report. Delivers the same dashboards an examiner or reader needs to see, without three persistent background daemons (Prometheus alone grows continuously as a live-scraping time-series database) competing for RAM against the LLM and training processes.** |

**4.5 Evaluation and Ground Truth**

| **Tool / Dataset** | **Role** | **Justification** |
|---|---|---|
| **HAI Dataset (HIL-based Augmented ICS Security Dataset)** | **Primary ground truth — process/sensor telemetry, VAE nodes** | **Open-access dataset (no approval/NDA process required) built from a real hardware-in-the-loop ICS testbed emulating steam-turbine power generation and pumped-storage hydropower, with labeled multi-stage attack scenarios. Feeds the VAE-based nodes directly, and its multiple distinct physical processes naturally support Non-IID partitioning across simulated clients. Replaces SWAT, whose access requires a formal request to iTrust/SUTD that can take days to weeks to approve.** |
| **CIC Modbus Dataset 2023** | **Secondary ground truth — network/protocol traffic, Isolation Forest nodes** | **Freely downloadable from the Canadian Institute for Cybersecurity (citation required, no approval process). Covers nine Modbus protocol attack types — reconnaissance, query flooding, false data injection, replay, and others — on a simulated substation network, with attacks explicitly mapped to MITRE ATT&CK for ICS techniques. This alignment directly benefits the Security Agent's MITRE-matching evaluation. Note: attack labels are provided as scenario/attack-log windows rather than a per-packet label column, so a labeling step during preprocessing (Phase 2) is required. Replaces CICIDS 2017/2018, which is a general network-IDS dataset rather than an ICS/SCADA-specific one.** |
| MITRE ATT&CK for ICS | Threat intelligence knowledge base | Authoritative framework for ICS/SCADA adversary tactics, techniques, and procedures. Technique descriptions are chunked, embedded via Sentence Transformers, and stored in embedded Qdrant with technique ID, tactic, and platform payload fields. Used by Security Agent for threat classification and remediation mapping. |
| Custom Evaluation Harness (Python) | Agent reasoning evaluation | Bespoke evaluation framework measuring: attack detection rate, false positive rate, root-cause attribution accuracy, mean time to investigation completion, and epsilon budget efficiency per detected event. |
| RAGAS (supplementary) | RAG retrieval quality measurement | Used specifically to evaluate the Qdrant memory retrieval quality within agent workflows, not for overall system evaluation. |

# 5. System Architecture

## 5.1 Architecture Layers

The system is organised into five layers, each with a distinct responsibility. Data and control flow upward through the layers; model updates and directives flow downward. All layers run as processes/threads within a single machine for this build.

| **Layer** | **Components** | **Responsibility** |
|---|---|---|
| Edge Layer | Flower simulation clients, VAE / Isolation Forest, asyncio-based telemetry replay generator, asyncio.Queue publisher | Local anomaly detection, Insight Embedding generation, simulated raw telemetry ingestion |
| Federated Learning Layer | Flower server (simulation mode), FedProx strategy, Opacus DP-SGD, SecAgg protocol, MLflow | Privacy-preserving gradient aggregation, global model maintenance, convergence tracking |
| Agentic Mesh Layer | LangGraph orchestrator, Triage / Investigator / Security / Dispatcher agents, Phi-3.5-mini SLM via Ollama | Autonomous anomaly investigation, threat classification, remediation dispatch |
| Shared Services Layer | Embedded Qdrant, SQLite, asyncio.Queue message bus, FastAPI microservices | Agent memory, metadata persistence, message delivery, service APIs |
| Observability Layer | Arize Phoenix, structured JSON/CSV logs, DVC | Reasoning-trace tracing, metrics logging, model versioning, experiment audit |

## 5.2 Data Flow

The end-to-end data flow through the system proceeds as follows:

1. Raw telemetry (sensor readings or Modbus network flows) is replayed from the HAI or CIC Modbus 2023 dataset by a Python generator/asyncio.Queue producer at each simulated edge node, standing in for a live industrial data source.

2. The edge node's preprocessing pipeline consumes from this queue, applies windowed feature extraction and normalisation, and passes the feature vector to the local anomaly detection model (VAE or Isolation Forest).

3. The local model produces an Insight Embedding — a fixed-dimensional latent representation of any detected anomaly — along with a reconstruction error score or anomaly score. This embedding is stored locally in the node's embedded Qdrant instance and published to the internal asyncio.Queue message bus.

4. At the end of a federated training round trigger condition (elapsed time or local data threshold), the edge node computes gradient updates using DP-SGD with gradient clipping and Gaussian noise, then transmits the noisy gradients to the Flower server (running in the same simulation) via SecAgg.

5. The Flower server aggregates gradients from all participating simulated nodes using FedProx, updates the global model, logs round statistics to MLflow, checks convergence criteria, and distributes the updated global model back to all nodes.

6. The updated global model is evaluated against the held-out test partition of the HAI or CIC Modbus 2023 dataset. If the anomaly score for any node cluster exceeds the Triage Agent's threshold, an investigation session is opened.

7. The Triage Agent queries embedded Qdrant for related historical embeddings, assigns severity, and delegates to the Investigator Agent with a session JWT.

8. The Investigator Agent queries up to three additional simulated nodes for corroborating Insight Embeddings, applies circuit breaker logic for non-responsive nodes, and compiles a structured evidence package.

9. The Security Agent embeds the evidence package, retrieves the closest MITRE ATT&CK for ICS technique entries from embedded Qdrant by cosine similarity, and generates a structured remediation recommendation with a confidence score.

10. The Action Dispatcher encodes the recommendation as a JSON payload and publishes it via the asyncio.Queue action topic for consumption by the target node's handler or the operator alert log.

11. All agent steps are traced by Arize Phoenix. Round and agent metrics are written to structured JSON/CSV logs and rendered as matplotlib charts for the evaluation report.

## 5.3 Security and Privacy Design Decisions

The following architectural decisions have been made specifically to satisfy privacy and security requirements:

- No raw data leaves any simulated edge node's logical boundary under any circumstances. The only outbound data from a node is DP-SGD gradient updates (during federated rounds) and Insight Embeddings (during agent investigations). Both are non-reversible mathematical representations.

- Gradient updates are protected by two independent mechanisms: DP-SGD noise injection at the node and SecAgg aggregation at the server. An adversary would need to simultaneously compromise both the node and the server to attempt gradient inversion.

- Agent-to-node queries are authenticated with short-TTL JWT tokens and transport-encrypted with mTLS at the code level. A compromised agent cannot query nodes without a valid session token issued by the orchestrator.

- The epsilon privacy budget is tracked cumulatively across all training rounds using Opacus. When the budget is exhausted, the federated round terminates and a new privacy accounting epoch begins. This ensures the total privacy cost remains bounded over the lifetime of the system.

- All model inference for agent reasoning is performed locally via Ollama. No prompt content, embeddings, or investigation data are transmitted to external cloud LLM APIs.

- All actions dispatched by the Action Dispatcher are logged with the full reasoning trace, enabling post-hoc legal and compliance review of every automated decision.

# 6. Implementation Plan

## Phase 1 — Foundation (Weeks 1 to 3)

- Set up the Flower simulation environment with six virtual edge-node clients as Python coroutines: three HAI process clients and three Modbus device clients.

- Implement the asyncio-based telemetry replay generator and the internal asyncio.Queue message bus for inter-service communication.

- Apply for HAI dataset access (open, typically fast) and download CIC Modbus Dataset 2023 (immediate, citation-only). Partition both into per-node allocations to simulate Non-IID distribution.

- Implement the node metadata registry in SQLite.

- Set up structured JSON/CSV logging conventions and deploy Arize Phoenix locally.

- Install and configure Ollama on the development machine; verify Phi-3.5-mini INT4 inference latency and RAM footprint.

## Phase 2 — Local Intelligence (Weeks 4 to 6)

- Implement VAE architecture in PyTorch for time-series anomaly detection nodes (HAI data).

- Implement Isolation Forest pipeline using scikit-learn for tabular/network flow nodes (CIC Modbus data).

- Derive attack-window labels from the CIC Modbus attack logs/scenario documentation, since the dataset does not ship a per-packet label column.

- Define Insight Embedding schema and implement embedding serialisation to embedded Qdrant.

- Evaluate local model AUROC and F1 against labeled HAI and CIC Modbus holdout sets.

- Integrate Opacus DP-SGD into the local training loop with configurable epsilon and delta parameters.

## Phase 3 — Federated Learning (Weeks 7 to 9)

- Configure the Flower server with FedProx aggregation strategy in simulation mode.

- Implement Secure Aggregation protocol between simulated nodes and server.

- Configure convergence criteria: maximum round budget and loss delta threshold.

- Implement stragglers timeout and async node fallback.

- Integrate MLflow for round-level experiment tracking and model registry.

- Integrate DVC for dataset versioning and model checkpoint management.

- Run baseline Non-IID convergence comparison: FedAvg vs FedProx across six simulated node configurations.

## Phase 4 — Agentic Mesh (Weeks 10 to 13)

- Implement LangGraph state machine with Triage, Investigator, Security, and Dispatcher agent nodes.

- Build MITRE ATT&CK for ICS knowledge base: chunk technique descriptions, embed via Sentence Transformers, and index in embedded Qdrant with technique ID, tactic, and platform payload fields.

- Implement JWT session management and code-level mTLS between agents and edge node handlers.

- Implement circuit breaker pattern in Investigator Agent with configurable failure threshold and recovery timeout.

- Integrate Arize Phoenix tracing into all agent tool calls and LLM completions.

- Run end-to-end integration test across all five layers and six simulated nodes with a single simulated attack scenario.

## Phase 5 — Evaluation and Documentation (Weeks 14 to 16)

- Run full end-to-end attack simulation using HAI and CIC Modbus labeled attack windows.

- Execute custom evaluation harness: measure detection rate, false positive rate, root-cause attribution accuracy, mean time to investigation completion, and epsilon budget efficiency.

- Run FedProx convergence analysis across six Non-IID simulated node configurations.

- Conduct chaos testing: simulate node termination mid-investigation and verify circuit breaker and investigation completion behaviour.

- Generate matplotlib-based metric charts and Phoenix trace analysis for representative investigation sessions.

- Write technical documentation, architecture diagrams, and evaluation report — including a dedicated section documenting the production-scaling path (Section 6 note above) for examiners interested in deployment realism.

# 7. Evaluation Framework

## 7.1 Federated Learning Evaluation

| **Metric** | **Target** | **Measurement Method** |
|---|---|---|
| Global model AUROC | >= 0.92 on HAI test set | Sklearn roc_auc_score on labeled attack windows |
| Global model F1-score | >= 0.88 | Sklearn f1_score on labeled attack events |
| FedProx convergence rounds | <= 50 rounds to within 2% of final validation loss | MLflow round loss curve |
| Non-IID performance delta | < 5% AUROC drop vs IID equivalent | Controlled experiment: IID vs Non-IID partition |
| Epsilon budget per round | <= 2.0 per federated round | Opacus privacy engine accounting |
| Total epsilon over full training | <= 10.0 (configurable; tighten to 1.0 for healthcare use) | Cumulative Opacus epsilon tracker |
| Straggler recovery rate | 100% of dropped-node rounds complete | Flower round completion logs |

## 7.2 Agentic Mesh Evaluation

| **Metric** | **Target** | **Measurement Method** |
|---|---|---|
| Attack detection rate | >= 90% of labeled HAI/CIC Modbus attack windows flagged | Ground truth comparison against dataset labels |
| False positive rate | <= 5% of normal operating windows flagged | False alarm count on normal partition |
| Root-cause attribution accuracy | >= 80% correct MITRE ATT&CK technique identified | Manual review of Security Agent outputs vs known attack labels |
| Mean time to investigation completion | <= 3 seconds per incident from triage to dispatch | Arize Phoenix trace span duration |
| Investigator hop compliance | 100% of investigations complete within 3 hops | LangGraph state machine transition log |
| Circuit breaker activation | Zero hung investigations on simulated node failure | Chaos test: simulate node termination mid-investigation, verify completion |
| Agent reasoning faithfulness | >= 0.85 RAGAS faithfulness on Qdrant retrieval steps | RAGAS evaluation on retrieval sub-tasks only |

# 8. Market Relevance and Business Impact

## 8.1 Regulatory Compliance Value

GDPR enforcement actions in the EU have levied fines exceeding 4 billion euros since 2018, with a significant portion attributable to unlawful cross-border data transfers and inadequate technical safeguards. HIPAA violations in the United States incurred over 135 million dollars in civil monetary penalties in 2023 alone. The proposed system eliminates the legal exposure associated with data transfer by architectural design — raw data never moves, so there is no transfer to regulate, no data processing agreement to negotiate, and no cross-border compliance assessment required.

## 8.2 Operational Cost Reduction

Traditional centralised ML systems incur cloud egress costs proportional to data volume. An industrial facility generating 10 TB per day of telemetry data may incur egress costs of 200 to 500 thousand dollars per year depending on cloud provider and region. The proposed system transmits only compressed gradient updates and Insight Embeddings — orders of magnitude smaller than raw data — reducing data transfer costs by an estimated 70 to 90 percent. Additionally, the use of Phi-3.5-mini served locally via Ollama rather than cloud-hosted LLMs such as GPT-4 eliminates per-token API costs for the reasoning layer entirely, which for high-throughput industrial investigation workloads would otherwise be substantial.

## 8.3 Operational Resilience

The modular mesh architecture ensures that no single failure terminates the system. If the federated learning server becomes unavailable, each edge node continues to perform local anomaly detection at its current model version. If one or more agents become unavailable, the remaining agents continue to process investigations with graceful degradation. This is categorically different from monolithic centralised systems where a single server failure halts all AI capability.

## 8.4 Industry Applicability

| **Industry Vertical** | **Data Type** | **Specific Use Case** |
|---|---|---|
| Manufacturing / ICS | PLC sensor readings, SCADA event logs | Multi-stage cyber attack detection on industrial control networks |
| Financial Services | Transaction event streams, account state | Distributed fraud ring detection across multiple banking nodes |
| Healthcare | Medical device telemetry, EHR event logs | Patient safety anomaly detection across federated hospital systems (epsilon tightened to 1.0) |
| Energy / Utilities | Smart meter readings, grid sensor data | Grid instability and equipment failure prediction without centralising customer data |
| Defence / Government | Network traffic, access logs | APT campaign detection across air-gapped or classified network segments |

# 9. Risk Register

| **Risk** | **Likelihood** | **Impact** | **Mitigation** |
|---|---|---|---|
| FedProx convergence failure on extreme Non-IID splits | Medium | High | Pre-train local models for 5 epochs before federation. Adjust proximal term mu via grid search. |
| Epsilon budget exhaustion before convergence | Low | Medium | Tune gradient clipping norm and noise multiplier. Increase round budget. Use Renyi DP accounting for tighter bounds. |
| Phi-3.5-mini reasoning quality insufficient for MITRE matching | Medium | Medium | Implement retrieval-augmented reasoning with Qdrant. Fall back to embedding similarity classification if LLM confidence score is below threshold. Llama 3.2 3B available as a swap-in alternative for comparison. |
| Agent investigation latency exceeds 3-second target | Medium | Medium | Profile each agent hop with Phoenix traces. Reduce Investigator max hops to 2 if needed. Cache frequent MITRE lookups in memory. |
| CIC Modbus 2023 attack labels require manual derivation from logs | Medium | Low | Budget explicit preprocessing days in Phase 2 for label derivation from attack-log timestamps/scenario windows rather than assuming a ready-made label column. |
| HAI dataset distribution mismatch with target domain | Low | Medium | Supplement with synthetic anomaly injection using domain-appropriate distributions via custom Faker pipelines. |
| Local machine resource contention during development (8GB RAM) | Medium | Low | Run FL training and full agent-mesh reasoning sequentially rather than concurrently during development; train and checkpoint first, then run the agent mesh against saved outputs. Reserved for the Phase 4 end-to-end integration test only. Ollama INT4 quantisation for Phi-3.5-mini keeps LLM RAM footprint to ~2.2–2.5GB. |

# 10. Success Criteria

This project will be considered successfully completed when all of the following criteria are met:

1. The federated global model achieves an AUROC of 0.92 or higher and an F1-score of 0.88 or higher on the labeled HAI test partition.

2. FedProx demonstrates measurably superior convergence compared to FedAvg under the Non-IID simulated node data partition, with a performance delta of less than 5 percent AUROC.

3. The total cumulative epsilon privacy budget across all federated training rounds remains at or below the configured maximum of 10.0.

4. The Agentic Mesh detects at least 90 percent of labeled attack windows in the HAI and CIC Modbus 2023 datasets end-to-end, from simulated telemetry ingestion to action dispatch.

5. Mean time from triage alert to action dispatch is at or below 3 seconds for all investigated incidents.

6. Zero hung or incomplete investigations occur during the chaos test in which simulated edge nodes are terminated mid-investigation.

7. All inter-agent communication is authenticated via JWT and encrypted via code-level mTLS. No plaintext credential exchange occurs between services.

8. All model inference for agent reasoning is performed locally via Ollama. No investigation data is transmitted to external LLM APIs.

9. Full reasoning traces for all investigation sessions are recoverable from Arize Phoenix with complete step-by-step auditability.

10. All model checkpoints, training round statistics, and dataset versions are reproducible from the DVC and MLflow registry.

11. The complete system — Flower simulation, agent mesh, and all shared services — runs end-to-end on an 8GB RAM development machine without out-of-memory failures.

# 11. Conclusion

This proposal presents a technically rigorous, industrially relevant, and architecturally complete system for privacy-preserving distributed anomaly detection. It addresses a genuine and pressing problem faced by organisations in every regulated industry: how to apply advanced AI to sensitive operational data without creating legal exposure, security risk, or operational fragility.

The Decentralised Agentic Mesh does not propose incremental improvements to existing approaches. It fundamentally changes the architecture of industrial AI — from centralised, static, and human-dependent to distributed, continuously learning, and autonomously responsive. Each technology choice in this build has been made to preserve the algorithmic and architectural substance of the original production design — FedProx for Non-IID robustness, DP-SGD with a formally justified epsilon budget for privacy guarantees, LangGraph for stateful agent reasoning, Phi-3.5-mini via Ollama for fully on-premise LLM inference, Flower's simulation runtime for federated realism — while ensuring the entire system is buildable and runnable end-to-end on a standard 8GB RAM laptop, with a documented path to full multi-node/containerized deployment as future work.

The evaluation framework is grounded in publicly available industrial benchmark datasets with labeled ground truth, ensuring that claims of performance are verifiable and reproducible. The MITRE ATT&CK for ICS knowledge base provides a formally recognised threat taxonomy for root-cause attribution. The implementation is phased to enable incremental validation at each stage, reducing technical risk and providing clear checkpoints for review.

This system is ready for implementation and is presented for technical review, feedback, and consideration as a contribution to the advancing field of privacy-preserving distributed AI. The revised version is intentionally shaped around three research modules — federated edge detection, privacy-preserving federated learning, and knowledge-grounded agentic attribution — which create a clearer publication narrative and a more defensible scientific contribution. The final implementation should also support a Docker-based reproducibility path so that the project is executable, benchmarkable, and reviewable on machines that cannot comfortably host the entire stack directly.

*— End of Proposal —*

**Allaudin Ansari**

August 2026
