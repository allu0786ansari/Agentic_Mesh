# Week 11 Implementation and Results

**Project:** Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

**Status:** Real Triage/Investigator agents and circuit breaker implemented; live service profiling pending

## 1. Objective

Week 11 replaces the Week 10 graph placeholders with operational agent components:

- Triage retrieves similar historical embeddings and asks local Phi for escalation.
- Investigator asks Phi for target nodes and queries node APIs with bounded hops.
- Circuit breakers isolate failing nodes and recover through half-open probes.
- SQLite and tracing remain part of the agent execution path.

## 2. Implemented Components

### Triage Agent

File: `agents/triage/triage_agent.py`

- Accepts the shared `AgentState` contract.
- Retrieves up to five similar records from embedded Qdrant when the alert contains a
  128-dimensional embedding.
- Sends alert metadata and retrieved evidence to `LocalPhiModel`.
- Validates `escalate`, `severity_level`, and `reasoning` output.
- Stores a triaged alert in SQLite when the node exists in `node_registry`.
- Emits a `triage_agent` tracing span.

### Circuit breaker

File: `agents/investigator/circuit_breaker.py`

- Implements `CLOSED`, `OPEN`, and `HALF_OPEN` states.
- Opens after a configurable consecutive failure threshold.
- Rejects calls while open until the recovery timeout expires.
- Closes after a successful half-open probe.
- `CircuitBreakerRegistry` maintains independent state per node ID.

### Investigator Agent

File: `agents/investigator/investigator_agent.py`

- Uses Phi to select valid configured node IDs when targets are not supplied.
- Supports explicit alert target nodes for deterministic orchestration.
- Enforces a maximum of three investigation hops.
- Uses `httpx.AsyncClient` with a default 500 ms request timeout per hop.
- Routes every node call through its node-specific circuit breaker.
- Returns structured success, unavailable, or circuit-open evidence instead of hiding
  node failures.
- Emits an `investigator_agent` tracing span.

### Production graph factory

File: `agents/graph.py`

`build_real_graph()` wires injected Triage and Investigator services into LangGraph,
then executes Security and Dispatcher transitions. The original `app` remains the
Week 10 deterministic graph for backward-compatible tests; Week 11 callers should use
`build_real_graph()` with real service instances.

## 3. Validation

```powershell
$py = 'agentvenv\Scripts\python.exe'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_week11_agents.py tests/unit/test_auth.py tests/unit/test_agent_graph.py tests/unit/test_api_security.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
& $py -m compileall -q agents api scripts
```

Expected result:

```text
8 passed
```

## 4. Runtime Setup

Start Phoenix first:

```powershell
& '.\agentvenv\Scripts\phoenix.exe' serve
```

In another terminal, configure tracing and verify Ollama is running:

```powershell
$env:PHOENIX_OTEL_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'
ollama ps
```

The model required by Week 11 is:

```text
agmesh-phi3.5:latest
```

## 5. Real Agent Graph Command

```powershell
$env:PHOENIX_OTEL_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'
& $py -c "from agents.tracing import configure_phoenix; configure_phoenix(); from agents.llm import LocalPhiModel; from agents.triage.triage_agent import TriageAgent; from agents.investigator.investigator_agent import InvestigatorAgent; from agents.graph import build_real_graph; import asyncio; triage=TriageAgent(llm=LocalPhiModel()); investigator=InvestigatorAgent(llm=LocalPhiModel(), node_urls={'node_01':'http://127.0.0.1:8443','node_02':'http://127.0.0.1:8444','node_03':'http://127.0.0.1:8445'}); result=asyncio.run(build_real_graph(triage_agent=triage, investigator_agent=investigator).ainvoke({'session_id':'week11-real','alert':{'node_id':'node_01','score':0.9,'embedding':[0.0]*128,'target_nodes':['node_01','node_02','node_03']}})); print(result)"
```

The configured node API URLs must be reachable and expose the authenticated
`/embeddings/search` endpoint. Supply an authenticated HTTP client or service-to-
service credential before enabling protected production routes.

## 6. Current Validation Result

The deterministic test transport verified:

```text
Triage escalation: true
Investigation hops: 3 maximum
Circuit breaker state transitions: CLOSED -> OPEN
Graph status: complete
```

The real Ollama adapter was verified in Week 10. Live node API querying and the mean
Triage + Investigator latency target of 1.5 seconds require running API services and
must be profiled with the real deployment configuration.

## 7. Remaining Work

1. Add authenticated service-to-service HTTP credentials to Investigator requests.
2. Connect real Qdrant insight collections and persist triage alerts for live runs.
3. Replace the Week 11 Security placeholder with the MITRE KB implementation planned
   for Week 12.
4. Add a live integration test with a running node API and injected failure/timeout.
5. Profile ten real Ollama + API runs and persist latency percentiles.
6. Add Dispatcher bus publication and full graph action persistence in Week 12.

## 8. Week 12 Handoff

Week 12 should implement the MITRE ATT&CK for ICS knowledge base, Security Agent
classification, confidence thresholds, and action dispatch over `edge/bus.py`.
