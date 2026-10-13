# Week 13 Implementation and Results

**Project:** Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

**Status:** Alert bus graph runner, real-agent local-stack orchestration, and federated alert trigger implemented; live full-stack validation pending

## 1. Objective

Week 13 connects the federated/edge alert boundary to the agent mesh:

- Alerts are published on `alerts.global`.
- An async subscriber consumes and deserializes each alert.
- Graph invocation does not block the subscriber event loop.
- Every completed session records status, hop count, action, and duration.
- A local-stack command injects multiple alerts and validates completion.

## 2. Implemented Components

### Graph runner

File: `agents/graph_runner.py`

`GraphRunner`:

- Subscribes to `alerts.global` through `AsyncBus`.
- Accepts dictionary or JSON-string alert messages.
- Creates a session state with a stable session ID.
- Invokes async graphs directly or sync graphs through an executor.
- Supports bounded `max_sessions` and `idle_timeout` processing.
- Writes `agent_graph_completed` events to JSONL metrics.
- Records final status, hop count, action, and duration in each event.

### Alert injector

File: `scripts/inject_alerts.py`

Publishes structured anomaly alerts with node ID, score, embedding, target nodes, and
session IDs to `alerts.global`.

### Federated evaluation trigger

File: `federated/simulate.py`

- Adds `--inject-attack` for deterministic alert injection after completed evaluation
  rounds.
- Publishes an alert when global evaluation loss exceeds `--alert-threshold`.
- Records source, round, loss, node, session ID, and embedding in the alert.
- Persists emitted alerts in the federated `summary.json`.

### Local stack runner

File: `scripts/run_local_stack.py`

Runs the bus, graph runner, and alert injection as concurrent asyncio tasks in one
process. It validates that the requested number of sessions completes with status
`complete`.

With `--real-agents`, it constructs the real Week 11/12 TriageAgent,
InvestigatorAgent, SecurityAgent, and ActionDispatcher using Ollama, Qdrant, SQLite,
and the configured node URLs.

### Smoke script

File: `scripts/run_smoke_test.sh`

Runs five local graph sessions and asserts that five `agent_graph_completed` events are
persisted in the configured metrics file.

## 3. Validation

```powershell
$py = 'agentvenv\Scripts\python.exe'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_graph_runner.py tests/unit/test_local_stack.py tests/unit/test_week12_agents.py tests/unit/test_week11_agents.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
& $py -m compileall -q agents api scripts
```

Result:

```text
8 passed
```

## 4. Run the Local Event Stack

Phoenix must be running and `PHOENIX_OTEL_ENDPOINT` must point to its OTLP endpoint:

```powershell
$py = 'agentvenv\Scripts\python.exe'
$env:PHOENIX_OTEL_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'
& $py -m scripts.run_local_stack `
  --count 5 `
  --real-agents `
  --qdrant-path qdrant_data_week12 `
  --database agmesh.db `
  --node-id node_01 `
  --score 0.9 `
  --endpoint $env:PHOENIX_OTEL_ENDPOINT `
  --metrics-path results/week13_smoke/metrics.jsonl
```

Expected result:

```text
{
  "sessions": 5,
  "status": "complete"
}
```

## 5. Trigger Alerts from Federated Evaluation

The trigger publishes qualifying evaluation results to `alerts.global`:

```powershell
& $py -m federated.simulate `
  --rounds 3 `
  --clients 3 `
  --nodes node_01 node_02 node_03 `
  --partitions-root data/partitions `
  --results-root results/week13_federated `
  --local-epochs 1 `
  --max-windows 256 `
  --inject-attack `
  --alert-threshold 0.2
```

Emitted alerts are saved in the resulting `summary.json` under `emitted_alerts`.

Verify persisted session events:

```powershell
(Select-String -Path results/week13_smoke/metrics.jsonl -Pattern 'agent_graph_completed').Count
```

Expected count: `5`.

The shell smoke script can be run in Git Bash:

```bash
bash scripts/run_smoke_test.sh
```

## 6. Inject Alerts and Run a Separate Subscriber

Terminal 1:

```powershell
$env:PHOENIX_OTEL_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'
& $py -m scripts.run_graph_runner --endpoint $env:PHOENIX_OTEL_ENDPOINT --max-sessions 5 --idle-timeout 30
```

Terminal 2 in the same process environment cannot share the in-process `AsyncBus`
state across processes. For cross-process deployment, replace `AsyncBus` with the
configured broker transport. For the local single-process build, use
`run_local_stack.py`, which owns both publisher and subscriber tasks.

## 7. Remaining Validation

1. Run the real four-agent graph rather than the Week 10 deterministic graph in the
   local stack command.
2. Add Dispatcher SQLite persistence assertions to the local-stack smoke path.
3. Run five complete sessions with live Ollama, Qdrant, node APIs, and Phoenix.
4. Export one complete trace to `docs/architecture/sample_trace.json`.
5. Add the full FL-to-agent smoke test and verify no Phoenix span has error status.

## 7. Week 14 Handoff

Week 14 should implement the evaluation harness over HAI and CIC Modbus attack windows,
including detection rate, false-positive rate, root-cause accuracy, and latency.
