# Week 10 Implementation and Results

**Project:** Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

**Status:** Week 10 implementation complete; Ollama verified; strict Phoenix OTLP export configured

## 1. Objective

Week 10 establishes the agentic mesh security and orchestration foundation without
changing the Week 9 model artifacts:

- Shared typed LangGraph state.
- Triage, Investigator, Security, and Dispatcher graph skeleton.
- Conditional escalation routing for low versus medium/high alerts.
- Short-TTL JWT session tokens.
- Local Ollama structured-JSON adapter for Phi.
- Self-signed development certificate generation.
- TLS-aware Uvicorn node API launcher.
- Local tracing facade compatible with Phoenix when available.

## 2. Implemented Components

### Agent state and graph

Files: `agents/state.py`, `agents/graph.py`

- `AgentState` defines session, alert, evidence, hop count, MITRE match, remediation,
  status, trace, and error fields.
- The compiled `app` routes every alert through Triage.
- Medium/high alerts proceed through Investigator, Security, and Dispatcher.
- Low alerts route directly to Dispatcher.
- Each node creates a tracing span through the existing project tracer.

### JWT authentication

File: `agents/auth.py`

`JWTAuthManager` issues tokens with subject, issue time, expiry, and unique token ID.
Validation rejects malformed, expired, unsigned, or subject-less tokens.

### Ollama Phi adapter

File: `agents/llm.py`

`LocalPhiModel` configures LangChain's Ollama wrapper with temperature zero and JSON
format output, then validates that every response is a JSON object. It does not hide
LLM parse failures behind a fallback response.

The repository `Modelfile` uses the installed Ollama 0.31.2 syntax and was verified
against the local runtime. The custom model returned valid JSON and was resident in
memory:

```text
Model: agmesh-phi3.5:latest
Response: { "ok": true }
Resident size: 3.3 GB
Context: 2048
```

### Development mTLS and API launcher

Files: `scripts/generate_dev_mtls.py`, `api/tls.py`, `scripts/run_node_api.py`

- Generates a 2048-bit RSA self-signed certificate with localhost and `127.0.0.1`
  SAN entries.
- Validates certificate/key paths before server startup.
- Starts the node API with Uvicorn TLS parameters when both files are supplied.
- Supports CA configuration and mandatory client certificates with `ssl_cert_reqs=2`.

### API JWT protection

Files: `api/node_api.py`, `scripts/run_node_api.py`

- Health remains available for liveness checks.
- Embedding storage and search accept an optional `JWTAuthManager` dependency.
- When a JWT manager is configured, both routes require a valid Bearer token.
- The API launcher can receive a development secret through `--jwt-secret`.

### Phoenix export configuration

File: `agents/tracing.py`

`configure_phoenix()` configures an OpenTelemetry OTLP HTTP exporter and is required
before any graph span can execute. There is no local span or silent no-op fallback.
Missing configuration raises an explicit runtime error.

The Phoenix server was launched locally and the graph export was verified with an
HTTP `200 OK` response from `POST /v1/traces`. The Phoenix process reported:

```text
Phoenix UI: http://localhost:6006
HTTP traces: http://localhost:6006/v1/traces
```

The installed Phoenix UI currently logs a Starlette/Jinja compatibility error for
the root page, but OTLP span ingestion is working. The UI dependency mismatch must
be resolved before claiming browser-level span inspection complete.

## 3. Validation

Focused Week 10 tests:

```powershell
$py = 'agentvenv\Scripts\python.exe'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_auth.py tests/unit/test_agent_graph.py tests/unit/test_dev_mtls.py tests/unit/test_tls.py tests/unit/test_llm_adapter.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
```

Expected result:

```text
13 passed
```

Compile the Week 10 packages:

```powershell
& $py -m compileall -q agents api scripts
```

## 4. Runtime Commands

### Generate development TLS files

```powershell
& $py -m scripts.generate_dev_mtls --output certs/dev --common-name localhost
```

### Run the node API over HTTPS

```powershell
& $py -m scripts.run_node_api `
  --host 127.0.0.1 `
  --port 8443 `
  --qdrant-path qdrant_data_week6 `
  --collection insight_embeddings_20260821T222733Z `
  --certfile certs/dev/dev-cert.pem `
  --keyfile certs/dev/dev-key.pem
```

Validate the endpoint from another PowerShell terminal:

```powershell
Invoke-RestMethod https://127.0.0.1:8443/health -SkipCertificateCheck
```

The development certificate is self-signed and should not be trusted in production.

### Run the LangGraph skeleton

```powershell
& $py -c "from agents.graph import app; print(app.invoke({'session_id':'week10-1','alert':{'node_id':'node_01','score':0.9}}))"
```

Expected result includes:

```text
status: complete
hop_count: 1
mitre_match: UNKNOWN
```

### Verify JWT sessions

```powershell
$env:JWT_SECRET_KEY = 'replace-with-a-development-secret'
& $py -c "from agents.auth import JWTAuthManager; manager=JWTAuthManager(); token=manager.create_token('week10-session'); print(manager.validate_token(token)['sub'])"
Remove-Item Env:JWT_SECRET_KEY
```

### Create and verify the local Phi model

Ollama must be installed and running separately:

```powershell
ollama create agmesh-phi3.5 -f Modelfile
ollama run agmesh-phi3.5 "Return JSON only: {\"ok\": true}"
ollama ps
```

The verified local runtime returned valid JSON and reported `agmesh-phi3.5:latest` at
3.3 GB resident with 2048 context. The exact footprint is machine-dependent.

### Launch Phoenix and verify exported spans

Launch Phoenix through the project interpreter in one terminal:

```powershell
$env:PHOENIX_HOST = '127.0.0.1'
$env:PHOENIX_PORT = '6006'
& $py -c "import phoenix; phoenix.launch_app()"
```

In a second terminal, emit the graph spans:

```powershell
$env:PHOENIX_OTEL_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'
& $py -m scripts.run_agent_graph --endpoint $env:PHOENIX_OTEL_ENDPOINT
```

Verify one trace in `http://127.0.0.1:6006` containing `triage`, `investigator`,
`security`, and `dispatcher` child spans.

## 5. Remaining Work

1. Resolve the Phoenix 12.35 / installed Starlette UI compatibility issue and verify
  the four child spans in the browser UI.
2. Replace graph stub nodes with the real Week 11 Triage and Investigator agents.
3. Add a production CA/client certificate workflow after development certificates are
  replaced for deployment.

## 6. Week 11 Handoff

Week 11 should implement the real Triage and Investigator agents, circuit-breaker
state transitions, bounded three-hop node querying, and latency measurements while
preserving this Week 10 state and authentication contract.
