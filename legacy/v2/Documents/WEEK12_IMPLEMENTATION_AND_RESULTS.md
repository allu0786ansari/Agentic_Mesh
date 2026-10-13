# Week 12 Implementation and Results

**Project:** Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

**Status:** Security Agent, MITRE KB builder, Action Dispatcher, and full graph wiring implemented; MITRE-only index validated; NIST/full KB pending

## 1. Objective

Week 12 adds knowledge-grounded classification and structured remediation dispatch:

- MITRE ATT&CK for ICS and NIST SP 800-82 text are chunked and embedded.
- Chunks are indexed in embedded Qdrant at 384 dimensions.
- SecurityAgent retrieves the top five knowledge matches and asks Phi for a structured
  classification.
- Low-confidence classifications fall back to `UNKNOWN`.
- ActionDispatcher maps tactics to actions, publishes through the asyncio bus, and
  updates SQLite alert status.
- The production graph can inject and execute all four real agents.

## 2. Implemented Components

### MITRE knowledge-base builder

File: `agents/security/kb_builder.py`

- Parses STIX JSON `attack-pattern` objects.
- Ignores revoked and deprecated techniques.
- Extracts technique IDs, names, descriptions, detection text, tactics, and platforms.
- Chunks long descriptions deterministically.
- Parses NIST text into guidance chunks.
- Embeds in batches and validates finite `(batch, 384)` vectors.
- Uses deterministic UUID5 point IDs for repeatable Qdrant indexing.
- Reports exact collection count.

### Security Agent

File: `agents/security/security_agent.py`

- Shares one injected Qdrant client when used with an in-process KB builder.
- Mean-pools evidence through the configured Sentence Transformer embedder.
- Retrieves five MITRE matches from `mitre_kb`.
- Requests `technique_id`, `tactic`, `confidence`, `reasoning`, and mitigations from Phi.
- Validates confidence range and returns `UNKNOWN` below the configured threshold.
- Updates the SQLite alert status to `classified`.
- Emits a `security_agent` span.

### Action Dispatcher

File: `agents/dispatcher/action_dispatcher.py`

- Maps `credential-access`, `impact`, and `execution` tactics to structured actions.
- Uses `investigate` for unknown tactics.
- Publishes to `actions.<node_id>` through `AsyncBus`.
- Adds session, severity, technique, confidence, and reasoning metadata.
- Persists SQLite status `dispatched`.
- Records action latency in the result state.
- Emits an `action_dispatcher` span.

### Full graph wiring

File: `agents/graph.py`

`build_real_graph()` now accepts real TriageAgent, InvestigatorAgent, SecurityAgent,
and ActionDispatcher instances. Triage controls escalation; Investigator evidence is
classified by SecurityAgent; the resulting remediation is published by Dispatcher.

## 3. Validation

```powershell
$py = 'agentvenv\Scripts\python.exe'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& $py -m pytest tests/unit/test_week12_agents.py tests/unit/test_week11_agents.py tests/unit/test_auth.py tests/unit/test_api_security.py -q
Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
& $py -m compileall -q agents api scripts
```

Validated result:

```text
9 passed
```

The tests cover Qdrant indexing, confidence fallback, circuit breakers, full graph
routing, action publication, and SQLite dispatch persistence.

## 4. Build the Real MITRE Knowledge Base

The indexer requires a downloaded MITRE ATT&CK for ICS STIX bundle. Set paths to the
actual files on the machine:

```powershell
& $py -m scripts.build_mitre_kb `
  --stix data/raw/mitre-ics/ics-attack.json `
  --nist data/raw/nist_800_82/nist-800-82.txt `
  --qdrant-path qdrant_data_week12 `
  --collection mitre_kb
```

Verify the indexed count:

```powershell
& $py -c "from qdrant_client import QdrantClient; client=QdrantClient(path='qdrant_data_week12'); print(client.count(collection_name='mitre_kb', exact=True).count)"
```

The project checkpoint expects approximately 1,500-2,000 points when the complete
MITRE/NIST sources are indexed. The builder reports the actual count; it does not
fabricate points when the source files are absent.

### MITRE-only indexing result

The downloaded MITRE-only build completed successfully:

```text
STIX objects: 2,173
Active ICS attack patterns: 97
Indexed points: 123
Qdrant collection: mitre_kb
Persisted count: 123
```

The 123-point count is expected for this command because it indexed only the active
MITRE ICS techniques. It is below the 1,500-2,000 project checkpoint because the NIST
PDF was not converted to text and supplied, and the current chunker produces roughly
one or a few chunks per technique description. The Hugging Face `hf_xet` message is a
download-performance warning; the embedding model downloaded successfully and the
index completed.

This result is valid engineering evidence, but it is not yet the complete Week 12 KB
target. The next KB improvement is to convert the downloaded NIST PDF to text, index
it, and add richer technique-source chunks or procedure/examples where available.

### NIST PDF conversion result

The downloaded NIST SP 800-82r3 PDF was converted with `pypdf`:

```text
Source: data/raw/nist_800_82/nist-800-82.pdf
Pages: 316
Output: data/raw/nist_800_82/nist-800-82.txt
Extracted characters: 740,532
Encoding: UTF-8
```

The text begins with the expected NIST SP 800-82r3 title and is ready for combined
MITRE/NIST indexing.

### Combined MITRE/NIST indexing result

The combined index was subsequently executed against the converted NIST text and is
persisted in `qdrant_data_week12`:

```text
Qdrant collection: mitre_kb
Vector dimensions: 384
Stored points: 741
MITRE source points: 123
NIST source points: 618
Unique technique IDs: 98
Payloads missing text: 0
Payloads missing technique_id: 0
```

The final count is exactly consistent with the current chunker: 123 MITRE chunks plus
618 NIST chunks. This is a complete, metadata-valid combined corpus for the downloaded
sources, but it remains below the original 1,500-2,000 planning checkpoint. Reaching
that larger count requires enriching the corpus with procedure/example sources or
using smaller semantic chunks; duplicating existing text would not improve retrieval
quality.

## 5. Run the Full Real Graph

Start Phoenix and ensure Ollama has `agmesh-phi3.5:latest` loaded. Start node APIs
with the authenticated `/embeddings/search` route enabled, then run:

```powershell
$env:PHOENIX_OTEL_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'
& $py -c "from agents.tracing import configure_phoenix; configure_phoenix(); from agents.llm import LocalPhiModel; from agents.triage.triage_agent import TriageAgent; from agents.investigator.investigator_agent import InvestigatorAgent; from agents.security.security_agent import SecurityAgent; from agents.dispatcher.action_dispatcher import ActionDispatcher; from agents.graph import build_real_graph; from edge.bus import AsyncBus; import asyncio; llm=LocalPhiModel(); triage=TriageAgent(llm=llm, database_path='agmesh.db'); investigator=InvestigatorAgent(llm=llm, node_urls={'node_01':'http://127.0.0.1:8443','node_02':'http://127.0.0.1:8444','node_03':'http://127.0.0.1:8445'}); security=SecurityAgent(llm=llm, qdrant_path='qdrant_data_week12', database_path='agmesh.db'); dispatcher=ActionDispatcher(database_path='agmesh.db'); graph=build_real_graph(triage_agent=triage, investigator_agent=investigator, security_agent=security, dispatcher=dispatcher); result=asyncio.run(graph.ainvoke({'session_id':'week12-real','alert':{'node_id':'node_01','score':0.9,'embedding':[0.0]*128,'target_nodes':['node_01','node_02','node_03']}})); print(result)"
```

Expected state fields include `mitre_match.technique_id`, `confidence`,
`recommended_mitigations`, and `remediation.action`. The action is available from:

```text
actions.node_01
```

## 6. Remaining Validation

1. Run five real graph sessions with live Phi, Qdrant, node APIs, and AsyncBus.
2. Record mean/p95 end-to-end latency and require mean below 3 seconds.
3. Verify four Phoenix child spans: triage, investigator, security, dispatcher.
4. Add authenticated service credentials to Investigator for protected production APIs.
5. Enrich the KB corpus or refine semantic chunking if the 1,500-2,000 point planning
  checkpoint is required for the final research report.
6. Export a complete trace and action payload for the final project evidence.

## 7. Week 13 Handoff

Week 13 should connect federated alerts to the graph through `alerts.global`, add the
async graph runner, orchestrate the local stack, and execute the end-to-end smoke test
without manual state injection.
