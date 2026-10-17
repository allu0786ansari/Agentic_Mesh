# ADR 0014 — Process boundaries, mTLS, JWT, redaction, tracing (Week 11)

**Status:** Accepted, 17 October 2026. Written before any multi-process run.

**Implements:**

- Plan Week 11, tasks 1–7;
- Proposal §5.2 (ADV-3, ADV-4, ADV-5), §9.2 and §9.5;
- engineering acceptance EA2 and EA3.

**Builds on:** ADR 0012 (the mesh) and ADR 0013 (the LLM path).

## 1. Processes

| Process | Instances | Listens on | Serves |
|---|---|---|---|
| `edge-runtime` | 1 per site | — | Already a separate process (`make edge`, ADR 0009). It writes the per-node evidence stores and publishes alerts. |
| `node-api` | 1 per node | `127.0.0.1`, HTTPS only (`configs/stack.yaml`) | One node's evidence store, read-only |
| `mesh` | 1 | — | The runner (ADR 0012), with `--nodes https` |
| `approval-api` | 0–1 | `127.0.0.1`, HTTPS only | Decisions on one mesh run's pending actions |
| `ollama` | 1 | loopback | ADR 0013 §7 |

- **Starting and stopping:** `make stack-up SITE=…` starts that site's node APIs as
  separate processes and waits until each answers; `make stack-down` stops them.
- **Memory:** only one site's node APIs run at a time.
- **No plaintext listener exists.**

## 2. Certificates (development PKI)

`make certs` writes `certs/dev/`. The directory is git-ignored, and private keys are
mode 0600. Keys are ECDSA P-256, and leaf certificates are valid for 30 days.

There are **three CAs, one per role**, so that a TLS server only trusts the clients
it serves:

| CA | Signs | Trusted by |
|---|---|---|
| `ca-server` | one server certificate per node API (`node-<id>.agmesh`) and the approval API (`approval-api.agmesh`), each with `serverAuth` | the mesh client and the operator client |
| `ca-mesh` | the mesh client certificate, with `clientAuth` | node APIs only |
| `ca-operator` | the operator client certificate, with `clientAuth` | the approval API only |

**Why separate CAs.** The ASGI server (uvicorn) does not pass the client
certificate to the application, so identity is enforced by the trust anchor
instead:

- a node API accepts only certificates issued by `ca-mesh`;
- a node's own server certificate is from `ca-server`, so it cannot act as a client.

**Server identity.** Every server certificate carries its own DNS name and
`127.0.0.1`. The mesh connects with SNI and hostname checking set to the node's
name, so a node API cannot stand in for another node.

**The JWT signing key** (`jwt-mesh.key`, ES256) is separate from every TLS key.
Node APIs hold only its public key.

## 3. mTLS

- **Protocol:** TLS 1.3 only, with `CERT_REQUIRED` on every server.
- **The mesh client:**
  - uses its own certificate;
  - verifies servers against `ca-server`;
  - keeps one keep-alive connection per node, so each node's handshake happens once
    per run.

## 4. Per-request JWT (node API)

One token is minted per evidence request. The mesh signs it with ES256.

**Claims:**

| Claim | Value |
|---|---|
| `iss` | `mesh` |
| `aud` | the node ID |
| `sub` | the session ID |
| `scope` | `evidence:read` |
| `site` | the node's site |
| `window_start`, `window_end` | the requested window, in epoch seconds |
| `iat`, `exp` | `exp − iat ≤ security.jwt.ttl_s` (120 s) |
| `jti` | a random 128-bit ID |

**What the node API checks.** It refuses with **401** for an invalid token and **403**
for a token that is valid but not allowed this request:

- the algorithm is exactly ES256;
- the signature verifies against the mesh's public key;
- `iss` and `aud` are right;
- `exp` and `iat` are valid, with 1 s leeway, and the lifetime is at most the TTL;
- the scope is right;
- `site` matches the node's site;
- the requested window lies inside the token's window;
- the `jti` has not been seen before.

**The replay cache:**

- it holds every `jti` until its `exp`, in memory, per node API process;
- it is bounded, with expired entries pruned first;
- when it is full of unexpired entries, the request is refused (503) rather than
  accepted unchecked.

**Limitation.** A node API restart empties the cache, so a token captured before the
restart could be replayed for at most 120 s afterwards. This is recorded, not
mitigated, in this development stack.

**How the mesh client treats errors:**

- 401, 403 and other 4xx responses are non-transient: no retry, and the breaker
  counts a failure;
- connection errors, timeouts and 503 are transient: one retry (ADR 0012 §3).

The response body is a list of `EvidenceRecord`, validated by the schema on both ends
(CC5).

## 5. Egress

The mesh runner and the LLM calibration refuse to start an LLM mode when
`llm.base_url` is not on `security.egress_allowlist` (ADV-5).

## 6. Redaction

A logging filter is installed at every process entry point:

- node API;
- approval API;
- mesh runner;
- calibration;
- stack.

It replaces the following with `[REDACTED]`:

- `Authorization` header values and `Bearer …` tokens;
- JWT-shaped strings (`eyJ…`.`…`.`…`);
- PEM private-key blocks;
- Hugging Face tokens (`hf_…`);
- the literal values of `MESH_APPROVAL_KEY` and `HF_TOKEN` when they are set.

The filter works on the formatted message and its arguments, before any handler
writes them.

## 7. Tracing

**The spans.** Spans use OpenTelemetry with OpenInference semantic conventions. There
is one trace per session:

| Span | OpenInference kind | Key attributes |
|---|---|---|
| `session` (root) | CHAIN | `session.id`, `agmesh.run_id`, site, mode, status |
| `triage`, `investigate`, `attribute`, `dispatch` | CHAIN | — |
| `hop` | TOOL | node, outcome, ms |
| `retrieve` | RETRIEVER | candidate IDs |
| `llm` | LLM | model, token counts, valid, error class |

**Contents.** Prompts and evidence text are not put into traces. They are in the
session store already; traces stay small and carry no evidence content.

**Spans are created by hand, not by an automatic instrumentor.** Two reasons:

- the LLM call is a raw Ollama HTTP call, which the LangChain instrumentor would not
  see;
- these spans carry the mesh's own attributes.

**Export:**

- every finished trace goes to `results/traces/<run_id>/<session_id>.json`;
- the trace ID is stored in a new `session_traces` table, so action → session →
  trace (EA3);
- when `PHOENIX_COLLECTOR_ENDPOINT` is set and the `obs` group is installed, spans
  also go to Phoenix over OTLP;
- Phoenix is optional, so it is not resident on the 8 GB laptop.

**Check.** `python -m agmesh.evaluation.checker --traces <mesh run>` checks that every
session in the run has exactly one trace file, with its root span carrying that
`session.id`.

## 8. Approval API

Exposed calls:

- `GET /v1/actions?status=pending_approval`;
- `POST /v1/approvals`, with `action_id`, `decision`, `approver`, `expires_at` and
  `token`.

Both go over mTLS with the operator certificate only. The POST calls
`approvals.decide` (ADR 0012 §5) and adds no check of its own. HMAC errors return
403.

## 9. Gate G3 decision rule, fixed now

G3 decides the default attribution mode and the final L_LLM from validation runs
only. The inputs are:

- the ADR 0013 §7 runs (prompt cache off);
- the Week 11 multi-process runs.

**The rule:**

- **Default mode stays `full`** if, on both sites:
  - JSON validity is at least 95% (PT5's validity bar);
  - `full` incident-to-dispatch p95 is at most 24 s;
  - investigation p95 over mTLS is at most 2 s.
- **Otherwise the default becomes `retrieval_only`,** with the LLM explaining after
  dispatch (Plan §7).
- **L_LLM stays 24 s,** as preregistered (PT7). The measured 26 s LLM timeout
  (ADR 0013) means a call that runs to its timeout misses PT7. That is reported, not
  re-targeted.

G3's decisions are recorded in ADR 0015. ADR 0003 is already taken by the scoring
selection, so the Plan's file name `0003-g3-decisions.md` is not used.

## 10. Cloud facts (before any laptop run)

**Security tests.** `tests/security` has 41 tests, all passing:

- **mTLS rejects:**
  - plaintext;
  - a client without a certificate;
  - a mesh certificate from a foreign CA;
  - a node's server certificate;
  - the operator certificate at a node;
  - TLS 1.2;
  - a server certificate for another node;
  - a server from a foreign CA.
- **JWT rejects:**
  - an expired token;
  - the wrong audience;
  - the wrong issuer;
  - a foreign signing key;
  - a lifetime above 120 s;
  - an `iat` in the future;
  - the wrong scope, or no scope;
  - another site;
  - a window that does not cover the request;
  - a token with no window;
  - no `jti`;
  - `alg=none`, HS256 and malformed tokens;
  - replay, over HTTP as 403;
  - a full replay cache (503).
- **Egress:** both LLM modes refuse a non-allowlisted endpoint before a run directory
  exists.
- **Redaction:** JWTs, bearer headers, PEM keys, HF tokens and env secrets are removed
  from every log record. Formatters that read `record.args` as a tuple still work.
- **Approval API:** operator certificate only; HMAC gate; a decision cannot be
  repeated.

**`tests/integration/test_stack_processes.py`.** Three node APIs run as separate
processes, the Investigator queries them over mTLS with a JWT per hop, and the stack
stops cleanly.

**Validation, `rule_only`, `--nodes https`, with each site's node APIs as separate
processes** (`results/W11/*__rule_only__https__summary.json`):

| Site | Sessions | Investigation p95 | Incident-to-dispatch p95 | Traces |
|---|---|---|---|---|
| A | 45, all completed | 17.1 ms | 27.8 ms | 45 of 45, checker OK |
| B | 220, all completed | 20.1 ms | 24.8 ms | 220 of 220, checker OK |

- The sessions, statuses, techniques, hops and actions are identical to the
  in-process `rule_only` runs (`results/W10`).
- In-process, the incident-to-dispatch p95 was 97 ms on Site A and 57 ms on Site B
  (ADR 0012, laptop). These cloud figures are from a different machine.
