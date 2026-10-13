# Threat Model (draft)

**Status:** Draft, Week 1. Finalised in Week 16.
**Source:** Proposal v3.0 §5. Where this file and the proposal disagree, update both.

## 1. Assets

| Asset | Where it lives | Why it matters |
|---|---|---|
| Raw telemetry (HAI process data, Modbus traffic) | Each node's own partition | Reveals process state, operations and security posture |
| Model updates | Sent by clients to the aggregation server each round | Can leak information about the training data |
| Evidence records | Node evidence stores; sent to the mesh during investigations | Describe what deviated, where and when |
| Prompts and knowledge-base content | Mesh process; Ollama | Contain investigation details |
| Dispatched actions | SQLite `actions` table; bus | Can change detection or isolate nodes |
| Credentials (certificates, JWT signing key) | `certs/`, environment | Grant access to node APIs |

## 2. Adversaries

| ID | Adversary | Assumed capability |
|---|---|---|
| ADV-1 | Honest-but-curious aggregation server | Follows the protocol; inspects everything it receives |
| ADV-2 | Curious peer client | Follows the protocol; inspects the global models it receives |
| ADV-3 | Network attacker between services | Observes, replays or injects traffic on local service links |
| ADV-4 | Compromised or faulty agent process | Issues node queries using any credentials it holds |
| ADV-5 | Exfiltration through external services | Investigation content sent to a third-party LLM API |

**Out of scope:** clients that poison updates (Byzantine clients), compromised node hosts, side channels, physical access.

## 3. Mechanisms

| Mechanism | Defends against | Guarantee or effect | Built in |
|---|---|---|---|
| DP-SGD (Opacus) at every client | ADV-1, ADV-2 | (ε, δ)-DP per client; the privacy unit is one non-overlapping training window | Week 5 |
| Noisy normalisation statistics | ADV-1 | Gaussian mechanism; its ε counts toward the client's budget | Week 6 |
| SecAgg+ (systems experiment X8) | ADV-1 | Server sees only the aggregate update | Week 7 |
| Federated evaluation | ADV-1 | Server receives aggregate metrics only, never data | Week 6 |
| mTLS between processes | ADV-3 | Encrypted, mutually authenticated links; plaintext refused | Week 11 |
| Per-request JWT (audience, scope, expiry, one-time `jti`) | ADV-3, ADV-4 | A query is limited to one session, site and time window, and cannot be replayed | Week 11 |
| LLM egress allowlist | ADV-5 | Mesh refuses to start if the LLM endpoint is not loopback or allowlisted | Week 1 (`agmesh.security.egress`), enforced in the mesh in Week 11 |
| Evidence minimisation | ADV-1 to ADV-4 | No raw values or latent vectors leave a node | Week 8 |
| Policy-gated dispatch | ADV-4 | High-impact actions require human approval | Week 10 |

## 4. What may leave a node

| Phase | Allowed outbound content |
|---|---|
| Training | Clipped, noised model updates; one set of noisy normalisation statistics |
| Federated evaluation | Aggregate metric counts; binned score histograms |
| Operation | Alerts and evidence records (schema in Proposal §6.6) |

Contract tests on every outbound message type enforce this list (completion criterion CC5).

## 5. Residual risks

- Evidence records are minimised but not formally differentially private.
- With 3–4 clients per federation, secure aggregation gives limited anonymity.
- Federated evaluation releases aggregate statistics, including binned score histograms.
- A historical development JWT secret exists in Git history (commit `fc1b2a3`, `.env.example`). It is not used by v3, but it must be treated as public and rotated if it was ever used outside local development. See `.gitleaks.toml`.

## 6. Open questions for later weeks

- Week 5: confirm the RDP accountant state can be saved and restored exactly across process restarts.
- Week 7: confirm the SecAgg+ dropout threshold that is meaningful with 4 clients.
- Week 11: decide whether node-to-mesh certificates are per node or per site gateway.
