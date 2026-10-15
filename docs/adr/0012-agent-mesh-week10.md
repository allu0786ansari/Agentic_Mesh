# ADR 0012 — Agent mesh: triage, investigation, attribution, dispatch (Week 10)

**Status:** Accepted, 15 October 2026. Written before any mesh run.

**Implements:** Plan Week 10 tasks 1–7, Proposal §8.1–8.6, §9.2.

**Builds on:**

- ADR 0009: alerts, the bus, sessions and fusion;
- ADR 0010 and ADR 0011: the renderer and retrieval v2, the default;
- ADR 0010 §6: the rule baseline.

## 1. Graph

The graph is a LangGraph `StateGraph` with four nodes: **triage → investigate →
attribute → dispatch**.

**State.** The state holds:

- the session ID and site;
- the incident (alert IDs, nodes, time span, fused score, severity);
- the evidence records;
- the hop count;
- the candidates and the attribution;
- the actions and the status.

**Every node persists before it returns.**

- **What is written:** each node writes a row to `session_steps` (session, step,
  status, JSON payload, time) and moves the session's status (ADR 0009 §6
  transitions).
- **What it guarantees:** a crashed run resumes from the last completed step, and a
  replayed session never repeats a step that already completed.
- **Schema change:** `session_steps` is added to the session schema, alongside the
  Plan's `sessions`, `hops` and `actions` tables.

## 2. Triage

**Incidents.**

- Alerts are grouped into incidents per site by the Δt chaining rule (ADR 0009 §7),
  with Δt from `configs/params.yaml` (60 s).
- **When an incident closes:** when an alert of the same site ends more than Δt
  after the incident's last alert, or when the bus is drained at the end of a
  replay.
- **Clock:** grouping uses data time (`window_end`), so replays are deterministic.
- **Session key:** the session's idempotency key is its sorted alert IDs (ADR 0009
  §6). Its alerts are acknowledged only after the session is committed.

**Fused score.**

- **What it is:** the noisy-OR of the nodes' largest calibrated probabilities.
  Latency-only alerts are excluded on Site B.
- **Site A:** its incidents normally involve one node, so the score is that node's
  largest probability.
- **Confirmation:** the k-of-n flag (k = 2) is recorded on Site B.

**Severity.** These bands are fixed here, not tuned:

- `high` for a fused score ≥ 0.9;
- `medium` for ≥ 0.5;
- `low` otherwise.

## 3. Investigator

**Where it looks.** The **same site only**, at most `mesh.max_hops` = 3 hops. One
hop is one evidence request to one node.

- **Site B:** each node of the site, alerting nodes first, then the others in node
  order. The window is `[incident start − 60 s, incident end + 60 s]`.
- **Site A:** each alerting node, at most 3. The window is the adjacent windows,
  `[start − 120 s, end + 120 s]`.

**Failure handling** (Proposal §8.3):

- **Timeouts:** connect 0.2 s, read 0.5 s.
- **Retries:**
  - one retry, and only for transient failures (connection error, timeout, 503);
  - the backoff is 50 ms × 2^attempt, with uniform jitter, from a seeded RNG.
- **Circuit breaker per node:**
  - it opens after 2 consecutive failures;
  - one half-open probe is allowed after 30 s;
  - each transition is written as a metric event.
- **Deadline:** 3 s for the whole investigation. When it expires, the session
  continues with what it has and ends `COMPLETED_DEGRADED`. A hop refused by an
  open breaker also makes the session degraded.

**Every hop is recorded** in `hops` before the next starts.

**The node interface** is `NodeClient.evidence(node, start, end)`.

- In Week 10 it is backed by:
  - `LocalNodeClient`, which reads the edge run's evidence stores;
  - `FakeNodeClient`, the test double, which can fail before, during or after a
    hop, and can be slow.
- Week 11 replaces it with the mTLS/JWT node API without changing the agent.

## 4. Attribution

| Mode | Candidates | Decision |
|---|---|---|
| `full` | retrieval v2: top 10, no Impact-only techniques | one LLM call, whose JSON schema enumerates the candidate IDs |
| `retrieval_only` | retrieval v2 | the top-1 retrieved |
| `llm_no_retrieval` | all 92 candidate techniques of v2, IDs and names only, no retrieved context | one LLM call over that enumeration |
| `rule_only` | — | `configs/policy/rule_baseline.yaml` |

**The LLM call.**

- **Model:** `llm.model` (Llama 3.2 3B, G0), temperature 0, seed 0, `num_ctx`
  2048.
- **Timeout:** `mesh.attribution.llm_timeout_s` (21 s).
- **Prompt:** the system prompt is the G0 benchmark's. The user message holds the
  rendered evidence, then one line per candidate: `"<ID> <name> (tactics: …)"`.

**Validation** follows Proposal §8.4. The output must:

- parse as JSON;
- match the schema;
- name one of the candidates;
- have confidence ≥ 0.5.

**Fallback.**

- **When:** any check fails, the call times out, or Ollama is unreachable.
- **What happens in `full`:** the result is the top-1 retrieved technique, with
  `attribution_mode = retrieval_fallback` and the reason recorded.
- **What happens in `llm_no_retrieval`:** the session **abstains** instead. This mode
  never uses retrieval, so the ablation isolates the retriever. This was amended
  before any run: the first draft fell back to retrieval, which would have mixed
  the two.
- **Grounding:** the raw output's ID and whether it was grounded are recorded
  before validation, for the grounding rate (Proposal §10.4).

**Ranked list for top-3 (fixed now for X4).**

- `full`: the chosen technique, then the remaining candidates in retrieval order.
- `llm_no_retrieval`: only the chosen technique.
- `retrieval_only`: the retrieval order.
- `rule_only`: only its one technique (or none).

**Tactic.** The tactic reported is the chosen technique's tactics in the pin, not
the LLM's free-text tactic.

**Mitigations** come from the pinned bundle's `mitigates` relationships
(`data/kb/techniques.json`). They are never generated.

**Abstention.** When `rule_only` matches nothing, or there is no evidence at all,
the session ends `ABSTAINED`.

## 5. Dispatcher and approvals

The action policy is `configs/policy/actions.yaml`:

| Action | Impact | When |
|---|---|---|
| `operator_alert` | low (dispatched) | every attributed session |
| `increase_monitoring` | low (dispatched) | severity medium or high |
| `isolate_or_block` | high (`pending_approval`) | severity high and confidence ≥ 0.5 |
| `change_threshold` | high (`pending_approval`) | never proposed automatically; operator-initiated only |

**How actions are written.**

- Each action is one row in `actions`, which is UNIQUE on `(session, action type)`.
- It is published to `actions.<site>` with the action ID as message ID.
- **This is an outbox by idempotent keys.** A restart repeats both steps, and both
  are no-ops the second time. Dispatch is therefore exactly once per
  `(session, type)`.

**High-impact actions execute only through `approvals.decide`.**

- It checks an HMAC-SHA256 approval token over `(action ID, decision, approver,
  expiry)`, with the key in `MESH_APPROVAL_KEY`.
- Only then does it move the action to `approved` or `rejected`.
- The authenticated HTTP endpoint is Week 11's node-API stack, and calls this same
  function.

## 6. Runner and outputs

**The runner.**

- `mesh.runner` consumes `alerts.<site>` from a **copy** of an edge run's
  `storage.db`, so every mode reads the same alerts.
- It runs each closed incident's graph in a worker thread, with the investigation
  deadline enforced there.

**Outputs.** It writes `results/mesh/<site>__<mode>__<split>__<time>/`:

- `storage.db`, holding the sessions, hops, steps and actions;
- `events.jsonl`;
- `summary.json`:
  - sessions by status and by attribution mode;
  - the technique distribution;
  - the grounding rate;
  - investigation and incident-to-dispatch latency p50 and p95.

**Scope.**

- **No accuracy against the ground truth is computed here.** That is X4 (Week 13).
- **Only validation can run** in Week 10. `test` needs `--final`.

---

## Cloud development run: `rule_only` on the validation alerts

This run needs neither the knowledge base nor the LLM. It used the laptop's
committed Week 8 models through the cloud's edge replay, and the uncommitted Week 10
tree (`results/W10/*__summary.json`).

| | Site A | Site B |
|---|---|---|
| Alerts → incidents (Δt 60 s) | 448 → 45 | 350 → 220 |
| Sessions | 45 `COMPLETED` | 220 `COMPLETED` |
| Hops per session | 1 (the alerting node) | 3 (every node) |
| Investigation latency p50 / p95 (target p95 ≤ 2 s) | 3.3 / 5.6 ms | 5.1 / 8.1 ms |
| Incident-to-dispatch p50 / p95 (target p95 ≤ 3 s, LLM-free) | 7.7 / 13.2 ms | 8.6 / 12.9 ms |
| Actions | 59, all low-impact, dispatched | 226, all low-impact, dispatched |

**Readings.**

- **Why there are no high-impact actions.** `rule_only` gives no confidence, so the
  policy's `min_confidence` 0.5 stops `isolate_or_block`. The same holds for
  `retrieval_only`. Only the LLM modes can propose it.
- **Most Site B incidents are false alarms.** They are the ~1% normal-window alerts
  of the two benign validation days. Week 13's X3 and X4 account for them; nothing
  is tuned here.
- **Latency is local only.** Evidence is read from local stores. Week 11 adds the
  mTLS node API, and X5 measures latency again then.
