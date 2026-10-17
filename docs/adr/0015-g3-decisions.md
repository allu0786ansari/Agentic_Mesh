# ADR 0015 — Gate G3 decisions: default attribution mode and L_LLM

**Status:** Accepted, 17 October 2026.

**Rule:** ADR 0014 §9, fixed before any multi-process run. The rule is computed by
`python -m agmesh.evaluation.g3` and written to `results/G3/g3_decision.json`.

**Scope:** validation runs only. Accuracy is not an input; attribution accuracy (X4)
is measured once, on the test incidents, after G4.

## 1. Inputs: Week 11 runs over mTLS

Both sites' node APIs ran as separate processes. LLM runs used the ADR 0013 §7
conditions (prompt cache off, warm-up).

| Site | Mode | Sessions | LLM answered | Validity | Grounding | Investigation p95 | Incident-to-dispatch p95 | Traces |
|---|---|---|---|---|---|---|---|---|
| B | `retrieval_only` | 220, all completed | — | — | — | 79.8 ms | 0.17 s | 220 of 220 |
| B | `full` | 220, all completed | 220 of 220 | 1.000 | 1.000 | 169.6 ms | **17.8 s** | 220 of 220 |
| A | `retrieval_only` | 45, all completed | — | — | — | 48.9 ms | 0.14 s | 45 of 45 |
| A | `full` | 45, all completed | 45 of 45 | 1.000 | 1.000 | 108.2 ms | **16.2 s** | 45 of 45 |

- **LLM call time:** p50 / p95 was 15.3 / 17.6 s on Site B and 14.4 / 16.1 s on
  Site A, with no timeouts.
- **Traces:** for every session, `checker --traces` finds one trace whose root span
  carries that `session.id`.

**Reproducibility.** These runs choose the same technique, incident by incident, as
the in-process runs:

| Mode | Site | Same technique as the in-process run | In-process run |
|---|---|---|---|
| `retrieval_only` | B | 220 of 220 | — |
| `retrieval_only` | A | 45 of 45 | — |
| `full` | B | 220 of 220 | 17 October 12:10, cache off |
| `full` | A | 44 of 45 | 17 October 10:40 |

The one Site A difference is incident 1, which was the cold-start fallback in the
earlier run (ADR 0013 §7.3). It is now answered. So moving to separate processes,
mTLS and JWT changed no decision.

## 2. The rule applied

| Check (both sites) | Threshold | Site A | Site B | Pass |
|---|---|---|---|---|
| JSON validity | ≥ 0.95 | 1.000 | 1.000 | yes |
| `full` incident-to-dispatch p95 | ≤ L_LLM = 24 s | 16.2 s | 17.8 s | yes |
| Investigation p95 over mTLS | ≤ 2 s | 108 ms | 170 ms | yes |

## 3. Decisions

1. **The default attribution mode stays `full`** (`mesh.attribution.mode`).
2. **L_LLM stays 24 s,** the preregistered PT7 value.
   - The LLM call timeout is 26 s (ADR 0013, `configs/mesh/llm_timeouts.lock.json`).
   - A call that ran to its timeout would miss PT7. None did in these runs. Such
     calls are reported, not re-targeted.
3. **`llm_no_retrieval` stays an ablation only,** with its own timeout of 74 s.
   - Validation outcome (in-process, ADR 0013 §7.4):
     - **Site B:** 220 of 220 answered; LLM p50 / p95 47.4 / 50.4 s.
     - **Site A:** 44 of 45 answered; 1 `Deadline` at 74 s, which abstained; p95
       60.2 s.
     - Validity and grounding were 1.000.
   - **Finding:** without retrieval, the 92-candidate prompt makes the LLM call
     about 3.1× slower than in `full` on the same laptop (p50 47 s against 15 s).

## 4. Engineering acceptance at G3

- **EA2 holds:**
  - in CI, `tests/security` (41 tests) rejects plaintext, wrong
    client certificates and TLS 1.2;
  - it rejects expired, wrong-audience, out-of-scope, wrong-window and replayed
    JWTs;
  - it refuses a non-allowlisted LLM endpoint;
  - the secret scan is clean.
- **EA3 (traces):**
  - every session's trace is recoverable from `results/traces/<run>` by
    `session_id`, and the trace ID is linked in `session_traces`;
  - viewing the traces in Phoenix is optional (`obs` group) and was not exercised in
    these runs.
- **EA4 holds** (no high-impact action without approval). The approval API tests
  pass: operator certificate only, HMAC gate, no repeated decision.

## 5. Limitations

- **Hardware:** all latencies come from the 8 GB development laptop (WSL, 5.8 GiB).
- **One replay:** a single replay with incidents processed back to back, so
  queueing is not measured (ADR 0013 §3.4).
- **Restarts:** the node API's replay cache is per process and lost on restart
  (ADR 0014 §4). This is recorded, not mitigated.
