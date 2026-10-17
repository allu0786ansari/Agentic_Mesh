# ADR 0013 — LLM path fix after the first Week 10 runs

**Status:** Accepted, 16 October 2026. Written before any re-run of the two LLM
modes.

**Amends:** ADR 0012 §4 (the LLM call) and §6 (the runner's summary).

**Scope:** validation only. Nothing here looks at the test split.

## 1. What happened

The first laptop runs of `full` and `llm_no_retrieval` (commit `c2d532e`) used
ADR 0012 as written. The error strings come from each run's `storage.db`.

| Run | LLM answered | Failed | Error |
|---|---|---|---|
| Site B `full` | 112 | 108 → `retrieval_fallback` | `ReadTimeout` |
| Site A `full` | 0 | 45 → `retrieval_fallback` | `ReadTimeout` |
| Site B `llm_no_retrieval` | 0 | 220 → `abstained` | 184 `ReadTimeout`, 35 `ConnectError` (Ollama not listening), 1 `RemoteProtocolError` |
| Site A `llm_no_retrieval` | 0 | 45 → `abstained` | `ReadTimeout` |

Those results stand and stay in the repository. They are reported as the first run
of ADR 0012.

## 2. Diagnosis

Four causes are visible. None of them involves accuracy.

1. **The timeout was calibrated on different prompts.** G0 measured about 207-token
   prompts with 5 candidates. Rebuilt from the cloud `rule_only` sessions and the
   G0 characters-per-token ratio, the v2 prompts are:
   - **`full`:** about 450–530 tokens (10 candidates);
   - **`llm_no_retrieval`:** about 1,830–1,930 tokens (92 candidates).

   At G0's measured rates (prompt reading about 90 tokens/s, writing about
   7 tokens/s), a `llm_no_retrieval` call needs about 20 s just to read the prompt,
   and about 28 s in total. That exceeds the 21 s timeout on every call.
2. **The context is too small for `llm_no_retrieval`.** About 1,900 prompt tokens
   plus the answer exceed `num_ctx` 2048.
3. **One G0 setting was missing.** G0 sent `num_predict` 256; the mesh client did
   not, so answers had no length cap.
4. **Failures cluster in time.** Site B `full`:
   - 74 of the first 80 incidents succeed;
   - 2 of incidents 80–99;
   - 0 of the last 40.

   Successful calls do not slow down (block means 12.6–17.8 s). Two explanations
   fit:
   - (a) the server keeps working on a request the client abandoned, so the next
     one starts late;
   - (b) particular incidents make the model write long answers.

   The committed logs cannot separate them, so §3 records the numbers that can.

## 3. Decision

**3.1 G0's settings, restored.** Every mesh LLM call sends `num_predict` 256,
together with temperature 0 and seed 0.

**3.2 Context size.** This follows from prompt size alone.

| Mode | `num_ctx` | Why |
|---|---|---|
| `full` | 2048 | about 530 prompt tokens + 256 fit |
| `llm_no_retrieval` | 4096 | about 1,930 prompt tokens + 256 do not fit in 2048 |

**3.3 Timeout, measured by G0's rule on the real prompts.** G0 decision 7 sets the
LLM timeout to 1.5 × the measured p95 of one call. `make llm-calibrate MODE=<mode>`
re-applies that rule:

- **Prompts:** 30 validation prompts, 15 per site, sampled with seed 0 from the
  first ADR 0012 run of that mode. Each is rebuilt from that session's stored
  evidence text and candidates.
- **Calls:**
  - one warm-up call is excluded;
  - there is no client timeout (a 600 s safety cap);
  - each call uses 3.1 and 3.2.
- **Result:** `timeout_s` = ceil(1.5 × p95), written to
  `configs/mesh/llm_timeouts.lock.json` and committed.
- **Use:** the runner refuses an LLM mode without that lock.
- **What it measures:** only time and token counts. The calibration does not
  compare answers with ground truth.

**3.4 A timed-out request is stopped.**

- **Streaming:** the client streams the reply and enforces the timeout as a total
  deadline. At the deadline it closes the connection, which cancels the request on
  the server.
- **Recovery:** after any LLM failure, before the next incident, the runner:
  1. waits until Ollama answers `/api/tags`;
  2. sends one probe call (one output token, the same options, so the model is not
     reloaded).

  The recovery is capped at 120 s and logged as an `llm_recovery` event. It runs
  between incidents, so it is not part of any incident's latency.
- **Not measured, in any mode:** the replay runs incidents back to back, not at
  data time, so queueing is not measured.

**3.5 What each call records.** Each call records:

- the prompt and output token counts and durations;
- the model load time;
- `done_reason`;
- on failure, the error class and the elapsed seconds.

The summary reports:

- failures by error class;
- prompt and output tokens (p50, p95);
- recoveries;
- the resolved retrieval version (it was `default`).

**3.6 PT7 does not change.** The preregistered full-mode target stays at
incident-to-dispatch p95 ≤ 24 s (L_LLM from G0).

- **If the new timeout is above 21 s,** some full-mode incidents can exceed 24 s,
  and PT7 is reported as missed for them.
- **The final L_LLM** and the default attribution mode are still Gate G3 decisions
  (Plan §G3; ADR 0001 decision 7: "re-measured … at G3").
- **`llm_no_retrieval`** is an accuracy ablation and has no latency target. Its
  latency is reported as a finding: the retrieved candidate list makes the prompt
  about 4× shorter.

**3.7 Two labels corrected.**

- `retrieval_eval` wrote `target: {recall@5: 0.85}` for v2. The v2 target is
  PT6-v2: Recall@10 ≥ 0.85 on the test rendered queries after `g4-frozen`
  (preregistration §3b).
- The mesh summary now names the retrieval version it used.

## 4. Re-run

Only `full` and `llm_no_retrieval` are re-run, on validation, for both sites, after
the two calibrations. `rule_only` and `retrieval_only` do not call the LLM, and
their first runs stand.

## 5. Not changed

The following are not changed:

- the prompt;
- the candidate lists;
- the JSON schema;
- the confidence floor;
- the fallback and abstention rules;
- the model.

If 3.1–3.4 still leave most calls failing, this ADR does not try a further fix. The
result goes to Gate G3, where the Plan's rule applies: if latency or validity is
poor, `retrieval_only` becomes the default, and the LLM explains after dispatch.

## 6. Addendum, 17 October 2026: memory exhaustion on the laptop

This addendum was written after the first ADR 0013 runs and before any further
LLM run.

**What the runs showed.**

- **`full` calibration:** all 30 calls took 8.8–17.0 s and decoded at
  7.2–8.2 tokens/s. p95 was 16.8 s, so the timeout is 26 s.
- **`llm_no_retrieval` calibration:**
  - 8 of the 30 calls decoded at 0.19–0.88 tokens/s, against 5–5.8 in healthy
    calls;
  - those calls took 124–405 s, against 47–67 s for healthy ones;
  - prompt reading slowed only from about 48 to about 31 tokens/s;
  - p95 was 387 s, so the timeout came out at 581 s.
- **Site B `llm_no_retrieval` run:** you stopped it after 143 of 220 incidents and
  7.7 h. It went through runs of 10–25 consecutive incidents at 290–540 s each.
- **Site B `full` re-run:** 185 of 220 incidents were answered. The 35 failures
  (34 `Deadline`, 1 `ReadTimeout`) came in four bursts.

**Machine state afterwards (17 October, 10:21).**

- **The server used about twice the memory Ollama reported:** `llama-server` held
  5.0 GiB (86.3% of WSL's 5.8 GiB). `ollama ps` reported 2.6 GB at context 4096.
- **Parallel slots are ruled out:** it ran with `-np 1`, one slot.
- **Memory was exhausted:**
  - `MemAvailable` was 160 MiB;
  - the page cache was down to 64 MiB;
  - 2.5 GiB of swap was in use.
- **It had restarted during the run:** the server process was started at 07:32.
- **It was the background service:** Ollama was running as a systemd service, so the
  runbook's `ollama serve` never started and `~/ollama.log` is empty.

**Interpretation.** The model's weights are memory-mapped. With the page cache gone,
they are re-read from disk while decoding. Decoding reads every weight once per
token, so it slows by 25–35×; prompt reading works in 512-token batches and slows
much less. That is what was measured. Why the server's memory grows during a run is
not identified. The per-incident readings in 6.3 will show it.

**Constraint.** The laptop has 8 GB, and WSL keeps its 5.8 GiB. The machine is not
enlarged.

### 6.1 Which measurement stands

- **`llm_no_retrieval`:** its calibration of 17 October is declared broken, because
  it was measured under memory exhaustion. It stays in the repository and is
  replaced once (`--force`).
- **`full`:** its calibration stands.

### 6.2 Conditions for every remaining LLM run

- **The service is gone:** the Ollama systemd service is disabled.
- **One server, started by hand:**
  - `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_MAX_LOADED_MODELS=1`;
  - its log goes to `~/ollama.log`;
  - it is restarted fresh before each calibration and each run.
- **Nothing else heavy runs alongside.**
- **Enough memory to start:** `MemAvailable` is at least 2 GB before each start.

### 6.3 Observation only

- The runner prints one line per incident.
- The runner and the calibration record, per incident or per call, `MemAvailable`
  and the LLM server's resident memory.
- Nothing decides on these numbers.

### 6.4 Stopping rule for the `llm_no_retrieval` re-calibration

The rule is fixed before that re-calibration.

- **When it stops:** if any calibration call decodes under 1 token/s, or p95 exceeds
  120 s. For reference, the healthy calls above took 47–67 s.
- **What follows from stopping:**
  - the ablation is not run on this machine;
  - the mode is removed from the lock, and the runner refuses it;
  - the result is reported as a finding (an 8 GB laptop cannot hold the
    92-candidate prompt at context 4096), with these memory readings.
- **Otherwise:** the timeout follows §3.3, and both sites are run.

### 6.5 The `full` runs

- **Site B:** the re-run of 17 October stands and is not repeated. Its 35 failures
  are reported together with the memory evidence.
- **Site A:** it is run under 6.2.
