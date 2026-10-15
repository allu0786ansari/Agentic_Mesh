# ADR 0009 — Edge runtime, evidence records, bus, sessions and fusion (Week 8)

**Status:** Accepted, 15 October 2026.

- Written before the edge runtime produced any output.
- Nothing in it is chosen on validation results.
- Week 8 reads validation data only. The runtime refuses `test` without `--final`,
  which needs `g4-frozen`.

**Implements:** Plan Week 8 tasks 1–6, Proposal §5.3–5.4, §6.5–6.6, §8.2–8.3 and
§9.2–9.3.

## 1. Which model the runtime loads

- **The runtime is model-agnostic.** It loads any saved global model:
  `federated.run --save-model` writes `<dir>/<config_id>/seed_<n>/`, holding the
  weights, the round-0 pooled normalisation and the spec.
  - A model is named by `model_version`, the SHA-256 of all three.
  - Every evidence record carries it.
- **Development model:** FedProx at ε = 3, seed 11, on each site. This is the
  headline privacy level (ADR 0008) with the configured strategy.
- **The final frozen model** is chosen at G4 (Week 14), from X1 validation results
  under the rules already preregistered. Week 8 decides nothing about it.

## 2. Node calibration (fitted on the node, kept on the node)

Each node fits the following on its own data, exactly as the detection results do
(`x1._evaluate`). They are written to `calibration_<node>.json` next to the run.

| Item | Fitted on | Rule |
|---|---|---|
| Training-error references (`ErrorReference`, R3 median) | node's training windows | ADR 0003, ADR 0005 |
| Threshold | normal validation windows that are scored | window FPR ≤ 1% (Proposal §6.5) |
| Calibrator | all validation windows (normal and attack), scored only | isotonic, score → P(attack) (Proposal §6.5) |
| Magnitude buckets | normal validation windows | per feature, p90 / p99 / p99.9 of its error (Proposal §6.6) |
| Request-rate bucket (Site B) | normal validation windows | p1 / p99 of `req_recv` |

R3 abstains for the first 7 minutes of a segment (ADR 0005). An abstained window is
never alerted on and never used in calibration.

## 3. Evidence record (Proposal §6.6, with the fields made exact)

**Top features** (at most 5).

- **Which features:** those whose error in the window is at or above their own p90.
- **Order:** by the ratio of the error to that feature's p99, largest first.
- **Ranking against each feature's own validation distribution** means a noisy
  feature does not outrank a quiet one that moved.
- **The list may be empty.** That happens when the score crossed the threshold
  without any single feature standing out.

**Each top feature carries:**

- **`direction`:** `up` if the input is above the model's reconstruction (the mean
  signed residual over the window's segments), otherwise `down`.
- **`magnitude`:**
  - `low` = p90–p99;
  - `medium` = p99–p99.9;
  - `high` = above p99.9.

**Site A also lists `constants_changed`.** These are the *names* of training-constant
variables whose value in the window differs from the training value (Plan v3.1 Week
8 task 3). The 22 such variables never vary in training, and the model does not see
them.

**Site B also carries `protocol_summary`,** computed from the window's counters:

| Field | From | Values |
|---|---|---|
| `read_ops`, `write_ops`, `diagnostic_ops` (other function codes), `exception_responses` | `req_read`, `req_write`, `req_other`, `resp_exception` | `0`, `1-10`, `11-100`, `>100` |
| `request_rate` | `req_recv` against the node's p1/p99 | `low`, `normal`, `high` |
| `new_peer`, `unseen_address`, `oversized_quantity`, `malformed_frames`, `stacked_frames`, `slow_responses` | `new_peers`, `unseen_addr` + `unseen_addr_write`, `oversized_qty`, `malformed_frames`, `stacked_frames`, `slow_responses` | `true` when the counter is > 0 |

**Minimisation.**

- **Allowed numbers:** the only numeric fields are `score`, `threshold` and
  `calibrated_prob`.
- **Allowed strings:**
  - identifiers;
  - ISO times;
  - feature names from the node's feature list;
  - values from the enumerations above.
- **Enforcement:** the pydantic models forbid extra fields, and a contract test with
  random raw values checks that none appear in a serialised record.
- **Not formally private:** records are minimised, not differentially private
  (Proposal §5.4).

## 4. Alert

**Fields:** those of Plan §6.3, plus `latency_only`.

- **What `latency_only` is:** `true` when the top features are non-empty and all of
  them are latency features (`latency_median_ms`, `slow_responses`).
- **Why it is needed:** fusion's `exclude_latency` rule (ADR 0001, MB-13) acts at
  Triage, before any evidence is fetched. It needs this one bit.
- **What it reveals:** it says nothing that the node's evidence store would not.

## 5. Bus (Plan §6.4, §6.6)

**Delivery: at-least-once.**

- **Publishing:** idempotent, because the message ID is the alert ID, so a replayed
  alert is stored once.
- **Claiming:**
  - a consumer claims messages under a lease (default 30 s);
  - it acknowledges only after its handler commits;
  - an expired lease makes the message available again.
- **Concurrency:**
  - claims run inside `BEGIN IMMEDIATE`;
  - the database runs in WAL mode;
  - this makes concurrent processes safe.

**Dead letters.**

- After `max_attempts` (5) claims without an acknowledgement, a message is
  dead-lettered.
- The plan's table gains one column, `failed_at`, so a poison message cannot loop
  forever.

**Idempotent handling** comes from `sessions.idempotency_key` and the `actions`
UNIQUE constraint.

## 6. Sessions (Plan §6.4, Proposal §8.3)

- **Schema:** the Plan's `sessions`, `hops` and `actions` tables, verbatim, plus a
  `session_alerts` table. It records which alerts opened each session; the
  Investigator and the X6 duplicate-replay test need that.
- **Idempotency key:** SHA-256 of `site_id`, then `|`, then the sorted IDs of the
  opening alerts joined by `,`. Opening a session twice returns the same session.
- **Status transitions:**
  - The allowed path is OPEN → INVESTIGATING → ATTRIBUTING → DISPATCHING → one of
    the terminal states: COMPLETED, COMPLETED_DEGRADED, ABSTAINED, FAILED.
  - Any non-terminal state may go to FAILED.
  - INVESTIGATING and ATTRIBUTING may go to ABSTAINED.
  - Anything else is refused.
- **Hop limit:** `hop_count` can never exceed `mesh.max_hops` (3).

## 7. Fusion (Site B)

**Incidents.**

- Alerts of one site are sorted by `window_end`.
- An alert joins the current incident if it ends within Δt of the incident's
  previous alert.
- Otherwise it opens a new incident.

**Scores and decisions.** Alerts that are latency-only do not count for either
score; they are attached as single-device context.

- **Noisy-OR:** `1 − ∏(1 − p_n)` over nodes, with `p_n` the node's largest
  calibrated probability in the incident. This is the incident's fused score and
  severity.
- **k-of-n:** the number of distinct nodes alerting in the incident. The incident
  is confirmed when it is at least k.

**Settings: Δt = 60 s, k = 2. Not tuned.**

- The Plan's Week 8 task 5 says "tune Δt and k on Site B validation data". Both
  values are fixed in `configs/params.yaml` at `g1-frozen`, and PT4 is preregistered
  for 2-of-3 fusion.
- Choosing them now, after the freeze, would change a frozen setting on data that
  has been seen. The task is therefore dropped.
- The edge run reports a **descriptive** sensitivity table over Δt ∈ {30, 60, 120}
  and k ∈ {1, 2, 3} on validation. It is labelled as such and never used to select
  anything.
- The decision threshold on the noisy-OR score is set by X3's matched-FPR protocol
  (Plan Week 13), which will be declared before X3 runs.

## 8. Replay and detection latency

**Replay.**

- Replay merges a site's node streams in time order.
- It emits each window at its end time, at a speed factor: 1 is real time, larger
  values are faster, and 0 means as fast as possible.
- Site A windows are 60 s long at a 10 s stride (`infer_stride_s`), built from the
  1 Hz rows. Site B windows are the partition's 10 s rows.

**Streaming versus batch.** The runtime scores one window at a time, carrying R3's
causal history.

- **R3 logic:** an exact replica of the batch definition. The unit test agrees to
  1e-12.
- **The network forward pass:** both paths are float32. A single-row matrix product
  and a batched one round differently in the last bit.
- **Where the difference grows:** Site B's `norm_top3` divides each feature error by
  its training IQR, which is floored at 1e-6 for near-constant features. That
  multiplies the last-bit difference.
- **Measured on the Site B dev model:** up to 0.4% relative difference in score, and
  1 alert decision in 33,051 windows, at a window whose score lies at the threshold.
- **Every run reports the difference:** `summary.json` → `stream_vs_batch`, with
  the size of the difference, the number of flipped decisions, and how far each
  flipped window was from the threshold.
- **Which path is authoritative:** the batch path (`x1`) for every detection result.
  The runtime is the operational path.

**Detection latency** is the wall time from a window's emission to its alert being
committed to the bus (Proposal §8.6). It is written as a metric event per alert. Its
p50 and p95 are reported against the 1 s target.

---

## Results: cloud development runs (validation only)

**What these are.** The artifacts are in `results/W8/`. They were run from the Week 8
working tree before it was committed, and their provenance says so (`git_dirty`). The
laptop runs from the committed code (`docs/runbooks/phase3_week8.md`) are the
record. Development models: FedProx, ε = 3, seed 11, 50 rounds.

| | Site A (VAE, R3) | Site B (AE, R0) |
|---|---|---|
| Federated validation result of the model | AUROC 0.888, AUPRC 0.702, max ε 2.999 | AUROC 1.000, AUPRC 1.000, max ε 3.000 |
| Windows replayed (normal + attack validation) | 23,183 | 33,051 |
| Alerts (in bus = records in node stores) | 444 | 350 |
| Window FPR on each node's normal validation | 0.0096–0.0098 | 0.0099–0.0101 |
| Attack windows alerted | 220 of 347 (63%); every node above 50% | b1: 12 of 12 (b2, b3 see no attacker traffic) |
| Stream vs batch: alert decisions that differ | 0 of 23,183 | 1 of 33,051, at a window whose batch score *is* the threshold (margin 0.0) |
| Detection latency p50 / p95 (target ≤ 1,000 ms) | 4.5 / 6.8 ms | 0.8 / 1.2 ms |
| Evidence: alerts with no top feature / latency-only | 0 / 0 | 0 / 0 |

**Readings.**

1. **The flipped Site B decision is expected.**
   - The threshold is an observed normal-validation score (ADR 0003's FPR rule), so
     one window always sits exactly on it.
   - Last-bit rounding can push that window either way.
   - Detection results remain those of the batch path.
2. **Site A's training-constant variables change during attacks.**
   - Seen in 90 of 310 alerts on the attack-validation split, and in none of the 134
     on normal validation.
   - The model never sees these variables (ADR 0002), so this is evidence the
     reconstruction error cannot give.
   - It is reported here as an observation. Week 9's renderer will carry it as
     evidence, and attribution (Weeks 9–10) must show whether it helps.
3. **Site B attack-day drift appears again (ADR 0002 decision 5).**
   - b1 raised 8 false alarms in the 135 attack-day normal windows.
   - At the configured Δt = 60 s and k = 2, 3 of the 10 attack-day incidents involve
     two or more nodes. On benign validation, 44 of 210 do.
   - The attacker is seen by b1 only, so 2-of-3 fusion cannot confirm the real
     attack episodes. This is the H2b trade-off that the preregistration already
     names.
   - Nothing is tuned on it.
