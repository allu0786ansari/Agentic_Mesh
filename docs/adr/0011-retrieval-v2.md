# ADR 0011 — Retrieval v2: mechanism candidates, ten for the LLM

**Status:** Accepted, 15 October 2026.

- Written after the PT6 result (ADR 0010: Recall@5 0.333) and before any v2 run.
- **This change is post hoc.** It was motivated by the PT6 diagnosis on the
  validation queries. Its validation numbers are therefore exploratory, and only the
  held-out test incidents can confirm it (§4).
- **PT6 is unchanged:** 0.333, not met. It is reported as such in every paper.

**Supersedes for the system default:** ADR 0010 §2 (candidates) and Proposal §8.4's
"top-5". v1 stays implemented and is reported alongside v2 in X4.

## 1. What changes

| | v1 (ADR 0010, PT6) | v2 (this ADR) |
|---|---|---|
| Candidate techniques | all 104 active | the 92 active techniques whose tactics are **not only Impact** |
| Candidates offered to the LLM (`full` mode) | 5 | **10** |
| Passages, encoder, renderer, query set, relevance, metrics | — | unchanged |

**What stays the same.** Everything else is unchanged, so v1 and v2 differ only in
these two points:

- the passages, the encoder and its pinned snapshot;
- the renderer and the 48 validation queries;
- relevance (canonical IDs, no parent match);
- Recall@k and MRR.

## 2. Why these two changes, and their risks

**Removing Impact-only techniques.**

- **Why:** attribution asks *how* the adversary acted: which technique was used. The
  12 Impact-only techniques describe *consequences*: Loss of Control, Loss of Safety,
  Manipulation of Control and others.
- **Consistency with the ground truth:** by protocol v1.1, no Impact technique is
  acceptable for any event, because by MITRE's own definitions they describe every
  manipulation event (rule 3).
- **The risk:** an analyst can no longer be given a consequence as the answer. A
  deployment that wants consequences would add them as context, not as the
  attributed technique.
- **Techniques that keep Impact among other tactics stay candidates.**
- **Membership is computed from the pinned extract,** not hand-listed. A test checks
  that 12 techniques are removed and 92 remain.

**Ten candidates.**

- **Why:** the LLM may only choose among the candidates it is given (grounding). With
  five, the right technique is absent for most Site A incidents.
- **What it costs:**
  - a longer prompt;
  - a larger enumeration in the JSON schema;
  - more ways for the LLM to choose wrongly.
- **Measured later:** latency and grounding are measured in X4 and X5.
- **The `retrieval_only` mode is unchanged:** it still takes the top-1.

## 3. Validation evaluation (exploratory)

```
make retrieval-eval VERSION=v2
```

- **Output:** `results/retrieval/val__v2__exploratory__<time>.json`.
- **Reported metrics:** Recall@5, Recall@10 and MRR, with the same breakdowns as
  ADR 0010 §5.
- **It is always labelled exploratory.** The same queries motivated the change.

## 4. Confirmation on held-out incidents (preregistered now)

**When.** After `g4-frozen` (Week 14), once per site, with `--final`.

**Queries.** Each **test** incident gets a rendered query, built exactly as in ADR
0010 §4:

- Site A: the 35 test events of the frozen mapping;
- Site B: episodes 4–6.

No paraphrases are written for test incidents.

**Target PT6-v2.**

- **Metric:** Recall@10 (the candidates the LLM sees) of v2 on the test incidents.
- **Target:** ≥ 0.85.
- **Also reported:** Recall@5 and MRR, and v1's numbers on the same queries.
- **Status:** this is a secondary target and does not replace PT6.
- **If it is missed,** it is reported as a finding, as PT6 was.

## 5. Implementation

- `configs/kb/retrieval.yaml` defines `v1` and `v2` and sets the system default to
  `v2`.
- `Retriever(..., exclude_only_tactics=...)` filters candidates after scoring, so
  the passages and their order are unchanged.
- `retrieval_eval --version`.
- Week 10's attribution agent reads the version and its candidate count from this
  file.
