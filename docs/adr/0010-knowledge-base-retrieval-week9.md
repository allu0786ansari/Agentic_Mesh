# ADR 0010 — Knowledge base, evidence rendering and retrieval evaluation (Week 9)

**Status:** Accepted, 15 October 2026.

- Written before any retrieval result exists. No embedding model has been run on
  any query.
- The renderer, passages, query set, metrics and rule baseline below are fixed
  here.
- The **first laptop run** of `make retrieval-eval` with this protocol is the PT6
  measurement. Any later change gets a new version of this ADR, and runs made
  after it are labelled exploratory.

**Implements:** Plan Week 9 tasks 1–7, Proposal §8.4 and §10.4, preregistration PT6.

## 1. Sources and pins

| Source | Pin |
|---|---|
| ATT&CK for ICS | v20.0 (`attack_ics.PIN`: commit `c2494b2`, SHA-256 `c9ca0ce5…1183`), fetched from GitHub raw and verified |
| Encoder | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, cosine |
| NIST SP 800-82 Rev. 3 | `https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-82r3.pdf` |

**Encoder revision.**

- This workspace cannot reach Hugging Face, so the revision cannot be pinned in
  advance.
- The first build records the snapshot commit and the SHA-256 of the weights in
  `configs/kb/encoder.lock.json`, which is committed.
- Later builds refuse a different snapshot.

**NIST SP 800-82 Rev. 3.**

- The same first-use rule applies: its SHA-256 is recorded in
  `configs/kb/sources.lock.json` on first download.
- It is supplementary context only. It is never a technique candidate, and PT6
  does not depend on it.
- If it is missing, the build continues without it and says so.

## 2. Passages (`mitre_ics`)

**Candidates.**

- Every **active** technique of the pin is a candidate: 104, including the parent
  T1692 and its sub-techniques.
- Revoked and deprecated techniques are left out.
- Their IDs are still accepted in predictions through `attack_ics.canonical`
  (ADR protocol v1.1 §5).

**Passages.**

- **Why more than one per technique:** all-MiniLM-L6-v2 reads at most 256
  word-pieces. A single chunk holding a technique's name, description and detection
  text would lose the detection text to truncation, and that text is the part that
  describes what can be observed.
- **What each technique gets:**
  - one **summary passage**: `"<name>. Tactics: <tactics>. <description>"`, with
    citations, markdown links and HTML removed;
  - one **detection passage** per analytic of its detection strategies (`detects`
    relationships): `"<name>. Detection: <analytic description>"`.
- **How a technique is scored:** by the *maximum* cosine similarity over its
  passages.
- **The Plan's check** ("every ICS technique in the bundle is present") is applied
  to techniques, not passages.

**Mitigations** are the `mitigates` relationships of each technique. They are
stored for Week 10's dispatcher and are never embedded.

The `nist_80082` collection holds the NIST passages, split at numbered section
headings. The `incident_memory` collection is created empty. It stays empty for
every evaluation, so no evaluated incident can retrieve itself.

## 3. Renderer (`kb/render.py`): evidence records → incident text

The renderer is a deterministic template over the minimised evidence records of
one incident. It never sees labels, scores or raw values.

**Vocabulary comes from the datasets' own documentation.**

- **Site A:** `configs/kb/hai_points.yaml`, transcribed from the HAI point list
  (technical details pp. 12–14).
  - The 21.03 aliases come from the document's own parenthesised names.
  - `P2_SIT02` and `P2_HILout` are not documented. Only their type is rendered.
- **Site B:** `configs/kb/modbus_features.yaml`, taken from the feature
  definitions in `data/site_b.py`.
- **Point types:** setpoint, command or sensor, from `ground_truth.variable_type`.
  This is the HAI naming convention.

**Site A text.**

- **Opening line:** `"Industrial control system incident at a boiler, turbine and
  water-treatment testbed (process <P>), node <node>: <n> anomalous 60-second
  windows over <m> minutes."`
- **Features:** up to five, aggregated over the incident's records. They are ranked
  by how many records name them, then by their highest magnitude, then by name.
  Each is rendered as `"<type phrase> <description> (<name>) <rose|fell>, <low |
  medium | high> deviation"`.
- **Training constants:** `"Variables that are constant in normal operation
  changed: <description> (<name>), …"`.

**Site B text.**

- **Opening line:** `"Modbus/TCP incident at <device role> (<node>): <n> anomalous
  10-second windows over <m> minutes."`
- **Counts:** read, write, other-function-code and exception-response counts, each
  rendered as the largest bucket seen.
- **Request rate:** `high` if any record says high, `low` if any says low,
  otherwise `normal`.
- **Flags:** one clause for each that is true in any record (new peer, unseen
  register address, oversized quantity, malformed frames, stacked frames, slow
  responses).
- **Features:** the top features, as for Site A.

**No technique names, IDs or tactic names appear in the template.** A test checks
the rendered text against every technique name in the pin.

## 4. Query set (`evaluation/retrieval_eval.py`)

**Validation incidents.**

- **Site A:** each of the 15 validation attack events of
  `mapping_site_a.json` (`partition = val_attack`). Its evidence is the records of
  its client node whose window overlaps `[start, start + labelled_s + 60 s]`.
- **Site B:** each of episodes 0–3 (`labeling_protocol.md` §4). Its evidence is the
  records of all three nodes in `[start, end + 60 s]`.
- **The labels set only the incident window.** The query text comes from the
  evidence alone.
- **A missed event produces no query.** An event with no records is counted and
  reported, but it has nothing to render. Retrieval is evaluated on detected
  incidents.

**Paraphrases.** These are 30 operator-style descriptions of the same validation
incidents, kept in `configs/kb/paraphrases.yaml`.

- Each is keyed to an incident and inherits its acceptable set.
- They are written from the rendered text and the evidence only, without opening
  any knowledge-base passage.
- `tests/unit/test_retrieval_set.py` fails if any paraphrase shares an 8-gram with
  any passage. This is the Plan's guard; tokens are lower-cased words.
- **Limitation to disclose:** the author is the assistant, a language model that has
  read ATT&CK. The 8-gram test is the operational guarantee, not a claim of
  ignorance.

**Relevant set.** The incident's acceptable set in the frozen ground truth
(`gt-frozen`, protocol v1.1), after `canonical`. A parent technique does not match a
sub-technique (protocol §5).

## 5. Metrics

**Primary.** These are computed on all queries pooled (the rendered incidents and
the paraphrases), and per site and per kind.

- **Recall@5:** the share of queries whose top 5 techniques contain at least one
  relevant technique. PT6 target: ≥ 0.85.
- **MRR:** the mean of 1 / rank of the first relevant technique, ranked over all
  104 candidates.

**Descriptive.**

- Recall@1, @3 and @10.
- Site A broken down by SP-led and CO-led events.

**The reference encoder is the only one that reports.** Every result names its
encoder. The hashing encoder used by CI and the fixture tests can never produce a
reported number: `retrieval_eval` refuses it unless `--fixture` is given.

## 6. Rule baseline (`configs/policy/rule_baseline.yaml`, `mesh/rules.py`)

**Frozen now, before any X4 run.** The rules are applied in order to an incident's
evidence, and the first rule that matches gives the technique.

**Site A.** This is the data-only rule of the ground truth (`ground_truth.rule`,
pass 1), carried over to evidence records.

- **The root** is the earliest record's highest-ranked top feature, using the
  setpoint > command > sensor precedence among that record's top features.
- **Changed constants also count:** a changed training constant of setpoint type
  counts as a setpoint deviation.
- **Rules, in order:**
  - root setpoint → T0836;
  - root command → T1692.001;
  - root sensor → T1692.002.

**Site B.** The rules come from the protocol semantics of the CIC attack
categories.

- **Rules, in order:**
  - write requests `>100` → T0806 (flood or brute-force write);
  - any write → T1692.001;
  - malformed frames, stacked frames or oversized quantity → T0814 (inputs that can
    crash a parser);
  - reads spanning ≥ 10 minutes → T0801 (monitoring);
  - any read → T0861;
  - otherwise, abstain.
- **Not independent of the ground truth:** the case-study mapping (protocol §3)
  rests on the same protocol semantics. This rule is labelled as such. Site B is a
  case study, not an accuracy estimate.

---

## Build facts (before any reference-encoder run)

These are counts, not results. No retrieval quality has been measured.

**Knowledge base.**

- The pinned bundle parses to **104 active techniques**, 26 of them
  sub-techniques.
- Every technique has at least one detection analytic.
- The collection holds **404 passages**: 104 summary and 300 detection.
- The embedded Qdrant store and the in-memory store give identical rankings on the
  same vectors.

**Query set.** It was rendered in the cloud from the laptop's committed Week 8
models (`results/models/*`); Site A reproduces the laptop's 448 alerts.

- **18 rendered queries:** 14 Site A, 4 Site B. They are kept in
  `results/W9/queries_val.json` for comparison with the laptop run.
- **One Site A validation event had no evidence** (`site_a/test2:e18`, missed by
  the detector), so there are 14 Site A queries, not 15.
- **30 paraphrases** cover 17 of the 18 incidents.
- **Overlap check:** none shares an 8-gram (or even a 5-gram) with any of the 404
  passages.
- **Total: 48 queries.**

**Fixture run.** The full pipeline (build → render → rank → metrics) was run
end-to-end with the hashing encoder, `--fixture`. That output is labelled fixture,
is not committed, and is not a result.

---

## Result: the PT6 measurement (laptop, 15 October 2026)

`results/retrieval/val__pt6.json` (`pt6_measurement: true`) comes from commit
`17a804c`. The tree's only change was the `uv.lock` update, committed in the same
push.

**Build details.**

- **Encoder:** `all-MiniLM-L6-v2` at snapshot `1110a243…`, with its weights'
  SHA-256 pinned in `configs/kb/encoder.lock.json`.
- **NIST SP 800-82r3:** 627 passages, SHA-256 pinned in
  `configs/kb/sources.lock.json`.
- **Rendered queries:** the laptop's 18 are identical to the cloud's.

| Queries | n | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | MRR |
|---|---|---|---|---|---|---|
| **All (PT6)** | 48 | 0.083 | 0.167 | **0.333** | 0.688 | 0.219 |
| Rendered | 18 | 0.111 | 0.222 | 0.500 | 0.833 | 0.280 |
| Paraphrases | 30 | 0.067 | 0.133 | 0.233 | 0.600 | 0.182 |
| Site A | 37 | 0.027 | 0.081 | 0.297 | 0.676 | 0.159 |
| Site B | 11 | 0.273 | 0.455 | 0.455 | 0.727 | 0.420 |
| Site A, SP-led | 31 | 0.032 | 0.097 | 0.323 | 0.677 | 0.168 |
| Site A, CO-led | 6 | 0.000 | 0.000 | 0.167 | 0.667 | 0.111 |

**PT6 (Recall@5 ≥ 0.85) is not met: 0.333.** Under the preregistration, this is
reported as a finding. The number stands, and nothing below changes it.

### Exploratory diagnosis (not a result)

These readings come from the per-query rankings of the PT6 artifact.

1. **On Site A, effects outrank mechanisms.**
   - 53% of all top-5 slots go to the 12 Impact-only techniques: Loss of Control,
     Loss of Protection, Loss of Safety, Manipulation of Control, Loss of
     Availability, Loss of Productivity and Revenue, and others.
   - The rendered evidence describes what happened to the process: valves moved,
     levels and speeds fell. That is the language of ATT&CK's *impact*
     descriptions.
   - The frozen ground truth names the *mechanism*: Modify Parameter, Command
     Message, Reporting Message. Under protocol v1.1 rule 3, no Impact technique is
     acceptable for any event.
   - So the embedding similarity is semantically reasonable but answers a different
     question than the attribution truth asks.
2. **Site B, whose evidence is protocol behaviour, works.**
   - All four rendered episodes put a relevant technique at rank 1 or 2.
   - The paraphrases are weaker (Recall@5 2/7).
3. **Relevant techniques sit just below the cut.** Recall@10 is 0.688 against
   Recall@5 of 0.333, and the median rank of the first relevant technique is 7.
4. **What it means for RQ3.**
   - In `full` mode the LLM may only choose among the top 5 candidates (Proposal
     §8.4).
   - With this retriever, its top-1 accuracy on Site A is therefore capped near the
     Recall@5 of 0.30.
   - Mapping observed *effects* to the *mechanism* that caused them takes inference
     that lexical or semantic similarity alone does not supply.

Any change made because of this result (candidate set, k, query form) is
**post hoc**. It needs a new version of this ADR, written before it is run. It is
evaluated on validation as exploratory, and it can only be confirmed on held-out
incidents (the test incidents, after G4).
