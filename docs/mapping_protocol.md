# Attribution Ground-Truth Mapping Protocol

**Status:** v1.1, reviewed.

- **Pass 1:** 14 October 2026.
- **Pass 2:** 15 October 2026, both by the assistant at the author's request (§6).
- **Next:** the mapping is tagged `gt-frozen` (preregistration, Status). The tag is
  required before Phase 3.
- **After the tag:** changing this protocol requires a new ADR and invalidates
  attribution results produced under the old version.
**Implements:** Proposal v3.1 §10.4; ADR 0002 decision 6.

## 1. Purpose and rules

The mapping says, for every attack event, which MITRE ATT&CK for ICS technique(s) and
tactic(s) a correct attribution should name. It is the ground truth for RQ3.

1. The mapping is written **before any attribution output exists** and frozen at G1.
2. It uses only dataset metadata, the dataset's labels and the *names* of variables
   or protocol features. It never uses a model score, a retrieval result or an LLM
   output.
3. Every event gets a primary technique, an acceptable set, and the tactic(s) of
   those techniques. The acceptable set holds the techniques an analyst would name as
   the main one. It is kept small enough that no constant answer is correct for most
   events (§2, "Constant-answer baseline").
4. Two review passes are made, on different days, with disagreements resolved in
   writing and logged in §6.
5. Technique IDs, names and tactics follow the pinned ATT&CK for ICS release.
   - **Pin:** v20.0, `mitre-attack/attack-stix-data` @ `c2494b2`, SHA-256 in
     `src/agmesh/evaluation/attack_ics.py`.
   - **Extract:** `evaluation/attack/ics_attack_20.0_techniques.json`.
   - **Check:** a test fails if the truth uses an ID that is not active in the pin, or
     a tactic that differs from it.
   - The pin was moved from Week 9 to pass 2, because v19.0 had revoked two of the
     rule's IDs (§6).

## 2. Site A (HAI 21.03): 50 events

### Source of truth: the attack timetable published with the dataset

The HAI technical details document lists every HAI 21.03 attack: 25 single
primitives and 25 combinations of two. For each it gives the target controller code,
the target point(s), the start time and the duration (*HAI Security Dataset Technical
Details*, icsdataset/hai, `hai_dataset_technical_details.pdf`, "HAI 21.03, Attack
operation", printed pp. 36-38, PDF pages 37-39). The table is transcribed in
`src/agmesh/evaluation/hai_attacks.py`.

- **Matching.** Each labelled event is matched to the documented attack whose start
  time is within 90 s and whose processes equal the event's `attack_P1`-`attack_P3`
  labels.
  - All **50 of 50** events match.
  - Every event starts between 1 s before and 21 s after the documented minute.
  - The labelled duration equals the documented duration for 49 events. The
    exception is A209: 199 s in the data, 159 s in the document.
- **Errata corrected.** The document lists `P2_LCV01D` as the target of `P3-LC-CO2`
  in seven attacks (A220, A402, A403, A405, A504, A506, A511). The target is
  `P3_LCV01D`:
  - HAI has no point named `P2_LCV01D`.
  - The events are labelled `attack_P3`.
  - The document's own primitive catalogue lists AP37 as `P3-LC CV2 P3_LCV01D`.
  - In the data, 50% of `attack_P3` rows put `P3_LCV01D` above its normal-row 99.9th
    percentile, against 0.1% of normal rows (pass 2).
- **Column names in 21.03.** The document uses the newer point names. In the 21.03
  data, `P2_AutoSD` is the column `P2_ASD` and `P2_SCO` is the column `P2_CO_rpm`.

**Rule v1.1** (from the documented primitive, not from the data; ATT&CK for ICS v20.0 IDs):

| Documented primitive | Primary | Acceptable | Tactics |
|---|---|---|---|
| `SP` (setpoint or trip limit: `P1_B2016`, `P1_B3004`, `P1_B3005`, `P2_AutoSD`, `P2_VTR01/02`, `P2_RTR`) | T0836 Modify Parameter | T0836 | Impair Process Control |
| `CO` (controller output: `P1_PCV01D`, `P1_FCV03D`, `P1_LCV01D`, `P2_SCO`, `P3_LCP01D`, `P3_LCV01D`) | T1692.001 Command Message (formerly T0855) | T1692.001 | Evasion, Impair Process Control |
| `…PV` (the controlled variable's sensor is frozen or replayed to hide the effect) | unchanged | adds T1692.002 Reporting Message (formerly T0856) | Evasion, Impair Process Control |
| `-ST` (short-term, repeated several times) | unchanged | adds T0806 Brute Force I/O | — |
| Combination of two primitives | `SP` before `CO` (the setpoint is upstream of the output) | union of both | union |

**Excluded on purpose:**

- **T0831 Manipulation of Control** (Impact). By MITRE's definition it covers
  changes to "set point values, tags, or other parameters" as well as commands, so it
  describes every event. v1.0 accepted it for CO events only, which was inconsistent.
  Accepting it everywhere would make one constant answer correct for 100% of events
  (rule 3), so v1.1 accepts it nowhere.
- **T0832 Manipulation of View.** Excluded for the same reason.
- **The HMI concealment of setpoint changes.** The document says every SP primitive
  hides its change in the HMI. That is part of the primitive, not a second technique
  to name.

**Result.** Primary T0836 for 32 events and T1692.001 for 18. Majority-class
baseline: 0.64.

**Constant-answer baseline.** The share of events for which always giving one answer
counts as correct:

| T0836 | T1692.001 | T1692.002 | T0806 |
|---|---|---|---|
| 0.64 | 0.64 | 0.38 | 0.20 |

Both the majority-class and the constant-answer baselines are reported next to every
top-1 result. **Tactics carry no information on Site A.**

- Every event includes Impair Process Control, so a constant tactic answer scores 1.0.
- This was already true under v1.0.
- Tactic accuracy is therefore reported with that baseline and is never used for a
  claim. PT5 is defined on techniques.

### Cross-check: what the data alone shows (the v0 inference, kept under `inferred`)

The tool still infers a technique from the data. A variable "deviates" if at least
20% of the event's rows lie outside its pre-event range, widened by 10% of that range,
and only variables of the labelled processes count. The root is the earliest
deviation; among deviations that start within five rows of it, a setpoint is taken
before a command and a command before a sensor. Point types follow the HAI point list
(technical details, pp. 12-14).

| Type | Points |
|---|---|
| Setpoint | `P1_B*`, `P1_STSP`, `P3_LH/LL`, `*_LD`, `P4_*_PS`, `P2_ASD`, `P2_MSD`, `P2_RTR`, `P2_VTR*` |
| Command | `…D` valve and pump demands, `P2_CO_rpm` (speed-controller output), `P2_AutoGO`, `P2_ManualGO`, `P2_OnOff`, `P2_Emerg`, `P2_TripEx`, `P4_ST_GOV` |
| Sensor | everything else |

Two of these types were wrong in v0. `P2_CO_rpm` was typed as a sensor; the HAI
point list shows it is the speed-controller output, a command. `P2_ASD`, `P2_RTR`
and `P2_VTR*` were typed as commands; they are setpoints.

**Agreement with the documented truth:**

- The data-only primary equals the documented primary for **18 of 50** events, and
  lies in the acceptable set for **30 of 50**.
- The data-only rule almost never sees a setpoint attack. HAI records the setpoint
  values shown on the HMI, and the attacks hide their setpoint changes there.
  Instead, the controller's output moves, so the inference says T1692.001.
- This matters for RQ3. An attributor that reads only the data will tend to call a
  hidden setpoint attack a command attack. Telling the two apart needs reasoning
  about whether the output's movement is consistent with the controller tracking an
  unseen setpoint. That is a real test of grounded attribution, and the paper reports
  it as a stratum (SP-led versus CO-led events).

## 3. Site B (CIC Modbus 2023): 7 episodes (case study)

Episodes are those of `docs/labeling_protocol.md` §4. Candidate mapping for the first
pass:

| Episode | Observed behaviour | Primary technique | Acceptable set | Tactic |
|---|---|---|---|---|
| 0 | Two reads, one asking for 65,535 values (a malformed length) | T0888 Remote System Information Discovery | T0888, T0861, T0846, T0814 | Discovery, Collection, Inhibit Response Function |
| 1 | About 20 input-register reads; a stacked 23-frame packet | T0861 Point & Tag Identification | T0861, T0888, T0814 | Collection, Discovery, Inhibit Response Function |
| 2 | Five reads across all four read function codes | T0861 Point & Tag Identification | T0861, T0888 | Collection, Discovery |
| 3 | Two coil reads | T0861 Point & Tag Identification | T0861, T0801 | Collection |
| 4 | Slow scan, about one discrete-input read a minute for 20 minutes | T0801 Monitor Process State | T0801, T0861 | Collection |
| 5 | 65,535 coil writes to one address in 4 minutes | T0806 Brute Force I/O | T0806, T1692.001, T0814 | Evasion, Impair Process Control, Inhibit Response Function |
| 6 | 21 coil writes to two addresses | T1692.001 Command Message | T1692.001 | Evasion, Impair Process Control |

Site B is reported as a case study (validity, grounding, chosen technique per episode),
not as an accuracy estimate.

The dataset's attack logs were not part of the obtained data (`docs/labeling_protocol.md`),
so this mapping rests on the observed traffic. The CIC page lists the attacks in the
scenarios as "reconnaissance, query flooding, loading payloads, delay response, modify
length parameters, false data injection, stacking Modbus frames, brute force write and
baseline replay". Episodes 0 and 1 match "modify length parameters" and "stacking
Modbus frames". Both are also ways to crash a parser, so pass 1 added T0814 Denial of
Service to their acceptable sets. Episode 5 matches "brute force write"; the others
match reconnaissance-style reads and a write.

**Pass 2 checked the traffic again** with an independent dissection of the attacker's
(185.175.0.7) requests to IED1A (`evaluation/ground_truth/pass2/site_b_episodes.txt`):

| Episode | Attacker requests found |
|---|---|
| 0 | two reads (coils, discrete inputs), quantity 65,535 |
| 1 | 20 packets, 42 input-register reads, one packet of 23 stacked frames |
| 2 | five reads over FC 1–4 |
| 3 | two coil reads |
| 4 | 20 discrete-input reads and one register read over 20 minutes |
| 5 | 65,535 write-single-coil to address 0 in 4 min 19 s, and one malformed frame |
| 6 | 21 coil writes to addresses 0 and 13 |

**The CIC attack logs.** The dataset page says per-attack logs exist (Timestamp,
TargetIP, Attack, TransactionID). Its download form returned a server error on
15 October 2026, so they could not be obtained. They would let the case study name
the dataset's own attack labels.

## 4. Output files

- `evaluation/ground_truth/mapping_site_a.json`: `{segment_id: {primary, acceptable, tactics, evidence}}`.
- `evaluation/ground_truth/mapping_site_b.json`: `{episode: {...}}` with the same fields.

Both carry the protocol version, the reviewer and the date of each pass, and are
committed together with `docs/preregistration.md` in the `gt-frozen` commit.

## 5. Scoring (for reference; fixed at v1.1, before any attribution output)

- **ID normalisation.** Each predicted ID is first mapped to its active technique in
  the pinned release by MITRE's `revoked-by` relationships (`attack_ics.canonical`).
  A predicted T0855 is scored as T1692.001, and T0856 as T1692.002. Models trained
  before April 2026 use the old IDs. An unknown ID matches nothing.
- **Technique top-1:** the first predicted technique is in the acceptable set.
- **Technique top-3:** any of the first three is in the acceptable set.
- **Sub-techniques.** A parent technique does not match its sub-technique: T1692
  does not match T1692.001. Answering T1692 alone would be right for 74% of Site A
  events, which rule 3 rules out. Parent-level agreement is reported as a
  sensitivity, with that baseline.
- **Tactic accuracy:** the predicted technique's tactics (in the pin) intersect the
  event's tactics. On Site A its constant baseline is 1.0 (§2).

## 6. Review log

| Pass | Date | Reviewer | Events changed | Notes |
|---|---|---|---|---|
| 1 | 14 Oct 2026 | Claude (AI assistant), at the author's request | Site A: all 50 (truth moved from data inference to the documented timetable); Site B: episodes 0 and 1 (T0814 added) | Timetable transcribed from the HAI technical details; 50/50 events matched; 7 errata recorded; P2 point types corrected; rule v1.0 |
| 2 | 15 Oct 2026 | Claude (AI assistant). The author delegated the verification ("do it yourself") | Site A: all 50 relabelled to v1.1 (IDs only; T0831 removed). Site B: episodes 5 and 6 (IDs; T0831 removed) | Independent re-transcription from fresh downloads: 0/50 differences; errata and aliases confirmed in the data; ATT&CK pinned at v20.0; rule v1.1 |

### Pass 2 report

**Independence and its limit.**

- The author delegated pass 2, so both passes were made by the same AI system, on
  different days.
- To make pass 2 a real check, it did not use pass-1 notes:
  - It re-downloaded the sources: the PDF is byte-identical to pass 1's copy, and the
    test CSVs match the laptop's checksums.
  - It transcribed the table with a new parser and checked the result against the
    rendered pages.
  - It compared every field with the code by script.
  - It tested the timetable against the labels and the raw signals.
- No human has verified the transcription. The papers disclose this.
- The scripts and outputs are in `evaluation/ground_truth/pass2/`.

**1. Transcription.**

- All 50 rows of `TIMETABLE` equal the PDF, field by field: ID, date (from the merged
  date cells) and start, duration, controller codes and points. There are 0
  differences.
- Matched against the labels without the project's matcher:
  - All 50 labelled events find their attack.
  - Starts are 1 s early to 21 s late (median +3 s).
  - The labelled processes agree for 50 of 50.
  - Durations agree within 5 s for 49.

**2. Errata and aliases.**

- **A209:** confirmed. 199 s labelled, 159 s documented. The timetable keeps the
  document's value; detection uses the labels.
- **`P2_LCV01D` → `P3_LCV01D`:** confirmed by the catalogue and the data (§2).
- **`P2_SCO` → `P2_CO_rpm`:** confirmed.
  - `P2_SCO` is not a 21.03 column.
  - `P2_CO_rpm` lies in `P2_SCO`'s documented range (0–100,000).
  - In 6 of 7 SC-CO events it moves from its pre-attack median by more than five
    times the larger of its robust SD and half its pre-attack range.
- **`P2_AutoSD` → `P2_ASD`:** accepted on name and range ("Auto speed demand",
  0–3,200 RPM). The data cannot confirm it: no setpoint column moves in any setpoint
  event, as the document's "hiding SP changes in HMI" predicts.

**3. Rule.** Accepted, with two changes, which make v1.1:

- **(a) ATT&CK IDs.**
  - The rule named T0855 and T0856, which ATT&CK for ICS v19.0 (28 April 2026)
    revoked.
  - MITRE's own `revoked-by` relationships map them to T1692.001 Command Message and
    T1692.002 Reporting Message. Both are sub-techniques of T1692 Unauthorized
    Message, and both carry Evasion and Impair Process Control.
  - The truth now uses the IDs of the pinned v20.0, and predictions are normalised
    (§5).
- **(b) T0831 removed from the CO set,** for the reasons in §2. This removes Impact
  from the CO events' tactics.
- **Checked and kept as they were:**
  - SP-led events accept only T0836: MITRE defines Modify Parameter by setpoints;
    T1692.001 is a command to act.
  - The frozen or replayed sensor adds T1692.002: the catalogue says "attempt to
    maintain/repeat previous sensor value", which is a spoofed reporting message.
  - `-ST` adds T0806 Brute Force I/O: the catalogue says ST "decrease or increase
    for a few seconds and restores to normal. Repeat several times", and MITRE says
    "repetitively or successively change I/O point values".
  - Combinations: the SP-before-CO precedence.
- **Effect:**
  - Primaries: 32 T0836, 18 T1692.001, the same counts as v1.0.
  - Constant-answer baselines: unchanged apart from T0831, which is gone.
  - Data-only agreement: unchanged (18/50 equal, 30/50 in the acceptable set).

**4. Site B.**

- All seven observed behaviours are confirmed (§3).
- The mappings are kept, apart from (a) and (b).
- T0814 on episodes 0 and 1 is kept, because a malformed length and stacked frames
  are both parser-crashing inputs.

**Applied by** `python -m agmesh.evaluation.ground_truth relabel`.

- It recomputes each Site A event's truth from its `attack_id` and the timetable, and
  translates the stored data-only inference.
- It does not read the test partitions again. The pass-1 read (logged under
  `ground_truth_site_a`) remains the only labelling access.
- The `reviews` field of both JSON files records this pass, and `reviewed` is `true`.
