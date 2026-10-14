# Attribution Ground-Truth Mapping Protocol

**Status:** v1.0. Pass 1 was done on 14 October 2026 by the assistant at the author's
request. It awaits the author's verification (§6). Changing this protocol after
`g1-frozen` requires a new ADR and invalidates attribution results produced under the
old version.
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
5. Technique IDs and names are checked against the pinned ATT&CK for ICS STIX bundle
   in Week 9. Any ID that does not exist there is a protocol error.

## 2. Site A (HAI 21.03): 50 events

### Source of truth: the attack timetable published with the dataset (v1.0)

The HAI technical details document lists every HAI 21.03 attack: 25 single
primitives and 25 combinations of two. For each it gives the target controller code,
the target point(s), the start time and the duration (*HAI Security Dataset Technical
Details*, icsdataset/hai, `hai_dataset_technical_details.pdf`, "HAI 21.03, Attack
operation", pp. 36-38). The table is transcribed in
`src/agmesh/evaluation/hai_attacks.py`.

- **Matching.** Each labelled event is matched to the documented attack whose start
  time is within 90 s and whose processes equal the event's `attack_P1`-`attack_P3`
  labels.
  - All **50 of 50** events match.
  - Every event starts between 1 s before and 21 s after the documented minute.
  - The labelled duration equals the documented duration for 49 events. The
    exception is A209: 199 s in the data, 159 s in the document.
- **Errata corrected.** The document lists `P2_LCV01D` as the target of `P3-LC-CO2`
  in seven attacks. HAI has no such point, and the events are labelled `attack_P3`,
  so the target is `P3_LCV01D`.
- **Column names in 21.03.** The document uses the newer point names. In the 21.03
  data, `P2_AutoSD` is the column `P2_ASD` and `P2_SCO` is the column `P2_CO_rpm`.

**Rule v1.0** (from the documented primitive, not from the data):

| Documented primitive | Primary | Acceptable | Tactics |
|---|---|---|---|
| `SP` (setpoint or trip limit: `P1_B2016`, `P1_B3004`, `P1_B3005`, `P2_AutoSD`, `P2_VTR01/02`, `P2_RTR`) | T0836 Modify Parameter | T0836 | Impair Process Control |
| `CO` (controller output: `P1_PCV01D`, `P1_FCV03D`, `P1_LCV01D`, `P2_SCO`, `P3_LCP01D`, `P3_LCV01D`) | T0855 Unauthorized Command Message | T0855, T0831 | Impair Process Control, Impact |
| `…PV` (the controlled variable's sensor is frozen or replayed to hide the effect) | unchanged | adds T0856 Spoof Reporting Message | adds Evasion |
| `-ST` (short-term, repeated several times) | unchanged | adds T0806 Brute Force I/O | — |
| Combination of two primitives | `SP` before `CO` (the setpoint is upstream of the output) | union of both | union |

**Result.** Primary T0836 for 32 events and T0855 for 18. Majority-class baseline:
0.64.

**Constant-answer baseline.** The share of events for which always giving one answer
counts as correct:

| T0836 | T0855 | T0831 | T0856 | T0806 |
|---|---|---|---|---|
| 0.64 | 0.64 | 0.64 | 0.38 | 0.20 |

Both the majority-class and the constant-answer baselines are reported next to every
top-1 result.

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
  Instead, the controller's output moves, so the inference says T0855.
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
| 5 | 65,535 coil writes to one address in 4 minutes | T0806 Brute Force I/O | T0806, T0855, T0814 | Impair Process Control, Inhibit Response Function |
| 6 | 21 coil writes to two addresses | T0855 Unauthorized Command Message | T0855, T0831 | Impair Process Control, Impact |

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

## 4. Output files

- `evaluation/ground_truth/mapping_site_a.json`: `{segment_id: {primary, acceptable, tactics, evidence}}`.
- `evaluation/ground_truth/mapping_site_b.json`: `{episode: {...}}` with the same fields.

Both carry the protocol version, the reviewer and the date of each pass, and are
committed together with `docs/preregistration.md` in the `g1-frozen` commit.

## 5. Scoring (for reference)

- Technique top-1: the first predicted technique is in the acceptable set.
- Technique top-3: any of the first three is in the acceptable set.
- Tactic accuracy: the predicted technique's tactics intersect the event's tactics.

## 6. Review log

| Pass | Date | Reviewer | Events changed | Notes |
|---|---|---|---|---|
| 1 | 14 Oct 2026 | Claude (AI assistant), at the author's request | Site A: all 50 (truth moved from data inference to the documented timetable); Site B: episodes 0 and 1 (T0814 added) | Timetable transcribed from the HAI technical details; 50/50 events matched; 7 errata recorded; P2 point types corrected; rule v1.0 |
| 2 | | Allaudin Ansari | | Verification (below) |

**Pass 2: what the author verifies.** The truth is now taken from the dataset's own
documentation, so the second pass checks the transcription and the rule. It is no
longer a re-labelling. For it to be a real check, do it without looking at the
assistant's notes:

1. Open `hai_dataset_technical_details.pdf`, pp. 36-38. For each of the 50 rows of
   `TIMETABLE` in `src/agmesh/evaluation/hai_attacks.py`, check the attack ID, start
   time, duration, controller code(s) and point(s).
2. Check the errata (`P2_LCV01D` → `P3_LCV01D`; A209's duration) and the two column
   aliases.
3. Decide whether you accept rule v1.0, in particular:
   - SP-led events accept only T0836, not T0855;
   - the frozen/replayed sensor adds T0856;
   - `-ST` adds T0806.
4. Check the seven Site B episodes, including T0814 on episodes 0 and 1.
5. Record the pass here and in `reviews` in both JSON files, then tell the assistant.
   It will set `"reviewed": true`, finalise `docs/preregistration.md`, and tag
   `g1-frozen`.
