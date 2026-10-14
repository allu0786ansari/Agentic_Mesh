# Attribution Ground-Truth Mapping Protocol

**Status:** Draft for Gate G1. The mappings are built and reviewed by the author; this
document fixes how. Changing it after `g1-frozen` requires a new ADR and invalidates
attribution results produced under the old version.
**Implements:** Proposal v3.1 §10.4; ADR 0002 decision 6.

## 1. Purpose and rules

The mapping says, for every attack event, which MITRE ATT&CK for ICS technique(s) and
tactic(s) a correct attribution should name. It is the ground truth for RQ3.

1. The mapping is written **before any attribution output exists** and frozen at G1.
2. It uses only the dataset's labels and the *names* of variables or protocol
   features that deviate during an event, never a model score, retrieval result or
   LLM output.
3. Every event gets a primary technique, an acceptable set (techniques that a
   reasonable analyst could also defend), and the tactic(s) of those techniques.
4. Two review passes are made on different days. The second pass is done without
   looking at the first; disagreements are resolved in writing and logged in §6.
5. Technique IDs and names are checked against the pinned ATT&CK for ICS STIX bundle
   in Week 9; any ID that does not exist there is a protocol error.

## 2. Site A (HAI 21.03): 50 events

**Evidence per event.** For each event segment the mapping tool (Week 4,
`python -m agmesh.evaluation.ground_truth site_a`, to be built once this protocol is
accepted) lists:

- the targeted processes (`attack_P1`–`attack_P3` over the event rows);
- the variables whose values leave their normal range during the event, grouped by
  variable type from the HAI naming convention: setpoints, actuator commands
  (`…D`), actuator positions (`…Z`), and sensor readings (transmitters `…FT`, `…LIT`,
  `…PIT`, `…TIT` and similar);
- whether a training-constant variable changed.

The tool reads the test events' label columns and variable values. This is a
labelling activity, done once, and logged in `results/test_access.log` with the
configuration `ground_truth_site_a`; no model, threshold or hyperparameter is chosen
from it.

**Rule (first pass).**

| Observed during the event | Primary technique | Acceptable set | Tactic |
|---|---|---|---|
| A setpoint changes and the process follows | T0836 Modify Parameter | T0836, T0855 | Impair Process Control |
| An actuator command changes without a matching setpoint change | T0855 Unauthorized Command Message | T0855, T0831 | Impair Process Control, Impact |
| A sensor reading is changed while the physical process does not follow | T0856 Spoof Reporting Message | T0856, T0832 | Evasion, Impair Process Control, Impact |
| Combination of the above | the technique of the variable that changes first | union of the rows that apply | union |

**Expected difficulty.** Almost all events will fall under *Impair Process Control*,
so tactic accuracy on Site A is near-degenerate; the technique-level primary is the
informative target. The majority-class baseline is reported for both levels.

## 3. Site B (CIC Modbus 2023): 7 episodes (case study)

Episodes are those of `docs/labeling_protocol.md` §4. Candidate mapping for the first
pass:

| Episode | Observed behaviour | Primary technique | Acceptable set | Tactic |
|---|---|---|---|---|
| 0 | Two reads, one asking for 65,535 values | T0888 Remote System Information Discovery | T0888, T0861, T0846 | Discovery, Collection |
| 1 | About 20 input-register reads; a stacked 23-frame packet | T0861 Point & Tag Identification | T0861, T0888 | Collection, Discovery |
| 2 | Five reads across all four read function codes | T0861 Point & Tag Identification | T0861, T0888 | Collection, Discovery |
| 3 | Two coil reads | T0861 Point & Tag Identification | T0861, T0801 | Collection |
| 4 | Slow scan, about one discrete-input read a minute for 20 minutes | T0801 Monitor Process State | T0801, T0861 | Collection |
| 5 | 65,535 coil writes to one address in 4 minutes | T0806 Brute Force I/O | T0806, T0855, T0814 | Impair Process Control, Inhibit Response Function |
| 6 | 21 coil writes to two addresses | T0855 Unauthorized Command Message | T0855, T0831 | Impair Process Control, Impact |

Site B is reported as a case study (validity, grounding, chosen technique per episode),
not as an accuracy estimate.

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
| 1 | | | | |
| 2 | | | | |
