# Labelling Protocol

**Status:** v1.0, Week 2. Implemented in `src/agmesh/data/site_a.py` and
`src/agmesh/data/site_b.py`; checked by `make check-data` against
`configs/partition_expectations.yaml`.
**Decisions it implements:** ADR 0001 decisions 2 and 4; Proposal v3.1 §6.1 and §6.2.

The labels decide every detection number the project reports, so they are fixed in
writing before any model is trained. Changing this protocol after Gate G1 requires a
new ADR and invalidates every result produced under the old version.

## 1. Site A — HAI 21.03

**Source.** The dataset's own `attack` column (HAI-1). It is 1 for every second of a
labelled attack and 0 otherwise.

**Process columns.** `attack_P1`, `attack_P2` and `attack_P3` record which process an
event targeted. They are used only to stratify the validation/test split and, in
Week 4, for tactic-level ground truth. They are never model inputs
(check `site_a.no_label_column_is_a_feature`). They are slightly shorter than `attack`
at event edges, which is why `attack` is the label.

**Events.** An event is a maximal run of consecutive rows with `attack = 1`. HAI 21.03
has 50 events in its five test files.

**Windows.** A 60-second window is an attack window if any of its rows has
`attack = 1`.

**Training data.** The three training files contain no labelled rows. Any that did
would be removed before splitting and counted in the manifest
(`load.train_attack_rows_dropped`).

## 2. Site B — CIC Modbus 2023

**Source.** The attacker's IP address, `data.modbus.attacker_ip` = 185.175.0.7
(G0 decision 4). The dataset's attack logs are not part of the obtained data. These
labels are therefore **derived** and are reported as such. The rule is valid only for
the external-attacker scenario, where the attacker is a host that never appears in
normal traffic; it would not work for a compromised IED or HMI.

**Window grid.** 10-second windows on a fixed epoch grid (window *k* covers
[10*k*, 10*k* + 10) seconds UTC), so windows on different nodes line up. A window
exists only if it lies fully inside one capture.

**Window label.** A window is an attack window **on a node** if that node's capture
holds at least one IPv4 packet to or from the attacker in that window. ARP frames are
not counted: they carry no Modbus content. Labels are per node; the same window can be
an attack window on IED1A and normal on IED1B.

**Idle stretches.** Windows that touch a stretch of more than `idle_gap_s` (60 s) with
no HMI request to or from the node are removed before splitting (MB-7), and counted in
the manifest. An attack window removed this way would be a labelling failure, so its
count must be zero (`site_b.attack_windows_dropped_as_idle`).

**Episodes.** Attacker packet times from all attack-day captures are sorted and split
at gaps longer than `episode_gap_s` = 125 s. The value comes from the attack day
itself: gaps inside one probe reach 121 s (the slow read scan sends one request a
minute) and gaps between probes are at least 133 s. An episode is the Site B
*attack event* for event-level metrics.

**Validation and test.** The first `attack_val_episodes` = 4 episodes, with the normal
attack-day windows around them, are attack validation; the remaining 3 episodes are
test. The boundary is the midpoint between episode 3 and episode 4, with one gap
window. The same boundary applies to every node.

## 3. Expected counts on the analysed data

| Item | Expected | Check |
|---|---|---|
| Site A modelled features | 56 (22 training-constant excluded, `P4_HT_PS` dropped) | `site_a.n_features` |
| Site A events | 50: 15 attack validation, 35 test | `site_a.events_*` |
| Site B episodes | 7: 4 attack validation, 3 test | `site_b.episodes`, `site_b.val_attack_episodes` |
| IED1A attack windows | 51 (12 in attack validation, 39 in test) | `site_b.ied1a.attack_windows` |
| IED1B and HMI attack windows | 0 | `site_b.ied1b.attack_windows`, `site_b.scada-hmi.attack_windows` |
| Attack windows lost to the idle cut or the split gap | 0 | `site_b.attack_windows_dropped_as_idle`, `site_b.*.no_attack_window_lost_at_boundary` |

## 4. Episodes on the attack day (1 February 2023, UTC)

| Episode | Start | End | What the attacker did | Partition |
|---|---|---|---|---|
| 0 | 03:51:37 | 03:52:46 | Two reads, one with quantity 65,535 | Attack validation |
| 1 | 03:58:25 | 04:01:29 | About 20 input-register reads; one stacked 23-frame packet; an MBAP length mismatch | Attack validation |
| 2 | 04:04:19 | 04:05:58 | Five reads across all four read function codes | Attack validation |
| 3 | 04:08:17 | 04:09:17 | Two coil reads | Attack validation |
| 4 | 04:17:05 | 04:37:34 | Slow scan: about one discrete-input read a minute for 20 minutes | Test |
| 5 | 13:03:42 | 13:08:01 | Flood of 65,535 coil writes to one address; a truncated frame | Test |
| 6 | 13:21:52 | 13:22:13 | 21 coil writes to two addresses | Test |

End times are the last IPv4 packet; the analysis report, which also counted ARP
frames, shows slightly later ends for some episodes.

## 5. Manual spot-check

Before Gate G1, check by hand, with Wireshark or `tcpdump -r` on the IED1A attack
capture, that:

1. the first and last attack window of every episode contain attacker packets, and the
   windows just outside them do not;
2. no IED1B or HMI window on the attack day contains the attacker's address;
3. a sample of 20 IED1A attack-day windows labelled normal contains no attacker
   packets.

Record the result, with the date and the capture's SHA-256, in
`docs/spot_check_week2.md`.
