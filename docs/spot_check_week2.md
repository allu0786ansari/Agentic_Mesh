# Week 2 Label Spot-Check

**Date:** 14 October 2026
**Protocol:** `docs/labeling_protocol.md` §5
**Method:** the Site B attack-day labels produced by the pipeline (`agmesh.data.site_b`)
were compared with an independent parse of the same captures made during the data
analysis with `dpkt` (a different decoder, written separately). Windows are 10 s on the
epoch grid.

**Captures checked (SHA-256, matching `data/raw_checksums.json`)**

| Capture | SHA-256 |
|---|---|
| `attack/ied1a/veth4edc015-0.pcap` | `919e99ca82b07a19f5c6ff4339a99aedf6fc04cebe98bdb9b11388651b0b27b5` |
| `attack/ied1b/vethd9e14c0-0.pcap` | `170fd8d3f6f3f10b875bc0349cbe65b6b2f88285893a0409a834e1fe45113b20` |
| `attack/scada-hmi/veth5bbeaa2-0.pcap` | `3392675e6af725865d2ec0318eeaeb0183b8e01011ee839d44d4649739a53b24` |

## Results

| Check | Result |
|---|---|
| Attack windows on IED1A, pipeline vs independent parse | 51 vs 51; the two sets of windows are identical |
| Attack windows on IED1B and the HMI | 0 in both parses |
| Attacker windows removed by the idle cut | 0 |
| Windows holding only the attacker's ARP frames (not attack windows by protocol) | 29 on IED1A, none elsewhere |
| 20 randomly sampled IED1A attack-day windows labelled normal (seed 2026) | none contain attacker IPv4 packets |

**Episode boundaries on IED1A (UTC, 1 February 2023)**

| Episode | First window | Window after the last | Attack windows | First and last window contain attacker packets | Neighbouring windows are clean |
|---|---|---|---|---|---|
| 0 | 03:51:30 | 03:52:50 | 2 | yes | yes |
| 1 | 03:58:20 | 04:01:30 | 3 | yes | yes |
| 2 | 04:04:10 | 04:06:00 | 5 | yes | yes |
| 3 | 04:08:10 | 04:09:20 | 2 | yes | yes |
| 4 | 04:17:00 | 04:37:40 | 21 | yes | yes |
| 5 | 13:03:40 | 13:08:10 | 16 | yes | yes |
| 6 | 13:21:50 | 13:22:20 | 2 | yes | yes |

Episodes 1 and 4 span more time than their attack-window counts suggest because the
attacker sent single requests minutes apart; the windows between them are normal.

**Conclusion:** the Site B labels follow the protocol exactly. No correction is needed.
