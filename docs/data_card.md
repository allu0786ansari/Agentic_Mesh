# Data Card

**Status:** v3.1. Both datasets were analysed in full before Week 2 (Proposal v3.1,
Appendix B). The measured facts below are also stored as machine-checkable
expectations in `configs/data_expectations.yaml`. On any machine,

```bash
uv run python scripts/inspect_data.py --expectations configs/data_expectations.yaml
```

checks the local copy against them (21 checks) and writes
`results/feasibility/data_inspection.json`. SHA-256 hashes are recorded in
`data/raw_checksums.json` by `scripts/verify_checksums.py`.

Raw data is never committed to Git. It is tracked by DVC (`data/raw.dvc`), and its
SHA-256 manifest is committed so anyone can verify they have identical files.

## Local layout

```
data/raw/
├── hai/
│   └── hai-21.03/
│       ├── train1.csv.gz … train3.csv.gz
│       └── test1.csv.gz  … test5.csv.gz
└── modbus2023/
    ├── benign/{ied1a,ied1b,scada-hmi}/*.pcap
    └── attack/{ied1a,ied1b,scada-hmi}/*.pcap
```

Captures may be `.pcap`, `.pcapng` or `.cap`; the reader handles all three.

## HAI 21.03 (Site A)

| Field | Value |
|---|---|
| Source | https://github.com/icsdataset/hai |
| Release | 21.03. HAI 22.04 and 23.05 could not be downloaded (the upstream Git LFS budget was exhausted, leaving pointer files). Accepted at Gate G0 (ADR 0001, decision 1) |
| Licence and citation | *(to confirm from the upstream repository before publication)* |
| Sampling | 1 Hz |
| Files | 3 training files (normal operation, 921,603 rows), 5 test files (402,005 rows) |
| Recording periods | `train1` 11–13 Jul 2020 (60 h), `train2` 31 Jul–3 Aug (63 h), `train3` 4–10 Aug (133 h). Test files fall between and before them (7 Jul–31 Jul) |
| Columns | `time`, 79 sensor and actuator features (P1 38, P2 22, P3 7, P4 12), and labels `attack`, `attack_P1`, `attack_P2`, `attack_P3` |
| Feature types in training | 55 continuous, 2 binary, 22 constant |
| Modelled features | 56: the 57 that vary in training, minus `P4_HT_PS` (an exact copy of `P4_ST_PS`) |
| Label | `attack` (the process columns mark targeted subsystems and are shorter at event edges) |
| Attack events in test files | 50 (8,947 attack rows). Per file: test1 5, test2 20, test3 8, test4 5, test5 12 |
| Event durations | Median 162 s, range 17–422 s; 3 events shorter than 60 s |
| Event targets | P1 23, P2 11, P1+P2 8, P1+P3 3, P2+P3 3, P3 2 |
| Gaps between events | Smallest 24 min, median 105 min |
| Operating regimes | Not separable. Only turbine demand (`P4_ST_PS` at 0 or 50, on 8–22% of each file) is a real regime; `P1_PCV02Z` is a calibration offset that separates `train1` from `train2` and `train3` |
| Drift between periods | 10 features have KS > 0.2 between `train1` and `train3`; turbine speed sensors shift by about 1.35 SD |
| Use in v3.1 | Site A process fleet, partitioned by recording period: a1 = `train1`, a2 = `train2`, a3 and a4 = the halves of `train3` (about 3,000–3,400 training windows each) |

## CIC Modbus Dataset 2023 (Site B)

| Field | Value |
|---|---|
| Source | https://www.unb.ca/cic/datasets/modbus-2023.html |
| Licence and citation | Citation required by the Canadian Institute for Cybersecurity *(add the exact citation)* |
| Obtained captures | Attack: one capture per device (`ied1a`, `ied1b`, `scada-hmi`). Benign: `ied1a` 8, `ied1b` 7, `scada-hmi` 19 |
| Hosts | SCADA HMI 185.175.0.3, IED1A 185.175.0.4 (secure IED), IED1B 185.175.0.5 (insecure IED), central agent 185.175.0.6, attacker 185.175.0.7, IED4C 185.175.0.8 (polled by the HMI, not captured), Docker gateway 185.175.0.1 |
| Benign period | 22–31 January 2023; about 194–228 hours of coverage per device. Ends about 11 hours before the attack day |
| Attack day | 1 February 2023 from 03:48:39 UTC, about 10 hours per device |
| Capture overlap | The three devices' captures overlap in time on every recorded day |
| Attack logs | **Not in the obtained data.** Labels are derived from the attacker's address, window by window |
| Attacker position | External attacker only (no compromised-IED or compromised-HMI captures) |
| Attacker traffic | 538,648 packets, all to or from IED1A. IED1B and the HMI saw none |
| Attack episodes | 7 on IED1A: four short reconnaissance probes (03:51–04:09), a 20-minute slow read scan (04:17–04:37), a 4-minute flood of 65,535 coil writes to one address (13:03–13:08), and a write burst to two addresses (13:21) |
| Attack windows (10 s) | IED1A: 51 of 3,681 (1.4%) with attacker IPv4 traffic. A further 29 windows contain only the attacker's ARP frames and are not attack windows |
| Signatures seen | Requested quantities up to 65,535; addresses never used in training; a new peer; two malformed frames (a 23-frame stack and an MBAP length mismatch) |
| Normal behaviour to model | HMI polls each IED about every 22 ms; occasional operator write bursts; the secure IEDs (IED1A, IED4C) stall for tens of seconds most days, often together, while IED1B never stalls; idle gaps up to 16 hours on IED1A on 22 January |
| MITRE ATT&CK mapping shipped with the dataset | None |
| Use in v3.1 | Site B substation: three nodes, one feature schema computed from each node's own side of the traffic. Training on benign days; attack day for attack validation (episodes 0–3) and test (episodes 4–6) |

## Partitions (Week 2)

Built by `dvc repro` (stages `prepare` and `partition`) and checked by
`make check-data`; labels follow `docs/labeling_protocol.md`. Counts are windows
(Site A 60 s, Site B 10 s); attack windows in brackets.

| Node | Data | train | val_normal | val_attack | test |
|---|---|---|---|---|---|
| a1 | `train1` | 3,060 | 539 | 320 (9) | 1,015 (31) |
| a2 | `train2` | 3,213 | 566 | 326 (20) | 1,023 (39) |
| a3 | `train3`, first half | 3,391 | 597 | 623 (17) | 1,457 (32) |
| a4 | `train3`, second half | 3,391 | 597 | 303 (13) | 1,561 (34) |
| b1 | IED1A | 64,792 | 11,433 | 147 (12) | 3,532 (39) |
| b2 | IED1B | 60,596 | 10,692 | 147 (0) | 3,531 (0) |
| b3 | SCADA HMI | 59,419 | 10,485 | 147 (0) | 3,532 (0) |

- Site A test events are split 15 validation / 35 test, by process group, across all
  four clients (each client gets 3–4 validation and 8–9 test events).
- Site B: idle stretches removed before splitting: 5,957 windows on IED1A (mostly the
  16-hour gap on 22 January), 141 on IED1B, 138 on the HMI; none on the attack day.
- Site B features: 18 per window (`site_b.FEATURES`), computed from each node's own
  side of the traffic. Normal traffic always requests quantity 1 and uses 8 addresses
  per IED (24 for the HMI); the attacker used unseen addresses and quantities up to
  65,535.

## Privacy budget implications

| Site | Training windows per client | σ at ε = 1 / 3 / 10 (batch 256, 50 rounds, 1 local epoch, RDP) |
|---|---|---|
| A | 3,059–3,391 | 8.4 / 3.2 / 1.3 |
| B | about 59,000–65,000 | 2.1 / 1.0 / 0.63 |

## Known limitations

- Both datasets come from testbeds, not production plants.
- HAI and CIC Modbus were recorded separately, so no event links the two sites.
- Site B has one attacker position and one attack day. Its labels are derived from the
  attacker's address, and its seven episodes are too few for accuracy estimates; Site B
  detection is reported per episode and its attribution as a case study.
- Only the external attack is real multi-device data, and it reaches one device. The
  multi-device fusion benefit is shown on a labelled synthetic scenario only.
