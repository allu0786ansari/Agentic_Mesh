# Data Card

**Status:** Week 1. Facts marked *(to confirm)* are filled in from
`results/feasibility/data_inspection.json` and `data/raw_checksums.json` after the
Week 1 runs on the development laptop.

Raw data is never committed to Git. It is tracked by DVC (`data/raw.dvc`), and its
SHA-256 manifest (`data/raw_checksums.json`) is committed so anyone can verify they
have identical files.

## Expected local layout

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

## HAI (Site A)

| Field | Value |
|---|---|
| Source | https://github.com/icsdataset/hai |
| Release | 21.03 — the release available locally. HAI 22.04 and 23.05 could not be downloaded (the upstream repository's Git LFS budget was exhausted, leaving pointer files). Confirmed at Gate G0. |
| Licence and citation | *(to confirm from the upstream repository)* |
| Sampling | 1 Hz *(to confirm)* |
| Files | 3 training files (normal operation), 5 test files (labelled attacks) *(to confirm)* |
| Label columns | `attack`, `attack_P1`, `attack_P2`, `attack_P3` (from the v2.1 pipeline) *(to confirm)* |
| Features | *(count and list from the inspection)* |
| Attack events in test files | *(from the inspection)* |
| Use in v3 | Site A process fleet: four nodes, all with the same features, partitioned by operating regime (Proposal §4.1) |

## CIC Modbus Dataset 2023 (Site B)

| Field | Value |
|---|---|
| Source | https://www.unb.ca/cic/datasets/modbus-2023.html |
| Licence and citation | Citation required by the Canadian Institute for Cybersecurity *(add the exact citation)* |
| Contents (from the v2.1 notes) | 34 benign and 3 attack captures, split by `ied1a`, `ied1b` and `scada-hmi` *(to confirm)* |
| Attack logs | **None found in the v2.1 download.** Labels are capture-level only (benign capture = 0, attack capture = 1) unless the inspection finds log files. |
| Capture time overlap between devices | *(from the inspection; decides how RQ2 fusion is evaluated)* |
| MITRE ATT&CK mapping shipped with the dataset | *(from the inspection; decides the Site B attribution ground truth)* |
| Use in v3 | Site B substation: three device nodes with a shared flow-window feature schema (Proposal §4.1) |

## Known limitations

- Both datasets come from testbeds, not production plants.
- HAI and CIC Modbus were recorded separately, so no event links the two sites.
- If no attack logs exist for CIC Modbus, Site B event-level metrics and technique-level attribution need a written labelling and mapping protocol (Weeks 2 and 4), and some claims may have to be limited to capture level. Gate G0 records this decision.
