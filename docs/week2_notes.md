# Week 2 Notes — Dataset Acquisition & Partitioning

## Datasets Used

### HAI 22.04 (nodes 01–03)
- Source: https://github.com/icsdataset/hai
- Download: git clone (no access request needed)
- Files: hai-train1.csv, hai-train2.csv, hai-test1.csv, hai-test2.csv, hai-test3.csv
- Sensor columns: ~59 columns per file (varies by HAI version)
- Attack label: native binary column 'attack' (0=normal, 1=attack)
- Attack rate: ~4–5% of total rows
- Subsystem split:
  - node_01 → first third of sensor columns  (Boiler)
  - node_02 → second third of sensor columns (Turbine)
  - node_03 → final third of sensor columns  (Water treatment)

### CIC Modbus 2023 (nodes 04–05)
- Source: https://www.unb.ca/cic/datasets/modbus-2023.html
- Download: manual CSV download (no account needed)
- Attack types: Reconnaissance, Query Flooding, False Data Injection, Brute Force Write, Baseline Replay
- All attack types map directly to MITRE ATT&CK for ICS techniques
- Split:
  - node_04 → attack-heavy traffic (recon, flood, injection)
  - node_05 → benign + remaining attacks (brute force, replay)

## Why Non-IID Partitioning Matters
Each node sees a genuinely different data distribution:
- Different subsystems (nodes 01–03) → different sensor ranges, failure modes
- Different attack type mixes (nodes 04–05) → different feature importance
This is exactly the heterogeneous condition FedProx is designed to handle better than FedAvg.

## PostgreSQL Schema
Three tables: node_registry, alert_records, fl_rounds.
alert_records.embedding uses pgvector(128) for future similarity search.
Schema is idempotent — safe to re-run init_db.py.

## Files Created This Week
- scripts/check_datasets.py   — Day 1 integrity check
- scripts/process_datasets.py — Day 2 cleaning pipeline
- scripts/partition_data.py   — Day 3 Non-IID partitioning
- scripts/init_db.py          — Day 4 PostgreSQL schema
- scripts/seed_registry.py    — Day 5 node registry seeding
- tests/unit/test_week2_data.py — 12-test gate
- notebooks/01_data_exploration.py