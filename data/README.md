# Dataset workspace

Raw datasets are stored locally under `data/raw/` and are intentionally excluded
from Git because the HAI and CIC Modbus archives are large. Keep the acquisition
source, release, checksum, and license/citation information in this file or in a
tracked manifest before running preprocessing.

## Project allocation

The project uses six logical edge nodes:

| Node | Raw source | Model | Training data | Evaluation data |
|---|---|---|---|---|
| `node_01` | HAI 21.03 process P1 | VAE | HAI `train*.csv.gz` | HAI `test*.csv.gz` |
| `node_02` | HAI 21.03 process P2 | VAE | HAI `train*.csv.gz` | HAI `test*.csv.gz` |
| `node_03` | HAI 21.03 process P3 | VAE | HAI `train*.csv.gz` | HAI `test*.csv.gz` |
| `node_04` | CIC Modbus IED1A | Isolation Forest | benign PCAP capture files | held-out benign plus attack PCAP windows |
| `node_05` | CIC Modbus IED1B | Isolation Forest | benign PCAP capture files | held-out benign plus attack PCAP windows |
| `node_06` | CIC Modbus SCADA HMI | Isolation Forest | benign PCAP capture files | held-out benign plus attack PCAP windows |

Raw files stay under `data/raw/`. The preprocessing command writes model-ready
artifacts under `data/partitions/node_XX/` and never copies raw data into Git.

## HAI

- Official source: https://github.com/icsdataset/hai
- Release currently available locally: HAI 21.03
- Local path: `data/raw/hai/hai-21.03/`
- Complete files: `train1.csv.gz`, `train2.csv.gz`, `train3.csv.gz`, and `test1.csv.gz` through `test5.csv.gz`
- Validation: `pandas.read_csv(..., compression="gzip")` successfully reads the files.
- Note: HAI 22.04 and HAI 23.05 are represented by Git LFS pointers in the upstream clone because that repository's LFS budget is exhausted. They must not be treated as downloaded data.

## CIC Modbus 2023

- Official source: https://www.unb.ca/cic/datasets/modbus-2023.html
- Download endpoint: https://cicresearch.ca/CICDataset/CICModbusDataset2023/
- Status: downloaded and inspected locally.
- Required local path: `data/raw/modbus2023/`
- Contents: 34 benign PCAPs and 3 attack PCAPs, separated by `ied1a`, `ied1b`, and `scada-hmi`.
- No CSV attack logs are present in the downloaded archive. Labels are therefore capture-level: benign windows are `0`, attack-capture windows are `1`.
- PCAP epoch timestamps are normalized to timezone-aware UTC timestamps during extraction. The source documentation records capture/log timezone considerations; no attack-log alignment is claimed because no attack logs are present locally.
- TShark is required because the repository stores PCAP, not tabular network-flow features. Wireshark/TShark is installed at `C:\Program Files\Wireshark\tshark.exe` on the development machine.

## Preparing the real Week 2 data

Run from the repository root with the project virtual environment active:

```powershell
python -m scripts.prepare_data `
	--hai-root data/raw/hai/hai-21.03 `
	--modbus-root data/raw/modbus2023 `
	--output data/partitions `
	--database agmesh.db `
	--tshark "C:\Program Files\Wireshark\tshark.exe"
```

The command:

1. Reads HAI train/test gzip CSVs and selects P1, P2, and P3 process columns.
2. Removes attack columns from model features, forward-fills short gaps, and fits min/max scaling on normal HAI training rows only.
3. Preserves timestamps and labels for later temporal windowing and evaluation.
4. Extracts Modbus packet fields with TShark and aggregates them into fixed 10-second windows.
5. Uses only benign capture files for Isolation Forest training.
6. Holds out benign captures and includes all attack captures in the Modbus evaluation partition.
7. Writes `train.parquet`, `test.parquet`, and `meta.json` for all six nodes.
8. Seeds six records in `agmesh.db`.

The Modbus split is by complete capture file, never by randomly splitting packets.
This prevents traffic from the same capture appearing in both training and testing.
Because attack logs are absent, the paper must describe Modbus results as capture-level
benign-versus-attack evaluation rather than exact attack-type or packet-level attribution.

## Week 2 generated results

The real-data preparation completed successfully and generated six partitions under
`data/partitions/`. The files were written as Parquet with `train.parquet`,
`test.parquet`, and `meta.json` for every node. The SQLite registry was seeded in
`agmesh.db` with one record per node.

| Node | Source | Features | Train rows | Test rows | Test labels |
|---|---|---:|---:|---:|---|
| `node_01` | HAI 21.03 P1 | 38 | 921,603 | 402,005 | 393,058 normal / 8,947 attack |
| `node_02` | HAI 21.03 P2 | 22 | 921,603 | 402,005 | 393,058 normal / 8,947 attack |
| `node_03` | HAI 21.03 P3 | 7 | 921,603 | 402,005 | 393,058 normal / 8,947 attack |
| `node_04` | CIC Modbus IED1A | 17 | 55,344 | 24,679 | 20,998 benign / 3,681 attack |
| `node_05` | CIC Modbus IED1B | 17 | 59,106 | 15,843 | 12,162 benign / 3,681 attack |
| `node_06` | CIC Modbus SCADA HMI | 17 | 54,703 | 18,834 | 15,153 benign / 3,681 attack |

### Result interpretation

- HAI training data contains normal rows only, which is appropriate for unsupervised VAE training.
- HAI test data contains normal and labeled attack rows for anomaly evaluation.
- Modbus training data contains benign windows only, which is appropriate for Isolation Forest fitting.
- Modbus test data combines held-out benign windows with all available attack-capture windows.
- The Modbus attack label is capture-level because the downloaded archive contains PCAPs but no attack-log CSV files.
- The SQLite `node_registry` contains six rows with three `vae/timeseries` nodes and three `isolation_forest/tabular` nodes.

### Validation evidence

The complete Week 1 and Week 2 automated suite passed after real-data generation:

```text
8 passed in 2.71s
```

The generated Parquet files were read with pandas and their row counts, feature
counts, labels, and SQLite registry values matched the metadata records.

## HAI 21.03 checksums

```text
test1.csv.gz 532DE41BED372206941609D05B763958F59ABECFD63C8C123649F823452E445E
test2.csv.gz C5572E0ED19906107CEB56C4B4F40E165153FC7BA87B3506230D53D2478C29BB
test3.csv.gz BFF835894BAFF0C73871BC01ECE4F5CF759D455A049E808B012128767420CAAF
test4.csv.gz 51380FD28FD985B94A10CAE3D8B80926BECB3AF9F233F2D3A38AB8BE9482A482
test5.csv.gz F0D59308B167ED95D0E81FB3E27C906682C9B21D3DBA0AB4C4D696565DF35179
train1.csv.gz FDD0162315B32817BA665E2CC8990EB09C47C4EDC29683A86B11DC9DF062B3A9
train2.csv.gz C330AF06A24F5FE3DF864DDDC0E9CBBBFD27DB2E005BD32DC8AB3EB2AB356337
train3.csv.gz DE09120C617BAD5F4F3CAB86705137FE7CED4B1123FB67FFDF67FD645F3D1B8F
```