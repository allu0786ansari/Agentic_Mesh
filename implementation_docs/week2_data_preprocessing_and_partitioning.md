# Week 2 Implementation Notes: Data Preprocessing and Partitioning

Author: Allaudin Ansari  
Project: Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics  
Runtime: Python 3.12

## Purpose

Week 2 prepares the raw HAI and CIC Modbus datasets for edge-node training. The output is five Non-IID node partitions:

- `node_01`, `node_02`, `node_03`: HAI time-series partitions for VAE models.
- `node_04`, `node_05`: CIC Modbus tabular partitions for Isolation Forest models.

Each node receives:

- `data/partitions/<node_id>/train.parquet`
- `data/partitions/<node_id>/test.parquet`
- `data/partitions/<node_id>/meta.json`

## Documentation Standard

For every new or updated file, document:

- File purpose
- Why it exists
- Main functions/classes
- Inputs and outputs
- Important design decisions
- How to run or test it
- Interview explanation

This keeps the codebase readable for implementation, technical review, and interview preparation.

## `scripts/check_datasets.py`

This is the raw dataset preflight checker.

Main responsibilities:

- Finds the configured HAI dataset folder.
- Detects Git LFS pointer files.
- Confirms HAI CSV files have timestamp and attack-label information.
- Confirms Modbus data is available either as CSV files or PCAP folders.
- Prints a Rich pass/fail table.

Interview explanation:

This file prevents expensive preprocessing from starting on missing or invalid data. It is an early failure gate.

## `scripts/process_dataset.py`

This script converts raw datasets into normalized processed files under `data/processed/`.

### HAI Processing

Function: `process_hai()`

Input:

- `data/raw/hai/<HAI_VERSION>/*.csv`

Output:

- `data/processed/hai_normal.parquet`
- `data/processed/hai_attacks.parquet`
- `data/processed/hai_combined.parquet`
- `data/processed/hai_scaler_meta.json`

Implementation behavior:

- Loads all HAI train and test CSV files.
- Normalizes column names to lowercase.
- Parses and sorts by timestamp.
- Forward-fills and backfills missing sensor values.
- Creates a binary `attack` column when needed.
- Fits `MinMaxScaler` only on normal rows to avoid attack-data leakage.
- Saves normal, attack, and combined parquet outputs.

### Modbus Processing

Function: `process_modbus()`

Input:

- `data/raw/modbus2023/attack/**/*.pcap`
- `data/raw/modbus2023/benign/**/*.pcap`

Output:

- `data/processed/modbus2023_raw_shards/*.parquet`
- `data/processed/modbus2023_clean_shards/*.parquet`
- `data/processed/modbus2023_combined.parquet`
- `data/processed/modbus2023_scaler_meta.json`

Industrial-memory behavior:

- Uses streaming `tshark` via `subprocess.Popen`.
- Does not capture full `tshark` output in memory.
- Writes PCAP rows in bounded parquet shards.
- Computes cleaning statistics from shards.
- Cleans and normalizes each shard independently.
- Combines cleaned shards into the expected `modbus2023_combined.parquet` file without pandas concat.

Important environment knobs:

```powershell
$env:MODBUS_PARSE_BATCH_ROWS="50000"
$env:MODBUS_STATS_SAMPLE_ROWS="250000"
```

Lower `MODBUS_PARSE_BATCH_ROWS` if the machine is memory constrained.

## `scripts/partition_data.py`

This script creates five edge-node partitions from processed parquet files.

### HAI Partitioning

Function: `partition_hai()`

Output nodes:

- `node_01`: HAI `p1_*` subsystem sensors.
- `node_02`: HAI `p2_*` subsystem sensors.
- `node_03`: HAI `p3_*` and `p4_*` subsystem sensors.

Implementation behavior:

- Reads `data/processed/hai_combined.parquet`.
- Keeps physical subsystem boundaries intact using sensor prefixes.
- Builds sliding windows with:
  - `WINDOW_SIZE = 60`
  - `WINDOW_STRIDE = 10`
- Flattens each window into `feature_0 ... feature_N`.
- Labels a window as attack if any row inside the window has `attack == 1`.
- Uses stratified train/test split so both splits retain attack examples.
- Writes metadata including `subsystem`, `feature_count`, `flat_feature_dim`, `sensor_columns`, and class distribution.

Current verified HAI layout:

- `node_01`: `p1`, 44 sensors, 2640 flattened features.
- `node_02`: `p2`, 24 sensors, 1440 flattened features.
- `node_03`: `p3/p4`, 18 sensors, 1080 flattened features.

### Modbus Partitioning

Function: `partition_modbus()`

Output nodes:

- `node_04`: attack-heavy tabular traffic.
- `node_05`: benign-heavy mixed tabular traffic.

Implementation behavior:

- Streams from `data/processed/modbus2023_clean_shards/*.parquet` when available.
- Falls back to `modbus2023_combined.parquet` row groups if shards are unavailable.
- Does not load the full 39M-row Modbus dataset into pandas.
- Excludes non-model fields:
  - `label`
  - `_source_file`
  - `node`
  - `ip.src`
  - `ip.dst`
- Uses deterministic hashing to route every row exactly once:
  - Most attack rows go to `node_04`.
  - Most benign rows go to `node_05`.
- Uses deterministic hashing again for train/test split.
- Writes append-only parquet output with `pyarrow.parquet.ParquetWriter`.

Current verified Modbus layout:

- `node_04`: 7 features, attack-heavy compared with node 05.
- `node_05`: 7 features, benign-heavy mixed traffic.

## `tests/unit/test_week2_data.py`

This is the Week 2 validation gate.

It checks:

- Processed HAI and Modbus parquet files exist.
- All five node partition directories contain `train.parquet`, `test.parquet`, and `meta.json`.
- VAE nodes are correctly typed as time-series.
- Isolation Forest nodes are correctly typed as tabular.
- HAI subsystem prefixes are not mixed across VAE nodes.
- All partition parquet files are readable.
- PostgreSQL registry contains five nodes when PostgreSQL is reachable.

Recommended command on this Windows workspace:

```powershell
agvenv\Scripts\python.exe -m pytest tests\unit\test_week2_data.py -v --tb=short -p no:cacheprovider
```

The `-p no:cacheprovider` flag avoids failures from the locked `.pytest_cache` directory.

## `scripts/init_db.py`

This is the PostgreSQL schema migration script.

Main responsibilities:

- Connects using `POSTGRES_DSN`.
- Enables `pgcrypto` for `gen_random_uuid()`.
- Enables `pgvector`.
- Creates `node_registry`.
- Creates `alert_records`.
- Creates `fl_rounds`.
- Adds useful indexes for alerts and FL rounds.
- Uses `DOUBLE PRECISION` for metric and privacy-budget values.
- Reports schema or extension failures separately from connection failures.

Interview explanation:

This file creates the structured persistence layer for node metadata, anomaly alerts, and federated training history.

## `scripts/seed_registry.py`

This is the PostgreSQL node registry seeding script.

Main responsibilities:

- Read each `data/partitions/node_*/meta.json`.
- Upsert one row per node into `node_registry`.
- Validate required metadata fields.
- Preserve model type, data type, feature count, status, and epsilon budget metadata.
- Set node status to `ready`.
- Update `last_seen` during each seed run.

Interview explanation:

This script connects generated dataset partitions to PostgreSQL so services can discover node capabilities from the registry.

## `scripts/verify_messaging.py`

This is the Kafka and NATS JetStream integration smoke test.

Main responsibilities:

- Sends and receives test messages on Kafka topic `telemetry.raw`.
- Sends and receives test messages through NATS JetStream.
- Confirms both messaging systems can round-trip JSON events.

Interview explanation:

Kafka is used for raw telemetry ingestion, while NATS JetStream is used for lightweight event and action dispatch. This script proves both messaging paths work before higher-level services depend on them.

## Validation Status

Latest validation result:

```text
22 passed
```

The PostgreSQL validation test now loads `.env`, so it uses the configured `POSTGRES_DSN`.

## Design Notes

- HAI partitioning is prefix-based because those prefixes represent physical plant subsystems.
- Modbus partitioning is streaming because the processed dataset contains about 39.7 million rows.
- The project keeps `modbus2023_combined.parquet` for compatibility, but the scalable path uses cleaned shards.
- No raw data leaves the preprocessing/partitioning stage; these files are local development artifacts for Week 2.
