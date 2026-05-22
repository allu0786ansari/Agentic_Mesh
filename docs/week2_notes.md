# Week 2 Notes: Data Handling, Preprocessing, Partitioning

Project: Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics  
Purpose: Interview and implementation reference for Week 2 engineering work.

## Week 2 Goal

Week 2 converts raw industrial datasets into clean, model-ready, Non-IID edge-node partitions.

The final Week 2 output is:

```text
data/partitions/
  node_01/train.parquet, test.parquet, meta.json
  node_02/train.parquet, test.parquet, meta.json
  node_03/train.parquet, test.parquet, meta.json
  node_04/train.parquet, test.parquet, meta.json
  node_05/train.parquet, test.parquet, meta.json
```

These five nodes are enough for the prototype because they prove heterogeneous data distribution:

- Three time-series HAI nodes for VAE anomaly detection.
- Two tabular Modbus nodes for Isolation Forest anomaly detection.

## File: `scripts/check_datasets.py`

### What This File Does

This is the Day 1 dataset integrity checker. It verifies that raw datasets exist before expensive preprocessing begins.

### Why It Exists

Industrial pipelines should fail early. If raw files are missing, incomplete, or are Git LFS pointer files, preprocessing should not start and produce misleading downstream errors.

### Main Functions

`locate_hai_dir()`

- Finds the configured HAI version folder under `data/raw/hai`.
- Uses `HAI_VERSION` from the environment, defaulting to `hai-22.04`.
- Falls back to the first HAI-like folder if the requested version is missing.

`is_git_lfs_pointer(path)`

- Opens the first line of a file.
- Detects Git LFS placeholder files.
- Prevents accidentally treating placeholder text as dataset CSV.

`check_hai()`

- Scans HAI CSV files.
- Reads only the first few rows for speed.
- Confirms required columns such as `timestamp` and attack label columns.
- Reports file sizes and column validity.

`check_modbus()`

- Supports either CSV-based Modbus data or PCAP-based Modbus data.
- For this project, the PCAP path is used.
- Verifies that `attack/` and `benign/` PCAP folders exist.

`main()`

- Prints a Rich table with pass/fail status.
- Returns exit code `0` only when all dataset checks pass.

### How To Run

```powershell
python scripts/check_datasets.py
```

### Interview Explanation

This file is a guardrail. It validates dataset presence and format cheaply before running memory-heavy parsing. In production pipelines, this kind of preflight check saves time and prevents silent corruption.

## File: `scripts/process_dataset.py`

### What This File Does

This is the Day 2 preprocessing pipeline. It converts raw HAI CSV files and Modbus PCAP files into normalized processed parquet files.

### Why It Exists

Raw industrial data is not directly model-ready. It may contain missing values, mixed schemas, unscaled numeric ranges, and large PCAP files that cannot be loaded into memory. This file creates stable processed artifacts for partitioning.

### HAI Processing

Main function: `process_hai()`

Input:

```text
data/raw/hai/<HAI_VERSION>/*.csv
```

Output:

```text
data/processed/hai_normal.parquet
data/processed/hai_attacks.parquet
data/processed/hai_combined.parquet
data/processed/hai_scaler_meta.json
```

Processing steps:

- Loads HAI train and test CSV files.
- Normalizes column names to lowercase.
- Parses and sorts timestamps.
- Identifies sensor columns.
- Forward-fills and backfills missing sensor values.
- Ensures attack labels are binary.
- Fits `MinMaxScaler` on normal rows only.
- Applies scaling to all rows.
- Saves normal, attack, and combined parquet files.

Important design decision:

The scaler is fitted only on normal data to avoid attack-data leakage. This matters because anomaly detection should learn the normal operating distribution, not the attack distribution.

### Modbus Processing

Main function: `process_modbus()`

Input:

```text
data/raw/modbus2023/attack/**/*.pcap
data/raw/modbus2023/benign/**/*.pcap
```

Output:

```text
data/processed/modbus2023_raw_shards/
data/processed/modbus2023_clean_shards/
data/processed/modbus2023_combined.parquet
data/processed/modbus2023_scaler_meta.json
```

The Modbus PCAP dataset is large, so it is processed with a streaming architecture.

Important functions:

`find_tshark()`

- Locates the Wireshark `tshark` executable.
- Supports PATH lookup and common Windows install paths.

`build_tshark_stream_command(path)`

- Builds a low-memory `tshark` command.
- Extracts only selected packet fields.
- Uses flags that reduce reassembly and defragmentation memory pressure.

`parse_pcap_to_raw_shards(...)`

- Streams one PCAP through `tshark`.
- Reads CSV rows in batches.
- Writes each batch immediately as a parquet shard.
- Avoids holding full PCAP output in Python memory.

`compute_modbus_cleaning_stats(shards)`

- Computes cleaning metadata from raw shards.
- Determines high-NaN columns to drop.
- Estimates medians, percentile clipping bounds, and min/max ranges.

`clean_modbus_shards(raw_shards, stats)`

- Reads one raw shard at a time.
- Fills missing numeric values.
- Replaces infinities.
- Clips extreme values.
- Normalizes numeric features.
- Writes cleaned shards.

`combine_parquet_shards(shards, output_path)`

- Combines cleaned parquet shards into `modbus2023_combined.parquet`.
- Uses PyArrow writer instead of pandas concat.

### Why Streaming Was Needed

The initial approach used:

```python
subprocess.run(..., capture_output=True)
rows = []
pd.concat(rows)
```

That failed because each PCAP expands into millions of packet rows. The new approach keeps memory bounded by processing batches and shards.

### How To Run

```powershell
python scripts/process_dataset.py
```

Optional memory tuning:

```powershell
$env:MODBUS_PARSE_BATCH_ROWS="20000"
python scripts/process_dataset.py
```

### Interview Explanation

This file demonstrates production-style data engineering: raw PCAP parsing is streamed, cleaning is shard-based, and the downstream contract is preserved by still writing `modbus2023_combined.parquet`.

## File: `scripts/partition_data.py`

### What This File Does

This is the Day 3 Non-IID partition generator. It turns processed datasets into five edge-node datasets.

### Why It Exists

Federated learning needs distributed clients with different local distributions. This script simulates realistic industrial heterogeneity.

### HAI Partitioning

Main function: `partition_hai()`

Input:

```text
data/processed/hai_combined.parquet
```

Output:

```text
data/partitions/node_01/
data/partitions/node_02/
data/partitions/node_03/
```

Node mapping:

- `node_01`: HAI `p1_*` subsystem sensors.
- `node_02`: HAI `p2_*` subsystem sensors.
- `node_03`: HAI `p3_*` and `p4_*` subsystem sensors.

Important functions:

`make_windows(df, sensor_cols, window_size, stride)`

- Converts raw time-series rows into sliding windows.
- Window size is 60 rows.
- Stride is 10 rows.
- A window is labeled attack if any row inside it is an attack.
- Output windows are flattened for parquet storage.

`save_partition(...)`

- Writes `train.parquet`, `test.parquet`, and `meta.json`.
- Uses `feature_0 ... feature_N` column names.
- Stores node metadata for model selection later.

Important design decision:

HAI is split by physical subsystem prefix, not by equal column count. This avoids mixing P1, P2, P3, and P4 process areas across nodes.

Verified HAI partition layout:

```text
node_01: p1 only, 44 sensors, 2640 flattened features
node_02: p2 only, 24 sensors, 1440 flattened features
node_03: p3/p4 only, 18 sensors, 1080 flattened features
```

### Modbus Partitioning

Main function: `partition_modbus()`

Input:

```text
data/processed/modbus2023_clean_shards/*.parquet
```

Fallback input:

```text
data/processed/modbus2023_combined.parquet
```

Output:

```text
data/partitions/node_04/
data/partitions/node_05/
```

Node mapping:

- `node_04`: attack-heavy tabular traffic.
- `node_05`: benign-heavy mixed traffic.

Important functions/classes:

`StreamingPartitionWriter`

- Writes train/test parquet outputs incrementally.
- Uses PyArrow `ParquetWriter`.
- Prevents loading the full 39M-row Modbus dataset into memory.

`to_binary_label(labels)`

- Converts label strings into `0 = normal`, `1 = attack`.

`stable_bucket(parts)`

- Creates deterministic hash buckets.
- Used for reproducible routing and train/test splits.

`iter_modbus_batches(paths)`

- Yields one parquet row group at a time.
- Keeps memory usage bounded.

Important design decision:

Every Modbus row is assigned exactly once. The split is Non-IID but complete:

- Most attack rows route to `node_04`.
- Most benign rows route to `node_05`.

### How To Run

```powershell
python scripts/partition_data.py
```

### Interview Explanation

This file creates realistic federated clients. HAI clients differ by physical subsystem; Modbus clients differ by attack/benign distribution. The Modbus path is streaming because industrial telemetry can easily exceed memory if processed as one DataFrame.

## File: `scripts/init_db.py`

### What This File Does

This is the Day 4 PostgreSQL schema migration script.

### Why It Exists

The system needs structured metadata for nodes, alerts, and federated learning rounds. PostgreSQL stores that operational state.

### Main Objects Created

`node_registry`

- One row per edge node.
- Stores node ID, data type, model type, feature count, privacy budget, status, and heartbeat time.
- Uses `DOUBLE PRECISION` for epsilon budget and epsilon consumed values.

`alert_records`

- One row per anomaly alert.
- Stores severity, investigation status, MITRE technique result, and remediation payload.
- Includes `embedding vector(128)` using pgvector for future similarity search.
- Uses `gen_random_uuid()` for alert IDs, backed by the `pgcrypto` extension.

`fl_rounds`

- One row per federated learning round.
- Stores round timing, participating node count, global loss, AUROC, epsilon total, and model version.
- Uses `DOUBLE PRECISION` for numeric metrics such as loss, AUROC, and epsilon.

### Extensions Enabled

- `pgcrypto`: provides `gen_random_uuid()` for UUID primary keys.
- `vector`: provides pgvector support for `embedding vector(128)`.

### Main Function

`run()`

- Connects to PostgreSQL using `POSTGRES_DSN`.
- Runs idempotent `CREATE TABLE IF NOT EXISTS` statements.
- Verifies required tables exist.
- Separates connection failures from schema or extension failures in error handling.

### How To Run

```powershell
python scripts/init_db.py
```

### Interview Explanation

This file provides the persistence layer for later orchestration. It separates operational metadata from vector memory and makes the system auditable.

## File: `scripts/seed_registry.py`

### What This File Does

This is the Day 5 PostgreSQL registry seeding script.

### Why It Exists

After partitioning, the system needs a database-level registry of available edge nodes. This file reads partition metadata and makes PostgreSQL aware of each simulated node.

### Input

```text
data/partitions/node_01/meta.json
...
data/partitions/node_05/meta.json
```

### Main Functions

`load_node_meta(meta_path)`

- Reads one node metadata file.
- Validates required fields: `node_id`, `data_type`, `model_type`, `feature_count`.
- Rejects unsupported model or data types.

`discover_node_rows(partition_dir)`

- Finds all `node_*/meta.json` files.
- Loads them in deterministic order.
- Detects duplicate node IDs.

`ensure_node_registry_exists(cursor)`

- Confirms `scripts/init_db.py` has already created the `node_registry` table.
- Fails clearly if the database schema is missing.

`upsert_rows(rows)`

- Inserts or updates one row per node.
- Uses `ON CONFLICT (node_id) DO UPDATE`.
- Sets node status to `ready`.
- Updates `last_seen` to the current database timestamp.

`run()`

- Coordinates discovery, connection, validation, and upsert.
- Handles database and metadata errors clearly.

### How To Run

```powershell
python scripts/seed_registry.py
```

Expected result:

```text
Seeded 5 node_registry rows
```

### Interview Explanation

This script bridges data engineering and infrastructure metadata. Partition files define what each node can train; PostgreSQL stores that information so later services can discover node capabilities.

## File: `scripts/verify_messaging.py`

### What This File Does

This is a Week 1 validation script, but it remains important for Week 2 and beyond because later edge nodes and agents depend on Kafka and NATS.

### Why It Exists

Kafka and NATS are separate messaging systems:

- Kafka handles raw telemetry ingestion.
- NATS JetStream handles lightweight inter-service events and agent actions.

This script verifies both are reachable and can round-trip messages.

### Main Functions

`check_kafka()`

- Sends 10 JSON messages to Kafka topic `telemetry.raw`.
- Creates a temporary consumer group.
- Reads messages back.
- Confirms all messages round-tripped.

`_nats_check()`

- Connects to NATS JetStream.
- Creates a temporary stream.
- Publishes 10 messages.
- Subscribes and reads them back.
- Acknowledges messages.
- Deletes the temporary stream.

`check_nats()`

- Runs the async NATS check from synchronous code.

`main()`

- Prints a Rich result table.
- Returns success only if Kafka and NATS both pass.

### How To Run

```powershell
python scripts/verify_messaging.py
```

### Interview Explanation

This file proves the messaging backbone works before ML and agents depend on it. It is a practical integration gate, not just a unit test.

## File: `tests/unit/test_week2_data.py`

### What This File Does

This is the automated Week 2 validation gate.

### What It Checks

- Processed HAI parquet exists.
- Processed Modbus parquet exists.
- All five node partitions contain `train.parquet`, `test.parquet`, and `meta.json`.
- VAE nodes are typed correctly.
- Isolation Forest nodes are typed correctly.
- HAI subsystem prefixes are not mixed.
- All partition parquet files are readable.
- PostgreSQL registry is populated when PostgreSQL is reachable.

### How To Run

```powershell
agvenv\Scripts\python.exe -m pytest tests\unit\test_week2_data.py -v --tb=short -p no:cacheprovider
```

### Current Result

```text
22 passed
```

The PostgreSQL registry test now loads `.env`, so it uses the same `POSTGRES_DSN` as the runtime scripts.

## Documentation Rule Going Forward

For every new or updated implementation file, add or update a note with:

- File purpose
- Why it exists
- Main functions/classes
- Inputs and outputs
- Important design decisions
- How to run or test it
- Interview explanation

This keeps the project understandable for review, interview preparation, and future handoff.
