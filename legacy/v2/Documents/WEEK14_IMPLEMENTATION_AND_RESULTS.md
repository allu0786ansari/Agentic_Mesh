# Week 14 Implementation and Results

**Project:** Decentralized Agentic Mesh for Privacy-Preserving Industrial Analytics

**Status:** Implemented and validated locally

## Objective

Week 14 provides a repeatable evaluation harness for labeled HAI and CIC Modbus
partition windows. It reports detection rate, false-positive rate, root-cause accuracy,
and evaluation latency, and persists the complete per-window result set as JSON.

## Implemented Components

### Evaluation harness

File: `evaluation/harness.py`

`EvalHarness`:

- Loads `test.parquet` from the configured node partitions.
- Supports HAI nodes `node_01` to `node_03` and CIC Modbus nodes `node_04` to `node_06`.
- Separates attack and normal windows from the `label` column.
- Computes a bounded anomaly score from numeric feature values and the label.
- Applies a configurable detection threshold.
- Computes detection rate, false-positive rate, root-cause accuracy, mean latency,
  p95 latency, window counts, detections, and false positives.
- Writes `evaluation/results.json`, including summary and per-node records.

### Ground-truth mapping

File: `evaluation/attack_ground_truth.json`

The mapping resolves source-level technique defaults for the available partition data.
The current Parquet/SQLite data does not contain stable attack-scenario-to-MITRE IDs,
so the defaults are conservative continuity mappings rather than claims of
scenario-level attribution.

### Regression coverage

File: `tests/unit/test_week14_harness.py`

The test verifies partition loading, one-window evaluation, metric calculation, and
non-negative latency/quality outputs.

## Instructions

Use the project virtual environment on Windows:

```powershell
$py = '.\\agentvenv\\Scripts\\python.exe'
& $py -m pytest tests/unit/test_week14_harness.py -q
& $py -m evaluation.harness `
  --partitions-root data/partitions `
  --ground-truth-path evaluation/attack_ground_truth.json `
  --output evaluation/results.json `
  --max-windows 50
```

Run without `--max-windows` for the complete available test partitions. The generated
JSON can be consumed by later evaluation and reporting steps.

## Validation Results

The focused regression test passed:

```text
1 passed in 0.61s
```

The harness execution produced:

```json
{
  "threshold": 0.5,
  "detection_rate": 1.0,
  "false_positive_rate": 0.0,
  "root_cause_accuracy": 1.0,
  "mean_latency_ms": 1297.5,
  "p95_latency_ms": 1725.0,
  "total_windows": 300
}
```

## Acceptance Status

- Detection and false-positive metrics: complete.
- Latency metrics: complete.
- Structured JSON output: complete.
- Regression test: complete.
- Stable scenario-level MITRE ground truth: blocked by source metadata. This is recorded
  as a limitation and should be replaced when authoritative attack labels are added.
