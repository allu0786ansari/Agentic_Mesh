#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-agentvenv/Scripts/python.exe}"
ENDPOINT="${PHOENIX_OTEL_ENDPOINT:-http://127.0.0.1:6006/v1/traces}"

"${PYTHON_BIN}" -m scripts.run_local_stack \
  --count 5 \
  --endpoint "${ENDPOINT}" \
  --metrics-path results/week13_smoke/metrics.jsonl

"${PYTHON_BIN}" -c "from pathlib import Path; p=Path('results/week13_smoke/metrics.jsonl'); text=p.read_text(encoding='utf-8'); assert text.count('agent_graph_completed') == 5; print('=== Smoke Test PASSED ===')"
