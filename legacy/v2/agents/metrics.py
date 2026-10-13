from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_LOG_PATH = Path(os.getenv("METRICS_LOG_PATH", "logs/metrics.jsonl"))


def log_event(event_name: str, payload: Mapping[str, Any], path: str | Path = DEFAULT_LOG_PATH) -> None:
    """Append one structured event atomically to the metrics JSONL stream."""
    if not event_name.strip():
        raise ValueError("event_name must not be empty")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
        "payload": dict(payload),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def read_events(path: str | Path = DEFAULT_LOG_PATH) -> list[dict[str, Any]]:
    """Read structured metric events, rejecting malformed records."""
    source = Path(path)
    if not source.exists():
        return []
    events: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed metrics record at line {line_number}") from exc
            if not isinstance(record, dict) or "event" not in record or "payload" not in record:
                raise ValueError(f"Invalid metrics record at line {line_number}")
            events.append(record)
    return events
