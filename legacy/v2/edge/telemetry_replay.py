from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Iterable, Iterator

import pandas as pd


class TelemetryReplay:
    """Read industrial telemetry rows as a replay stream.

    This class is designed to mimic a live data ingestion source while remaining
    fully in-process. It can consume a parquet file or an in-memory iterator and
    yields rows one at a time at a controlled cadence.
    """

    def __init__(self, source: str | Path | Iterable[dict[str, Any]], interval: float = 0.0):
        self.source = Path(source) if isinstance(source, (str, Path)) else source
        self.interval = interval

    async def stream(self) -> AsyncGenerator[dict[str, Any], None]:
        rows = self._load_rows()
        for row in rows:
            yield row
            if self.interval > 0:
                await asyncio.sleep(self.interval)

    def _load_rows(self) -> Iterator[dict[str, Any]]:
        if hasattr(self.source, "__iter__") and not isinstance(self.source, (str, Path)):
            for item in self.source:
                yield item
            return

        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"Telemetry source does not exist: {path}")

        if path.suffix.lower() == ".parquet":
            dataframe = pd.read_parquet(path)
            for row in dataframe.to_dict(orient="records"):
                yield row
            return

        if path.suffix.lower() in {".json", ".jsonl"}:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            yield json.loads(line)
            else:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    for item in payload:
                        yield item
                else:
                    yield payload
            return

        raise ValueError(f"Unsupported telemetry source format: {path.suffix}")
