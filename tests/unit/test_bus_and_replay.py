import asyncio
from pathlib import Path

import pandas as pd
import pytest

from edge.bus import AsyncBus
from edge.telemetry_replay import TelemetryReplay


@pytest.mark.asyncio
async def test_async_bus_round_trip() -> None:
    bus = AsyncBus()
    await bus.drain("unit.test")

    await bus.publish("unit.test", {"id": 1, "status": "ok"})
    received = await bus.get("unit.test", timeout=1.0)

    assert received == {"id": 1, "status": "ok"}


@pytest.mark.asyncio
async def test_telemetry_replay_reads_parquet() -> None:
    path = Path("tests/unit/test_data.parquet")
    df = pd.DataFrame([{"timestamp": "2026-08-19T00:00:00", "value": 1.0}, {"timestamp": "2026-08-19T00:00:01", "value": 2.0}])
    df.to_parquet(path, index=False)

    try:
        replay = TelemetryReplay(path)
        rows = [row async for row in replay.stream()]  # type: ignore[attr-defined]
        assert len(rows) == 2
        assert rows[0]["value"] == 1.0
        assert rows[1]["value"] == 2.0
    finally:
        path.unlink(missing_ok=True)
