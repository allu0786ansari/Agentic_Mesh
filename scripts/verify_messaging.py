from __future__ import annotations

import asyncio

from edge.bus import AsyncBus


async def main() -> None:
    bus = AsyncBus()
    await bus.drain("week1.test")

    for index in range(10):
        await bus.publish("week1.test", {"id": index, "payload": f"message-{index}"})

    received = []
    for _ in range(10):
        received.append(await bus.get("week1.test", timeout=1.0))

    assert len(received) == 10, "Message count mismatch"
    assert all(item["id"] == idx for idx, item in enumerate(received)), "Incorrect ordering in message bus"
    print("All messaging checks passed")


if __name__ == "__main__":
    asyncio.run(main())
