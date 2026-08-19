from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional


class AsyncBus:
    """In-process message bus for edge and agent communication.

    This replaces a distributed broker such as Kafka/NATS in the local development
    build while preserving the semantics needed by the project: topic-based publish/
    subscribe, ordered delivery within a process, and simple decoupled messaging.
    """

    _instance: Optional["AsyncBus"] = None
    _lock: asyncio.Lock

    def __new__(cls) -> "AsyncBus":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._topics: Dict[str, asyncio.Queue] = {}
            cls._instance._lock = asyncio.Lock()
        return cls._instance

    async def publish(self, topic: str, message: Any) -> None:
        if topic not in self._topics:
            self._topics[topic] = asyncio.Queue()
        await self._topics[topic].put(message)

    async def subscribe(self, topic: str) -> asyncio.Queue:
        if topic not in self._topics:
            self._topics[topic] = asyncio.Queue()
        return self._topics[topic]

    async def get(self, topic: str, timeout: Optional[float] = None) -> Any:
        queue = await self.subscribe(topic)
        try:
            if timeout is None:
                return await queue.get()
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Timed out waiting for message on topic '{topic}'") from exc

    async def drain(self, topic: str) -> None:
        queue = await self.subscribe(topic)
        while not queue.empty():
            queue.get_nowait()

    @asynccontextmanager
    async def topic_context(self, topic: str):
        queue = await self.subscribe(topic)
        try:
            yield queue
        finally:
            await self.drain(topic)


async def publish(topic: str, message: Any) -> None:
    await AsyncBus().publish(topic, message)


async def subscribe(topic: str) -> asyncio.Queue:
    return await AsyncBus().subscribe(topic)
