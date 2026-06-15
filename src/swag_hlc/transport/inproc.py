"""In-process / multiprocessing transport backend.

Uses ``multiprocessing.Queue`` so the exact same wiring works whether modules
run as asyncio tasks in one process or as separate OS processes (the hybrid
model).  Fan-out is supported: a publisher writes to every subscriber queue
registered for its topic.

Static-wiring rule: declare topics and create *all* subscribers before creating
publishers and spawning child processes.  A Publisher captures a snapshot of the
subscriber queues for its topic at creation time; subscribers added afterwards
(or in a child after fork) will not receive its messages.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
from typing import Any

from swag_hlc.transport.base import Publisher, Subscriber, Transport


class _QueuePublisher(Publisher):
    def __init__(self, topic: str, queues: list[mp.Queue]) -> None:
        self._topic = topic
        self._queues = queues

    def publish(self, message: Any) -> None:
        for q in self._queues:
            # Drop-newest backpressure: never block a real-time producer.
            try:
                q.put_nowait(message)
            except _queue.Full:
                pass


class _QueueSubscriber(Subscriber):
    def __init__(self, topic: str, q: mp.Queue) -> None:
        self._topic = topic
        self._q = q

    def poll(self, timeout: float | None = None) -> Any | None:
        try:
            if timeout is None:
                return self._q.get_nowait()
            return self._q.get(timeout=timeout)
        except _queue.Empty:
            return None


class InProcTransport(Transport):
    """Multiprocessing-queue pub/sub with per-topic fan-out."""

    def __init__(self, ctx: mp.context.BaseContext | None = None, maxsize: int = 256) -> None:
        # 'fork' keeps wiring simple (children inherit queues); override if needed.
        self._ctx = ctx or mp.get_context("fork")
        self._maxsize = maxsize
        self._subs: dict[str, list[mp.Queue]] = {}

    def declare_topic(self, topic: str) -> None:
        self._subs.setdefault(topic, [])

    def publisher(self, topic: str) -> Publisher:
        self.declare_topic(topic)
        # Snapshot current subscriber queues for this topic.
        return _QueuePublisher(topic, list(self._subs[topic]))

    def subscriber(self, topic: str) -> Subscriber:
        self.declare_topic(topic)
        q = self._ctx.Queue(maxsize=self._maxsize)
        self._subs[topic].append(q)
        return _QueueSubscriber(topic, q)
