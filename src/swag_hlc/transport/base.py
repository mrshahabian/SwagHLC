"""Abstract transport interface.

A minimal pub/sub contract.  Producers get a ``Publisher`` for a topic;
consumers get a ``Subscriber`` they can poll.  Backends decide how messages
actually move (in-process queues, ROS 2 topics, sockets, ...).

Design constraint: topics and subscriptions are declared *up front* (static
wiring) so a multiprocessing backend can hand the right queue endpoints to each
child process before it is spawned.  This keeps the hot path lock-light and
makes the wiring explicit and inspectable.
"""

from __future__ import annotations

import abc
from typing import Any


class Publisher(abc.ABC):
    """Sends messages on a single topic."""

    @abc.abstractmethod
    def publish(self, message: Any) -> None: ...


class Subscriber(abc.ABC):
    """Receives messages from a single topic."""

    @abc.abstractmethod
    def poll(self, timeout: float | None = None) -> Any | None:
        """Return the next message, or ``None`` if none arrived within timeout."""


class Transport(abc.ABC):
    """A message bus. Create publishers/subscribers, then ``start``."""

    @abc.abstractmethod
    def declare_topic(self, topic: str) -> None: ...

    @abc.abstractmethod
    def publisher(self, topic: str) -> Publisher: ...

    @abc.abstractmethod
    def subscriber(self, topic: str) -> Subscriber: ...

    def start(self) -> None:  # optional lifecycle hooks
        pass

    def close(self) -> None:
        pass
