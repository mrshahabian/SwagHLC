"""Base class for dummy stream sources.

A source pretends to be one sensor device: it emits ``SensorFrame`` chunks onto
its topic at the device's real-time cadence (``chunk_size / rate_hz`` seconds per
chunk).  Subclasses only implement ``next_chunk`` — how the samples are produced.
"""

from __future__ import annotations

import abc
import asyncio
import time

import numpy as np

from swag_hlc.config import SourceConfig
from swag_hlc.messages import SensorFrame
from swag_hlc.transport.base import Publisher


class StreamSource(abc.ABC):
    def __init__(self, cfg: SourceConfig, publisher: Publisher) -> None:
        self.cfg = cfg.resolved()
        self.publisher = publisher
        self._seq = 0
        self._period = self.cfg.chunk_size / float(self.cfg.rate_hz)
        # Native per-sample shape streamed on the wire (subclasses override).
        self.feature_shape: tuple[int, ...] = (self.cfg.channels,)
        # Subclasses replaying labelled data set these before returning a chunk.
        self._chunk_label: float | None = None  # raw activity code (e.g. 2.1)
        self._chunk_pos: dict = {}

    @abc.abstractmethod
    def next_chunk(self) -> np.ndarray:
        """Return the next ``(chunk_size, *feature_shape)`` float32 array.

        Data is emitted in its NATIVE shape (e.g. HD-EMG as (chunk, 4, 16)); the
        inference side flattens. Subclasses replaying labelled data may also set
        ``self._chunk_label`` and ``self._chunk_pos`` to annotate the frame.
        """

    def _make_frame(self, data: np.ndarray) -> SensorFrame:
        frame = SensorFrame(
            device_id=self.cfg.id,
            modality=self.cfg.modality,
            feature_shape=tuple(self.feature_shape),
            channels=int(np.prod(self.feature_shape)),
            sample_rate_hz=self.cfg.rate_hz,
            seq=self._seq,
            t_send=time.monotonic(),
            data=np.ascontiguousarray(data, dtype=np.float32),
            label=self._chunk_label,
            source_pos=dict(self._chunk_pos),
        )
        self._seq += 1
        return frame

    async def run(self, stop: asyncio.Event) -> None:
        """Emit chunks paced to wall-clock until ``stop`` is set."""
        next_t = time.monotonic()
        while not stop.is_set():
            self.publisher.publish(self._make_frame(self.next_chunk()))
            next_t += self._period
            delay = next_t - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
            else:
                # Falling behind real time; yield without extra sleep.
                next_t = time.monotonic()
                await asyncio.sleep(0)
