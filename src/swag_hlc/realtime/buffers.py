"""Per-device circular sample buffer + latest-window extraction.

Each input device gets its own RingBuffer.  This realizes the agreed
"per-modality independent windows" design: every modality fills its own buffer
at its own rate, and the model consumer pulls the *latest* window from each
buffer when it runs inference — no cross-modality resampling/alignment.

The buffer stores samples in their NATIVE per-sample shape (e.g. HD-EMG (4, 16)):
``buffer`` has shape ``(capacity, *feature_shape)`` and a window comes out as
``(window_size, *feature_shape)``.  Flattening is the model's job (inference side).
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    def __init__(self, feature_shape: tuple[int, ...], window_size: int, chunk_hint: int = 0) -> None:
        self.feature_shape = tuple(feature_shape)
        self.window_size = window_size
        # Capacity holds at least a full window plus one incoming chunk.
        self.capacity = max(window_size, window_size + max(chunk_hint, 0))
        self._buf = np.zeros((self.capacity, *self.feature_shape), dtype=np.float32)
        self._w = 0  # write cursor
        self._count = 0  # samples currently buffered (saturates at capacity)
        self.total_written = 0

    def append(self, chunk: np.ndarray) -> None:
        n = chunk.shape[0]
        self.total_written += n
        if n >= self.capacity:
            chunk = chunk[-self.capacity:]
            n = chunk.shape[0]
        end = self._w + n
        if end <= self.capacity:
            self._buf[self._w:end] = chunk
        else:
            first = self.capacity - self._w
            self._buf[self._w:] = chunk[:first]
            self._buf[: n - first] = chunk[first:]
        self._w = (self._w + n) % self.capacity
        self._count = min(self.capacity, self._count + n)

    def has_window(self) -> bool:
        return self._count >= self.window_size

    def latest_window(self) -> np.ndarray | None:
        """Most recent ``window_size`` samples as ``(window_size, *feature_shape)``."""
        if not self.has_window():
            return None
        start = (self._w - self.window_size) % self.capacity
        end = start + self.window_size
        if end <= self.capacity:
            return self._buf[start:end].copy()
        first = self.capacity - start
        return np.concatenate([self._buf[start:], self._buf[: self.window_size - first]], axis=0)
