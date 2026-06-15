"""Temporal smoothing of the intent stream for the controller.

Raw per-window argmax is jittery; the mid-level controller wants a *stable*
intent. A ``Smoother`` turns the live probability stream into a debounced class
index. Methods (config ``smoothing.method``):

  * ``none``     — passthrough (stable = argmax of the current window).
  * ``majority`` — most frequent argmax over the last ``window`` predictions
    (mirrors BioActLab's majority voting), optional ``min_confidence`` gate.
  * ``ema``      — exponential moving average of the probability vector
    (``alpha``), then argmax. Smooths probabilities, not just labels.
  * ``hold``     — hysteresis: only switch to a new class after it persists for
    ``min_count`` consecutive windows (debounce against flicker).
"""

from __future__ import annotations

import collections

import numpy as np


class Smoother:
    def __init__(
        self,
        num_classes: int,
        method: str = "majority",
        window: int = 5,
        alpha: float = 0.3,
        min_count: int = 3,
        min_confidence: float = 0.0,
    ) -> None:
        self.num_classes = num_classes
        self.method = (method or "none").lower()
        self.window = max(1, int(window))
        self.alpha = float(alpha)
        self.min_count = max(1, int(min_count))
        self.min_confidence = float(min_confidence)
        self._hist: collections.deque[int] = collections.deque(maxlen=self.window)
        self._ema: np.ndarray | None = None
        self._stable: int | None = None
        self._cand: int | None = None
        self._cand_n = 0

    def update(self, probs: np.ndarray) -> int:
        raw = int(np.argmax(probs))
        conf = float(np.max(probs))

        if self.method == "ema":
            self._ema = probs if self._ema is None else self.alpha * probs + (1 - self.alpha) * self._ema
            self._stable = int(np.argmax(self._ema))
            return self._stable

        if self.method == "hold":
            if self._stable is None:
                self._stable = raw
            elif raw == self._stable:
                self._cand, self._cand_n = None, 0
            else:  # a different class is proposed — require persistence
                if raw == self._cand:
                    self._cand_n += 1
                else:
                    self._cand, self._cand_n = raw, 1
                if self._cand_n >= self.min_count and conf >= self.min_confidence:
                    self._stable = raw
                    self._cand, self._cand_n = None, 0
            return self._stable

        if self.method == "majority":
            if conf >= self.min_confidence:
                self._hist.append(raw)
            if not self._hist:
                self._stable = raw
            else:
                self._stable = collections.Counter(self._hist).most_common(1)[0][0]
            return self._stable

        return raw  # "none"
