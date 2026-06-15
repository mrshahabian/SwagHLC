"""Stub inference model — pure numpy, no torch, no checkpoint.

Stands in for a trained BioActLab model so the streaming pipeline can be built
and exercised end to end before real weights are wired in.  ``options.mode``:
  * ``energy`` — class derived from window signal energy (default). Output
    visibly *reacts* to the incoming stream, which is handy for eyeballing the
    pipeline; still completely fake.
  * ``cycle``  — cycle deterministically through classes over time.
  * ``random`` — fresh random distribution each inference.

``options.latency_ms`` lets you simulate model compute time (the engine sleeps),
so you can feel backpressure / GPU-per-model latency before real models exist.
"""

from __future__ import annotations

import numpy as np

from swag_hlc.config import ModelConfig
from swag_hlc.realtime.models.base import InferenceModel


class StubModel(InferenceModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        opts = cfg.options
        self.mode = str(opts.get("mode", "energy")).lower()
        self.latency_ms = float(opts.get("latency_ms", 0.0))
        self.temperature = float(opts.get("temperature", 1.0))
        seed = int(opts.get("seed", abs(hash(cfg.id)) % (2**32)))
        self._rng = np.random.default_rng(seed)
        self._tick = 0

    def warmup(self) -> None:
        # Real models would load weights & move to self.device here.
        _ = self.device

    def predict(self, windows: dict[str, np.ndarray]) -> np.ndarray:
        self._tick += 1
        if self.mode == "cycle":
            logits = np.full(self.num_classes, -2.0, dtype=np.float32)
            logits[self._tick % self.num_classes] = 2.0
            return self.softmax(logits / self.temperature)
        if self.mode == "random":
            return self.softmax(self._rng.standard_normal(self.num_classes).astype(np.float32))

        # "energy": fold all input windows into a single scalar feature and use
        # it to pick a *peaked* class, so the fake prediction tracks (and varies
        # with) the stream instead of always saturating one class.
        # Flatten native shapes (e.g. (T,4,16)->(T,64)) here on the inference side.
        flat = self.flatten_windows(windows)
        energy = 0.0
        for w in flat.values():
            energy += float(np.mean(np.abs(w)))
        energy /= max(len(flat), 1)
        scale = float(self.cfg.options.get("energy_scale", 3.0))
        center = (energy * scale) % self.num_classes
        idx = np.arange(self.num_classes, dtype=np.float32)
        logits = -((idx - center) ** 2) + 0.1 * self._rng.standard_normal(self.num_classes)
        return self.softmax(logits / self.temperature)
