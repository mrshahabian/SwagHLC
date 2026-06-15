"""Synthetic stream source — pure numpy, zero dataset/torch dependency.

Generates plausible-looking multi-channel signals so the whole pipeline can run
on any machine.  ``options.pattern`` selects the waveform:
  * ``noise``  — gaussian noise (default).
  * ``sine``   — per-channel sinusoids at staggered frequencies + noise.
  * ``walk``   — bounded random walk (slow drift, useful to eyeball windows).
"""

from __future__ import annotations

import numpy as np

from swag_hlc.config import SourceConfig
from swag_hlc.modalities import get_modality_spec
from swag_hlc.transport.base import Publisher
from swag_hlc.dummy_stream.base_source import StreamSource


class SyntheticSource(StreamSource):
    def __init__(self, cfg: SourceConfig, publisher: Publisher) -> None:
        super().__init__(cfg, publisher)
        opts = self.cfg.options
        self.pattern = str(opts.get("pattern", "noise")).lower()
        self.amplitude = float(opts.get("amplitude", 1.0))
        self.noise = float(opts.get("noise", 0.1))
        # Deterministic per-device seed keeps runs reproducible & devices distinct.
        seed = int(opts.get("seed", abs(hash(self.cfg.id)) % (2**32)))
        self._rng = np.random.default_rng(seed)
        self._t0 = 0  # running sample index for phase continuity
        ch = self.cfg.channels
        self._freqs = (0.5 + np.arange(ch)) * float(opts.get("base_freq", 1.0))
        self._walk_state = np.zeros(ch, dtype=np.float32)
        # Emit in the modality's native shape (e.g. HD-EMG (4,16)) when it matches
        # the channel count; otherwise fall back to a flat vector.
        try:
            ns = get_modality_spec(self.cfg.modality).native_shape
            self.feature_shape = ns if int(np.prod(ns)) == ch else (ch,)
        except KeyError:
            self.feature_shape = (ch,)

    def next_chunk(self) -> np.ndarray:
        n = self.cfg.chunk_size
        ch = self.cfg.channels
        if self.pattern == "sine":
            t = (self._t0 + np.arange(n)) / float(self.cfg.rate_hz)
            sig = self.amplitude * np.sin(2 * np.pi * np.outer(t, self._freqs))
            sig = sig + self.noise * self._rng.standard_normal((n, ch))
            self._t0 += n
        elif self.pattern == "walk":
            steps = self.noise * self._rng.standard_normal((n, ch))
            sig = self._walk_state + np.cumsum(steps, axis=0)
            sig = np.clip(sig, -self.amplitude, self.amplitude)
            self._walk_state = sig[-1].copy()
        else:  # noise
            sig = self.amplitude * self._rng.standard_normal((n, ch))
        return sig.astype(np.float32).reshape(n, *self.feature_shape)
