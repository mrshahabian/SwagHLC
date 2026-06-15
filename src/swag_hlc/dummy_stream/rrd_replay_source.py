"""Replay the real RRD dataset as a live stream.

Because the dataset's trials are *sequential recordings*, we treat the selected
(subject -> days -> trials) sequence as one continuous stream: a source emits
fixed-size chunks of one modality's channels, walking trial after trial and
(optionally) looping at the end.  Trials are loaded one at a time (lazy) to keep
memory bounded regardless of how many trials are selected.

What to stream is fully configurable (see SourceConfig / DatasetConfig):
subject, days, trials, and the modality (or explicit ``params``).  Ground-truth
labels travel with each chunk so the monitor can show predicted-vs-true intent.

Requires h5py (``pip install h5py`` / it's in the project venv).
"""

from __future__ import annotations

import numpy as np

from swag_hlc.activities import normalize_code
from swag_hlc.config import SourceConfig
from swag_hlc.dummy_stream import rrd_index
from swag_hlc.dummy_stream.base_source import StreamSource
from swag_hlc.modalities import resolve_params
from swag_hlc.transport.base import Publisher


class RrdReplaySource(StreamSource):
    def __init__(self, cfg: SourceConfig, publisher: Publisher) -> None:
        super().__init__(cfg, publisher)
        self.root = self.cfg.root or str(rrd_index.default_root())
        self.subject = self.cfg.subject
        self.params = resolve_params(self.cfg.modality, self.cfg.params)
        self.loop = True if self.cfg.loop is None else bool(self.cfg.loop)
        # Optional: stream only these activity classes (keep matching samples).
        self.activities = (
            {normalize_code(a) for a in self.cfg.activities} if self.cfg.activities else None
        )
        include_mvc = bool(self.cfg.options.get("include_mvc", False))
        include_nolabel = bool(self.cfg.options.get("include_nolabel", False))

        self._pairs = rrd_index.resolve_pairs(
            self.root, self.subject, self.cfg.days, self.cfg.trials,
            include_mvc=include_mvc, include_nolabel=include_nolabel,
        )
        if not self._pairs:
            raise ValueError(
                f"No trials selected for subject={self.subject} days={self.cfg.days} "
                f"trials={self.cfg.trials} (root={self.root})."
            )

        self._ti = 0  # index into self._pairs
        self._cur: tuple[np.ndarray, np.ndarray, str, str] | None = None
        self._pos = 0
        self._exhausted = False

        # Fix the real native shape from the first trial (e.g. HD-EMG (4,16)).
        feat, _ = self._read_trial(*self._pairs[0])
        self.feature_shape = tuple(feat.shape[1:])
        self.cfg.channels = int(np.prod(self.feature_shape))

    # -- loading -----------------------------------------------------------
    def _read_trial(self, day: str, trial: str) -> tuple[np.ndarray, np.ndarray]:
        import h5py

        path = rrd_index.subject_path(self.root, self.subject)
        with h5py.File(path, "r") as f:
            grp = f[day][trial]
            label = np.asarray(grp["Label"][...]).reshape(-1)
            t_len = label.shape[0]
            cols = []
            for p in self.params:
                if p not in grp:
                    raise KeyError(
                        f"Param '{p}' missing in {self.subject}/{day}/{trial}. "
                        f"Available: {sorted(grp.keys())[:8]}..."
                    )
                arr = np.asarray(grp[p][...], dtype=np.float32)
                if arr.shape[0] != t_len:
                    # Skip non-time-aligned datasets (e.g. markers at 100 Hz).
                    raise ValueError(
                        f"Param '{p}' length {arr.shape[0]} != Label length {t_len}."
                    )
                cols.append(arr)
            # Single param -> keep its NATIVE shape (HD-EMG stays (T,4,16)).
            # Multi-param modality -> flatten each to (T,-1) and concatenate.
            if len(cols) == 1:
                feat = cols[0]
            else:
                feat = np.concatenate([c.reshape(t_len, -1) for c in cols], axis=1)
        feat = np.ascontiguousarray(feat, dtype=np.float32)
        label = label.astype(np.float32)
        if self.activities is not None:  # keep only selected activity samples
            keep = np.array([normalize_code(v) in self.activities for v in label])
            feat, label = feat[keep], label[keep]
        return feat, label

    def _load_next_trial(self) -> bool:
        if self._ti >= len(self._pairs):
            if not self.loop:
                self._exhausted = True
                return False
            self._ti = 0
        day, trial = self._pairs[self._ti]
        feat, label = self._read_trial(day, trial)
        self._cur = (feat, label, day, trial)
        self._pos = 0
        return True

    # -- streaming ---------------------------------------------------------
    def next_chunk(self) -> np.ndarray:
        n = self.cfg.chunk_size
        feats: list[np.ndarray] = []
        labs: list[np.ndarray] = []
        got = 0
        empty_trials = 0  # guard: a selection may match nothing
        while got < n:
            if self._cur is None:
                if not self._load_next_trial():
                    break
                if self._cur[0].shape[0] == 0:  # filtered-empty trial
                    self._cur = None
                    self._ti += 1
                    empty_trials += 1
                    if empty_trials > len(self._pairs):
                        break  # no selected activities anywhere
                    continue
            feat, label, day, trial = self._cur  # type: ignore[misc]
            take = min(feat.shape[0] - self._pos, n - got)
            feats.append(feat[self._pos : self._pos + take])
            labs.append(label[self._pos : self._pos + take])
            self._pos += take
            got += take
            self._chunk_pos = {"day": day, "trial": trial, "i": int(self._pos)}
            if self._pos >= feat.shape[0]:
                self._cur = None
                self._ti += 1

        if got == 0:  # exhausted, not looping — emit silence to hold cadence
            self._chunk_label = None
            return np.zeros((n, *self.feature_shape), dtype=np.float32)

        out = np.concatenate(feats, axis=0)
        lab = np.concatenate(labs, axis=0)
        self._chunk_label = float(lab[-1])  # raw "current" activity code (e.g. 2.1)
        if out.shape[0] < n:  # tail without loop: pad to keep frame size stable
            pad = np.zeros((n - out.shape[0], *self.feature_shape), dtype=np.float32)
            out = np.concatenate([out, pad], axis=0)
        return out
