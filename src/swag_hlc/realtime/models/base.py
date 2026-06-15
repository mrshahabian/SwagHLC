"""Abstract inference model.

A model declares which input devices it consumes and its window geometry, then
maps a set of latest per-device windows to a probability vector over intent
classes.  This is the seam where a real BioActLab torch checkpoint plugs in
later (behind a guarded ``import torch``) without the engine changing.
"""

from __future__ import annotations

import abc

import numpy as np

from swag_hlc.config import ModelConfig


class InferenceModel(abc.ABC):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.id = cfg.id
        self.num_classes = cfg.num_classes
        self.input_ids = list(cfg.inputs)
        self.window = cfg.window
        self.device = cfg.device  # "cpu" | "cuda[:N]" — GPU-per-model
        self.labels = cfg.labels or [f"class_{i}" for i in range(cfg.num_classes)]

    def warmup(self) -> None:
        """Optional: load weights, move to device, run a dummy forward pass."""

    @abc.abstractmethod
    def predict(self, windows: dict[str, np.ndarray]) -> np.ndarray:
        """Map per-device windows -> probs ``(num_classes,)``.

        Windows arrive in the stream's NATIVE shape, ``{device_id:
        (window_size, *feature_shape)}`` (e.g. HD-EMG ``(T, 4, 16)``). Flattening
        to the model's channel vector is done HERE, on the inference side — use
        ``flatten_window`` / ``flatten_windows`` below.
        """

    @staticmethod
    def flatten_window(w: np.ndarray) -> np.ndarray:
        """``(window_size, *feature_shape)`` -> ``(window_size, channels)``."""
        return w.reshape(w.shape[0], -1)

    @classmethod
    def flatten_windows(cls, windows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: cls.flatten_window(w) for k, w in windows.items()}

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return (e / np.sum(e)).astype(np.float32)
