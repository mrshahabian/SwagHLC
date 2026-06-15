"""Transport-agnostic message schemas.

These dataclasses are the *contract* between the dummy stream and the real-time
inference module, and between the inference module and the mid-level controller.

They are plain dataclasses carrying numpy arrays so the in-process transport can
pass them by reference/pickle with zero ceremony.  Each type also provides
``to_dict``/``from_dict`` (numpy -> list) so a ROS 2 / network backend can map
them onto real message types or JSON without changing producers/consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SensorFrame:
    """A chunk of samples from one sensor *device*, in its NATIVE on-disk shape.

    A "device" is one physical (or simulated) sensor unit. The RRD set has a
    single HD-EMG grid, but we simulate N of them — each gets its own
    ``device_id`` and its own SensorFrame stream.

    The stream preserves the dataset's native per-sample layout rather than
    flattening it: ``data`` has shape ``(n_samples, *feature_shape)``. For HD-EMG
    that is ``(n_samples, 4, 16)`` (the electrode grid), for bipolar EMG
    ``(n_samples, 16)``, for IMU ``(n_samples, 66)``. The *inference side* is
    responsible for any flattening/reshaping the model needs (e.g. 4x16 -> 64).
    ``channels`` is the flattened channel count = prod(feature_shape).
    """

    device_id: str
    modality: str
    feature_shape: tuple[int, ...]  # native per-sample shape, e.g. (4, 16)
    channels: int  # flattened channel count = prod(feature_shape)
    sample_rate_hz: float
    seq: int  # monotonically increasing chunk counter for this device
    t_send: float  # producer monotonic clock at send time (seconds)
    data: np.ndarray  # (n_samples, *feature_shape), float32
    # Ground-truth activity CODE for this chunk when replaying a labelled dataset
    # (None for real hardware / synthetic). Raw RRD code (e.g. 2.1), not an index;
    # the engine maps it to a class index/name via the ActivityRegistry.
    label: float | None = None
    # Provenance of the current chunk in the dataset (day/trial), for debugging.
    source_pos: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["feature_shape"] = list(self.feature_shape)
        d["data"] = self.data.astype(np.float32).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SensorFrame":
        d = dict(d)
        d["feature_shape"] = tuple(d["feature_shape"])
        d["data"] = np.asarray(d["data"], dtype=np.float32)
        return cls(**d)


@dataclass
class IntentPrediction:
    """The high-level controller's output to the mid-level controller.

    Per the agreed contract this carries the **full probability vector** plus the
    argmax and timing/provenance so the mid-level controller can apply its own
    thresholds / smoothing.
    """

    model_id: str
    seq: int
    t_infer: float  # monotonic clock when inference finished (seconds)
    probs: np.ndarray  # (num_classes,), float32, sums to 1
    argmax: int
    source_ids: list[str] = field(default_factory=list)  # devices that fed this
    label: str | None = None  # human-readable name of the raw argmax class
    # Debounced/smoothed intent for the controller (None if smoothing disabled).
    stable_index: int | None = None
    stable_label: str | None = None
    latency_ms: float | None = None  # window-newest-sample -> inference latency
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["probs"] = self.probs.astype(np.float32).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IntentPrediction":
        d = dict(d)
        d["probs"] = np.asarray(d["probs"], dtype=np.float32)
        return cls(**d)
