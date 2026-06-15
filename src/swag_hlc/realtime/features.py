"""Handcrafted EMG features — faithful reproduction of BioActLab's
``EMGFeatureExtractor`` (utils/feature_extraction.py).

For a window ``(T, C)`` it produces a ``(C*6,)`` vector, channel-major, with the
6 features per channel in the exact order BioActLab uses:
``[WL, MAV, VAR, ZC, RMS, IEMG]``. This matches the trained EMGCNN's 96-dim input
(16 bipolar channels x 6) and the feature ordering in its ``params.json``.
"""

from __future__ import annotations

import numpy as np

FEATURE_TYPES = ("WL", "MAV", "VAR", "ZC", "RMS", "IEMG")
NUM_FEATURES = len(FEATURE_TYPES)


def emg_features(window: np.ndarray, zc_threshold: float = 0.0) -> np.ndarray:
    """``(T, C)`` raw EMG window -> ``(C*6,)`` feature vector (channel-major)."""
    if window.ndim != 2:
        raise ValueError(f"expected 2D (T, C) window, got {window.shape}")
    t, c = window.shape
    out = np.zeros(c * NUM_FEATURES, dtype=np.float32)
    for ch in range(c):
        s = window[:, ch].astype(np.float64)
        wl = float(np.sum(np.abs(np.diff(s)))) if t > 1 else 0.0          # waveform length
        mav = float(np.mean(np.abs(s))) if t else 0.0                      # mean abs value
        var = float(np.var(s)) if t else 0.0                               # variance
        zc = float(np.sum(np.diff(np.sign(s - zc_threshold)) != 0)) if t > 1 else 0.0  # zero crossings
        rms = float(np.sqrt(np.mean(s ** 2))) if t else 0.0               # root mean square
        iemg = float(np.sum(np.abs(s)))                                    # integrated EMG
        out[ch * NUM_FEATURES : (ch + 1) * NUM_FEATURES] = (wl, mav, var, zc, rms, iemg)
    return out
