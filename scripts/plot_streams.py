"""Plot the three streamed modalities for a visual sanity check + print stats.

Pulls data the SAME way the live streamer does (via RrdReplaySource, native
shapes), then plots HD-EMG / bipolar-EMG / IMU-Gyr and the activity-label
timeline, and prints per-modality stats. Figures are written to --out.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/plot_streams.py \
        --subject MP301 --seconds 3 --out figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swag_hlc.config import SourceConfig
from swag_hlc.dummy_stream.rrd_replay_source import RrdReplaySource

RATE = 1000.0
BP_PARAMS = [
    "EMG_Left_AM", "EMG_Left_BF", "EMG_Left_GM", "EMG_Left_Gmax", "EMG_Left_RF",
    "EMG_Left_ST", "EMG_Left_TA", "EMG_Left_VL", "EMG_Right_AM", "EMG_Right_BF",
    "EMG_Right_GM", "EMG_Right_Gmax", "EMG_Right_RF", "EMG_Right_ST", "EMG_Right_TA", "EMG_Right_VL",
]
GYR_PARAMS = [
    "Gyr_Left_Foot", "Gyr_Left_Shank", "Gyr_Left_Thigh", "Gyr_Pelvis",
    "Gyr_Right_Foot", "Gyr_Right_Shank", "Gyr_Right_Thigh", "Gyr_T8",
]


class _Sink:
    def publish(self, msg):
        pass


def _collect(source: RrdReplaySource, n_samples: int):
    """Pull chunks from the live source until we have n_samples; keep labels too."""
    frames, labels = [], []
    got = 0
    while got < n_samples:
        chunk = source.next_chunk()
        frames.append(chunk)
        labels.append(np.full(chunk.shape[0], source._chunk_label, dtype=np.float32))
        got += chunk.shape[0]
    data = np.concatenate(frames, axis=0)[:n_samples]
    lab = np.concatenate(labels, axis=0)[:n_samples]
    return data, lab


def _src(subject, modality, params=None, chunk=100):
    cfg = SourceConfig(
        id=modality, modality=modality, params=params, chunk_size=chunk,
        generator="rrd_replay", subject=subject, days="all", trials="all",
    ).resolved()
    return RrdReplaySource(cfg, _Sink())


def _stats(name, data):
    print(f"  {name:8s} shape={tuple(data.shape)} "
          f"min={data.min():.3g} max={data.max():.3g} mean={data.mean():.3g} "
          f"nan={int(np.isnan(data).sum())} inf={int(np.isinf(data).sum())}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="MP301")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    n = int(args.seconds * RATE)
    t = np.arange(n) / RATE

    hd_src = _src(args.subject, "hd_emg")
    bp_src = _src(args.subject, "bp_emg", BP_PARAMS)
    gyr_src = _src(args.subject, "imu", GYR_PARAMS)

    hd, lab = _collect(hd_src, n)              # (n, 4, 16)
    bp, _ = _collect(bp_src, n)               # (n, 16)
    gyr, _ = _collect(gyr_src, n)             # (n, 24)

    print(f"Streamed {args.seconds}s ({n} samples) from {args.subject}:")
    _stats("HD-EMG", hd)
    _stats("BP-EMG", bp)
    _stats("IMU-Gyr", gyr)
    print(f"  labels present in window: {sorted(set(np.round(lab, 3).tolist()))}")

    # 1) HD-EMG: a few channels over time + a grid-activation heatmap snapshot.
    hd_flat = hd.reshape(n, -1)               # (n, 64)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for i, ch in enumerate(range(0, 64, 8)):
        ax[0].plot(t, hd_flat[:, ch] + i * 4 * hd_flat.std(), lw=0.5)
    ax[0].set(title=f"HD-EMG (8 of 64 ch, offset) — {args.subject}", xlabel="s")
    grid = np.abs(hd[: int(0.2 * RATE)]).mean(axis=0)   # mean |activation| over 0.2 s, (4,16)
    im = ax[1].imshow(grid, aspect="auto", cmap="viridis")
    ax[1].set(title="HD-EMG 4x16 grid mean |activation| (0.2 s)")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout(); fig.savefig(f"{args.out}/hd_emg.png", dpi=110); plt.close(fig)

    # 2) BP-EMG: 16 channels, offset.
    fig, ax = plt.subplots(figsize=(13, 5))
    off = 4 * np.nanstd(bp)
    for i in range(bp.shape[1]):
        ax.plot(t, bp[:, i] + i * off, lw=0.5)
    ax.set_yticks([i * off for i in range(16)]); ax.set_yticklabels(BP_PARAMS, fontsize=7)
    ax.set(title=f"Bipolar EMG (16 ch) — {args.subject}", xlabel="s")
    fig.tight_layout(); fig.savefig(f"{args.out}/bp_emg.png", dpi=110); plt.close(fig)

    # 3) IMU Gyr: 8 sensors x 3 axes, grouped.
    fig, ax = plt.subplots(figsize=(13, 5))
    off = 4 * np.nanstd(gyr)
    for i in range(gyr.shape[1]):
        ax.plot(t, gyr[:, i] + i * off, lw=0.5)
    ax.set_yticks([3 * j * off + off for j in range(8)]); ax.set_yticklabels(GYR_PARAMS, fontsize=7)
    ax.set(title=f"IMU gyroscopes (8x3=24) — {args.subject}", xlabel="s")
    fig.tight_layout(); fig.savefig(f"{args.out}/imu_gyr.png", dpi=110); plt.close(fig)

    # 4) Activity label timeline over the streamed window.
    fig, ax = plt.subplots(figsize=(13, 2.6))
    ax.step(t, lab, where="post")
    ax.set(title=f"Activity label (code) — {args.subject}", xlabel="s", ylabel="code")
    fig.tight_layout(); fig.savefig(f"{args.out}/labels.png", dpi=110); plt.close(fig)

    print(f"\nSaved figures to {os.path.abspath(args.out)}/: hd_emg.png, bp_emg.png, imu_gyr.png, labels.png")


if __name__ == "__main__":
    main()
