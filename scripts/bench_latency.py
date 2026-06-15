"""Benchmark the real-time pipeline latency across configurations.

Runs the HD-EMG CustomResNet model (GPU) on the streamed RRD data under a matrix
of {inference rate, overlap, majority-voting window} and prints a timing table:
measured compute / data-age / inference-interval, plus the derived buffering and
majority-voting reaction latencies.

    PYTHONPATH=src .venv/bin/python scripts/bench_latency.py
"""

from __future__ import annotations

import math

import swag_hlc.app as app
from swag_hlc.config import load_config

app.QUIET = True  # suppress per-prediction streaming lines

BASE = "configs/integration_customresnet.yaml"
RATE = 1000.0  # dataset sample rate (Hz)
SECONDS = 8.0


def _pct(xs, q):
    return app._pct(xs, q)


def reaction_ms(method, window, infer_period_ms):
    """Worst-case frames to flip the *stable* intent after a true change."""
    if method == "majority":
        return math.ceil((window + 1) / 2) * infer_period_ms  # need a new majority
    if method == "hold":
        return window * infer_period_ms  # min_count consecutive frames
    return infer_period_ms  # none/ema (ema is gradual; ~1 frame to start moving)


def run_row(label, infer_rate, overlap, smoothing, chunk_size=100):
    cfg = load_config(BASE)
    cfg.run.duration_s = SECONDS
    cfg.dataset.activities = [1, 2, 2.1, 3, 4]
    for s in cfg.sources:
        s.chunk_size = chunk_size
    m = cfg.models[0]
    m.infer_rate_hz = infer_rate
    m.window.window_size = 100
    m.window.hop = int(round(100 * (1 - overlap)))  # informational; engine uses latest window
    m.smoothing = smoothing
    res = app.run_process_mode(cfg)
    counts, totals, lat_sum, correct, stable_correct, labelled, lat = res
    mid = m.id
    L = lat[mid]
    n = labelled[mid] or 1
    infer_period = 1000.0 / infer_rate
    method = smoothing.get("method", "none")
    win = smoothing.get("window", smoothing.get("min_count", 0))
    return {
        "label": label,
        "compute50": _pct(L["compute"], .5), "compute95": _pct(L["compute"], .95),
        "age50": _pct(L["age"], .5), "age95": _pct(L["age"], .95),
        "intv50": _pct(L["intv"], .5),
        "chunk_fill": chunk_size / RATE * 1000,
        "window_span": 100 / RATE * 1000,       # window covers 100 ms of data
        "infer_period": infer_period,
        "reaction": reaction_ms(method, win, infer_period),
        "acc_raw": correct[mid] / n, "acc_stable": stable_correct[mid] / n,
    }


# (label, infer_rate, overlap, smoothing, chunk_size)
MATRIX = [
    ("20Hz ov5% noMV chunk100",   20, 0.05, {}, 100),
    ("20Hz ov5% MV5 chunk100",    20, 0.05, {"method": "majority", "window": 5}, 100),
    ("20Hz ov5% MV7 chunk100",    20, 0.05, {"method": "majority", "window": 7}, 100),
    ("20Hz ov5% MV10 chunk100",   20, 0.05, {"method": "majority", "window": 10}, 100),
    ("10Hz ov5% MV7 chunk100",    10, 0.05, {"method": "majority", "window": 7}, 100),
    ("20Hz NOov MV7 chunk100",    20, 0.00, {"method": "majority", "window": 7}, 100),
    ("20Hz ov5% MV7 chunk20",     20, 0.05, {"method": "majority", "window": 7}, 20),
    ("50Hz ov5% MV7 chunk20",     50, 0.05, {"method": "majority", "window": 7}, 20),
]


def main():
    rows = [run_row(*cfg) for cfg in MATRIX]
    h = (f"{'config':28} | {'compute p50/p95':14} | {'data_age p50/p95':16} | "
         f"{'infer dt':8} | {'react':6} | {'acc raw/stbl':12}")
    print("\n\n" + "=" * len(h))
    print("LATENCY TABLE  (HD-EMG CustomResNet on GPU, 1 kHz stream, window=100)")
    print("=" * len(h)); print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['label']:28} | {r['compute50']:5.1f}/{r['compute95']:5.1f} ms | "
              f"{r['age50']:6.0f}/{r['age95']:6.0f} ms | {r['intv50']:5.0f} ms | "
              f"{r['reaction']:4.0f}ms | {r['acc_raw']:.0%}/{r['acc_stable']:.0%}")
    print("-" * len(h))
    c = rows[0]
    print(f"fixed components: chunk_fill={c['chunk_fill']:.0f} ms (100 samp@1kHz), "
          f"window_span={c['window_span']:.0f} ms (data covered by one window)")
    print("compute = model forward; data_age = newest-sample -> prediction published; "
          "infer dt = actual gap between inferences; react = worst-case MV switch delay.")


if __name__ == "__main__":
    main()
