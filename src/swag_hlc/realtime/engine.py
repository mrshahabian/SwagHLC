"""Inference engine — one model, its input buffers, and its output topic.

Async-within: one task per input device drains its subscriber into a ring
buffer; one inference task runs the model on the latest windows at a capped
cadence and publishes ``IntentPrediction``.  Multiple engines (one per model)
run as separate processes for the hybrid procs+async design.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from swag_hlc.activities import build_registry
from swag_hlc.config import ModelConfig
from swag_hlc.messages import IntentPrediction, SensorFrame
from swag_hlc.realtime.buffers import RingBuffer
from swag_hlc.realtime.models import build_model
from swag_hlc.realtime.smoothing import Smoother
from swag_hlc.transport.base import Publisher, Subscriber


class InferenceEngine:
    def __init__(
        self,
        cfg: ModelConfig,
        subscribers: dict[str, Subscriber],
        publisher: Publisher,
        log=print,
    ) -> None:
        self.cfg = cfg
        self.model = build_model(cfg)
        # Maps raw ground-truth activity codes into this model's class space.
        self.registry = build_registry(cfg.activities, cfg.activity_names)
        self.subscribers = subscribers  # device_id -> Subscriber
        self.publisher = publisher
        self.log = log
        self._buffers: dict[str, RingBuffer] = {}
        self._rates: dict[str, float] = {}  # learned from incoming frames
        self._labels: dict[str, int | None] = {}  # latest ground-truth per device
        self._pos: dict[str, dict] = {}  # latest dataset position per device
        # Latency instrumentation (monotonic clock is shared across processes).
        self._newest_tsend: dict[str, float] = {}  # device -> t_send of newest frame
        self._recv_lag_ms: dict[str, float] = {}   # device -> transport recv lag
        self._last_infer_t: float | None = None
        self._seq = 0
        # Default inference cadence if not capped: 20 Hz is a sane controller rate.
        self.infer_rate_hz = float(cfg.infer_rate_hz or 20.0)

    def _buffer_for(self, frame: SensorFrame) -> RingBuffer:
        buf = self._buffers.get(frame.device_id)
        if buf is None:
            buf = RingBuffer(
                feature_shape=frame.feature_shape,
                window_size=self.cfg.window.window_size,
                chunk_hint=frame.data.shape[0],
            )
            self._buffers[frame.device_id] = buf
            self._rates[frame.device_id] = frame.sample_rate_hz
        return buf

    async def _drain(self, device_id: str, sub: Subscriber, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            frame = await loop.run_in_executor(None, sub.poll, 0.1)
            if frame is None:
                continue
            self._buffer_for(frame).append(frame.data)
            self._labels[frame.device_id] = frame.label
            self._newest_tsend[frame.device_id] = frame.t_send
            self._recv_lag_ms[frame.device_id] = (time.monotonic() - frame.t_send) * 1000.0
            if frame.source_pos:
                self._pos[frame.device_id] = frame.source_pos

    async def _infer_loop(self, stop: asyncio.Event) -> None:
        self.model.warmup()
        # Build the smoother now that warmup may have finalized num_classes.
        self.smoother = Smoother(self.model.num_classes, **self.cfg.smoothing)
        period = 1.0 / self.infer_rate_hz
        next_t = time.monotonic()
        while not stop.is_set():
            windows: dict[str, np.ndarray] = {}
            ready = True
            for dev in self.cfg.inputs:
                buf = self._buffers.get(dev)
                w = buf.latest_window() if buf is not None else None
                if w is None:
                    ready = False
                    break
                windows[dev] = w
            if ready:
                t_window = time.monotonic()
                probs = self.model.predict(windows)
                latency_ms = (time.monotonic() - t_window) * 1000.0
                if self.model.cfg.options.get("latency_ms"):
                    await asyncio.sleep(float(self.model.cfg.options["latency_ms"]) / 1000.0)
                argmax = int(np.argmax(probs))
                stable_idx = self.smoother.update(probs)  # debounced intent
                primary = self.cfg.inputs[0] if self.cfg.inputs else None
                true_code = self._labels.get(primary)
                true_index, true_name = (None, None)
                if true_code is not None:
                    true_index, true_name = self.registry.map_label(true_code)
                labels = self.model.labels
                # --- latency breakdown ---
                now = time.monotonic()
                newest = self._newest_tsend.get(primary)
                data_age_ms = (now - newest) * 1000.0 if newest is not None else None
                interval_ms = (now - self._last_infer_t) * 1000.0 if self._last_infer_t else None
                self._last_infer_t = now
                pred = IntentPrediction(
                    model_id=self.cfg.id,
                    seq=self._seq,
                    t_infer=time.monotonic(),
                    probs=probs,
                    argmax=argmax,
                    source_ids=list(self.cfg.inputs),
                    label=labels[argmax] if argmax < len(labels) else None,
                    stable_index=stable_idx,
                    stable_label=labels[stable_idx] if stable_idx < len(labels) else None,
                    latency_ms=latency_ms,
                    meta={
                        "device": self.cfg.device,
                        "true_code": true_code,
                        "true_index": true_index,
                        "true_name": true_name,
                        "correct": (true_index is not None and true_index == argmax),
                        "stable_correct": (true_index is not None and true_index == stable_idx),
                        "pos": self._pos.get(primary, {}),
                        "compute_ms": latency_ms,
                        "recv_lag_ms": self._recv_lag_ms.get(primary),
                        "data_age_ms": data_age_ms,
                        "interval_ms": interval_ms,
                    },
                )
                self.publisher.publish(pred)
                self._seq += 1
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
            else:
                next_t = time.monotonic()
                await asyncio.sleep(0)

    async def run(self, stop: asyncio.Event) -> None:
        tasks = [
            asyncio.create_task(self._drain(dev, sub, stop))
            for dev, sub in self.subscribers.items()
        ]
        tasks.append(asyncio.create_task(self._infer_loop(stop)))
        await asyncio.gather(*tasks)
