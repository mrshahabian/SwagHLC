"""High-level (late) fusion of multiple model probability streams.

Subscribes to several models' ``IntentPrediction`` topics, keeps the latest
probability vector from each, and combines them into one fused distribution over
the shared class space (the sum rule, optionally weighted) — then publishes a
fused ``IntentPrediction`` for the controller. Per-model argmaxes are kept in
``meta`` so the monitor can show every model's vote next to the fused result.

Runs as its own process/async task; multi-rate inputs are handled by the
"keep latest from each model" policy and emitting on a fixed cadence.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from swag_hlc.activities import build_registry
from swag_hlc.config import FusionConfig
from swag_hlc.messages import IntentPrediction
from swag_hlc.realtime.smoothing import Smoother
from swag_hlc.transport.base import Publisher, Subscriber


class FusionNode:
    def __init__(self, cfg: FusionConfig, subscribers: dict[str, Subscriber], publisher: Publisher) -> None:
        self.cfg = cfg
        self.subscribers = subscribers  # model_id -> Subscriber on intent/<model_id>
        self.publisher = publisher
        self.registry = build_registry(cfg.activities, cfg.activity_names)
        self.num_classes = self.registry.num_classes
        self.labels = self.registry.labels
        n = len(cfg.inputs)
        w = cfg.weights or [1.0] * n
        self._weights = {mid: float(w[i]) for i, mid in enumerate(cfg.inputs)}
        self._latest: dict[str, np.ndarray] = {}  # model_id -> probs
        self._latest_argmax: dict[str, int] = {}
        self._true_index: int | None = None
        self._true_name: str | None = None
        self._seq = 0

    async def _drain(self, model_id: str, sub: Subscriber, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            pred = await loop.run_in_executor(None, sub.poll, 0.1)
            if pred is None:
                continue
            self._latest[model_id] = np.asarray(pred.probs, dtype=np.float32)
            self._latest_argmax[model_id] = int(pred.argmax)
            ti = pred.meta.get("true_index")
            if ti is not None:  # all models share ground truth; cache the latest
                self._true_index, self._true_name = ti, pred.meta.get("true_name")

    def _fuse(self) -> np.ndarray:
        acc = np.zeros(self.num_classes, dtype=np.float64)
        for mid, probs in self._latest.items():
            if probs.shape[0] != self.num_classes:
                continue
            acc += self._weights.get(mid, 1.0) * probs
        s = acc.sum()
        if s <= 0:
            return np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float32)
        return (acc / s).astype(np.float32)  # sum rule, renormalized to a distribution

    async def _fuse_loop(self, stop: asyncio.Event) -> None:
        smoother = Smoother(self.num_classes, **self.cfg.smoothing)
        period = 1.0 / float(self.cfg.rate_hz)
        next_t = time.monotonic()
        while not stop.is_set():
            if len(self._latest) == len(self.cfg.inputs):  # all models have reported
                probs = self._fuse()
                argmax = int(np.argmax(probs))
                stable = smoother.update(probs)
                pred = IntentPrediction(
                    model_id=self.cfg.id,
                    seq=self._seq,
                    t_infer=time.monotonic(),
                    probs=probs,
                    argmax=argmax,
                    source_ids=list(self.cfg.inputs),
                    label=self.labels[argmax] if argmax < len(self.labels) else None,
                    stable_index=stable,
                    stable_label=self.labels[stable] if stable < len(self.labels) else None,
                    meta={
                        "fusion": self.cfg.method,
                        "votes": dict(self._latest_argmax),  # per-model argmax
                        "true_index": self._true_index,
                        "true_name": self._true_name,
                        "correct": (self._true_index is not None and self._true_index == argmax),
                        "stable_correct": (self._true_index is not None and self._true_index == stable),
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
        tasks = [asyncio.create_task(self._drain(m, s, stop)) for m, s in self.subscribers.items()]
        tasks.append(asyncio.create_task(self._fuse_loop(stop)))
        await asyncio.gather(*tasks)
