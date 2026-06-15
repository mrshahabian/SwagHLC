"""Orchestrator / CLI — launch dummy streams + inference engines + a monitor.

Run:
    PYTHONPATH=src python -m swag_hlc.app --config configs/demo_single_model.yaml

Two run modes (config ``run.mode`` or ``--mode``):
  * ``process`` — each dummy source and each model runs as its own OS process,
    each with its own asyncio loop (the hybrid procs+async design). Default.
  * ``async``   — everything in one process on one event loop (simplest).

The monitor subscribes to every model's intent topic and prints predictions,
then prints a per-model class summary at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import multiprocessing as mp
import time

from swag_hlc.config import AppConfig, intent_topic, load_config, sensor_topic
from swag_hlc.dummy_stream import build_source
from swag_hlc.realtime.engine import InferenceEngine
from swag_hlc.realtime.fusion import FusionNode
from swag_hlc.transport import build_transport
from swag_hlc.transport.base import Subscriber

# Suppress per-prediction streaming lines (set True by the latency benchmark).
QUIET = False


# --------------------------------------------------------------------------- #
# async bodies (shared by both run modes)
# --------------------------------------------------------------------------- #
async def _watch_stop(mp_stop, stop: asyncio.Event) -> None:
    while not mp_stop.is_set():
        await asyncio.sleep(0.05)
    stop.set()


async def _run_sources(source_cfgs, publishers, stop: asyncio.Event) -> None:
    sources = [build_source(c, publishers[c.id]) for c in source_cfgs]
    await asyncio.gather(*(s.run(stop) for s in sources))


async def _run_engine(model_cfg, subscribers, publisher, stop: asyncio.Event) -> None:
    engine = InferenceEngine(model_cfg, subscribers, publisher)
    await engine.run(stop)


# --------------------------------------------------------------------------- #
# process targets (module-level so they survive fork/spawn)
# --------------------------------------------------------------------------- #
def _sources_proc(source_cfgs, publishers, mp_stop) -> None:
    async def main():
        stop = asyncio.Event()
        await asyncio.gather(
            _watch_stop(mp_stop, stop),
            _run_sources(source_cfgs, publishers, stop),
        )

    asyncio.run(main())


def _engine_proc(model_cfg, subscribers, publisher, mp_stop) -> None:
    async def main():
        stop = asyncio.Event()
        await asyncio.gather(
            _watch_stop(mp_stop, stop),
            _run_engine(model_cfg, subscribers, publisher, stop),
        )

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# monitor
# --------------------------------------------------------------------------- #
def _monitor(intent_subs: dict[str, Subscriber], deadline: float, throttle_s: float = 0.25):
    counts: dict[str, collections.Counter] = {m: collections.Counter() for m in intent_subs}
    totals: collections.Counter = collections.Counter()
    last_print: dict[str, float] = {m: 0.0 for m in intent_subs}
    lat_sum: dict[str, float] = collections.defaultdict(float)
    correct: collections.Counter = collections.Counter()
    stable_correct: collections.Counter = collections.Counter()
    labelled: collections.Counter = collections.Counter()
    lat: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: {"compute": [], "age": [], "recv": [], "intv": []}
    )
    while time.monotonic() < deadline:
        any_msg = False
        for model_id, sub in intent_subs.items():
            pred = sub.poll(timeout=0.02)
            if pred is None:
                continue
            any_msg = True
            counts[model_id][pred.stable_label or pred.label or pred.argmax] += 1
            totals[model_id] += 1
            lat_sum[model_id] += pred.latency_ms or 0.0
            if pred.meta.get("true_index") is not None:
                labelled[model_id] += 1
                correct[model_id] += int(pred.meta.get("correct", False))
                stable_correct[model_id] += int(pred.meta.get("stable_correct", False))
            m = pred.meta
            for key, mk in (("compute", "compute_ms"), ("age", "data_age_ms"),
                            ("recv", "recv_lag_ms"), ("intv", "interval_ms")):
                if m.get(mk) is not None:
                    lat[model_id][key].append(float(m[mk]))
            now = time.monotonic()
            if not QUIET and now - last_print[model_id] >= throttle_s:
                last_print[model_id] = now
                top = ", ".join(f"{p:.2f}" for p in pred.probs)
                tname = pred.meta.get("true_name")
                pos = pred.meta.get("pos") or {}
                where = f" @{pos.get('trial')}" if pos.get("trial") else ""
                truth = f" true={tname}" if tname is not None else ""
                stable = f" stable={pred.stable_label}" if pred.stable_label is not None else ""
                votes = pred.meta.get("votes")
                votes_s = f" votes={votes}" if votes else ""
                print(
                    f"[{model_id}] seq={pred.seq:>4} pred={pred.label}{stable}{truth} "
                    f"p=[{top}]{votes_s}{where}"
                )
        if not any_msg:
            time.sleep(0.005)
    return counts, totals, lat_sum, correct, stable_correct, labelled, lat


def _pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[i]


def _print_summary(counts, totals, lat_sum, correct, stable_correct, labelled, lat) -> None:
    print("\n=== summary ===")
    for model_id, total in totals.items():
        if total == 0:
            print(f"  {model_id}: no predictions")
            continue
        dist = ", ".join(f"{c}:{n}" for c, n in counts[model_id].most_common())
        acc = ""
        if labelled[model_id]:
            n = labelled[model_id]
            acc = (
                f", acc(raw)={correct[model_id] / n:.0%} "
                f"acc(stable)={stable_correct[model_id] / n:.0%}"
            )
        L = lat[model_id]
        comp, age, recv, intv = L["compute"], L["age"], L["recv"], L["intv"]
        print(f"  {model_id}: {total} preds{acc}")
        if comp:  # inference engines report a latency breakdown; the fusion node doesn't
            print(f"      latency ms  compute p50/p95={_pct(comp,.5):.1f}/{_pct(comp,.95):.1f}"
                  f" | data_age p50/p95={_pct(age,.5):.0f}/{_pct(age,.95):.0f}"
                  f" | recv_lag p50={_pct(recv,.5):.1f}"
                  f" | infer_interval p50={_pct(intv,.5):.0f}")
        print(f"      stable intent counts: {{{dist}}}")


async def _run_fusion(fcfg, subscribers, publisher, stop: asyncio.Event) -> None:
    await FusionNode(fcfg, subscribers, publisher).run(stop)


def _fusion_proc(fcfg, subscribers, publisher, mp_stop) -> None:
    async def main():
        stop = asyncio.Event()
        await asyncio.gather(_watch_stop(mp_stop, stop), _run_fusion(fcfg, subscribers, publisher, stop))

    asyncio.run(main())


# --------------------------------------------------------------------------- #
# run modes
# --------------------------------------------------------------------------- #
def _wire(cfg: AppConfig, transport):
    """Static wiring: declare topics, create subscribers, then publishers.

    Model intent topics may have TWO subscribers (the fusion node + the monitor);
    the in-process transport fans out to both.
    """
    sources = cfg.expanded_sources()
    models = cfg.active_models()
    fusion = cfg.fusion

    # Fail fast on wiring mistakes (otherwise an engine silently produces nothing).
    source_ids = {s.id for s in sources}
    for m in models:
        missing = [d for d in m.inputs if d not in source_ids]
        if missing:
            raise ValueError(
                f"model '{m.id}' inputs {missing} have no matching source. "
                f"Available device ids: {sorted(source_ids)} "
                f"(note: a source with count=N expands to <id>_0..<id>_{{N-1}})."
            )
    if fusion:
        model_ids = {m.id for m in models}
        missing = [mid for mid in fusion.inputs if mid not in model_ids]
        if missing:
            raise ValueError(
                f"fusion inputs {missing} are not active model ids. Available: {sorted(model_ids)}."
            )
    # Validate torch checkpoints up front (a child crash here would otherwise
    # silently starve the fusion node, which waits for every model).
    from swag_hlc.realtime.models.torch_model import resolve_checkpoint
    for m in models:
        if m.type == "torch":
            ckpt = m.checkpoint or m.options.get("checkpoint")
            try:
                resolve_checkpoint(ckpt) if ckpt else None
            except (FileNotFoundError, ValueError) as e:
                raise ValueError(f"model '{m.id}' checkpoint: {e}") from e

    for s in sources:
        transport.declare_topic(sensor_topic(s.id))
    for m in models:
        transport.declare_topic(intent_topic(m.id))
    if fusion:
        transport.declare_topic(intent_topic(fusion.id))

    # --- Subscribers FIRST ---
    engine_subs = {m.id: {dev: transport.subscriber(sensor_topic(dev)) for dev in m.inputs} for m in models}
    fusion_subs = {}
    if fusion:
        fusion_subs = {mid: transport.subscriber(intent_topic(mid)) for mid in fusion.inputs}
    # The monitor watches every model + the fused output.
    monitor_ids = [m.id for m in models] + ([fusion.id] if fusion else [])
    monitor_subs = {mid: transport.subscriber(intent_topic(mid)) for mid in monitor_ids}

    # --- Publishers AFTER all subscribers exist (publishers snapshot subscribers) ---
    source_pubs = {s.id: transport.publisher(sensor_topic(s.id)) for s in sources}
    intent_pubs = {m.id: transport.publisher(intent_topic(m.id)) for m in models}
    fusion_pub = transport.publisher(intent_topic(fusion.id)) if fusion else None
    return {
        "sources": sources, "models": models, "fusion": fusion,
        "engine_subs": engine_subs, "fusion_subs": fusion_subs, "monitor_subs": monitor_subs,
        "source_pubs": source_pubs, "intent_pubs": intent_pubs, "fusion_pub": fusion_pub,
    }


def run_process_mode(cfg: AppConfig) -> None:
    ctx = mp.get_context("fork")
    transport = build_transport(cfg.transport.kind, ctx=ctx, **cfg.transport.options)
    w = _wire(cfg, transport)

    mp_stop = ctx.Event()
    procs: list[mp.process.BaseProcess] = []
    for s in w["sources"]:  # one process per source (each its own device)
        procs.append(ctx.Process(target=_sources_proc, args=([s], {s.id: w["source_pubs"][s.id]}, mp_stop)))
    for m in w["models"]:  # one process per active model (GPU-per-model, isolated)
        procs.append(ctx.Process(target=_engine_proc, args=(m, w["engine_subs"][m.id], w["intent_pubs"][m.id], mp_stop)))
    if w["fusion"]:  # one process for high-level fusion
        procs.append(ctx.Process(target=_fusion_proc, args=(w["fusion"], w["fusion_subs"], w["fusion_pub"], mp_stop)))

    fuse_msg = " + 1 fusion proc" if w["fusion"] else ""
    print(
        f"Launching {len(w['sources'])} source proc(s) + {len(w['models'])} engine proc(s)"
        f"{fuse_msg} for {cfg.run.duration_s:.1f}s (transport={cfg.transport.kind})\n"
    )
    for p in procs:
        p.start()
    try:
        deadline = time.monotonic() + cfg.run.duration_s
        res = _monitor(w["monitor_subs"], deadline)
        _print_summary(*res)
        return res
    finally:
        mp_stop.set()
        for p in procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()


def run_async_mode(cfg: AppConfig) -> None:
    transport = build_transport(cfg.transport.kind, **cfg.transport.options)
    w = _wire(cfg, transport)

    async def main():
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()

        async def monitor_until():
            deadline = time.monotonic() + cfg.run.duration_s
            res = await loop.run_in_executor(None, _monitor, w["monitor_subs"], deadline)
            stop.set()
            return res

        tasks = [asyncio.create_task(_run_sources(w["sources"], w["source_pubs"], stop))]
        for m in w["models"]:
            tasks.append(asyncio.create_task(_run_engine(m, w["engine_subs"][m.id], w["intent_pubs"][m.id], stop)))
        if w["fusion"]:
            tasks.append(asyncio.create_task(_run_fusion(w["fusion"], w["fusion_subs"], w["fusion_pub"], stop)))
        mon = asyncio.create_task(monitor_until())
        await mon
        _print_summary(*mon.result())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    fuse_msg = " + fusion" if w["fusion"] else ""
    print(
        f"Launching {len(w['sources'])} source(s) + {len(w['models'])} engine(s){fuse_msg} (async) "
        f"for {cfg.run.duration_s:.1f}s (transport={cfg.transport.kind})\n"
    )
    asyncio.run(main())


def _parse_list_tokens(tokens: list[str]):
    """['all'] -> 'all'; ['1','Trial_05'] -> [1, 'Trial_05']."""
    if tokens == ["all"]:
        return "all"
    out = []
    for t in tokens:
        try:
            out.append(int(t))
        except ValueError:
            try:
                out.append(float(t))
            except ValueError:
                out.append(t)
    return out


def _apply_selection_overrides(cfg, args) -> None:
    """Apply CLI dataset/activity selection over the loaded config.

    Sources inherit dataset defaults at expansion time, so updating the dataset
    block is enough for them; models resolved their class space at load time, so
    re-resolve them when --activities changes.
    """
    if args.subject is not None:
        cfg.dataset.subject = args.subject
    if args.days is not None:
        cfg.dataset.days = _parse_list_tokens(args.days)
    if args.trials is not None:
        cfg.dataset.trials = _parse_list_tokens(args.trials)
    if args.activities is not None:
        # This is a STREAM filter only — it does NOT change a model's class space.
        # A trained model's output classes are fixed by its checkpoint; you can
        # stream a subset of activities through it, but you can't re-shape its head
        # from the CLI. (To change the class space, edit the model's `activities`.)
        cfg.dataset.activities = _parse_list_tokens(args.activities)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SwagHLC dummy real-time inference demo")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--duration", type=float, default=None, help="Override run.duration_s")
    ap.add_argument("--mode", choices=["process", "async"], default=None, help="Override run.mode")
    ap.add_argument("--subject", default=None, help="Override dataset.subject")
    ap.add_argument("--days", nargs="+", default=None, help="Override dataset.days (or 'all')")
    ap.add_argument("--trials", nargs="+", default=None, help="Override dataset.trials (ints/names/'all')")
    ap.add_argument("--activities", nargs="+", default=None,
                    help="Override the streamed/classified activity codes (or 'all')")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.duration is not None:
        cfg.run.duration_s = args.duration
    if args.mode is not None:
        cfg.run.mode = args.mode
    _apply_selection_overrides(cfg, args)

    if cfg.run.mode == "async":
        run_async_mode(cfg)
    else:
        run_process_mode(cfg)


if __name__ == "__main__":
    main()
