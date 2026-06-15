"""Configuration schema + YAML loader.

Everything that varies between runs lives here.  The dummy stream now replays the
**real RRD dataset** by default: a top-level ``dataset:`` block sets the default
subject / days / trials / root, and each ``sources:`` entry streams one modality
from that selection (overridable per source).  ``models:`` configures one or more
inference models.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from swag_hlc.activities import build_registry, normalize_code
from swag_hlc.modalities import RRD_SAMPLE_RATE_HZ, get_modality_spec


def sensor_topic(device_id: str) -> str:
    return f"sensor/{device_id}"


def intent_topic(model_id: str) -> str:
    return f"intent/{model_id}"


@dataclass
class TransportConfig:
    kind: str = "inproc"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    duration_s: float = 5.0
    mode: str = "process"  # "process" (procs+async) | "async" (single process)


@dataclass
class DatasetConfig:
    """Default RRD selection that sources inherit unless they override it."""

    root: str = "/vol/storage/common/SWAG/RRD"
    subject: str = "MP201"  # MP2xx+ have HD-EMG; MP1xx are bipolar+IMU only
    days: Any = "all"  # "all" | [Day_1, ...] | [1, 2]
    trials: Any = "all"  # "all" | [Trial_01, ...] | [1, 2, 3]
    include_mvc: bool = False  # MVC trial is a calibration block, skipped by default
    include_nolabel: bool = False  # NoLabel_* recordings have no intent, skipped
    activities: Any = None  # None/"all" = stream every activity; else list of codes
    loop: bool = True  # loop the trial sequence to keep streaming


@dataclass
class SourceConfig:
    """One dummy sensor device (or a template for ``count`` of them)."""

    id: str
    modality: str
    channels: int | None = None
    rate_hz: float | None = None
    chunk_size: int = 100  # samples per published frame (100 @ 1 kHz = 10 ms)
    generator: str = "rrd_replay"  # "rrd_replay" (default) | "synthetic"
    count: int = 1  # replicate into id_0..id_{count-1} (e.g. N HD-EMG devices)
    # --- dataset selection (None => inherit from DatasetConfig) ---
    root: str | None = None
    subject: str | None = None
    days: Any = None
    trials: Any = None
    params: list[str] | None = None  # explicit HDF5 dataset names (override modality)
    activities: Any = None  # None = inherit dataset; "all" = no filter; else codes
    loop: bool | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def with_dataset_defaults(self, ds: DatasetConfig) -> "SourceConfig":
        acts = self.activities if self.activities is not None else ds.activities
        if acts in ("all", ["all"]):
            acts = None
        return replace(
            self,
            root=self.root if self.root is not None else ds.root,
            subject=self.subject if self.subject is not None else ds.subject,
            days=self.days if self.days is not None else ds.days,
            trials=self.trials if self.trials is not None else ds.trials,
            activities=acts,
            loop=self.loop if self.loop is not None else ds.loop,
            options={
                "include_mvc": ds.include_mvc,
                "include_nolabel": ds.include_nolabel,
                **dict(self.options),
            },
        )

    def resolved(self) -> "SourceConfig":
        spec = get_modality_spec(self.modality)
        ch = self.channels if self.channels is not None else spec.channels
        rate = self.rate_hz if self.rate_hz is not None else spec.default_rate_hz
        return replace(self, channels=ch, rate_hz=rate)

    def expand(self) -> list["SourceConfig"]:
        base = self.resolved()
        if base.count <= 1:
            return [replace(base, count=1)]
        return [replace(base, id=f"{base.id}_{i}", count=1) for i in range(base.count)]


@dataclass
class WindowConfig:
    window_size: int = 100
    hop: int = 50


@dataclass
class ModelConfig:
    id: str
    type: str = "stub"
    active: bool = True
    num_classes: int = 5
    labels: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    window: WindowConfig = field(default_factory=WindowConfig)
    infer_rate_hz: float | None = None
    device: str = "cpu"
    # Activity class space this model predicts (raw RRD codes). When set, it drives
    # num_classes and labels via the ActivityRegistry. None => use num_classes/labels.
    activities: list[float] | None = None
    activity_names: dict[float, str] = field(default_factory=dict)
    # For type: torch — path to a BioActLab .pth (or run dir) and the arch name.
    checkpoint: str | None = None
    arch: str | None = None
    # Intent smoothing for the controller: {method, window, alpha, min_count, ...}
    smoothing: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionConfig:
    """High-level (late) fusion of several models' probability outputs."""

    id: str = "fusion"
    inputs: list[str] = field(default_factory=list)  # model ids to fuse
    method: str = "sum"  # sum | weighted (both renormalized to a distribution)
    weights: list[float] = field(default_factory=list)
    rate_hz: float = 10.0
    activities: list[float] | None = None
    activity_names: dict[float, str] = field(default_factory=dict)
    smoothing: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    transport: TransportConfig = field(default_factory=TransportConfig)
    run: RunConfig = field(default_factory=RunConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sources: list[SourceConfig] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    fusion: "FusionConfig | None" = None
    activity_names: dict[float, str] = field(default_factory=dict)

    def expanded_sources(self) -> list[SourceConfig]:
        out: list[SourceConfig] = []
        for s in self.sources:
            out.extend(s.with_dataset_defaults(self.dataset).expand())
        return out

    def active_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.active]


def _window_from(d: dict[str, Any] | None) -> WindowConfig:
    d = d or {}
    return WindowConfig(window_size=int(d.get("window_size", 100)), hop=int(d.get("hop", 50)))


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    t = raw.get("transport", {}) or {}
    transport = TransportConfig(kind=t.get("kind", "inproc"), options=t.get("options", {}) or {})

    r = raw.get("run", {}) or {}
    run = RunConfig(duration_s=float(r.get("duration_s", 5.0)), mode=r.get("mode", "process"))

    activity_names = {
        normalize_code(k): v for k, v in (raw.get("activity_names") or {}).items()
    }

    d = raw.get("dataset", {}) or {}
    dataset = DatasetConfig(
        root=d.get("root", DatasetConfig.root),
        subject=d.get("subject", DatasetConfig.subject),
        days=d.get("days", "all"),
        trials=d.get("trials", "all"),
        include_mvc=bool(d.get("include_mvc", False)),
        include_nolabel=bool(d.get("include_nolabel", False)),
        activities=d.get("activities"),
        loop=bool(d.get("loop", True)),
    )

    sources = [
        SourceConfig(
            id=s["id"], modality=s.get("modality", ""), channels=s.get("channels"),
            rate_hz=s.get("rate_hz"), chunk_size=int(s.get("chunk_size", 100)),
            generator=s.get("generator", "rrd_replay"), count=int(s.get("count", 1)),
            root=s.get("root"), subject=s.get("subject"), days=s.get("days"),
            trials=s.get("trials"), params=s.get("params"),
            activities=s.get("activities"), loop=s.get("loop"),
            options=s.get("options", {}) or {},
        )
        for s in raw.get("sources", []) or []
    ]

    ds_acts = dataset.activities if dataset.activities not in (None, "all", ["all"]) else None
    models = []
    for m in raw.get("models", []) or []:
        acts = m.get("activities")
        if acts in ("all", ["all"]):
            acts = None
        if acts is None:
            acts = ds_acts  # inherit the dataset's class set when unspecified
        num_classes = int(m.get("num_classes", 5))
        labels = m.get("labels", []) or []
        if acts:  # registry drives the class space (canonical, dataloader-matching)
            reg = build_registry([float(a) for a in acts], activity_names)
            num_classes = reg.num_classes
            labels = labels or reg.labels
            acts = reg.codes  # canonicalized + validated order
        models.append(
            ModelConfig(
                id=m["id"], type=m.get("type", "stub"), active=bool(m.get("active", True)),
                num_classes=num_classes, labels=labels, inputs=m.get("inputs", []) or [],
                window=_window_from(m.get("window")), infer_rate_hz=m.get("infer_rate_hz"),
                device=m.get("device", "cpu"),
                activities=[float(a) for a in acts] if acts else None,
                activity_names=activity_names,
                checkpoint=m.get("checkpoint"), arch=m.get("arch"),
                smoothing=m.get("smoothing", {}) or {},
                options=m.get("options", {}) or {},
            )
        )

    fusion = None
    fblk = raw.get("fusion")
    if fblk:
        f_acts = fblk.get("activities")
        if f_acts in (None, "all", ["all"]):
            f_acts = ds_acts  # default to the dataset's class set
        fusion = FusionConfig(
            id=fblk.get("id", "fusion"), inputs=fblk.get("inputs", []) or [],
            method=fblk.get("method", "sum"), weights=fblk.get("weights", []) or [],
            rate_hz=float(fblk.get("rate_hz", 10.0)),
            activities=[float(a) for a in f_acts] if f_acts else None,
            activity_names=activity_names, smoothing=fblk.get("smoothing", {}) or {},
        )

    return AppConfig(
        transport=transport, run=run, dataset=dataset, sources=sources,
        models=models, fusion=fusion, activity_names=activity_names,
    )
