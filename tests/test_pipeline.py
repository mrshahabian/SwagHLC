"""Smoke + unit tests.

The numpy-only tests run anywhere; the RRD streaming test is skipped when h5py
or the dataset is unavailable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from swag_hlc.config import ModelConfig, SourceConfig, WindowConfig, load_config
from swag_hlc.dummy_stream import rrd_index
from swag_hlc.realtime.buffers import RingBuffer
from swag_hlc.realtime.models import build_model


def test_ring_buffer_latest_window():
    buf = RingBuffer(feature_shape=(2,), window_size=4, chunk_hint=3)
    buf.append(np.arange(6, dtype=np.float32).reshape(3, 2))
    buf.append(np.arange(6, 12, dtype=np.float32).reshape(3, 2))  # 6 total
    w = buf.latest_window()
    assert w is not None and w.shape == (4, 2)
    assert np.allclose(w, np.arange(4, 12, dtype=np.float32).reshape(4, 2))


def test_ring_buffer_preserves_native_grid_shape():
    buf = RingBuffer(feature_shape=(4, 16), window_size=8, chunk_hint=10)
    buf.append(np.random.randn(10, 4, 16).astype(np.float32))
    w = buf.latest_window()
    assert w is not None and w.shape == (8, 4, 16)  # HD-EMG grid kept intact


def test_source_expand_replicates_devices():
    expanded = SourceConfig(id="hd_emg", modality="hd_emg", count=3).expand()
    assert [c.id for c in expanded] == ["hd_emg_0", "hd_emg_1", "hd_emg_2"]
    assert all(c.channels == 64 for c in expanded)


def test_stub_model_returns_valid_distribution():
    cfg = ModelConfig(id="m", type="stub", num_classes=5, inputs=["d"], window=WindowConfig(8, 4))
    model = build_model(cfg)
    model.warmup()
    probs = model.predict({"d": np.random.randn(8, 4).astype(np.float32)})
    assert probs.shape == (5,) and np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert (probs >= 0).all()


def test_activity_registry_maps_codes():
    from swag_hlc.activities import build_registry

    reg = build_registry([1.0, 2.0, 2.1, 3.0, 4.0], {2.1: "walk_turn"})
    assert reg.num_classes == 5
    assert reg.index_of(2.1) == 2
    assert reg.map_label(2.1) == (2, "walk_turn")
    assert reg.map_label(9.0) == (None, "other(9.0)")  # valid code, not in class set


def test_registry_matches_bioactlab_label_map():
    """Selection is canonicalized to BioActLab's sorted LABEL_MAP convention."""
    from swag_hlc.activities import build_registry

    reg = build_registry([4, 1, 2.1, 2, 3])  # deliberately unsorted
    assert reg.codes == [1.0, 2.0, 2.1, 3.0, 4.0]
    assert reg.label_map() == {1.0: 0, 2.0: 1, 2.1: 2, 3.0: 3, 4.0: 4}


def test_registry_drops_invalid_codes():
    from swag_hlc.activities import build_registry

    reg = build_registry([1, 2, 999])  # 999 not in ORIGINAL_LABEL_MAP
    assert 999 not in reg.codes and reg.num_classes == 2


def test_majority_smoother_debounces_flicker():
    from swag_hlc.realtime.smoothing import Smoother

    sm = Smoother(num_classes=3, method="majority", window=5)
    onehot = lambda i: np.eye(3, dtype=np.float32)[i]
    for _ in range(4):
        sm.update(onehot(0))
    assert sm.update(onehot(1)) == 0  # one flicker can't outvote four 0s
    for _ in range(5):
        sm.update(onehot(1))
    assert sm.update(onehot(1)) == 1  # sustained change does switch


def test_hold_smoother_requires_persistence():
    from swag_hlc.realtime.smoothing import Smoother

    sm = Smoother(num_classes=3, method="hold", min_count=3)
    onehot = lambda i: np.eye(3, dtype=np.float32)[i]
    sm.update(onehot(0))
    assert sm.update(onehot(1)) == 0 and sm.update(onehot(1)) == 0  # not yet
    assert sm.update(onehot(1)) == 1  # third consecutive -> switch


def test_load_demo_config_resolves_activities():
    import pathlib

    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "demo_single_model.yaml"
    cfg = load_config(cfg_path)
    assert cfg.dataset.subject  # some subject is configured
    assert len(cfg.expanded_sources()) == 1
    m = cfg.active_models()[0]
    assert m.id == "stub_lstm"
    # class space derived from dataset.activities = [1, 2, 2.1, 3, 4]
    assert m.num_classes == 5 and m.activities == [1.0, 2.0, 2.1, 3.0, 4.0]
    # source inherits the activity filter
    assert cfg.expanded_sources()[0].activities == [1.0, 2.0, 2.1, 3.0, 4.0]


# --- RRD streaming (needs h5py + dataset) ---------------------------------- #
def _dataset_available() -> bool:
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    root = rrd_index.default_root()
    return root.exists() and bool(rrd_index.list_subjects(root))


@pytest.mark.skipif(not _dataset_available(), reason="RRD dataset / h5py unavailable")
def test_rrd_replay_streams_real_chunks():
    from swag_hlc.dummy_stream.rrd_replay_source import RrdReplaySource

    captured = []

    class _Pub:
        def publish(self, msg):
            captured.append(msg)

    # Pick a subject that actually has the HD-EMG grid param.
    subject = None
    for s in rrd_index.list_subjects(rrd_index.default_root()):
        day = rrd_index.list_days(rrd_index.default_root(), s)[0]
        tr = rrd_index.list_trials(rrd_index.default_root(), s, day)
        import h5py

        with h5py.File(rrd_index.subject_path(rrd_index.default_root(), s), "r") as f:
            if tr and "EMG_Right_MA" in f[day][tr[0]]:
                subject = s
                break
    if subject is None:
        pytest.skip("no subject with HD-EMG found")
    cfg = SourceConfig(
        id="hd_emg", modality="hd_emg", chunk_size=64, generator="rrd_replay",
        subject=subject, days="all", trials="all", loop=True,
    ).resolved()
    src = RrdReplaySource(cfg, _Pub())
    chunk = src.next_chunk()
    # Streamed in NATIVE shape: 64 samples x 4x16 HD-EMG grid (not pre-flattened).
    assert chunk.shape == (64, 4, 16)
    assert src.feature_shape == (4, 16)
    assert src._chunk_label is not None  # ground-truth label travels with stream


_CKPT = (
    "/vol/storage/common/SWAG/Integration_models/HD_EMG/"
    "CustomResNet_Day_1-Day_2-Day_3-Day_7_20260614_132709/100/ealry_stop_model.pth"
)


def _torch_ckpt_available() -> bool:
    import importlib.util
    import os

    have = all(importlib.util.find_spec(m) for m in ("torch", "torchvision"))
    return have and os.path.exists(_CKPT)


def test_emg_features_layout():
    from swag_hlc.realtime.features import emg_features

    win = np.random.randn(100, 16).astype(np.float32)
    feat = emg_features(win)
    assert feat.shape == (96,) and np.isfinite(feat).all()
    # channel-major: a constant channel has WL=0, ZC small, RMS=|c|
    win2 = np.zeros((50, 2), dtype=np.float32); win2[:, 1] = 5.0
    f2 = emg_features(win2)
    assert f2[0] == 0.0 and abs(f2[6 + 4] - 5.0) < 1e-4  # ch1 RMS == 5


def test_multimodal_config_loads():
    import pathlib

    p = pathlib.Path(__file__).resolve().parents[1] / "configs" / "multimodal_fusion.yaml"
    cfg = load_config(p)
    srcs = {s.id: s for s in cfg.expanded_sources()}
    assert set(srcs) == {"hd_emg", "bp_emg", "imu_gyr"}
    assert len(srcs["bp_emg"].params) == 16 and len(srcs["imu_gyr"].params) == 8
    assert [m.id for m in cfg.active_models()] == ["hd_emg", "imu", "bp_emg"]
    assert cfg.fusion and cfg.fusion.inputs == ["hd_emg", "imu", "bp_emg"]


def test_fusion_sum_normalizes():
    from swag_hlc.config import FusionConfig
    from swag_hlc.realtime.fusion import FusionNode

    fc = FusionConfig(id="f", inputs=["a", "b"], method="sum", activities=[1, 2, 2.1, 3, 4])
    node = FusionNode(fc, subscribers={}, publisher=None)
    node._latest = {"a": np.array([0.8, 0.1, 0.1, 0, 0], np.float32),
                    "b": np.array([0.2, 0.7, 0.1, 0, 0], np.float32)}
    fused = node._fuse()
    assert np.isclose(fused.sum(), 1.0, atol=1e-5) and int(np.argmax(fused)) == 0


@pytest.mark.skipif(not _torch_ckpt_available(), reason="torch/torchvision/checkpoint unavailable")
def test_real_customresnet_checkpoint_runs():
    from swag_hlc.config import ModelConfig, WindowConfig
    from swag_hlc.realtime.models import build_model

    cfg = ModelConfig(
        id="cr", type="torch", arch="CustomResNet", checkpoint=_CKPT,
        activities=[1.0, 2.0, 2.1, 3.0, 4.0], num_classes=5, inputs=["hd_emg"],
        window=WindowConfig(100, 95), device="cpu", options={"torch_threads": 4},
    )
    model = build_model(cfg)
    model.warmup()  # loads the real .pth into the reproduced architecture
    # native HD-EMG window (T, 4, 16) -> model flattens to (T, 64) internally
    probs = model.predict({"hd_emg": np.random.randn(100, 4, 16).astype(np.float32)})
    assert probs.shape == (5,) and np.isclose(probs.sum(), 1.0, atol=1e-5)
