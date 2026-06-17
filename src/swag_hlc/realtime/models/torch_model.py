"""Real trained-model inference — loads a BioActLab checkpoint and runs it.

Faithfully reproduces BioActLab's ``CustomResNet`` (window_cls) so a ``.pth``
trained there runs unchanged here:
  * architecture rebuilt with identical attribute names so the raw ``state_dict``
    loads (``resnet``/``dropout``/``fc``);
  * input tensor ``[B, 1, T_win, F]`` (float32), built by flattening the native
    stream window on the inference side (HD-EMG (T,4,16) -> (T,64));
  * **no normalization** (BioActLab trained/evaluated this model with
    ``normalization=None`` — windows are used raw);
  * labels {1,2,2.1,3,4} -> {0,1,2,3,4}, matching the ActivityRegistry order.

Everything torch is imported lazily so the rest of the package runs without it.
"""

from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np

from swag_hlc.config import ModelConfig
from swag_hlc.realtime.models.base import InferenceModel

log = logging.getLogger(__name__)

# Preferred checkpoint filenames when a run *directory* is given.
_CKPT_PREFERENCE = ("ealry_stop", "early_stop", "best", "final")


def resolve_checkpoint(path: str) -> tuple[str, dict]:
    """Accept a .pth file OR a run directory and return (pth_path, params_meta).

    BioActLab runs look like ``<run>/<window>/ealry_stop_model.pth`` with a
    ``params.json`` in ``<run>/``. Pointing at the **run dir**, the **window dir**,
    or the **.pth** all work — we find the checkpoint (searching one level into
    window subfolders if needed) and the params.json.
    """
    if os.path.isdir(path):
        cands = sorted(glob.glob(os.path.join(path, "*.pth")))
        if not cands:  # e.g. run dir -> look inside window subfolders (100/, ...)
            cands = sorted(glob.glob(os.path.join(path, "*", "*.pth")))
        if not cands:
            raise FileNotFoundError(
                f"No .pth in {path} or its immediate subfolders (expected e.g. 100/ealry_stop_model.pth)"
            )
        pth = next((c for key in _CKPT_PREFERENCE for c in cands if key in os.path.basename(c).lower()), cands[0])
    else:
        pth = path
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Checkpoint not found: {pth}")
    # params.json lives next to the .pth or in the parent run dir.
    d = os.path.dirname(os.path.abspath(pth))
    meta: dict = {}
    for cand in (os.path.join(d, "params.json"), os.path.join(os.path.dirname(d), "params.json")):
        if os.path.exists(cand):
            with open(cand) as f:
                meta = json.load(f)
            break
    return pth, meta


def _build_custom_resnet(num_classes: int, meta: dict):
    """Mirror of BioActLab models/custom_models.py CustomResNet (weights=None:
    we load trained weights, so no pretrained download is needed).

    torchvision dependency removed: ResNet-18 reimplemented in pure torch.nn with
    identical layer names so BioActLab .pth state dicts load without modification.
    Only weights=None was ever used here — no pretrained weights are needed.
    """
    import torch.nn as nn

    class BasicBlock(nn.Module):
        def __init__(self, in_planes, planes, stride=1, downsample=None):
            super().__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            identity = self.downsample(x) if self.downsample is not None else x
            return self.relu(out + identity)

    def _make_layer(in_planes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(in_planes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    class ResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = _make_layer(64, 64, 2)
            self.layer2 = _make_layer(64, 128, 2, stride=2)
            self.layer3 = _make_layer(128, 256, 2, stride=2)
            self.layer4 = _make_layer(256, 512, 2, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 1000)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            return x.flatten(1)

    class CustomResNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            dropout_rate = 0.01
            self.resnet = ResNet18()
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            num_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()
            self.dropout = nn.Dropout(p=dropout_rate)
            self.fc = nn.Linear(num_features, num_classes)

        def forward(self, x):
            x = self.resnet(x)
            x = self.dropout(x)
            return self.fc(x)

    return CustomResNet(num_classes)


def _build_emgcnn(num_classes: int, meta: dict):
    """Mirror of BioActLab models/custom_models.py EMGCNN (1D-CNN over the
    per-window feature vector). input_size = total_dims (e.g. 96 = 16ch x 6)."""
    import torch
    import torch.nn as nn

    input_size = int(meta.get("total_dims") or 96)
    p = {"kernel_size": 3, "pool_size": 2, "dropout": 0.2}

    class EMGCNN(nn.Module):
        def __init__(self, input_size, num_classes, parameters):
            super().__init__()
            self.input_size = input_size
            self.conv1 = nn.Conv1d(1, 32, kernel_size=parameters["kernel_size"], stride=1, padding=2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=parameters["kernel_size"], stride=1, padding=2)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=parameters["kernel_size"], stride=1, padding=2)
            self.fc1 = None
            self.fc2 = nn.Linear(256, num_classes)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(parameters["pool_size"])
            self.dropout = nn.Dropout(parameters.get("dropout", 0.2))
            with torch.no_grad():
                x = torch.zeros(1, 1, input_size)
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                flat = x.view(1, -1).size(1)
            self.fc1 = nn.Linear(flat, 256)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            elif x.dim() == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = self.dropout(self.relu(self.fc1(x)))
            return self.fc2(x)

    return EMGCNN(input_size, num_classes, p)


# arch name (params.json "model") -> builder(num_classes, meta)
ARCH_BUILDERS = {
    "CustomResNet": _build_custom_resnet,
    "EMGCNN": _build_emgcnn,
}


class TorchModel(InferenceModel):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        opts = cfg.options
        raw_ckpt = cfg.checkpoint or opts.get("checkpoint")
        if not raw_ckpt:
            raise ValueError(f"model '{cfg.id}': torch type needs a 'checkpoint' path (file or run dir)")
        # (c) Auto-discover the .pth + params.json from a file or run directory.
        self.checkpoint, self.meta = resolve_checkpoint(raw_ckpt)
        # arch precedence: explicit config > params.json "model" > default.
        self.arch = cfg.arch or opts.get("arch") or self.meta.get("model", "CustomResNet")
        if self.arch not in ARCH_BUILDERS:
            raise ValueError(
                f"Unknown arch '{self.arch}'. Known: {list(ARCH_BUILDERS)}. "
                f"Add a builder in torch_model.py to support more architectures."
            )
        # Cross-check window size against the checkpoint's training config.
        meta_ws = self.meta.get("w_s")
        if meta_ws and int(meta_ws) != cfg.window.window_size:
            log.warning(
                "model '%s': window_size %d != checkpoint w_s %d — set window.window_size: %d",
                cfg.id, cfg.window.window_size, int(meta_ws), int(meta_ws),
            )
        # Feature-extraction models (e.g. EMGCNN) take handcrafted features, not
        # raw windows. Auto-detected from params.json.
        self.feature_extraction = bool(self.meta.get("feature_extraction", False))
        self._torch = None
        self._model = None
        self._device = None

    def warmup(self) -> None:
        import torch

        self._torch = torch
        dev = self.device
        if dev.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested for %s but unavailable; using CPU.", self.id)
            dev = "cpu"
        self._device = torch.device(dev)

        # CPU thread tuning: the default (all logical cores) badly oversubscribes
        # for a single small sample here (~770ms); a modest cap is far faster.
        if self._device.type == "cpu":
            threads = self.cfg.options.get("torch_threads")
            if threads is None:
                threads = min(8, os.cpu_count() or 1)
            torch.set_num_threads(int(threads))

        state = torch.load(self.checkpoint, map_location=self._device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Infer num_classes from the classifier head; trust the checkpoint.
        if "fc.weight" in state:
            nc = int(state["fc.weight"].shape[0])
            if nc != self.num_classes:
                log.warning("model '%s': num_classes %d -> %d (from checkpoint)", self.id, self.num_classes, nc)
                self.num_classes = nc
                if len(self.labels) != nc:
                    self.labels = [f"class_{i}" for i in range(nc)]

        model = ARCH_BUILDERS[self.arch](self.num_classes, self.meta)
        model.load_state_dict(state)
        model.eval().to(self._device)
        self._model = model
        log.info(
            "Loaded %s (%s) on %s | num_classes=%d w_s=%s feat=%s prm=%s",
            self.arch, self.id, dev, self.num_classes, self.meta.get("w_s"),
            self.feature_extraction, self.meta.get("prm"),
        )

    def predict(self, windows: dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        w = self.flatten_window(next(iter(windows.values())))  # (T_win, C)
        if self.feature_extraction:
            # EMGCNN: raw window (T,16) -> 6 features/channel -> (96,) -> [1,1,96].
            from swag_hlc.realtime.features import emg_features

            x = torch.from_numpy(emg_features(w)).float().unsqueeze(0).unsqueeze(0)
        else:
            # CustomResNet: [B=1, C=1, T_win, F] — BioActLab's window_cls layout.
            x = torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x.to(self._device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return probs.astype(np.float32)
