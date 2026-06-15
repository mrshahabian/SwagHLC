"""Model registry — config-driven model construction.

The registry supports *many* model types and *many* instances; the brief wants
the inference module configurable to run multiple models with different configs.
We ship single-active first (pick via ``active`` in YAML) but the registry and
engine already support running several concurrently (an ensemble).

Add a real model by registering a builder here, e.g. a ``"torch"`` type that
loads a BioActLab ``.pth`` + ``params.json`` behind a guarded ``import torch``.
"""

from __future__ import annotations

from typing import Callable

from swag_hlc.config import ModelConfig
from swag_hlc.realtime.models.base import InferenceModel
from swag_hlc.realtime.models.stub import StubModel


def _build_torch(cfg: ModelConfig) -> InferenceModel:
    # Imported lazily so the package runs without torch/torchvision installed.
    from swag_hlc.realtime.models.torch_model import TorchModel

    return TorchModel(cfg)


_REGISTRY: dict[str, Callable[[ModelConfig], InferenceModel]] = {
    "stub": StubModel,
    "torch": _build_torch,
}


def register_model(type_name: str, builder: Callable[[ModelConfig], InferenceModel]) -> None:
    _REGISTRY[type_name.lower()] = builder


def build_model(cfg: ModelConfig) -> InferenceModel:
    try:
        builder = _REGISTRY[cfg.type.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model type '{cfg.type}'. Known: {known}") from exc
    return builder(cfg)
