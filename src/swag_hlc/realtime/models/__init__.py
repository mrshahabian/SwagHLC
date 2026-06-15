"""Inference models and the model registry."""

from swag_hlc.realtime.models.base import InferenceModel
from swag_hlc.realtime.models.registry import build_model

__all__ = ["InferenceModel", "build_model"]
