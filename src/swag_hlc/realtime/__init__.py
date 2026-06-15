"""Real-time inference module.

Buffers incoming sensor streams *per modality/device* (independent ring buffers,
independent window cadence), runs one or more models on the latest windows, and
publishes ``IntentPrediction`` messages for the mid-level controller.
"""

from swag_hlc.realtime.buffers import RingBuffer
from swag_hlc.realtime.engine import InferenceEngine

__all__ = ["RingBuffer", "InferenceEngine"]
