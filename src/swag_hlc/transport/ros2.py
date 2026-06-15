"""ROS 2 transport backend — interface stub.

ROS 2 is the agreed target middleware but is **not installed on this machine**
yet, so this backend is intentionally a documented stub: it shows exactly how
the SwagHLC topics/messages map onto ROS 2, and raises a clear, actionable error
if you try to use it without rclpy.

Mapping plan (when rclpy is available):
  * SensorFrame      -> a custom msg ``swag_msgs/SensorFrame`` (or, to avoid
    custom-msg builds early, ``std_msgs/Float32MultiArray`` + a companion
    ``std_msgs/Header`` carrying device_id/modality/seq in the frame_id).
  * IntentPrediction -> ``swag_msgs/IntentPrediction`` carrying the full
    probability vector, argmax, source_ids and timing.
  * Each ``device_id`` -> topic ``/swag/sensor/<device_id>``.
  * Each ``model_id``  -> topic ``/swag/intent/<model_id>``.

Because everything upstream only depends on ``transport.base`` and the message
dataclasses (which already provide ``to_dict``/``from_dict``), switching from
InProc to ROS 2 is a config change plus filling in the bodies below.
"""

from __future__ import annotations

from typing import Any

from swag_hlc.transport.base import Publisher, Subscriber, Transport

_INSTALL_HINT = (
    "ROS 2 (rclpy) is not available. Install ROS 2 and source its setup, or use "
    "transport.kind: inproc for the dummy demo. This backend is a documented "
    "interface stub — see swag_hlc/transport/ros2.py for the topic/message map."
)


def _require_rclpy():
    try:
        import rclpy  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on host
        raise RuntimeError(_INSTALL_HINT) from exc
    return rclpy


class Ros2Transport(Transport):
    def __init__(self, namespace: str = "swag", **_: Any) -> None:
        self._ns = namespace
        _require_rclpy()  # fail fast with a helpful message

    def declare_topic(self, topic: str) -> None:  # pragma: no cover
        raise NotImplementedError(_INSTALL_HINT)

    def publisher(self, topic: str) -> Publisher:  # pragma: no cover
        raise NotImplementedError(_INSTALL_HINT)

    def subscriber(self, topic: str) -> Subscriber:  # pragma: no cover
        raise NotImplementedError(_INSTALL_HINT)
