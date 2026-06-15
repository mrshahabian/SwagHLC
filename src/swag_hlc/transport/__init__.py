"""Pluggable message transport.

The whole point of this layer: the dummy stream and the inference module never
import each other.  They publish/subscribe to *topics* through a Transport, so
the backend (in-process today, ROS 2 tomorrow) is swappable from config alone.
"""

from swag_hlc.transport.base import Publisher, Subscriber, Transport
from swag_hlc.transport.inproc import InProcTransport


def build_transport(kind: str, **kwargs) -> Transport:
    """Factory: map a config string to a transport backend."""
    kind = kind.lower()
    if kind in ("inproc", "in_process", "local"):
        return InProcTransport(**kwargs)
    if kind in ("ros2", "ros"):
        # Imported lazily so the package works without rclpy installed.
        from swag_hlc.transport.ros2 import Ros2Transport

        return Ros2Transport(**kwargs)
    raise ValueError(f"Unknown transport kind '{kind}'. Use 'inproc' or 'ros2'.")


__all__ = [
    "Transport",
    "Publisher",
    "Subscriber",
    "InProcTransport",
    "build_transport",
]
