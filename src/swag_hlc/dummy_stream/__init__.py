"""Dummy data stream — the *replaceable* sensor front-end.

Kept strictly separate from ``realtime`` so it can later be swapped for a real
hardware adapter with a different data format, without the inference module
noticing: both sides only share the ``SensorFrame`` schema and the transport.
"""

from swag_hlc.dummy_stream.base_source import StreamSource
from swag_hlc.dummy_stream.synthetic_source import SyntheticSource

__all__ = ["StreamSource", "SyntheticSource", "build_source"]


def build_source(cfg, publisher):
    """Factory: map a resolved SourceConfig to a concrete source instance."""
    gen = cfg.generator.lower()
    if gen == "synthetic":
        return SyntheticSource(cfg, publisher)
    if gen in ("rrd_replay", "rrd", "replay"):
        # Lazy import: pulls in h5py, which is optional / not needed for the demo.
        from swag_hlc.dummy_stream.rrd_replay_source import RrdReplaySource

        return RrdReplaySource(cfg, publisher)
    raise ValueError(f"Unknown source generator '{cfg.generator}'.")
