"""SwagHLC — SWAG High-Level Controller (real-time intent inference).

This package simulates the *high-level* controller of the SWAG control system:
an AI module that classifies/predicts user intent from streamed biomechanical
sensor data and publishes a probability distribution over intent classes for the
mid-level controller to consume.

Two deliberately separated halves:
  * ``dummy_stream`` — a *replaceable* dummy data source that pretends to be the
    sensor front-end (synthetic generators today; real hardware tomorrow).
  * ``realtime``     — the *real-time inference* module that buffers streams,
    runs one or more models, and publishes intent predictions.

They only ever talk through ``transport`` (a pluggable message bus), so the
dummy source can be swapped for a different real-time data format without
touching the inference module.
"""

__version__ = "0.1.0"
