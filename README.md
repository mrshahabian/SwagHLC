# SwagHLC

**SWAG High-Level Controller** — a dummy real-time intent-inference pipeline.

This subproject simulates the *high-level* controller of the SWAG control system:
an AI module that classifies user **intent** from streamed biomechanical sensor
data (RRD modalities: HD-EMG, BP-EMG, IMU) and publishes a probability
distribution for the mid-level controller. It *pretends* sensor data streams in,
so trained models can be exercised as if running live.

Two intentionally separate halves, connected only by a pluggable transport:

- **`dummy_stream`** — the replaceable sensor front-end (synthetic now, real
  hardware later).
- **`realtime`** — the real-time inference module (buffers, models, predictions).


The dummy stream **replays the real RRD dataset** (the sequential trials *are*
the stream); a synthetic generator is available as a portable fallback.

## Setup

```bash
cd SwagHLC
python3 -m venv .venv && .venv/bin/pip install numpy pyyaml h5py pytest
```

## Quickstart

Browse the dataset to choose what to stream:

```bash
PYTHONPATH=src .venv/bin/python -m swag_hlc.dataset_info                  # list subjects
PYTHONPATH=src .venv/bin/python -m swag_hlc.dataset_info --subject MP301  # days + trials
```

Stream real HD-EMG from a subject into one model:

```bash
PYTHONPATH=src .venv/bin/python -m swag_hlc.app --config configs/demo_single_model.yaml --duration 5
```

## Configuring

Everything is YAML (see `configs/`). Three blocks:

- **`dataset:`** — default RRD selection inherited by sources: `root`, `subject`,
  `days` (`all`/list), `trials` (`all` / `[Trial_01,…]` / `[1,2,3]`),
  `activities` (`all` or a list of codes to stream, e.g. `[1, 2, 2.1, 3, 4]`),
  `include_mvc`, `include_nolabel`, `loop`.
- **`sources:`** — dummy devices. `modality` (`hd_emg`/`bp_emg`/`imu`, or explicit
  `params:`), `chunk_size`, `generator` (`rrd_replay` default | `synthetic`),
  `count:` to replicate a device into N independent streams (e.g. multiple
  HD-EMG units), and per-source `subject`/`days`/`trials`/`activities` overrides.
- **`models:`** — inference models. `type` (`stub` | `torch`), `inputs:` (device
  ids it consumes), `activities:` (the intent class space — codes; sets
  `num_classes`/`labels` automatically; inherits `dataset.activities`),
  `window:`, `infer_rate_hz`, `device:` (`cpu`/`cuda:0`, GPU-per-model), `active:`.
- **`activity_names:`** (top-level) — map activity codes to human-readable names,
  e.g. `{1: stand, 2: walk, 2.1: walk_turn, 3: stairs_up, 4: stairs_down}`. These
  flow to predictions, ground-truth display, and `dataset_info`.

### Activities / intent classes

The RRD `Label` holds numeric activity **codes** (e.g. `1, 2, 2.1, 3, 4 …`).
Model trained on the 5-class set `{1, 2, 2.1, 3, 4}`. `swag_hlc.activities`
is the single source of truth tying *raw code ↔ class index ↔ name*, so
ground-truth labels and model outputs share one class space (the monitor prints
predicted-vs-true and an accuracy %). **No human-readable names exist** in
the HDF5 — names default to `activity_<code>`; set real ones via
`activity_names:`. Browse what's available:
`python -m swag_hlc.dataset_info --subject MP301` (lists trials + activity codes
with sample counts).

**Selectable like subjects/trials.** Choose activities in YAML (`dataset.activities`
/ per-source / `models[].activities`) or at the CLI:

```bash
# stream & classify only activities 1 and 3, from subject MP301, trials 1-2
... -m swag_hlc.app --config configs/demo_single_model.yaml \
      --subject MP301 --trials 1 2 --activities 1 3
```

**Dataloader parity.** Activity selection reproduces BioActLab's `RRDTorchDataset`
exactly: codes are validated against `ORIGINAL_LABEL_MAP` (unknowns dropped) and
the class index is assigned by **canonical sorted order** (`LABEL_MAP =
{old: idx for idx, old in enumerate(sorted(...))}`). So a selection of
`[1, 2, 2.1, 3, 4]` → `{1:0, 2:1, 2.1:2, 3:3, 4:4}`, matching a model trained
with `actions=[1,2,2.1,3,4]`. (Note: the *sequence* dataloader indexes by list
order; it coincides because the canonical set is passed sorted.)

`run.mode`: `process` (a process per source/model, asyncio within — default) or
`async` (single process). `transport.kind`: `inproc` (default) or `ros2` (stub
until rclpy is installed).

## Native streaming format

Data is streamed in the dataset's **native per-sample shape** — it is *not*
pre-flattened. A `SensorFrame.data` is `(n_samples, *feature_shape)`: HD-EMG
`(n, 4, 16)`, bipolar `(n, 16)`, IMU `(n, 66)`. The **inference side** flattens
(4×16 → 64) inside `model.predict` (`InferenceModel.flatten_window(s)`), so the
wire stays faithful to the hardware and the model owns reshaping.

## Dataset facts (calibrated from the files)

- All main signals are co-sampled at **1000 Hz** (markers 100 Hz).
- **HD-EMG** (`EMG_Right_MA`, 4×16 = 64 ch) exists only for **MP2xx+** subjects;
  **MP1xx** have **bipolar EMG (16 ch) + IMU** but no HD-EMG. No single subject
  has all three modalities — fuse across subjects via per-source `subject:`.


### Multi-modality + high-level fusion

`configs/multimodal_fusion.yaml` streams **HD-EMG + bipolar-EMG + IMU-Gyr** at
once, runs three separately-trained models, and **late-fuses** their
probabilities (sum rule) into one final distribution for the controller:

```bash
PYTHONPATH=src .venv/bin/python scripts/plot_streams.py --subject MP301   # visual + stats first
PYTHONPATH=src .venv/bin/python -m swag_hlc.app --config configs/multimodal_fusion.yaml --duration 14
```

