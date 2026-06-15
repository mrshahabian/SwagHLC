"""Modality / channel specifications for the RRD (MyPredict) dataset.

Calibrated against the *actual* dataset on disk
(``/vol/storage/common/SWAG/RRD/MP*.hdf5``):

  * All main signals (EMG bipolar, HD-EMG, IMU) are **co-sampled at 1000 Hz**
    in the files (a real ``Time`` vector confirms it); motion markers are 100 Hz.
  * HD-EMG ``EMG_Right_MA`` is stored ``(T, 4, 16)`` = 64 channels, and exists
    only for some subjects (MP2xx+); MP1xx have bipolar EMG + IMU only.
  * Bipolar muscles per side: BF, GM, Gmax, Gmed, RF, ST, TA, VL (8 -> 16 both).

Each modality maps to the concrete HDF5 dataset names ("params") that compose
it.  The replay source concatenates these along the channel axis (and reshapes
HD-EMG's grid to flat 64), mirroring BioActLab's loader.  Channel counts are
ultimately taken from the data at load time, so these are just defaults/sane
fallbacks (e.g. for the synthetic generator).
"""

from __future__ import annotations

from dataclasses import dataclass

MODALITY_HD_EMG = "hd_emg"
MODALITY_BP_EMG = "bp_emg"
MODALITY_IMU = "imu"

# Real per-dataset sampling rate (Hz). Everything main is 1000 Hz here.
RRD_SAMPLE_RATE_HZ = 1000.0


# Modality -> ordered list of HDF5 dataset names that compose it.
# Bipolar set used by the MP3xx cohort + the trained BP model: AM, BF, GM, Gmax,
# RF, ST, TA, VL (8 per side), Left block then Right block (matches the model's
# params.json `prm` order). NOTE: the MP1xx/MP2xx cohort recorded 'Gmed' instead
# of 'AM' (a different muscle) — for those subjects pass explicit `params:`.
_BP_LEFT = [
    "EMG_Left_AM", "EMG_Left_BF", "EMG_Left_GM", "EMG_Left_Gmax",
    "EMG_Left_RF", "EMG_Left_ST", "EMG_Left_TA", "EMG_Left_VL",
]
_BP_RIGHT = [
    "EMG_Right_AM", "EMG_Right_BF", "EMG_Right_GM", "EMG_Right_Gmax",
    "EMG_Right_RF", "EMG_Right_ST", "EMG_Right_TA", "EMG_Right_VL",
]
_IMU_ACC = [
    "Acc_Left_Foot", "Acc_Left_Shank", "Acc_Left_Thigh", "Acc_Pelvis",
    "Acc_Right_Foot", "Acc_Right_Shank", "Acc_Right_Thigh", "Acc_T8",
]
_IMU_GYR = [
    "Gyr_Left_Foot", "Gyr_Left_Shank", "Gyr_Left_Thigh", "Gyr_Pelvis",
    "Gyr_Right_Foot", "Gyr_Right_Shank", "Gyr_Right_Thigh", "Gyr_T8",
]
_IMU_ANG = [
    "Ang_Left_Ankle", "Ang_Left_Hip", "Ang_Left_Knee",
    "Ang_Right_Ankle", "Ang_Right_Hip", "Ang_Right_Knee",
]

MODALITY_PARAMS: dict[str, list[str]] = {
    MODALITY_HD_EMG: ["EMG_Right_MA"],          # (T,4,16) -> 64
    MODALITY_BP_EMG: _BP_LEFT + _BP_RIGHT,       # 16 (Left block then Right block)
    MODALITY_IMU: _IMU_ACC + _IMU_GYR + _IMU_ANG,  # 24+24+18 = 66
}


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    channels: int  # flattened channel count = prod(native_shape)
    default_rate_hz: float
    params: list[str]
    native_shape: tuple[int, ...]  # native per-sample shape streamed on the wire
    description: str = ""


DEFAULT_MODALITY_SPECS: dict[str, ModalitySpec] = {
    MODALITY_HD_EMG: ModalitySpec(
        MODALITY_HD_EMG, 64, RRD_SAMPLE_RATE_HZ, MODALITY_PARAMS[MODALITY_HD_EMG],
        (4, 16),  # stream the electrode grid as-is; flatten to 64 on inference side
        "High-density EMG 4x16 grid (subjects MP2xx+). One in RRD; we simulate N.",
    ),
    MODALITY_BP_EMG: ModalitySpec(
        MODALITY_BP_EMG, 16, RRD_SAMPLE_RATE_HZ, MODALITY_PARAMS[MODALITY_BP_EMG],
        (16,),
        "Bipolar surface EMG, 8 muscles x 2 sides.",
    ),
    MODALITY_IMU: ModalitySpec(
        MODALITY_IMU, 66, RRD_SAMPLE_RATE_HZ, MODALITY_PARAMS[MODALITY_IMU],
        (66,),
        "IMU suite: 8 Acc + 8 Gyr + 6 joint angles (3-axis each).",
    ),
}


def get_modality_spec(name: str) -> ModalitySpec:
    try:
        return DEFAULT_MODALITY_SPECS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        known = ", ".join(DEFAULT_MODALITY_SPECS)
        raise KeyError(f"Unknown modality '{name}'. Known: {known}") from exc


def resolve_params(modality: str | None, params: list[str] | None) -> list[str]:
    """Explicit ``params`` win; otherwise expand the modality to its param list."""
    if params:
        return list(params)
    if modality and modality in MODALITY_PARAMS:
        return list(MODALITY_PARAMS[modality])
    raise ValueError(
        f"Cannot resolve params: modality='{modality}', params={params}. "
        f"Set 'params' explicitly or use a known modality: {list(MODALITY_PARAMS)}"
    )
