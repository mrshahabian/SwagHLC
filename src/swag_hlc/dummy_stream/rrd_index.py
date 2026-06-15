"""Helpers to browse and resolve selections in the RRD dataset.

The dataset is a directory of ``<subject>.hdf5`` files; each file has ``Meta`` +
``Day_*`` groups; each day has ``MVC`` and ``Trial_*`` subgroups; each trial has
per-param datasets + a ``Label`` vector.  These helpers turn loose config
selections ("all" / lists / ints) into concrete, existing names — and power the
``swag-hlc-dataset`` CLI so you can see what's available to stream.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def default_root() -> Path:
    return Path(os.environ.get("RRD_ROOT", "/vol/storage/common/SWAG/RRD"))


def subject_path(root: str | os.PathLike, subject: str) -> Path:
    root = Path(root)
    name = subject if subject.endswith(".hdf5") else f"{subject}.hdf5"
    return root / name


def list_subjects(root: str | os.PathLike) -> list[str]:
    root = Path(root)
    return sorted(p.stem for p in root.glob("*.hdf5"))


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def list_days(root: str | os.PathLike, subject: str) -> list[str]:
    import h5py

    with h5py.File(subject_path(root, subject), "r") as f:
        return sorted((k for k in f.keys() if k != "Meta"), key=_natural_key)


def list_trials(
    root: str | os.PathLike,
    subject: str,
    day: str,
    include_mvc: bool = False,
    include_nolabel: bool = False,
) -> list[str]:
    """Trials in acquisition order.

    By default returns only the labelled activity trials (``Trial_*``); the
    ``MVC`` calibration block and ``NoLabel_*`` (no-intent) recordings are
    excluded unless explicitly requested.
    """
    import h5py

    with h5py.File(subject_path(root, subject), "r") as f:
        trials = list(f[day].keys())
    trials = sorted(trials, key=_natural_key)
    if not include_mvc:  # exclude MVC / MVCf / MVC_* calibration blocks
        trials = [t for t in trials if not t.upper().startswith("MVC")]
    if not include_nolabel:
        trials = [t for t in trials if not t.lower().startswith("nolabel")]
    return trials


def resolve_days(root, subject, days_spec) -> list[str]:
    avail = list_days(root, subject)
    if days_spec in (None, "all", ["all"]):
        return avail
    out = []
    for d in days_spec:
        name = d if str(d).startswith("Day") else f"Day_{d}"
        if name in avail:
            out.append(name)
    return out or avail


def resolve_trials(
    root, subject, day, trials_spec, include_mvc: bool = False, include_nolabel: bool = False
) -> list[str]:
    """Accept 'all', a list of names ('Trial_03'), or a list of ints (3 -> Trial_03)."""
    avail = list_trials(
        root, subject, day, include_mvc=include_mvc, include_nolabel=include_nolabel
    )
    if trials_spec in (None, "all", ["all"]):
        return avail
    avail_set = set(avail)
    out = []
    for t in trials_spec:
        if isinstance(t, int) or (isinstance(t, str) and t.isdigit()):
            name = f"Trial_{int(t):02d}"
        else:
            name = str(t)
        if name in avail_set:
            out.append(name)
    return out


def activity_counts(
    root, subject, days_spec="all", include_mvc=False, include_nolabel=False
) -> dict[float, int]:
    """Total samples per activity code across the selected trials (one file open)."""
    import collections

    import h5py
    import numpy as np

    counts: collections.Counter = collections.Counter()
    with h5py.File(subject_path(root, subject), "r") as f:
        for day in resolve_days(root, subject, days_spec):
            trials = list_trials(
                root, subject, day, include_mvc=include_mvc, include_nolabel=include_nolabel
            )
            for trial in trials:
                lab = np.asarray(f[day][trial]["Label"][...]).reshape(-1)
                u, c = np.unique(lab, return_counts=True)
                for v, n in zip(u, c):
                    counts[round(float(v), 3)] += int(n)
    return dict(counts)


def resolve_pairs(
    root, subject, days_spec, trials_spec, include_mvc=False, include_nolabel=False
) -> list[tuple[str, str]]:
    """Ordered (day, trial) pairs to stream sequentially."""
    pairs: list[tuple[str, str]] = []
    for day in resolve_days(root, subject, days_spec):
        for trial in resolve_trials(
            root, subject, day, trials_spec, include_mvc=include_mvc, include_nolabel=include_nolabel
        ):
            pairs.append((day, trial))
    return pairs
