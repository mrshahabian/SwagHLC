"""Activity / intent class registry — the single source of truth for classes.

The RRD ``Label`` vector holds numeric activity codes (e.g. 1, 2, 2.1, 3, 4 …).
BioActLab reindexes the full set of 23 codes to 0..22, but actually *trains* on a
5-class subset: codes {1, 2, 2.1, 3, 4} -> indices {0,1,2,3,4}
(``action_mapping`` in sequential_evaluator.py / cli.py / core.py).

This module ties together, in one place:
  * raw dataset code  <->  model class index  <->  human-readable name,

so ground-truth labels, model outputs, and config all agree on the class space.

IMPORTANT: no human-readable activity names exist in BioActLab or in the HDF5
files (only numeric codes). Names therefore default to ``activity_<code>`` and
are meant to be overridden via the ``activity_names:`` block in YAML once the
real names are known. We deliberately do NOT invent semantic names here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def normalize_code(x) -> float:
    """Canonicalise a code so 2.1 stays 2.1 and 2.0/2 compare equal."""
    return round(float(x), 3)


# Full RRD code -> contiguous index map (mirrors BioActLab ORIGINAL_LABEL_MAP in
# src/bioactlab/data/rrd_torch_dataset.py).
ORIGINAL_LABEL_MAP: dict[float, int] = {
    -2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 2.1: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10,
    7.1: 11, 7.2: 12, 7.3: 13, 8.1: 14, 8.2: 15, 8.3: 16, 8.4: 17, 8.5: 18,
    8.6: 19, 8.7: 20, 8.8: 21, 9: 22,
}
# Normalized-key view for safe float lookups (2.1, 1.0, ...).
_ORIG = {normalize_code(k): v for k, v in ORIGINAL_LABEL_MAP.items()}
ALL_CODES: list[float] = sorted(_ORIG, key=lambda c: _ORIG[c])

# The subset BioActLab models are trained on (core.py: actions=[1,2,2.1,3,4]).
DEFAULT_ACTIVE_CODES: list[float] = [1.0, 2.0, 2.1, 3.0, 4.0]

# Placeholder names only — fill the real ones via YAML `activity_names:`.
DEFAULT_ACTIVITY_NAMES: dict[float, str] = {c: f"activity_{c}" for c in ALL_CODES}


def is_valid_code(code) -> bool:
    return normalize_code(code) in _ORIG


@dataclass
class ActivityRegistry:
    """A selectable class space: a set of activity codes + their names.

    Faithful to BioActLab's window dataloader (``RRDTorchDataset``): codes are
    validated against ORIGINAL_LABEL_MAP (unknowns dropped) and the class
    ``index`` is assigned by **canonical (sorted) code order** — exactly the
    ``LABEL_MAP = {old: new_idx for new_idx, old in enumerate(sorted(...))}`` rule.
    For the standard set [1, 2, 2.1, 3, 4] this yields {1:0, 2:1, 2.1:2, 3:3, 4:4}.

    Codes outside the set map to ``None`` (ground-truth samples of an activity the
    model wasn't trained on).
    """

    codes: list[float]
    names: dict[float, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate + canonicalize order to match the dataloader's LABEL_MAP.
        valid = [normalize_code(c) for c in self.codes if is_valid_code(c)]
        dropped = [c for c in self.codes if not is_valid_code(c)]
        if dropped:
            import logging

            logging.getLogger(__name__).warning(
                "Dropping unknown activity codes not in ORIGINAL_LABEL_MAP: %s", dropped
            )
        self.codes = sorted(set(valid), key=lambda c: _ORIG[c])
        merged = dict(DEFAULT_ACTIVITY_NAMES)
        merged.update({normalize_code(k): v for k, v in (self.names or {}).items()})
        self.names = merged
        self._code_to_idx = {c: i for i, c in enumerate(self.codes)}

    def original_index(self, code) -> int | None:
        """The full-taxonomy index (ORIGINAL_LABEL_MAP), regardless of selection."""
        return _ORIG.get(normalize_code(code))

    def label_map(self) -> dict[float, int]:
        """BioActLab-equivalent {raw_code: class_idx} for this selection."""
        return dict(self._code_to_idx)

    @property
    def num_classes(self) -> int:
        return len(self.codes)

    @property
    def labels(self) -> list[str]:
        """Display names in index order (what the model's outputs mean)."""
        return [self.names.get(c, f"activity_{c}") for c in self.codes]

    def index_of(self, code) -> int | None:
        return self._code_to_idx.get(normalize_code(code))

    def name_of_code(self, code) -> str:
        return self.names.get(normalize_code(code), f"activity_{code}")

    def map_label(self, raw_code) -> tuple[int | None, str]:
        """Raw dataset code -> (class index or None, display name)."""
        idx = self.index_of(raw_code)
        if idx is None:
            return None, f"other({normalize_code(raw_code)})"
        return idx, self.labels[idx]


def build_registry(codes: list[float] | None, names: dict | None = None) -> ActivityRegistry:
    return ActivityRegistry(codes=codes or DEFAULT_ACTIVE_CODES, names=names or {})
