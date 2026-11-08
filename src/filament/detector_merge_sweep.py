"""Settings and diagnostics of the "merge only" detector sweep.

``notebooks/sweep_detector_merge.ipynb`` scores :func:`filament.detector_fusion.merge_components`
on the date holdout. The claim fusion of phase F lost clearly (pooled PQ 0.3603
against the base's 0.4255) and the cause was not measured. Here the base labels
stay and the detector may only join base components, so the pixel set equals the
base's by construction.

One axis at a time from the defaults (``cut`` 0.5 on the raw score, ``coverage``
0.5, no deduplication). The grids are untuned placeholders chosen before any
result, not a fine grid. The verdict is the cross-fit over all of them
(:func:`filament.detector_sweep.crossfit_summary`), not the best row.

Numpy only. Nothing in this module ran on real data: the tests are synthetic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from filament.detector_fusion import (
    DEFAULT_DILATION,
    Detection,
    merge_components,
)
from filament.metrics import PqCounts

# Untuned. Chosen before any result, not read from the holdout.
MERGE_CUTS = (0.2, 0.3, 0.5, 0.7, 0.9)
MERGE_COVERAGES = (0.5, 0.8)
DEFAULT_CUT = 0.5
DEFAULT_COVERAGE = 0.5


@dataclass(frozen=True, slots=True)
class MergeSetting:
    """One setting of the merge. Hashable, so it can key a grid for ``stats``.

    ``dedup`` runs :func:`filament.detector_fusion.dedup_detections` (its default
    IoU) after the merge across windows. ``axis`` is not compared.
    """

    axis: str = field(compare=False)
    cut: float = DEFAULT_CUT
    coverage: float = DEFAULT_COVERAGE
    dedup: bool = False

    @property
    def label(self) -> str:
        return (
            f"merge cut={self.cut:g} coverage={self.coverage:g} dedup={int(self.dedup)}"
        )

    def __str__(self) -> str:
        return self.label


def merge_axis_settings(
    cuts: Sequence[float] = MERGE_CUTS,
    coverages: Sequence[float] = MERGE_COVERAGES,
) -> dict[str, list[MergeSetting]]:
    """The settings of each axis, every one a single change from the defaults."""
    return {
        "merge_cut": [MergeSetting("merge_cut", cut=value) for value in cuts],
        "merge_coverage": [
            MergeSetting("merge_coverage", coverage=value) for value in coverages
        ],
        "merge_dedup": [
            MergeSetting("merge_dedup", dedup=flag) for flag in (False, True)
        ],
    }


@dataclass(frozen=True, slots=True)
class MergeOutcome:
    """The labels of one setting on one frame and how many joins it made."""

    labels: np.ndarray
    joins: int


def merge_outcomes(
    labels: np.ndarray,
    merged: Sequence[Detection],
    unique: Sequence[Detection],
    settings: Sequence[MergeSetting],
    dilation: int = DEFAULT_DILATION,
) -> dict[MergeSetting, MergeOutcome]:
    """Apply every setting to one frame's base ``labels``.

    ``merged`` is the detections after the merge across windows and ``unique``
    the same after deduplication; the caller computes both once per frame. A
    setting with ``dedup`` uses ``unique``, the others ``merged``. ``joins`` is
    the drop in the number of components (ids with no pixel are not counted); a frame with ``joins == 0`` has the
    base partition, so a caller can reuse the base counts for it.
    """
    before = int(np.count_nonzero(np.bincount(labels.ravel())[1:]))
    outcomes = {}
    for setting in settings:
        joined = merge_components(
            labels,
            unique if setting.dedup else merged,
            setting.cut,
            setting.coverage,
            dilation,
        )
        outcomes[setting] = MergeOutcome(joined, before - int(joined.max()))
    return outcomes


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """Pooled PQ split into its parts: ``pq = sq * rq``.

    ``rq`` is recognition quality, ``tp / (tp + fp/2 + fn/2)``, and ``sq`` is
    segmentation quality, the mean IoU of the matched pairs, ``iou_sum / tp``.
    Losing ``rq`` means matching fewer filaments; losing ``sq`` means matching
    them worse. Both are NaN when undefined.
    """

    pq: float
    tp: int
    fp: int
    fn: int
    sq: float
    rq: float

    def text(self) -> str:
        return (
            f"pq={self.pq:.4f} tp={self.tp} fp={self.fp} fn={self.fn} "
            f"sq={self.sq:.4f} rq={self.rq:.4f}"
        )


def pooled_diagnostics(counts: Sequence[PqCounts]) -> Diagnostics:
    """Pool per-frame counts (add them, never average the PQ values)."""
    total = sum(counts, PqCounts())
    denominator = total.tp + 0.5 * total.fp + 0.5 * total.fn
    nan = float("nan")
    return Diagnostics(
        pq=total.pq,
        tp=total.tp,
        fp=total.fp,
        fn=total.fn,
        sq=total.iou_sum / total.tp if total.tp else nan,
        rq=total.tp / denominator if denominator else nan,
    )


def diagnostic_note(
    diagnostics: Diagnostics, *, joins: int | None = None, frames: int | None = None
) -> str:
    """The text of the CSV ``note`` column for one row of the diagnostics table."""
    parts = [diagnostics.text()]
    if joins is not None:
        parts.append(f"merges={joins}")
    if frames is not None:
        parts.append(f"frames_changed={frames}")
    return " ".join(parts)
