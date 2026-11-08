"""Score the detector fusion on the holdout, one axis at a time.

``notebooks/sweep_detector.ipynb`` runs this on the U-Net probability cache and
the detections of ``notebooks/export_detections.ipynb``. No network runs there.
The logic lives here, in numpy, so it is tested on synthetic frames.

How the cost is kept low. A fusion setting only changes which pixels each
detection claims (``dilation``, ``emit_unclaimed``) and which claimed ids are
emitted (a score cut). The match of a claimed id with a ground-truth mask does
not depend on which other ids are emitted, because the claims are disjoint. So a
frame is claimed once per ``(dilation, emit_unclaimed)`` and its matches are
recorded once (:class:`filament.components.ViewHits`). Every score cut is then
:func:`filament.components.selection_counts` on those matches, with no pixels.
Merging across windows and deduplication do not depend on any swept setting and
run once per frame.

One axis at a time, from the defaults of :mod:`filament.detector_fusion`:
the score cut on raw detector scores, the same cut on cross-fitted calibrated
scores, the dilation, and ``emit_unclaimed``. The grids below are untuned
placeholders chosen before any result was seen. They are not a fine grid on
purpose: the verdict is the cross-fit over all of them
(:func:`crossfit_summary`), not the best row.

Nothing in this module ran on real data.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from filament.axis_report import MIN_PARTITIONS_ADOPTED
from filament.components import (
    ViewHits,
    expected_matched_iou,
    selection_counts,
)
from filament.detector_fusion import (
    DEFAULT_CONFIG,
    Detection,
    FusionConfig,
    Rect,
    claim_pixels,
    crossfit_calibrated_scores,
    dedup_detections,
    fusion_targets,
    merge_across_windows,
)
from filament.metrics import PqCounts
from filament.stats import (
    adopts_change,
    crossfit_select,
    date_folds,
    paired_bootstrap,
    weighted_pq,
)
from filament.sweep import format_pq

# Untuned. Chosen before any result, not read from the holdout.
RAW_CUTS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
CALIBRATED_CUTS = (0.1, 0.2, 0.3, 0.4, 0.5)
DILATIONS = (0, 3, 5, 9, 15)
# The same partitions as ``filament.axis_report.crossfit_rows``.
CROSSFIT_SEEDS = (0, 1, 2, 3, 4)
CROSSFIT_FOLDS = 2
CALIBRATION_FOLDS = 5
# The roadmap's rule for a submission without asking: a gain on the cross-fit
# number of more than this (AGENTS.md, "Kaggle").
SUBMIT_GAIN = 0.01


@dataclass(frozen=True, slots=True)
class Setting:
    """A fusion configuration and a score cut. Hashable; ``axis`` is not compared.

    ``config.min_score`` must stay ``None``: the cut is ``cut``, applied to the
    raw scores or, with ``calibrated``, to the cross-fitted calibrated ones.
    """

    axis: str = field(compare=False)
    config: FusionConfig = DEFAULT_CONFIG
    cut: float | None = None
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.config.min_score is not None:
            raise ValueError("put the cut in Setting.cut, not in config.min_score")
        if self.calibrated and self.cut is None:
            raise ValueError("a calibrated setting needs a cut")
        if self.calibrated and self.config.emit_unclaimed:
            raise ValueError("unclaimed pieces have no score to calibrate")

    @property
    def label(self) -> str:
        parts = []
        if self.cut is not None:
            parts.append(f"{'cal' if self.calibrated else 'raw'}>={self.cut:g}")
        if self.config.dilation != DEFAULT_CONFIG.dilation:
            parts.append(f"dilation={self.config.dilation}")
        if self.config.emit_unclaimed:
            parts.append("emit_unclaimed")
        return ", ".join(parts) if parts else "default"

    def __str__(self) -> str:
        return self.label


def axis_settings(
    raw_cuts: Sequence[float] = RAW_CUTS,
    calibrated_cuts: Sequence[float] = CALIBRATED_CUTS,
    dilations: Sequence[int] = DILATIONS,
) -> dict[str, list[Setting]]:
    """The settings of each axis, every one a single change from the defaults."""

    def replaced(**changes: object) -> FusionConfig:
        values = {
            "windows_iou": DEFAULT_CONFIG.windows_iou,
            "dedup_iou": DEFAULT_CONFIG.dedup_iou,
            "dilation": DEFAULT_CONFIG.dilation,
            "min_area": DEFAULT_CONFIG.min_area,
            "min_score": None,
            "emit_unclaimed": DEFAULT_CONFIG.emit_unclaimed,
        }
        values.update(changes)
        return FusionConfig(**values)  # type: ignore[arg-type]

    return {
        "score_raw": [Setting("score_raw", DEFAULT_CONFIG, cut) for cut in raw_cuts],
        "score_calibrated": [
            Setting("score_calibrated", DEFAULT_CONFIG, cut, calibrated=True)
            for cut in calibrated_cuts
        ],
        "dilation": [
            Setting("dilation", replaced(dilation=value)) for value in dilations
        ],
        "emit_unclaimed": [
            Setting("emit_unclaimed", replaced(emit_unclaimed=flag))
            for flag in (False, True)
        ],
    }


def unique_settings(axes: Mapping[str, Sequence[Setting]]) -> list[Setting]:
    """Every distinct setting over all axes, first occurrence first."""
    seen: dict[Setting, None] = {}
    for settings in axes.values():
        for setting in settings:
            seen.setdefault(setting)
    return list(seen)


@dataclass(frozen=True, slots=True)
class ClaimHits:
    """One claim of a frame: its matches per view and its raw scores.

    ``scores[i]`` is the detector score of claimed id ``i + 1``, NaN for an
    unclaimed U-Net piece (``emit_unclaimed``).
    """

    hits: tuple[ViewHits, ...]
    scores: np.ndarray


@dataclass(frozen=True, slots=True)
class DedupStats:
    """What deduplication did to the detections of one or more frames.

    ``dropped_cross_window`` counts the dropped detections whose first
    suppressor (the best-scored kept detection they overlap above ``iou``) came
    from another window, so merging across windows had left them as two. The window
    of a merged detection is the window of its best member.
    """

    detections: int = 0
    merged: int = 0
    kept: int = 0
    dropped: int = 0
    dropped_cross_window: int = 0

    def __add__(self, other: object) -> DedupStats:
        if not isinstance(other, DedupStats):
            return NotImplemented
        return DedupStats(
            self.detections + other.detections,
            self.merged + other.merged,
            self.kept + other.kept,
            self.dropped + other.dropped,
            self.dropped_cross_window + other.dropped_cross_window,
        )


def _iou_in_boxes(
    first: Detection, second: Detection, crops: dict[int, np.ndarray]
) -> float:
    top = max(first.bbox[0], second.bbox[0])
    left = max(first.bbox[1], second.bbox[1])
    bottom = min(first.bbox[2], second.bbox[2])
    right = min(first.bbox[3], second.bbox[3])
    if bottom <= top or right <= left:
        return 0.0
    for item in (first, second):
        if id(item) not in crops:
            crops[id(item)] = item.crop()
    a, b = crops[id(first)], crops[id(second)]
    inside_a = a[top - first.bbox[0] : bottom - first.bbox[0],
                 left - first.bbox[1] : right - first.bbox[1]]  # fmt: skip
    inside_b = b[top - second.bbox[0] : bottom - second.bbox[0],
                 left - second.bbox[1] : right - second.bbox[1]]  # fmt: skip
    both = int(np.count_nonzero(inside_a & inside_b))
    either = int(np.count_nonzero(a)) + int(np.count_nonzero(b)) - both
    return both / either if either else 0.0


def dedup_stats(
    merged: Sequence[Detection], kept: Sequence[Detection], iou: float
) -> DedupStats:
    """Count what :func:`filament.detector_fusion.dedup_detections` dropped.

    ``kept`` is its result for ``merged``. For each dropped detection the
    suppressor is the first kept one (in score order) whose IoU with it is above
    ``iou``, the one the greedy pass met first.
    """
    kept_ids = {id(item) for item in kept}
    crops: dict[int, np.ndarray] = {}
    dropped = cross = 0
    for detection in merged:
        if id(detection) in kept_ids:
            continue
        dropped += 1
        for other in kept:
            if _iou_in_boxes(detection, other, crops) > iou:
                cross += int(other.window != detection.window)
                break
    return DedupStats(
        detections=0,
        merged=len(merged),
        kept=len(kept),
        dropped=dropped,
        dropped_cross_window=cross,
    )


@dataclass(frozen=True, slots=True)
class FileClaims:
    """Everything one frame contributes to the sweep."""

    claims: Mapping[FusionConfig, ClaimHits]
    detector_only: ClaimHits
    dedup: DedupStats


def _hits(labels: np.ndarray, scores: np.ndarray, views) -> ClaimHits:
    hits, _ = fusion_targets(labels, views)
    return ClaimHits(hits=hits, scores=np.asarray(scores, dtype=np.float64))


def claim_file(
    detections: Sequence[Detection],
    unet_mask: np.ndarray,
    disk: np.ndarray,
    views: Sequence[Sequence[np.ndarray]],
    windows: Sequence[Rect],
    configs: Sequence[FusionConfig],
) -> FileClaims:
    """Merge and deduplicate once, then claim and record the matches per config.

    ``unet_mask`` is the current pipeline's output for the frame (the base), and
    ``disk`` its solar-disk gate (:func:`filament.baseline.on_disk`), applied in
    :func:`filament.detector_fusion.claim_pixels`. ``configs`` may differ only in
    ``dilation`` and ``emit_unclaimed``: merging, deduplication and ``min_area``
    are shared. The detector-only claim is the sanity row: the merged and
    deduplicated detections claim their own pixels on the disk, with no U-Net and
    no dilation.
    """
    first = configs[0]
    for config in configs:
        if (config.windows_iou, config.dedup_iou, config.min_area) != (
            first.windows_iou,
            first.dedup_iou,
            first.min_area,
        ):
            raise ValueError("configs may differ only in dilation and emit_unclaimed")
        if config.min_score is not None:
            raise ValueError("configs must have min_score=None; cut with a Setting")
    merged = merge_across_windows(detections, windows, first.windows_iou)
    unique = dedup_detections(merged, first.dedup_iou)
    stats = dedup_stats(merged, unique, first.dedup_iou)
    stats = DedupStats(
        len(detections), stats.merged, stats.kept, stats.dropped,
        stats.dropped_cross_window,
    )  # fmt: skip
    gate = np.asarray(disk, dtype=bool)
    claims = {}
    for config in dict.fromkeys(configs):
        labels, scores = claim_pixels(
            unique,
            unet_mask,
            config.dilation,
            config.min_area,
            emit_unclaimed=config.emit_unclaimed,
            disk=gate,
        )
        claims[config] = _hits(labels, scores, views)
    labels, scores = claim_pixels(unique, gate, 0, first.min_area)
    return FileClaims(
        claims=claims, detector_only=_hits(labels, scores, views), dedup=stats
    )


def counts_for(
    claim: ClaimHits,
    cut: float | None = None,
    *,
    scores: np.ndarray | None = None,
    keep_unscored: bool = False,
) -> PqCounts:
    """PQ counts of a frame when the ids with ``score >= cut`` are emitted.

    ``scores`` replaces the claim's raw scores (the calibrated ones), one per id.
    ``cut=None`` emits every id. A NaN score fails any cut unless
    ``keep_unscored``, as in :func:`filament.detector_fusion.keep_scored`.
    """
    values = claim.scores if scores is None else np.asarray(scores, dtype=np.float64)
    if values.shape != claim.scores.shape:
        raise ValueError(f"{values.size} scores for {claim.scores.size} ids")
    emit = np.ones(values.size, dtype=bool) if cut is None else values >= cut
    if keep_unscored:
        emit = emit | np.isnan(values)
    return selection_counts(claim.hits, emit)


def calibrate_crossfit(
    claims: Sequence[ClaimHits],
    dates: Sequence[dt.date],
    n_splits: int = CALIBRATION_FOLDS,
    seed: int = 0,
) -> list[np.ndarray]:
    """Calibrated scores per frame where no date fits and predicts together.

    The target of an id is the roadmap's ``(1/R) * sum_r IoU_r * [IoU_r > 0.5]``
    (:func:`filament.components.expected_matched_iou`). The calibrator is the
    isotonic fit of :func:`filament.detector_fusion.crossfit_calibrated_scores`,
    out of fold by date. ``claims`` must have finite scores (no unclaimed pieces).
    """
    if len(dates) != len(claims):
        raise ValueError(f"{len(dates)} dates for {len(claims)} frames")
    scores = [claim.scores for claim in claims]
    targets = [expected_matched_iou(claim.hits, claim.scores.size) for claim in claims]
    sizes = [values.size for values in scores]
    if sum(sizes) == 0:
        raise ValueError("no claimed ids to calibrate")
    groups = [day for day, size in zip(dates, sizes, strict=True) for _ in range(size)]
    calibrated = crossfit_calibrated_scores(
        np.concatenate(scores), np.concatenate(targets), groups, n_splits, seed
    )
    return [chunk for chunk in np.split(calibrated, np.cumsum(sizes)[:-1])]


@dataclass(frozen=True, slots=True)
class CrossfitSummary:
    """The cross-fit over several date partitions, as one line.

    ``config`` is the setting chosen most often over the folds of every partition
    (the first one on a tie). ``pooled_pq`` is the mean over partitions of the
    out-of-fold pooled PQ. ``diff`` is that minus the base's pooled PQ.
    ``fraction_better`` is the mean over partitions of the share of paired
    bootstrap resamples in which the out-of-fold counts beat the base.
    ``adopted`` partitions pass the roadmap's rule (a gain of at least 0.005 in at
    least 90% of resamples, :func:`filament.stats.adopts_change`).
    """

    config: str
    pooled_pq: float
    diff: float
    fraction_better: float
    adopted: int
    partitions: int
    picks: Mapping[str, int]


def crossfit_summary(
    base: Sequence[PqCounts],
    settings: Mapping[Hashable, Sequence[PqCounts]],
    dates: Sequence[dt.date],
    *,
    seeds: Sequence[int] = CROSSFIT_SEEDS,
    n_folds: int = CROSSFIT_FOLDS,
    resamples: int = 2000,
) -> CrossfitSummary:
    """Cross-fit ``settings`` against ``base`` as :func:`filament.axis_report.crossfit_rows` does.

    The same partitions, the same bootstrap and the same rule, so its
    ``pooled_pq`` equals the verdict row of ``crossfit_rows``. ``settings`` should
    include the base itself, so a partition may keep it.
    """
    pooled = []
    fractions = []
    picks: Counter[str] = Counter()
    adopted = 0
    for seed in seeds:
        result = crossfit_select(settings, date_folds(dates, n_folds, seed=seed))
        comparison = paired_bootstrap(base, result.out_of_fold, n=resamples, seed=seed)
        adopted += bool(adopts_change(comparison))
        pooled.append(result.pooled_pq)
        fractions.append(comparison.fraction_better)
        picks.update(str(key) for key in result.selected)
    top = max(picks.values())
    chosen = next(key for key, count in picks.items() if count == top)
    mean_pq = float(np.mean(pooled))
    return CrossfitSummary(
        config=chosen,
        pooled_pq=mean_pq,
        diff=mean_pq - weighted_pq(base),
        fraction_better=float(np.mean(fractions)),
        adopted=adopted,
        partitions=len(list(seeds)),
        picks=dict(picks),
    )


def verdict_line(summary: CrossfitSummary) -> str:
    """The one-line verdict.

    ``VERDICT <config> pq=<crossfit> diff=<vs base> fraction_better=<f>
    adopted=<k>/<n> base_fixed_in_sample=<base pq>``. ``diff`` is measured against
    the base at its fixed setting, in sample (0.4255 on the holdout), which is
    stricter than the cross-fitted base (0.4210): both arms share the same frozen
    U-Net mask, so the selection bias of the threshold and the filter is in both.
    The last field prints that base so the line says what ``diff`` is against.
    """
    base_pq = summary.pooled_pq - summary.diff
    return (
        f"VERDICT {summary.config} pq={format_pq(summary.pooled_pq)} "
        f"diff={summary.diff:+.6f} fraction_better={summary.fraction_better:.3f} "
        f"adopted={summary.adopted}/{summary.partitions} "
        f"base_fixed_in_sample={format_pq(base_pq)}"
    )


def submit_without_asking(
    summary: CrossfitSummary, gain: float = SUBMIT_GAIN, *, pilot: bool = False
) -> bool:
    """True when a submission needs no OK from Nacho (AGENTS.md).

    All three must hold: the cross-fit gain over the base exceeds ``gain``; at
    least :data:`MIN_PARTITIONS_ADOPTED` partitions pass the roadmap's adoption
    rule (``summary.adopted``); and the run is not a pilot. The gain is compared at
    the six decimals the VERDICT line prints, so float noise on a gain of exactly
    0.01 does not decide it.
    """
    if pilot:
        return False
    if summary.adopted < MIN_PARTITIONS_ADOPTED:
        return False
    return round(summary.diff, 6) > gain
