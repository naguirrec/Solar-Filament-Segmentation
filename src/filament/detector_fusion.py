"""Fuse instance detections with the U-Net mask: the detector groups, the U-Net draws.

Phase F of ``docs/ROADMAP_V2.md``. A detector (RF-DETR-Seg) predicts on overlapping
windows of a 2048x2048 frame. It says which pixels go together and how sure it is,
but at its native resolution it loses thin structures. The U-Net has the pixels.
This module joins them:

1. :func:`merge_across_windows`: one filament seen by two windows becomes one
   detection. Two detections are compared only on the pixels both windows see.
2. :func:`dedup_detections`: of detections that overlap, keep the best-scored.
3. :func:`claim_pixels`: in score order, each detection claims the U-Net pixels
   inside its dilated mask that nobody claimed before. The labels are disjoint by
   construction, so :func:`filament.metrics.fast_view_counts` scores them exactly.
4. :func:`keep_scored`: a score cut that renumbers the ids, so a sweep can score
   many cuts from one claim.
5. :func:`isotonic_calibrator`, :func:`crossfit_calibrated_scores` and
   :func:`emit_by_pooled_pq`: calibrate the scores out of fold, by date, and
   emit a detection when its calibrated score is above ``lambda / 2``, with
   ``lambda`` the pooled PQ (``docs/ROADMAP_V2.md`` section 4).
6. :func:`per_file_detector_counts`: per-file :class:`filament.metrics.PqCounts`
   of a configuration, the input of :mod:`filament.stats`.
7. :func:`merge_components`: the opposite use, "merge only". The base labels stay and
   a detection may only join the base components it covers; it never splits,
   drops or moves anything.

Every default in this module is untuned. None was fitted on the holdout. Each
docstring says so where it names one. The 0.05 of :func:`dedup_detections` is another
team's plateau (0.03 to 0.1 on their own detectors), not a measurement here. Nothing
in this module has run on real data: the tests are synthetic.

Numpy and OpenCV. scikit-learn is used only for the isotonic fit. No torch.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from sklearn.isotonic import IsotonicRegression

from filament.components import (
    FEATURE_NAMES,
    ViewHits,
    component_view_hits,
    expected_matched_iou,
    selection_counts,
)
from filament.metrics import PqCounts, fast_view_counts
from filament.rle import decode_mask, encode_mask
from filament.selector import FileCandidates, dinkelbach_lambda
from filament.stats import date_folds

# A rectangle in frame coordinates, ``(top, left, bottom, right)``, ends excluded.
Rect = tuple[int, int, int, int]

# Untuned defaults.
DEFAULT_WINDOWS_IOU = 0.5
# Another team's plateau, 0.03 to 0.1, on their own detectors. Not tuned here.
DEFAULT_DEDUP_IOU = 0.05
DEFAULT_DILATION = 5
DEFAULT_MIN_AREA = 400
EMISSION_ROUNDS = 5

_NO_DAY = dt.date(1970, 1, 1)


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected filament, already in frame coordinates.

    ``mask`` is the pycocotools compressed RLE counts string of the frame-sized
    mask (see :mod:`filament.rle`), and ``shape`` is the frame ``(height, width)``
    that goes with it. It is not a frame-sized boolean array on purpose: a frame
    holds many detections, and 2048x2048 booleans are 4 MiB each, so a few hundred
    detections would be gigabytes while their RLE is a few kilobytes at most for a
    long thin filament (the tests check one 800 px mask: about 1.6 KB). ``bbox``
    is the box of the mask, kept next to it so that most pair tests need no decoding.
    Build one with :meth:`from_mask`, which fills ``mask`` and ``bbox`` from an
    array. ``window`` is the index of the window that produced it. ``score`` is
    in ``[0, 1]``.
    """

    score: float
    mask: str
    shape: tuple[int, int]
    window: int
    bbox: Rect

    def __post_init__(self) -> None:
        if not np.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score!r}")
        if self.window < 0:
            raise ValueError(f"window index must be >= 0, got {self.window!r}")
        height, width = self.shape
        top, left, bottom, right = self.bbox
        if height < 1 or width < 1:
            raise ValueError(f"shape must be positive, got {self.shape!r}")
        if not (0 <= top < bottom <= height and 0 <= left < right <= width):
            raise ValueError(f"bbox {self.bbox!r} is empty or outside {self.shape!r}")

    @classmethod
    def from_mask(cls, mask: np.ndarray, score: float, window: int) -> Detection:
        """Encode a 2D mask. An empty mask is an error: it is not a detection."""
        array = np.asarray(mask) != 0
        if array.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {array.shape}")
        rows = np.flatnonzero(array.any(axis=1))
        if rows.size == 0:
            raise ValueError("a detection needs at least one pixel")
        cols = np.flatnonzero(array.any(axis=0))
        box = (int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1)
        return cls(
            score=float(score),
            mask=encode_mask(array),
            shape=(int(array.shape[0]), int(array.shape[1])),
            window=int(window),
            bbox=box,
        )

    def decode(self) -> np.ndarray:
        """The frame-sized boolean mask."""
        return decode_mask(self.mask, *self.shape).astype(bool)

    def crop(self) -> np.ndarray:
        """The mask inside ``bbox`` only, as its own array (not a view of the frame)."""
        top, left, bottom, right = self.bbox
        return self.decode()[top:bottom, left:right].copy()


def _order_key(detection: Detection) -> tuple:
    """Highest score first; equal scores by window, box and mask, so it is total."""
    return (-detection.score, detection.window, detection.bbox, detection.mask)


def _rect_and(first: Rect, second: Rect) -> Rect | None:
    top, left = max(first[0], second[0]), max(first[1], second[1])
    bottom, right = min(first[2], second[2]), min(first[3], second[3])
    if bottom <= top or right <= left:
        return None
    return (top, left, bottom, right)


def _sub(crop: np.ndarray, box: Rect, rect: Rect) -> np.ndarray:
    """The part of a box's crop that lies inside ``rect``. ``rect`` is within the box."""
    return crop[
        rect[0] - box[0] : rect[2] - box[0], rect[1] - box[1] : rect[3] - box[1]
    ]


def _inter_union(
    crop_a: np.ndarray, box_a: Rect, crop_b: np.ndarray, box_b: Rect, region: Rect
) -> tuple[int, int]:
    """Pixels in both masks, and in either, counting only pixels inside ``region``."""
    area_a = area_b = both = 0
    inside_a = _rect_and(box_a, region)
    inside_b = _rect_and(box_b, region)
    if inside_a is not None:
        area_a = int(np.count_nonzero(_sub(crop_a, box_a, inside_a)))
    if inside_b is not None:
        area_b = int(np.count_nonzero(_sub(crop_b, box_b, inside_b)))
    if inside_a is not None and inside_b is not None:
        shared = _rect_and(inside_a, inside_b)
        if shared is not None:
            both = int(
                np.count_nonzero(
                    _sub(crop_a, box_a, shared) & _sub(crop_b, box_b, shared)
                )
            )
    return both, area_a + area_b - both


def _same_shape(detections: Sequence[Detection]) -> tuple[int, int] | None:
    shapes = {d.shape for d in detections}
    if len(shapes) > 1:
        raise ValueError(f"detections have different frame shapes: {sorted(shapes)}")
    return next(iter(shapes)) if shapes else None


def merge_across_windows(
    detections: Sequence[Detection],
    windows: Sequence[Rect],
    windows_iou: float = DEFAULT_WINDOWS_IOU,
) -> list[Detection]:
    """Merge detections of different windows that are one filament seen twice.

    ``windows[i]`` is the frame rectangle ``(top, left, bottom, right)`` that
    window ``i`` sees, and ``Detection.window`` indexes it. (The signature has
    ``windows`` besides the threshold because a window index alone does not say
    which pixels two windows share.) A filament that crosses a window edge is
    clipped in both views, so their masks differ even when both are right, and a
    plain IoU would call them different filaments. Two detections are therefore
    compared on the rectangle both windows can see: the IoU of their masks
    restricted to it. When that IoU is above ``windows_iou`` they are merged.
    Detections of the same window never merge here (see :func:`dedup_detections`),
    nor do two whose windows share no pixels, nor two with no pixel in the shared
    rectangle.

    Merging is transitive, so a filament seen by three windows is one detection.
    A merged detection has the union of the masks, the highest score, and the
    window of the member with that score. The result is sorted by score, highest
    first, and does not depend on the input order. ``windows_iou`` is untuned.
    """
    if not 0.0 <= windows_iou < 1.0:
        raise ValueError(f"windows_iou must be in [0, 1), got {windows_iou!r}")
    shape = _same_shape(detections)
    for detection in detections:
        if detection.window >= len(windows):
            raise ValueError(
                f"window {detection.window} has no rectangle, {len(windows)} given"
            )
    ordered = sorted(detections, key=_order_key)
    count = len(ordered)
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    crops: dict[int, np.ndarray] = {}

    def crop_of(index: int) -> np.ndarray:
        if index not in crops:
            crops[index] = ordered[index].crop()
        return crops[index]

    for i in range(count):
        first = ordered[i]
        for j in range(i + 1, count):
            second = ordered[j]
            if first.window == second.window:
                continue
            region = _rect_and(windows[first.window], windows[second.window])
            if region is None:
                continue
            # Cheap reject on boxes before decoding anything.
            in_a = _rect_and(first.bbox, region)
            in_b = _rect_and(second.bbox, region)
            if in_a is None or in_b is None or _rect_and(in_a, in_b) is None:
                continue
            both, either = _inter_union(
                crop_of(i), first.bbox, crop_of(j), second.bbox, region
            )
            if either > 0 and both / either > windows_iou:
                low, high = sorted((find(i), find(j)))
                parent[high] = low

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    merged: list[Detection] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(ordered[members[0]])
            continue
        assert shape is not None
        union = np.zeros(shape, dtype=bool)
        for index in members:
            top, left, bottom, right = ordered[index].bbox
            union[top:bottom, left:right] |= crop_of(index)
        best = ordered[members[0]]  # members are in order: the best score is first
        merged.append(Detection.from_mask(union, best.score, best.window))
    return sorted(merged, key=_order_key)


def dedup_detections(
    detections: Sequence[Detection], iou: float = DEFAULT_DEDUP_IOU
) -> list[Detection]:
    """Greedy suppression: highest score first, drop what overlaps a kept one.

    A detection is dropped when its IoU with an already kept detection is above
    ``iou``. Order is by descending score, then by window, box and mask, so the
    result does not depend on the input order and a tie always breaks the same way.
    The default 0.05 is the plateau (0.03 to 0.1) another team found for their own
    detectors. It has not been tuned here, and the threshold is low on purpose: two
    masks that touch at all are very likely one filament.
    """
    if not 0.0 <= iou < 1.0:
        raise ValueError(f"iou must be in [0, 1), got {iou!r}")
    shape = _same_shape(detections)
    if shape is None:
        return []
    frame: Rect = (0, 0, shape[0], shape[1])
    kept: list[tuple[Detection, np.ndarray]] = []
    for detection in sorted(detections, key=_order_key):
        crop = None
        duplicate = False
        for other, other_crop in kept:
            if _rect_and(detection.bbox, other.bbox) is None:
                continue
            if crop is None:
                crop = detection.crop()
            both, either = _inter_union(
                crop, detection.bbox, other_crop, other.bbox, frame
            )
            if either > 0 and both / either > iou:
                duplicate = True
                break
        if not duplicate:
            kept.append((detection, crop if crop is not None else detection.crop()))
    return [detection for detection, _ in kept]


def _ellipse(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def claim_pixels(
    detections: Sequence[Detection],
    unet_mask: np.ndarray,
    dilation: int = DEFAULT_DILATION,
    min_area: int = DEFAULT_MIN_AREA,
    *,
    emit_unclaimed: bool = False,
    disk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Give each detection the U-Net pixels it covers, best score first.

    In descending score order (ties as in :func:`dedup_detections`, so the result
    does not depend on the input order) each detection claims the pixels of
    ``unet_mask`` that lie inside its mask dilated by an ellipse of radius
    ``dilation`` and that no earlier detection claimed. The detector's mask is
    coarse, hence the dilation; the U-Net decides which pixels are filament.

    Returns ``(labels, scores)``. ``labels`` is an int32 image, ``0`` background and
    ids ``1..n`` with no gaps, disjoint by construction. ``scores[i - 1]`` is the
    detector score of id ``i`` (float64). An id whose total claimed area is under
    ``min_area`` is dropped and the ids are compacted; ids with no pixel are always
    dropped. The area is the id's whole claim, not each connected piece: a detection
    that claims several disconnected pieces stays one id, which is what the
    detector is for. Pixels of a dropped id are not returned to the unclaimed pool.

    By default U-Net pixels nobody claimed are not emitted. With
    ``emit_unclaimed`` each 8-connected component of them with at least
    ``min_area`` pixels gets its own id after the detections' ids, and a NaN score.
    A score cut then needs ``keep_unscored`` in :func:`keep_scored`. Off by default
    because such pieces are mostly false positives (``docs/ROADMAP_V2.md`` phase F).

    ``disk`` is the solar-disk gate of the current pipeline
    (:func:`filament.baseline.on_disk`). When given, only U-Net pixels on the disk
    can be claimed or emitted, the gate :func:`filament.instances.labels_for_min_areas`
    applies before it labels. A U-Net mask that came out of that function is already
    gated, so this only matters for a mask that did not. ``None`` applies no gate.

    ``dilation`` (5 px) and ``min_area`` (400, the repo-wide cut) are untuned here.
    """
    if dilation < 0:
        raise ValueError(f"dilation must be >= 0, got {dilation!r}")
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area!r}")
    unet = np.asarray(unet_mask) != 0
    if unet.ndim != 2:
        raise ValueError(f"unet_mask must be 2D, got shape {unet.shape}")
    if disk is not None:
        gate = np.asarray(disk, dtype=bool)
        if gate.shape != unet.shape:
            raise ValueError(f"disk {gate.shape} and unet_mask {unet.shape} differ")
        unet = unet & gate
    shape = _same_shape(detections)
    if shape is not None and shape != unet.shape:
        raise ValueError(f"detections {shape} and unet_mask {unet.shape} differ")
    height, width = unet.shape
    ordered = sorted(detections, key=_order_key)
    count = len(ordered)
    labels = np.zeros((height, width), dtype=np.int32)
    kernel = _ellipse(dilation) if dilation > 0 else None
    for index, detection in enumerate(ordered, start=1):
        top, left, bottom, right = detection.bbox
        # Pad by the radius so the dilation is not cut at the box, only at the frame.
        row0, col0 = max(top - dilation, 0), max(left - dilation, 0)
        row1, col1 = min(bottom + dilation, height), min(right + dilation, width)
        patch = np.zeros((row1 - row0, col1 - col0), dtype=np.uint8)
        patch[top - row0 : bottom - row0, left - col0 : right - col0] = detection.crop()
        if kernel is not None:
            patch = cv2.dilate(patch, kernel)
        window = labels[row0:row1, col0:col1]
        window[(patch != 0) & unet[row0:row1, col0:col1] & (window == 0)] = index

    claimed_anywhere = labels > 0
    area = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    keep = (area >= min_area) & (area > 0)
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[1:][keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    labels = lookup[labels]
    scores = np.array([d.score for d in ordered], dtype=np.float64)[keep]

    if emit_unclaimed:
        leftover = (unet & ~claimed_anywhere).astype(np.uint8)
        total, pieces, stats, _ = cv2.connectedComponentsWithStats(
            leftover, connectivity=8
        )
        extra = [
            piece
            for piece in range(1, total)
            if stats[piece, cv2.CC_STAT_AREA] >= max(min_area, 1)
        ]
        mapping = np.zeros(total, dtype=np.int32)
        for offset, piece in enumerate(extra):
            mapping[piece] = scores.size + 1 + offset
        labels = np.where(mapping[pieces] > 0, mapping[pieces], labels).astype(np.int32)
        scores = np.concatenate([scores, np.full(len(extra), np.nan)])
    return labels, scores


def keep_scored(
    labels: np.ndarray,
    scores: np.ndarray,
    min_score: float,
    *,
    keep_unscored: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the ids with ``score >= min_score``, renumbered ``1..K`` with no gaps.

    ``labels`` and ``scores`` are the output of :func:`claim_pixels` (or of a
    previous cut), so one claim serves a sweep over many cuts. The pixels of a kept
    id do not change. A NaN score (an unclaimed U-Net piece) fails any cut unless
    ``keep_unscored`` is set.
    """
    values = np.asarray(scores, dtype=np.float64)
    label_image = np.asarray(labels)
    count = int(label_image.max()) if label_image.size else 0
    if values.size != count:
        raise ValueError(f"{values.size} scores for {count} ids")
    passes = values >= min_score
    if keep_unscored:
        passes |= np.isnan(values)
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[1:][passes] = np.arange(1, int(passes.sum()) + 1, dtype=np.int32)
    return lookup[label_image], values[passes]


@dataclass(frozen=True, slots=True, eq=False)
class IsotonicCalibrator:
    """A fitted monotone map from a detector score to an expected matched IoU.

    The knots of the isotonic fit, evaluated by linear interpolation and clamped
    at both ends. NaN stays NaN.
    """

    knots: np.ndarray
    values: np.ndarray

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(scores, dtype=np.float64), self.knots, self.values)


def _checked_pairs(scores: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, ...]:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f"scores {x.shape} and targets {y.shape} must be 1D and equal")
    if x.size == 0:
        raise ValueError("no pairs to fit")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("scores and targets must be finite")
    # No upper bound: the roadmap target sums IoU over every matched pair, so one id
    # that matches two overlapping masks of a view can exceed 1. It is not clipped,
    # because the unclipped value is what the metric counts.
    if np.any(y < 0):
        raise ValueError("targets must be >= 0")
    return x, y


def isotonic_calibrator(scores: np.ndarray, targets: np.ndarray) -> IsotonicCalibrator:
    """Fit an increasing (monotone) map from ``scores`` to ``targets``.

    ``targets`` is the roadmap's ``(1/R) * sum_r IoU_r * [IoU_r > 0.5]`` per
    detection (see :func:`fusion_targets`). To report a number, fit it on pairs
    that are out of fold (:func:`crossfit_calibrated_scores`): a calibrator fitted
    and scored on the same pairs is optimistic. Scores outside the fitted range are
    clamped to the end values.
    """
    x, y = _checked_pairs(scores, targets)
    fit = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
    return IsotonicCalibrator(
        knots=np.asarray(fit.X_thresholds_, dtype=np.float64),
        values=np.asarray(fit.y_thresholds_, dtype=np.float64),
    )


def crossfit_calibrated_scores(
    scores: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[dt.date],
    n_splits: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Calibrated scores where no date is fitted and predicted together.

    ``groups[i]`` is the date of candidate ``i``. The dates are dealt into
    ``n_splits`` folds by :func:`filament.stats.date_folds` (all candidates of a
    date stay in one fold, folds of similar size, a different partition per
    ``seed``). Each fold is scored by a calibrator fitted on the other folds. Pass
    only scored candidates: a NaN score cannot be calibrated. ``n_splits`` and the
    partition are untuned.
    """
    x, y = _checked_pairs(scores, targets)
    if len(groups) != x.size:
        raise ValueError(f"{len(groups)} groups for {x.size} candidates")
    folds = date_folds(groups, n_splits, seed=seed)
    out = np.zeros_like(x)
    for fold in sorted(set(folds.tolist())):
        inside = folds == fold
        out[inside] = isotonic_calibrator(x[~inside], y[~inside])(x[inside])
    return out


def fusion_targets(
    labels: np.ndarray, views: Sequence[Sequence[np.ndarray]]
) -> tuple[tuple[ViewHits, ...], np.ndarray]:
    """Each id's matches per view, and its calibration target.

    ``views`` is one list of ground-truth masks per annotator view, as
    :func:`filament.sweep.view_masks` returns. The target of id ``i`` is
    ``expected_matched_iou``: ``(1/R) * sum_r IoU_r * [IoU_r > 0.5]``.
    """
    hits = tuple(component_view_hits(labels, masks) for masks in views)
    count = int(np.asarray(labels).max()) if np.asarray(labels).size else 0
    return hits, expected_matched_iou(hits, count)


def emission_candidates(
    file_name: str, views: Sequence[ViewHits], count: int
) -> FileCandidates:
    """Wrap one file's hits as the selector's :class:`FileCandidates`.

    The features are zeros: the emission rule reads only the hits and the scores, so
    :func:`filament.selector.dinkelbach_lambda` and
    :func:`filament.selector.crossfit_emission` accept the result as it is. The
    ``day`` is a placeholder.
    """
    return FileCandidates(
        file_name=file_name,
        day=_NO_DAY,
        features=np.zeros((count, len(FEATURE_NAMES)), dtype=np.float32),
        views=tuple(views),
        target=expected_matched_iou(views, count),
    )


@dataclass(frozen=True, slots=True)
class Emission:
    """The result of :func:`emit_by_pooled_pq`: ``lam`` and what it emits per file."""

    lam: float
    emit: tuple[np.ndarray, ...]
    counts: tuple[PqCounts, ...]


def emit_by_pooled_pq(
    views_per_file: Sequence[Sequence[ViewHits]],
    scores_per_file: Sequence[np.ndarray],
    *,
    rounds: int = EMISSION_ROUNDS,
) -> Emission:
    """Emit a detection when its calibrated score is above ``lambda / 2``.

    ``lambda`` is the pooled PQ of the set that rule emits, which depends on
    ``lambda`` itself, so :func:`filament.selector.dinkelbach_lambda` iterates it
    from 0 until it stops changing (1e-4) or for at most ``rounds`` rounds
    (5, untuned). The iteration is the selector's, not a copy. ``lambda`` here is
    pooled over the files given, so on the files that fitted the calibrator it is
    optimistic. To report a number, take ``lambda`` from the other folds, as
    :func:`filament.selector.crossfit_emission` does with :func:`emission_candidates`.
    """
    if len(views_per_file) != len(scores_per_file):
        raise ValueError(
            f"{len(views_per_file)} files of hits, {len(scores_per_file)} of scores"
        )
    files = []
    scores = []
    for index, (views, score) in enumerate(
        zip(views_per_file, scores_per_file, strict=True)
    ):
        values = np.asarray(score, dtype=np.float64)
        files.append(emission_candidates(f"file{index}", views, values.size))
        scores.append(values)
    lam = dinkelbach_lambda(files, scores, iterations=rounds)
    emit = tuple(values > lam / 2.0 for values in scores)
    counts = tuple(
        selection_counts(item.views, chosen)
        for item, chosen in zip(files, emit, strict=True)
    )
    return Emission(lam=lam, emit=emit, counts=counts)


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """One setting of the fusion. Hashable, so it can key a grid for ``stats``.

    Every default is untuned: none was fitted on the holdout. ``min_score`` is a
    cut on the scores in use (raw, or the ones passed to
    :func:`per_file_detector_counts`); ``None`` applies no cut.
    """

    windows_iou: float = DEFAULT_WINDOWS_IOU
    dedup_iou: float = DEFAULT_DEDUP_IOU
    dilation: int = DEFAULT_DILATION
    min_area: int = DEFAULT_MIN_AREA
    min_score: float | None = None
    emit_unclaimed: bool = False


DEFAULT_CONFIG = FusionConfig()


def _claim_with(
    detections: Sequence[Detection],
    unet_mask: np.ndarray,
    windows: Sequence[Rect],
    config: FusionConfig,
    disk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    merged = merge_across_windows(detections, windows, config.windows_iou)
    unique = dedup_detections(merged, config.dedup_iou)
    return claim_pixels(
        unique,
        unet_mask,
        config.dilation,
        config.min_area,
        emit_unclaimed=config.emit_unclaimed,
        disk=disk,
    )


def fuse(
    detections: Sequence[Detection],
    unet_mask: np.ndarray,
    windows: Sequence[Rect],
    config: FusionConfig = DEFAULT_CONFIG,
    disk: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge across windows, deduplicate, claim, then cut by ``config.min_score``.

    ``disk`` is the optional solar-disk gate of :func:`claim_pixels`.
    """
    labels, scores = _claim_with(detections, unet_mask, windows, config, disk)
    if config.min_score is not None:
        labels, scores = keep_scored(
            labels, scores, config.min_score, keep_unscored=config.emit_unclaimed
        )
    return labels, scores


@dataclass(frozen=True, slots=True)
class FusionFile:
    """What one frame needs: its detections, the U-Net mask and the annotator views.

    ``views`` is one list of ground-truth masks per annotator view, as
    :func:`filament.sweep.view_masks` returns (an annotator who drew nothing is an
    empty list and still a view). ``disk`` is the optional gate of
    :func:`claim_pixels`.
    """

    file_name: str
    detections: Sequence[Detection]
    unet_mask: np.ndarray
    views: Sequence[Sequence[np.ndarray]]
    disk: np.ndarray | None = None


def per_file_detector_counts(
    files: Iterable[FusionFile],
    windows: Sequence[Rect],
    config: FusionConfig = DEFAULT_CONFIG,
    *,
    scores: Sequence[np.ndarray] | None = None,
) -> tuple[PqCounts, ...]:
    """Per-file :class:`PqCounts` of one configuration, in the order of ``files``.

    This is what :func:`filament.stats.paired_bootstrap` and
    :func:`filament.stats.crossfit_select` take, so a notebook can compare a
    configuration with the current best on the same files. Each view is scored on
    its own with :func:`filament.metrics.fast_view_counts` and the counts of a file
    are added, as :func:`filament.sweep.per_file_counts` does.

    ``scores``, when given, replaces the detector scores of each file, in the
    order of the ids that :func:`fuse` returns before ``config.min_score`` is
    applied (for example the cross-fitted calibrated scores). Each entry must have
    one value per claimed id. The claim itself always uses the raw scores.
    """
    results: list[PqCounts] = []
    for index, item in enumerate(files):
        labels, raw = _claim_with(
            item.detections, item.unet_mask, windows, config, item.disk
        )
        if scores is not None:
            if index >= len(scores):
                raise ValueError(f"no scores for file {index} ({item.file_name})")
            ids = int(labels.max()) if labels.size else 0
            raw = np.asarray(scores[index], dtype=np.float64)
            if raw.shape != (ids,):
                raise ValueError(f"{item.file_name}: {raw.size} scores for {ids} ids")
        if config.min_score is not None:
            labels, raw = keep_scored(
                labels, raw, config.min_score, keep_unscored=config.emit_unclaimed
            )
        total = PqCounts()
        for masks in item.views:
            total = total + fast_view_counts(labels, masks)
        results.append(total)
    if scores is not None and len(scores) != len(results):
        raise ValueError(f"{len(scores)} score arrays for {len(results)} files")
    return tuple(results)


def merge_components(
    labels: np.ndarray,
    detections: Sequence[Detection],
    cut: float = 0.5,
    coverage: float = 0.5,
    dilation: int = DEFAULT_DILATION,
) -> np.ndarray:
    """Join components of ``labels`` that one detection covers. Nothing else changes.

    The "merge only" use of the detector (``docs/ROADMAP_V2.md`` phase F, second
    try). ``labels`` is the base pipeline's label image: ``0`` background, ids
    ``1..n``, disjoint. A detection with ``score >= cut`` has its mask dilated by an
    ellipse of radius ``dilation`` (the same as :func:`claim_pixels`). A component
    is *covered* by it when at least ``coverage`` of the component's area lies
    inside that dilated mask. A detection that covers two or more components joins
    them, and joins are transitive (union-find): if one detection joins A and B and
    another B and C, the result is one component A+B+C.

    The detector never splits a component, never drops one and never changes a
    pixel: the set of non-zero pixels is the input's, and each output id is a union
    of whole input ids. Detections should already be merged across windows (and
    optionally deduplicated); this function does neither.

    Returns an int32 label image with ids ``1..m`` and no gaps, numbered by the
    smallest input id of each group, so with nothing to join it equals ``labels``.
    The number of joins performed is ``labels.max() - result.max()``. ``cut``,
    ``coverage`` and ``dilation`` are untuned placeholders.
    """
    if not 0.0 <= cut <= 1.0:
        raise ValueError(f"cut must be in [0, 1], got {cut!r}")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage!r}")
    if dilation < 0:
        raise ValueError(f"dilation must be >= 0, got {dilation!r}")
    base = np.asarray(labels)
    if base.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {base.shape}")
    if not np.issubdtype(base.dtype, np.integer):
        raise ValueError(f"labels must be an integer image, got dtype {base.dtype}")
    if base.size and int(base.min()) < 0:
        raise ValueError("labels must be non-negative")
    shape = _same_shape(detections)
    if shape is not None and shape != base.shape:
        raise ValueError(f"detections {shape} and labels {base.shape} differ")
    count = int(base.max()) if base.size else 0
    if count == 0:
        return base.astype(np.int32)
    area = np.bincount(base.ravel(), minlength=count + 1)
    parent = list(range(count + 1))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    height, width = base.shape
    kernel = _ellipse(dilation) if dilation > 0 else None
    for detection in detections:
        if detection.score < cut:
            continue
        top, left, bottom, right = detection.bbox
        # Pad by the radius so the dilation is not cut at the box, only at the frame.
        row0, col0 = max(top - dilation, 0), max(left - dilation, 0)
        row1, col1 = min(bottom + dilation, height), min(right + dilation, width)
        patch = np.zeros((row1 - row0, col1 - col0), dtype=np.uint8)
        patch[top - row0 : bottom - row0, left - col0 : right - col0] = detection.crop()
        if kernel is not None:
            patch = cv2.dilate(patch, kernel)
        inside = np.bincount(
            base[row0:row1, col0:col1][patch != 0], minlength=count + 1
        )
        inside[0] = 0
        covered = np.flatnonzero((area > 0) & (inside >= coverage * area))
        for other in covered[1:]:
            low, high = sorted((find(int(covered[0])), find(int(other))))
            parent[high] = low

    present = np.flatnonzero(area[1:] > 0) + 1
    roots = np.array([find(int(item)) for item in present], dtype=np.int64)
    # A root is the smallest id of its group, so sorting the roots numbers the groups.
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[present] = np.searchsorted(np.unique(roots), roots) + 1
    return lookup[base]
