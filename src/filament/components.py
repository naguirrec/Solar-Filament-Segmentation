"""Score each predicted component by its probability, and drop the weak ones.

Connected components keep every blob above ``min_area``, however unsure the
network was about it. ``docs/ROADMAP_V2.md`` section 4 shows that a candidate only
helps the pooled PQ when its expected matched IoU per view is above PQ / 2. A low
probability over a whole component is a cheap sign that it is not.

:func:`component_scores` gives each component its mean, maximum and 90th percentile
probability. :func:`keep_components` drops some ids and leaves the others
consecutive, and the pixels of a kept component are not changed. Masks stay
disjoint, since one label image cannot hold two ids on a pixel.

:func:`component_features` describes each component for the selector in
:mod:`filament.selector`, and :func:`component_view_hits` records which ground-truth
masks it matches, so the metric can be recomputed for any subset of components
without touching the pixels again.

Pure numpy, scikit-image and OpenCV. The component labels come from
:mod:`filament.instances`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import skeletonize

from filament.annotations import TrainImage, annotator_views
from filament.baseline import on_disk
from filament.instances import labels_for_min_areas
from filament.metrics import PqCounts, fast_view_counts
from filament.sweep import GridPoint, PerFileCounts, view_masks

OBJECT_STATISTICS = ("mean", "max", "p90")

FEATURE_NAMES = (
    "log_area",
    "skeleton_length",
    "mean_width",
    "elongation",
    "solidity",
    "probability_mean",
    "probability_max",
    "probability_p90",
    "flip_std_mean",
    "radial_distance",
    "touches_limb",
    "contrast",
)

# The ring around a component that its contrast is measured against, in pixels.
_RING_INNER = 5
_RING_OUTER = 15
# A component reaching this fraction of the disk radius touches the limb.
_LIMB_FRACTION = 0.985


@dataclass(frozen=True, slots=True)
class ComponentScores:
    """One value per component, indexed by ``label - 1``."""

    area: np.ndarray
    mean: np.ndarray
    max: np.ndarray
    p90: np.ndarray

    def __len__(self) -> int:
        return int(self.area.size)

    def statistic(self, name: str) -> np.ndarray:
        if name not in OBJECT_STATISTICS:
            raise ValueError(
                f"unknown statistic {name!r}, expected one of {OBJECT_STATISTICS}"
            )
        return getattr(self, name)


def component_scores(labels: np.ndarray, probability: np.ndarray) -> ComponentScores:
    """Summarize ``probability`` over each id ``1..labels.max()`` of a label image.

    The 90th percentile is nearest-rank, with no interpolation: the value at index
    ``ceil(0.9 * area) - 1`` of the component's sorted pixels. An id with no pixels
    gets zeros.
    """
    label_image = np.asarray(labels)
    values = np.asarray(probability, dtype=np.float32)
    if label_image.shape != values.shape:
        raise ValueError(
            f"labels {label_image.shape} and probability {values.shape} differ"
        )
    count = int(label_image.max()) if label_image.size else 0
    empty = np.zeros(count, dtype=np.float32)
    if count == 0:
        return ComponentScores(np.zeros(0, np.int64), empty, empty, empty)
    rows, cols = np.nonzero(label_image)
    ids = label_image[rows, cols]
    pixels = values[rows, cols]
    # Sorted by id, then by probability, so each component is one ascending run.
    order = np.lexsort((pixels, ids))
    ids, pixels = ids[order], pixels[order]
    area = np.bincount(ids, minlength=count + 1)[1:].astype(np.int64)
    ends = np.cumsum(area)
    starts = ends - area
    totals = np.bincount(ids, weights=pixels, minlength=count + 1)[1:]
    present = area > 0
    mean = np.zeros(count, dtype=np.float32)
    maximum = np.zeros(count, dtype=np.float32)
    p90 = np.zeros(count, dtype=np.float32)
    mean[present] = (totals[present] / area[present]).astype(np.float32)
    maximum[present] = pixels[ends[present] - 1]
    rank = np.ceil(0.9 * area[present]).astype(np.int64) - 1
    p90[present] = pixels[starts[present] + rank]
    return ComponentScores(area=area, mean=mean, max=maximum, p90=p90)


def keep_components(labels: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Drop the ids whose ``keep[id - 1]`` is false, and renumber the rest ``1..K``."""
    label_image = np.asarray(labels)
    mask = np.asarray(keep, dtype=bool)
    count = int(label_image.max()) if label_image.size else 0
    if mask.size != count:
        raise ValueError(f"{mask.size} flags for {count} components")
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[1:][mask] = np.arange(1, int(mask.sum()) + 1, dtype=np.int32)
    return lookup[label_image]


@dataclass(frozen=True, slots=True, order=True)
class ObjectCut:
    """Keep a component when its ``statistic`` probability is at least ``cut``."""

    statistic: str
    cut: float

    def __post_init__(self) -> None:
        if self.statistic not in OBJECT_STATISTICS:
            raise ValueError(
                f"unknown statistic {self.statistic!r}, "
                f"expected one of {OBJECT_STATISTICS}"
            )
        if not 0.0 <= self.cut < 1.0:
            raise ValueError(f"cut must be in [0, 1), got {self.cut!r}")

    def __str__(self) -> str:
        return f"{self.statistic}>={self.cut:g}"


# Cut 0 keeps everything, so it is the setting without the filter.
NO_OBJECT_FILTER = ObjectCut("mean", 0.0)


# The object filter adopted for submissions, and the post-processing it was
# validated on. From the B1 sweep (notebooks/sweep_object_filter.ipynb on the r34
# 80-epoch flip-TTA cache, results/2026-11-03_holdout_object_filter_r34.csv): the
# cross-fitted verdict adopted it (4 of 5 partitions), and the cut mean >= 0.8 is
# the one chosen in 7 of the 10 folds. The folds disagree on min_area, so it stays
# 400. See docs/RESULTS.md.
ADOPTED_OBJECT_FILTER: dict[str, object] | None = {"stat": "mean", "min": 0.8}
OBJECT_FILTER_VALIDATED_MODELS = {"r34": 1.0}
OBJECT_FILTER_VALIDATED_ON = {
    "threshold": 0.45,
    "close_size": 11,
    "min_area": 400,
    "tta": True,
}


def object_filter_mismatches(
    *,
    threshold: float,
    close_size: int,
    min_area: int,
    tta: bool,
    models: Mapping[str, float] | None = None,
) -> list[str]:
    """Settings that differ from the ones the adopted filter was validated on.

    The cut was chosen on flip-TTA probability maps of the r34 model alone, at the
    settings of ``OBJECT_FILTER_VALIDATED_ON``. With others it is untested, though
    it still runs. ``models`` are the model weights the maps come from; they are
    compared after normalizing to sum to 1. Weights without a positive sum (all zero,
    none at all, or NaN) raise ``ValueError``: they cannot be normalized.
    """
    used = {
        "threshold": threshold,
        "close_size": close_size,
        "min_area": min_area,
        "tta": tta,
    }
    found = [
        f"{name}={used[name]!r}, validated with {wanted!r}"
        for name, wanted in OBJECT_FILTER_VALIDATED_ON.items()
        if used[name] != wanted
    ]
    if models is not None:
        total = sum(float(weight) for weight in models.values())
        if not total > 0:
            raise ValueError(f"model weights must have a positive sum, got {models!r}")
        normalized = {name: float(weight) / total for name, weight in models.items()}
        same = set(normalized) == set(OBJECT_FILTER_VALIDATED_MODELS) and all(
            math.isclose(normalized[name], weight)
            for name, weight in OBJECT_FILTER_VALIDATED_MODELS.items()
        )
        if not same:
            found.append(
                f"models={normalized!r}, validated with "
                f"{OBJECT_FILTER_VALIDATED_MODELS!r}"
            )
    return found


def object_cut_from_config(config: Mapping[str, object] | None) -> ObjectCut | None:
    """Read a notebook setting such as ``{"stat": "mean", "min": 0.8}``.

    ``None`` means no filter. Any other key, or a missing one, is an error, so a
    typo cannot silently turn the filter off.
    """
    if config is None:
        return None
    keys = set(config)
    if keys != {"stat", "min"}:
        raise ValueError(
            f"an object filter needs exactly the keys 'stat' and 'min', got {sorted(keys)}"
        )
    return ObjectCut(str(config["stat"]), float(config["min"]))


def filter_components(
    labels: np.ndarray, probability: np.ndarray, cut: ObjectCut | None
) -> np.ndarray:
    """Drop the components whose probability statistic is below ``cut``.

    The kept ids are renumbered ``1..K`` and their pixels are not changed. With
    ``cut=None`` the label image is returned as it is, the same object, so the
    output without the filter is bit for bit the output before it.
    """
    if cut is None:
        return labels
    scores = component_scores(labels, probability)
    return keep_components(labels, scores.statistic(cut.statistic) >= cut.cut)


# Untuned grids for notebooks/sweep_object_filter.ipynb. A component's mean is at
# least the pixel threshold, so a cut at or below it would do nothing.
OBJECT_CUTS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
LOW_MIN_AREAS = (100, 160, 240, 320, 400, 500)
HIGH_THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def per_file_object_counts(
    holdout: Sequence[TrainImage],
    read_gray: Callable[[TrainImage], np.ndarray],
    load_probability: Callable[[TrainImage], np.ndarray],
    base: GridPoint,
    cuts: Sequence[ObjectCut],
    *,
    progress: Callable[[int, int, TrainImage], None] | None = None,
) -> PerFileCounts:
    """Score object filters on top of one post-processing setting, file by file.

    ``base`` gives the threshold, the closing and the area cut. Each file is
    labeled once and each cut only renumbers those labels. :data:`NO_OBJECT_FILTER`
    reproduces :func:`filament.sweep.per_file_counts` at ``base``.
    """
    if len(holdout) == 0:
        raise ValueError("holdout is empty")
    wanted = list(cuts)
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("cuts must be non-empty and distinct")
    rows: dict[ObjectCut, list[PqCounts]] = {cut: [] for cut in wanted}
    total = len(holdout)
    for index, image in enumerate(holdout, start=1):
        gray = read_gray(image)
        probability = np.asarray(load_probability(image))
        if probability.shape != gray.shape:
            raise ValueError(
                f"{image.file_name} probability {probability.shape} does not "
                f"match the frame {gray.shape}"
            )
        semantic = (probability >= base.threshold).astype(np.uint8)
        labels = labels_for_min_areas(
            semantic,
            gray,
            [base.min_area],
            close_size=base.close_size,
            disk=on_disk(gray),
        )[base.min_area]
        views = [view_masks(image, view) for view in annotator_views(image)]
        for cut in wanted:
            kept = filter_components(labels, probability, cut)
            running = PqCounts()
            for masks in views:
                running = running + fast_view_counts(kept, masks)
            rows[cut].append(running)
        if progress is not None:
            progress(index, total, image)
    return PerFileCounts(
        file_names=tuple(image.file_name for image in holdout),
        views=tuple(len(annotator_views(image)) for image in holdout),
        counts={cut: tuple(values) for cut, values in rows.items()},
    )


def _disk_geometry(disk: np.ndarray) -> tuple[float, float, float]:
    """Center row, center column and radius of a boolean disk mask."""
    rows, cols = np.nonzero(disk)
    if rows.size == 0:
        raise ValueError("the disk mask is empty")
    return float(rows.mean()), float(cols.mean()), float(np.sqrt(rows.size / np.pi))


def _ellipse(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _contrast(
    region_mask: np.ndarray,
    top: int,
    left: int,
    labels: np.ndarray,
    gray: np.ndarray,
    disk: np.ndarray,
) -> float:
    """How much darker a component is than the ring around it, relative to the ring.

    The ring leaves out every component and everything off the disk. A component
    with no ring pixels has contrast 0.
    """
    height, width = labels.shape
    pad = _RING_OUTER
    row0, col0 = max(top - pad, 0), max(left - pad, 0)
    row1 = min(top + region_mask.shape[0] + pad, height)
    col1 = min(left + region_mask.shape[1] + pad, width)
    mask = np.zeros((row1 - row0, col1 - col0), dtype=np.uint8)
    mask[
        top - row0 : top - row0 + region_mask.shape[0],
        left - col0 : left - col0 + region_mask.shape[1],
    ] = region_mask
    inner = cv2.dilate(mask, _ellipse(_RING_INNER))
    outer = cv2.dilate(mask, _ellipse(_RING_OUTER))
    ring = (
        (outer != 0)
        & (inner == 0)
        & (labels[row0:row1, col0:col1] == 0)
        & disk[row0:row1, col0:col1]
    )
    if not ring.any():
        return 0.0
    patch = gray[row0:row1, col0:col1]
    inside = float(np.median(patch[mask != 0]))
    around = float(np.median(patch[ring]))
    return (around - inside) / max(around, 1.0)


def component_features(
    labels: np.ndarray,
    probability: np.ndarray,
    flip_std: np.ndarray,
    gray: np.ndarray,
    disk: np.ndarray,
) -> np.ndarray:
    """Return ``(components, len(FEATURE_NAMES))`` float32 features, in id order.

    Shape comes from the component's own pixels, certainty from the probability and
    the spread between flips, position from the solar disk, and contrast from the
    H-alpha frame. Nothing here reads a label. No annotation metadata is used.
    A closing can give one id to separated fragments, and they are then measured
    as one component.
    """
    label_image = np.asarray(labels)
    if not (
        label_image.shape
        == np.shape(probability)
        == np.shape(flip_std)
        == np.shape(gray)
        == np.shape(disk)
    ):
        raise ValueError("labels, maps, frame and disk must share a shape")
    scores = component_scores(label_image, probability)
    count = len(scores)
    features = np.zeros((count, len(FEATURE_NAMES)), dtype=np.float32)
    if count == 0:
        return features
    center_row, center_col, radius = _disk_geometry(np.asarray(disk, dtype=bool))
    spread = np.asarray(flip_std, dtype=np.float32)
    frame = np.asarray(gray)
    gate = np.asarray(disk, dtype=bool)
    for region in regionprops(label_image.astype(np.int32)):
        index = region.label - 1
        coords = region.coords
        area = float(coords.shape[0])
        skeleton = float(np.count_nonzero(skeletonize(region.image)))
        major, minor = region.axis_major_length, region.axis_minor_length
        distances = np.hypot(coords[:, 0] - center_row, coords[:, 1] - center_col)
        centroid = np.hypot(
            region.centroid[0] - center_row, region.centroid[1] - center_col
        )
        top, left = int(region.bbox[0]), int(region.bbox[1])
        features[index] = (
            np.log(area),
            skeleton,
            area / max(skeleton, 1.0),
            major / max(minor, 1.0),
            region.solidity,
            scores.mean[index],
            scores.max[index],
            scores.p90[index],
            float(spread[coords[:, 0], coords[:, 1]].mean()),
            centroid / radius,
            float(distances.max() / radius >= _LIMB_FRACTION),
            _contrast(region.image, top, left, label_image, frame, gate),
        )
    return features


@dataclass(frozen=True, slots=True)
class ViewHits:
    """Every (component, ground-truth mask) pair with IoU > 0.5 in one view.

    ``candidate`` holds zero-based component indexes. ``ground_truths`` is how many
    masks the view has, matched or not.
    """

    candidate: np.ndarray
    ground_truth: np.ndarray
    iou: np.ndarray
    ground_truths: int


def component_view_hits(labels: np.ndarray, masks: Sequence[np.ndarray]) -> ViewHits:
    """Record the matches :func:`filament.metrics.fast_view_counts` counts."""
    label_image = np.asarray(labels)
    count = int(label_image.max()) if label_image.size else 0
    area = np.bincount(label_image.ravel(), minlength=count + 1).astype(np.float64)
    candidate: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    iou_values: list[np.ndarray] = []
    for index, mask in enumerate(masks):
        rows, cols = np.nonzero(np.asarray(mask))
        inter = np.bincount(label_image[rows, cols], minlength=count + 1).astype(
            np.float64
        )
        inter[0] = 0.0
        union = area + float(rows.size) - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        hits = np.flatnonzero(iou > 0.5)
        candidate.append(hits - 1)
        ground_truth.append(np.full(hits.size, index, dtype=np.int64))
        iou_values.append(iou[hits])
    empty = np.zeros(0, dtype=np.int64)
    return ViewHits(
        candidate=np.concatenate(candidate) if candidate else empty,
        ground_truth=np.concatenate(ground_truth) if ground_truth else empty,
        iou=np.concatenate(iou_values) if iou_values else np.zeros(0),
        ground_truths=len(masks),
    )


def selection_counts(views: Sequence[ViewHits], emit: np.ndarray) -> PqCounts:
    """PQ counts of one file if only the components in ``emit`` are submitted.

    The same rule as :func:`filament.metrics.fast_view_counts`, without the pixels.
    Components are disjoint, so whether one matches a mask does not depend on which
    others are emitted.
    """
    chosen = np.asarray(emit, dtype=bool)
    emitted = int(chosen.sum())
    total = PqCounts()
    for view in views:
        keep = chosen[view.candidate]
        matched_masks = np.unique(view.ground_truth[keep]).size
        matched_components = np.unique(view.candidate[keep]).size
        total = total + PqCounts(
            tp=int(keep.sum()),
            fp=emitted - matched_components,
            fn=view.ground_truths - matched_masks,
            iou_sum=float(view.iou[keep].sum()),
        )
    return total


def expected_matched_iou(views: Sequence[ViewHits], components: int) -> np.ndarray:
    """``(1/R) * sum over views of matched IoU``, per component: the selector's target.

    ``docs/ROADMAP_V2.md`` section 4: emitting a component helps the pooled PQ
    exactly when this is above PQ / 2.
    """
    target = np.zeros(components, dtype=np.float64)
    for view in views:
        np.add.at(target, view.candidate, view.iou)
    return target / max(len(views), 1)
