"""Semantic targets and a calendar-date holdout for U-Net training.

Images in the competition are 2048×2048. Training starts at 512×512 so a
free Kaggle GPU session can finish. 512 is an untuned starting size, not
a tuned hyperparameter.

The semantic target is the union of every filament on an image. All four
categories are one class. Building that target does not read ``category_id``.

The holdout is the most recent 20% of calendar dates. A date is taken
whole. Pixels are not split. The date rule lives in
:func:`filament.split.split_by_date`.

Repeated ``file_name`` values are grouped first, then that date split
runs. A file is one sample: it is not counted twice and it is not placed
on both sides. The semantic target is the union of that file's polygons.
The audit does not forbid a union as the training target.

The collapsed ``annotations`` list is a stand-in from an earlier score.
Exact duplicate polygons are dropped. Instances from different entries
of the same file with IoU greater than 0.5 keep one mask, not the union
of their polygons. The other polygons stay on ``semantic_only`` so the
semantic union still covers them. That reduced list is not the metric.
Each original entry is stored on ``views`` and scored on its own.
Whether a repeated file name means the same image was annotated more
than once is [VERIFICAR].

Panoptic quality pools those views with
:func:`filament.metrics.fast_view_counts`. Instances still come from
:func:`filament.instances.labels_from_semantic`. The area cut is that
module's default (400 at 2048, 25 at 512). The close size is that
module's default (0). This module does not choose another cutoff or
kernel. Both defaults come from the 2026-11-01 holdout sweep
(``docs/RESULTS.md``). One image's PQ is
not averaged with other images.
:func:`filament.baseline_holdout.score_holdout` pools the holdout.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from filament.annotations import (
    FilamentAnnotation,
    TrainImage,
    annotator_views,
    rasterize_polygons,
)
from filament.instances import labels_from_semantic
from filament.metrics import PqCounts, fast_view_counts
from filament.split import observation_date, split_by_date

logger = logging.getLogger(__name__)

# Untuned. 2048 divided by 4. Chosen so a free T4 session can finish.
FULL_DISK_SIDE = 2048
# Share of calendar dates held out. The holdout every model is scored on.
DEFAULT_HOLDOUT_FRACTION = 0.2
TRAIN_SIDE = 512


@dataclass(frozen=True, slots=True)
class TrainingSplit:
    """Train images and the date holdout from :func:`split_by_date`.

    ``holdout`` is that function's validation side. Dates are oldest first
    on each side. A file name that occurred once, and whose annotations
    had no exact duplicate polygon, is the same object that was passed in.
    A repeated file name is one collapsed image. ``annotations`` is the
    stand-in panoptic list. ``views`` is one tuple per original entry.
    """

    train: tuple[TrainImage, ...]
    holdout: tuple[TrainImage, ...]


def semantic_target(
    segmentations: Sequence[Sequence[Sequence[float]]],
    height: int,
    width: int,
) -> np.ndarray:
    """Union every polygon list into one uint8 mask of shape ``(height, width)``.

    Each item of ``segmentations`` is one annotation: a sequence of COCO
    polygons (flat ``x, y`` coordinates). Pieces and annotations are
    unioned. An empty list is an all-zero mask. Overlap stays 1, not a sum.
    """
    _require_size(height, width)
    polygons: list[Sequence[float]] = []
    for segmentation in segmentations:
        polygons.extend(segmentation)
    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)
    mask = rasterize_polygons(polygons, height, width)
    return np.ascontiguousarray(mask, dtype=np.uint8)


def semantic_target_from_annotations(
    annotations: Sequence[FilamentAnnotation],
    height: int,
    width: int,
) -> np.ndarray:
    """Union every polygon on these annotations. ``category_id`` is not read.

    Each annotation contributes its ``segmentation``. ``semantic_only``,
    when the object has it, contributes polygons that were left out of
    the panoptic mask. A duck-typed annotation without that attribute
    contributes only ``segmentation``.
    """
    segmentations: list[Sequence[Sequence[float]]] = []
    for annotation in annotations:
        segmentations.append(annotation.segmentation)
        segmentations.extend(getattr(annotation, "semantic_only", ()))
    return semantic_target(segmentations, height, width)


def downscale_pair(
    image: np.ndarray,
    target: np.ndarray,
    *,
    side: int = TRAIN_SIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Downscale an image and its mask together to ``side`` × ``side``.

    The mask uses nearest-neighbor sampling, so it stays binary. The image
    uses area averaging when both axes are an integer multiple of ``side``,
    and bilinear interpolation otherwise. The default ``side`` is 512, an
    untuned starting size for a 2048 frame on a free Kaggle GPU.

    Nearest maps output pixel ``i`` to input index
    ``floor((i + 0.5) * length / side)``. Area averaging is the mean of
    each block. The image is returned as ``float32`` in the input's
    numeric range. The mask keeps its dtype.
    """
    if isinstance(side, bool) or not isinstance(side, int) or side <= 0:
        raise ValueError("side must be a positive int")
    array = np.asarray(image)
    mask = np.asarray(target)
    if array.ndim not in (2, 3) or mask.ndim != 2:
        raise ValueError("image must be HxW or HxWxC, and target must be HxW")
    if array.shape[:2] != mask.shape:
        raise ValueError(
            f"image shape {array.shape[:2]} and target shape {mask.shape} differ"
        )
    _require_shrink(array.shape[0], side, "height")
    _require_shrink(array.shape[1], side, "width")
    small_image = _downscale_image(array, side, side)
    small_mask = _nearest(mask, side, side)
    return small_image, small_mask


def instance_pq(semantic: np.ndarray, gray: np.ndarray, image: TrainImage) -> float:
    """Pooled panoptic quality of one prediction against one file's views.

    ``semantic`` and ``gray`` share a shape. Instances come from
    :func:`filament.instances.labels_from_semantic` with its default
    area cut (400 at 2048, 25 at 512) and its default close size (0).
    No other cutoff or kernel is passed. Those defaults come from the
    2026-11-01 holdout sweep (``docs/RESULTS.md``).

    Each annotator entry is a view (:func:`filament.annotations.annotator_views`).
    The stand-in ``annotations`` list, which keeps one mask when two
    entries overlap, is not the ground truth. Masks are rasterized at
    the image size and nearest-downscaled to ``semantic.shape``. An
    annotation that disappears under nearest is omitted. ``category_id``
    is not read.

    Counts are summed across the views of this file, then one PQ is
    returned. Averaging this float across files is not the holdout
    score. A file with no prediction and no instances in any view has
    PQ NaN, not 1.
    """
    labels = labels_from_semantic(semantic, gray)
    mask = np.asarray(semantic)
    out_h, out_w = int(mask.shape[0]), int(mask.shape[1])
    _require_shrink(image.height, out_h, "height")
    _require_shrink(image.width, out_w, "width")
    total = PqCounts()
    for view in annotator_views(image):
        actual: list[np.ndarray] = []
        for annotation in view:
            full = annotation.rasterize(image.height, image.width)
            small = _nearest(np.asarray(full), out_h, out_w)
            if np.any(small):
                actual.append(small)
        total += fast_view_counts(labels, actual)
    return total.pq


def split_training_images(
    images: Sequence[TrainImage],
    val_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> TrainingSplit:
    """Hold out the most recent 20% of calendar dates, one row per file.

    ``val_fraction`` is the share of dates held out. The default,
    :data:`DEFAULT_HOLDOUT_FRACTION` (0.2), is the holdout every model in this
    repo is scored on. The detector split
    (:func:`filament.detector_data.build_detector_splits`) reuses that
    constant and refuses any other value unless its caller opts in, so a
    different holdout there is a deliberate act and not a silent drift.

    Entries that share a ``file_name`` are collapsed, then
    :func:`filament.split.split_by_date` sees one mapping per file. Whole
    dates stay together. This function does not split pixels.

    When every entry of a file has the same ``date_captured`` (blanks
    count as one value), that value is passed through. When the values
    differ, a warning is logged and the split uses the calendar date
    parsed from ``file_name``. The collapsed image stores that date as
    ``YYYY-MM-DD 00:00:00``.

    Returned ``annotations`` are the stand-in panoptic list, not the
    score. Exact duplicate polygons are dropped. Across different
    entries, instances with IoU greater than 0.5 keep the earlier mask,
    and the other segmentation is stored on ``semantic_only`` so the
    semantic union still covers it. The kept mask is not the union of
    the polygons. Instances from the same entry are not merged unless
    those polygons are exact duplicates. ``category_id`` is copied onto
    a replaced annotation and is not read to decide a merge.

    ``views`` on a rebuilt image is one tuple per original entry, before
    that reduction. Scoring uses those views. An image that is returned
    unchanged has empty ``views``; its annotations are the only view.

    Logs, on ``filament.train_data``, how many file names were repeated,
    how many extra image entries that was, how many exact polygons were
    dropped, and how many near-duplicate instances were merged.
    """
    collapsed, counts = _collapse_file_names(images)
    logger.info(
        "collapsed train file names: duplicated_file_names=%d "
        "extra_image_entries=%d exact_polygons_dropped=%d "
        "near_duplicate_instances_merged=%d",
        counts.duplicated_file_names,
        counts.extra_image_entries,
        counts.exact_polygons_dropped,
        counts.near_duplicate_instances_merged,
    )
    by_name = {image.file_name: image for image in collapsed}
    payload = [
        {"file_name": image.file_name, "date_captured": image.date_captured}
        for image in collapsed
    ]
    split = split_by_date(payload, val_fraction=val_fraction)
    return TrainingSplit(
        train=tuple(by_name[name] for name in split.train),
        holdout=tuple(by_name[name] for name in split.val),
    )


class _CollapseCounts(NamedTuple):
    """Counts logged by :func:`split_training_images`."""

    duplicated_file_names: int
    extra_image_entries: int
    exact_polygons_dropped: int
    near_duplicate_instances_merged: int


def _collapse_file_names(
    images: Sequence[TrainImage],
) -> tuple[tuple[TrainImage, ...], _CollapseCounts]:
    """One ``TrainImage`` per ``file_name``, in first-seen order."""
    groups: dict[str, list[TrainImage]] = {}
    order: list[str] = []
    for image in images:
        group = groups.get(image.file_name)
        if group is None:
            group = []
            groups[image.file_name] = group
            order.append(image.file_name)
        group.append(image)

    duplicated = 0
    extra = 0
    exact = 0
    merged = 0
    collapsed: list[TrainImage] = []
    for file_name in order:
        group = groups[file_name]
        extra += len(group) - 1
        if len(group) > 1:
            duplicated += 1
        image, dropped, near = _collapse_group(group)
        exact += dropped
        merged += near
        collapsed.append(image)
    return (
        tuple(collapsed),
        _CollapseCounts(duplicated, extra, exact, merged),
    )


def _collapse_group(
    group: Sequence[TrainImage],
) -> tuple[TrainImage, int, int]:
    """Return the panoptic-quality image, polygons dropped, and merges."""
    if len(group) == 1 and not _has_exact_duplicate(group[0].annotations):
        return group[0], 0, 0
    _require_same_shape(group)
    kept, exact_dropped = _drop_exact_polygons(group)
    clusters, near_merged = _cluster_across_entries(
        kept, group[0].height, group[0].width
    )
    annotations = tuple(
        _annotation_for_cluster(cluster, group[0].id) for cluster in clusters
    )
    # Original entries, including a mask the stand-in list dropped.
    views = tuple(tuple(image.annotations) for image in group)
    image = TrainImage(
        id=group[0].id,
        file_name=group[0].file_name,
        date_captured=_date_for_split(group),
        height=group[0].height,
        width=group[0].width,
        annotations=annotations,
        views=views,
    )
    return image, exact_dropped, near_merged


def _has_exact_duplicate(annotations: Sequence[FilamentAnnotation]) -> bool:
    seen: set[tuple[tuple[float, ...], ...]] = set()
    for annotation in annotations:
        key = _segmentation_key(annotation)
        if key in seen:
            return True
        seen.add(key)
    return False


def _drop_exact_polygons(
    group: Sequence[TrainImage],
) -> tuple[list[tuple[int, FilamentAnnotation]], int]:
    """Drop a later annotation whose polygon list exactly matches an earlier one.

    The match is the stored coordinate tuple, across entries and inside one
    entry. Each polygon in the dropped annotation counts.
    """
    seen: set[tuple[tuple[float, ...], ...]] = set()
    kept: list[tuple[int, FilamentAnnotation]] = []
    dropped = 0
    for entry_index, image in enumerate(group):
        for annotation in image.annotations:
            key = _segmentation_key(annotation)
            if key in seen:
                dropped += len(key)
                continue
            seen.add(key)
            kept.append((entry_index, annotation))
    return kept, dropped


def _segmentation_key(
    annotation: FilamentAnnotation,
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(piece) for piece in annotation.segmentation)


def _cluster_across_entries(
    kept: Sequence[tuple[int, FilamentAnnotation]],
    height: int,
    width: int,
) -> tuple[list[list[tuple[int, FilamentAnnotation]]], int]:
    """Merge instances from different entries when IoU is greater than 0.5.

    A cluster holds at most one instance from each entry, so two overlapping
    filaments that share an entry stay apart. Matching is transitive: three
    entries of one filament become one cluster. The caller keeps the first
    mask in the cluster and does not concatenate polygons.
    """
    if not kept:
        return [], 0
    if len({entry for entry, _ in kept}) < 2:
        return [[item] for item in kept], 0

    masks = [
        rasterize_polygons(annotation.segmentation, height, width)
        for _, annotation in kept
    ]
    parent = list(range(len(kept)))
    entries: list[set[int]] = [{entry} for entry, _ in kept]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pairs: list[tuple[float, int, int]] = []
    for left in range(len(kept)):
        for right in range(left + 1, len(kept)):
            if kept[left][0] == kept[right][0]:
                continue
            intersection = int(
                np.count_nonzero((masks[left] != 0) & (masks[right] != 0))
            )
            union = int(np.count_nonzero((masks[left] != 0) | (masks[right] != 0)))
            # IoU > 1/2 iff 2 * intersection > union.
            if union == 0 or intersection * 2 <= union:
                continue
            pairs.append((intersection / union, left, right))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    for _, left, right in pairs:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        if entries[left_root] & entries[right_root]:
            continue
        parent[right_root] = left_root
        entries[left_root] = entries[left_root] | entries[right_root]

    buckets: dict[int, list[tuple[int, FilamentAnnotation]]] = {}
    order: list[int] = []
    for index, item in enumerate(kept):
        root = find(index)
        if root not in buckets:
            buckets[root] = []
            order.append(root)
        buckets[root].append(item)
    clusters = [buckets[root] for root in order]
    merged = sum(len(cluster) - 1 for cluster in clusters)
    return clusters, merged


def _annotation_for_cluster(
    cluster: Sequence[tuple[int, FilamentAnnotation]],
    image_id: int | str,
) -> FilamentAnnotation:
    """One panoptic instance: the first mask, not the union.

    Later masks in the cluster are kept on ``semantic_only`` so the
    semantic target still covers their polygons. Rasterize does not
    draw them.
    """
    first = cluster[0][1]
    extras: list[tuple[tuple[float, ...], ...]] = []
    for _, annotation in cluster[1:]:
        extras.append(annotation.segmentation)
        extras.extend(annotation.semantic_only)
    extras.extend(first.semantic_only)
    if not extras and first.image_id == image_id:
        return first
    return FilamentAnnotation(
        id=first.id,
        image_id=image_id,
        category_id=first.category_id,
        spine=first.spine,
        segmentation=first.segmentation,
        semantic_only=tuple(extras),
    )


def _date_for_split(group: Sequence[TrainImage]) -> str | None:
    """``date_captured`` when the entries agree, else the file-name date.

    Agreement compares stripped strings. A blank value agrees with another
    blank. A disagreement is logged. The stored clock is midnight because
    ``file_name`` is parsed as a calendar date.
    """
    if len(group) == 1:
        return group[0].date_captured
    tokens = {_date_token(image.date_captured) for image in group}
    if len(tokens) == 1:
        return group[0].date_captured
    file_name = group[0].file_name
    day = observation_date({"file_name": file_name})
    logger.warning(
        "file_name %s has differing date_captured values %s; "
        "using the date parsed from file_name (%s)",
        file_name,
        [image.date_captured for image in group],
        day.isoformat(),
    )
    return f"{day.isoformat()} 00:00:00"


def _date_token(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"date_captured must be a str or None, got {type(value).__name__}"
        )
    text = value.strip()
    return text or None


def _require_same_shape(group: Sequence[TrainImage]) -> None:
    height = group[0].height
    width = group[0].width
    for image in group[1:]:
        if image.height != height or image.width != width:
            raise ValueError(
                f"file_name {group[0].file_name!r} has shapes "
                f"{height}x{width} and {image.height}x{image.width}"
            )


def _require_size(height: object, width: object) -> None:
    for label, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive int")


def _require_shrink(length: int, out_length: int, label: str) -> None:
    if out_length > length:
        raise ValueError(f"cannot downscale {label} from {length} to {out_length}")


def _downscale_image(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height % out_h == 0 and width % out_w == 0:
        return _area(image, out_h, out_w)
    return _bilinear(image, out_h, out_w)


def _area(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Mean of each integer block. Channels are not mixed."""
    height, width = image.shape[:2]
    factor_y = height // out_h
    factor_x = width // out_w
    contiguous = np.ascontiguousarray(image)
    if contiguous.ndim == 2:
        blocks = contiguous.reshape(out_h, factor_y, out_w, factor_x)
        mean = blocks.mean(axis=(1, 3))
    else:
        channels = contiguous.shape[2]
        blocks = contiguous.reshape(out_h, factor_y, out_w, factor_x, channels)
        mean = blocks.mean(axis=(1, 3))
    return np.ascontiguousarray(mean, dtype=np.float32)


def _bilinear(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize. Output pixel centers map back onto the input grid."""
    height, width = image.shape[:2]
    source = np.ascontiguousarray(image, dtype=np.float32)
    row_pos = (np.arange(out_h, dtype=np.float64) + 0.5) * (height / out_h) - 0.5
    col_pos = (np.arange(out_w, dtype=np.float64) + 0.5) * (width / out_w) - 0.5
    row_pos = np.clip(row_pos, 0.0, height - 1)
    col_pos = np.clip(col_pos, 0.0, width - 1)
    row0 = np.floor(row_pos).astype(np.int64)
    col0 = np.floor(col_pos).astype(np.int64)
    row1 = np.minimum(row0 + 1, height - 1)
    col1 = np.minimum(col0 + 1, width - 1)
    row_weight = (row_pos - row0).astype(np.float32)
    col_weight = (col_pos - col0).astype(np.float32)
    top_left = source[np.ix_(row0, col0)]
    top_right = source[np.ix_(row0, col1)]
    bottom_left = source[np.ix_(row1, col0)]
    bottom_right = source[np.ix_(row1, col1)]
    if source.ndim == 2:
        col_axis = col_weight.reshape(1, out_w)
        row_axis = row_weight.reshape(out_h, 1)
    else:
        col_axis = col_weight.reshape(1, out_w, 1)
        row_axis = row_weight.reshape(out_h, 1, 1)
    top = top_left * (1.0 - col_axis) + top_right * col_axis
    bottom = bottom_left * (1.0 - col_axis) + bottom_right * col_axis
    mixed = top * (1.0 - row_axis) + bottom * row_axis
    return np.ascontiguousarray(mixed, dtype=np.float32)


def _nearest(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    rows = _nearest_index(mask.shape[0], out_h)
    cols = _nearest_index(mask.shape[1], out_w)
    return np.ascontiguousarray(mask[np.ix_(rows, cols)])


def _nearest_index(length: int, out_length: int) -> np.ndarray:
    """Input index under the center of each output pixel."""
    scale = length / out_length
    index = np.floor((np.arange(out_length, dtype=np.float64) + 0.5) * scale)
    return np.clip(index.astype(np.int64), 0, length - 1)
