"""Score the date holdout at full resolution for two predictors.

Kaggle scores 2048×2048 frames. Both scores here are computed at the gray
image's height and width, which is 2048 on real train images. The classic
baseline is not downscaled to 512, and the U-Net mask is not scored at 512.
No leaderboard figure is stored here.

The holdout is the tuple :func:`filament.train_data.split_training_images`
already built. This module does not split dates and does not merge
near-duplicate ``file_name`` values. One sample per file is that split's
job. Each annotator entry already stored on the image is one view.

Both paths call :func:`filament.metrics.fast_view_counts`. TP, FP, FN,
and matched IoU are summed over every view of every image, then one PQ
is taken from that pool. There is no per-image average. A zero
denominator is NaN, not 1. The match is every pair with IoU > 0.5, not
a one-to-one greedy match. The stand-in list that keeps one overlapping
mask is not the ground truth.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from filament.annotations import TrainImage, annotator_views
from filament.baseline import segment_filaments
from filament.instances import labels_from_semantic
from filament.metrics import PqCounts, fast_view_counts

# A mapping keyed by ``file_name``, or a callable that receives that name.
# The caller loads pixels. This module has no Kaggle path.
ImageSource = Mapping[str, np.ndarray] | Callable[[str], np.ndarray]


@dataclass(frozen=True, slots=True)
class HoldoutScores:
    """Pooled panoptic quality on the date holdout, plus how many images.

    ``baseline_pq`` and ``unet_pq`` are each one PQ from the summed
    counts of every annotator view of every holdout image. They are not
    means. Both are at the gray image's full resolution. Either one is
    NaN when its denominator is zero.
    """

    baseline_pq: float
    unet_pq: float
    image_count: int


def score_holdout(
    holdout: Sequence[TrainImage],
    gray_images: ImageSource,
    unet_semantics: ImageSource,
) -> HoldoutScores:
    """Return full-resolution baseline and U-Net PQ on ``holdout``.

    ``holdout`` is the validation side of
    :func:`filament.train_data.split_training_images`. Each item is scored
    as it stands. Annotations are not merged across ``file_name``.

    ``gray_images`` supplies one grayscale uint8 frame per ``file_name``.
    It is either a mapping or a callable. The frame's shape must be the
    ``TrainImage`` height and width. On real data that is 2048×2048.

    ``unet_semantics`` supplies the semantic mask predicted at 512 on real
    data, again by ``file_name``. The mask is nearest-upsampled to the gray
    frame, then passed to :func:`filament.instances.labels_from_semantic`
    with no area cut and no close size. At 2048 that default cut is 400,
    from the 2026-11-01 holdout sweep. The default close size is 0. That is
    the same order as test inference: upsample, then components at full
    resolution.

    The baseline is :func:`filament.baseline.segment_filaments` with no
    extra arguments, so the current defaults apply (``min_area`` 400,
    ``close_size`` 0, ``block_size`` 151, ``offset`` 5) on the
    full-resolution frame. Those defaults are not scaled down to 512.

    Ground truth is each annotator view, rasterized at the image height
    and width. ``category_id`` is not read. An annotation that rasterizes
    to no foreground pixels is omitted: background is not an instance.
    An entry that then has no masks is still scored, as an empty view.
    The same views are passed to both scores. Counts are summed. The
    returned PQ is not the mean of per-image scores.

    An empty holdout raises ``ValueError``. A holdout whose pooled
    denominator is zero has PQ NaN, not 1.
    """
    if len(holdout) == 0:
        raise ValueError("holdout is empty")

    baseline_counts = PqCounts()
    unet_counts = PqCounts()
    for image in holdout:
        gray = _gray_uint8(_load(gray_images, image.file_name), image)
        views = _view_masks(image)
        baseline_labels = segment_filaments(gray)
        semantic = _semantic_mask(_load(unet_semantics, image.file_name), image)
        upsampled = _nearest_upsample(semantic, gray.shape[0], gray.shape[1])
        unet_labels = labels_from_semantic(upsampled, gray)
        for masks in views:
            baseline_counts += fast_view_counts(baseline_labels, masks)
            unet_counts += fast_view_counts(unet_labels, masks)

    return HoldoutScores(
        baseline_pq=baseline_counts.pq,
        unet_pq=unet_counts.pq,
        image_count=len(holdout),
    )


def _load(source: ImageSource, file_name: str) -> np.ndarray:
    if isinstance(source, Mapping):
        return source[file_name]
    if callable(source):
        return source(file_name)
    raise TypeError("image source must be a mapping or a callable")


def _gray_uint8(image: np.ndarray, train_image: TrainImage) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(
            f"{train_image.file_name} gray image must be 2D, got shape {array.shape}"
        )
    if array.dtype != np.uint8:
        raise TypeError(
            f"{train_image.file_name} gray image must be uint8, got dtype {array.dtype}"
        )
    expected = (train_image.height, train_image.width)
    if array.shape != expected:
        raise ValueError(
            f"{train_image.file_name} gray shape {array.shape} "
            f"does not match the train image {expected}"
        )
    return np.ascontiguousarray(array)


def _semantic_mask(mask: np.ndarray, train_image: TrainImage) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(
            f"{train_image.file_name} semantic mask must be 2D, got shape {array.shape}"
        )
    if array.dtype != np.uint8 and array.dtype != np.dtype(bool):
        raise TypeError(
            f"{train_image.file_name} semantic mask must be uint8 or bool, "
            f"got dtype {array.dtype}"
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{train_image.file_name} semantic mask is empty")
    return np.ascontiguousarray(array)


def _nearest_upsample(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Repeat each input pixel under the output pixel centers.

    Output index ``i`` reads input ``floor((i + 0.5) * in_length / out_length)``.
    That is the training nearest-neighbor rule, run from the 512 prediction
    up to the gray frame. A mask that is already the gray shape is unchanged.
    A mask larger than the gray frame is rejected: this path only upsamples.
    """
    height, width = int(mask.shape[0]), int(mask.shape[1])
    if out_h < height or out_w < width:
        raise ValueError(
            f"semantic mask {(height, width)} is larger than the gray image "
            f"{(out_h, out_w)}; this score only upsamples"
        )
    rows = _nearest_index(height, out_h)
    cols = _nearest_index(width, out_w)
    return np.ascontiguousarray(mask[np.ix_(rows, cols)])


def _nearest_index(length: int, out_length: int) -> np.ndarray:
    """Input index under the center of each output pixel."""
    scale = length / out_length
    index = np.floor((np.arange(out_length, dtype=np.float64) + 0.5) * scale)
    return np.clip(index.astype(np.int64), 0, length - 1)


def _view_masks(image: TrainImage) -> list[list[np.ndarray]]:
    """Rasterize each annotator view. ``category_id`` is unread.

    A mask with no foreground pixels is omitted. An empty view is kept
    as an empty list so its predictions still count as false positives.
    """
    expected = (image.height, image.width)
    views: list[list[np.ndarray]] = []
    for view in annotator_views(image):
        masks: list[np.ndarray] = []
        for annotation in view:
            mask = np.asarray(annotation.rasterize(image.height, image.width))
            if mask.shape != expected:
                raise ValueError(
                    f"{image.file_name} ground truth has shape {mask.shape}, "
                    f"expected {expected}"
                )
            if not np.any(mask):
                continue
            masks.append(np.ascontiguousarray(mask))
        views.append(masks)
    return views
