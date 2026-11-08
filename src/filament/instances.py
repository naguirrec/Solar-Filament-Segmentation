"""Turn one semantic filament mask into a submission CSV.

The mask is already foreground versus background. Pixels outside the
solar disk are cleared, then connected components (8-neighborhood) are
labeled. A small closing/gap kernel can join nearby fragments. The
dilation only assigns labels: background pixels, including the bridged
gap, stay background. There is no morphological opening.

``min_area`` defaults to 400 on a 2048×2048 mask. That value is the best
U-Net setting in the 2026-11-01 date-holdout sweep (``docs/RESULTS.md``).
It was not chosen by counting components on test images.

Other resolutions scale that area by pixel count against 2048×2048.
For a square mask the factor is ``(side/2048)**2``. A mask that is
512 px on a side uses ``(512/2048)**2 * 400 = 25``. That scale is only
for an already-semantic mask. The classic baseline keeps the same 400
px default on a full-disk frame and does not scale it.

``close_size`` defaults to 0 (``filament.baseline.DEFAULT_CLOSE_SIZE``).
The 2026-11-01 sweep showed closing lowers the baseline PQ and changes
nothing for the U-Net. ``0`` leaves fragments split.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from filament.baseline import (
    DEFAULT_CLOSE_SIZE,
    DEFAULT_MIN_AREA,
    iter_filament_counts,
    label_foreground,
    on_disk,
)
from filament.metrics import panoptic_quality
from filament.submission import filament_id, write_submission

# Full-disk side used only to scale the default area cut.
_FULL_DISK_SIDE = 2048


def starting_min_area(height: int, width: int) -> int:
    """Return the default area cut for a semantic mask of this shape.

    2048×2048 uses 400, the best U-Net value in the 2026-11-01 date-holdout
    sweep (``docs/RESULTS.md``). Any other shape scales that area by
    ``height * width / 2048**2``, rounded to the nearest integer and at
    least 1. A 512×512 mask uses 25.
    """
    if isinstance(height, bool) or isinstance(width, bool):
        raise TypeError("height and width must be ints")
    if not isinstance(height, (int, np.integer)) or not isinstance(
        width, (int, np.integer)
    ):
        raise TypeError("height and width must be ints")
    rows = int(height)
    cols = int(width)
    if rows < 1 or cols < 1:
        raise ValueError(f"mask shape {(rows, cols)} must be at least 1×1")
    numerator = DEFAULT_MIN_AREA * rows * cols
    denominator = _FULL_DISK_SIDE * _FULL_DISK_SIDE
    # Round half away from zero. 512×512 lands on 25 exactly.
    scaled = (numerator + denominator // 2) // denominator
    return max(scaled, 1)


def labels_from_semantic(
    semantic: np.ndarray,
    gray: np.ndarray,
    *,
    min_area: int | None = None,
    close_size: int | None = None,
) -> np.ndarray:
    """Return one int32 label image of on-disk connected components.

    ``semantic`` is a 2D uint8 or bool mask. Nonzero pixels are
    foreground. ``gray`` is the uint8 H-alpha image of the same shape
    and is used only to drop the off-disk sky. ``min_area`` defaults to
    :func:`starting_min_area`. Components smaller than that are dropped.
    ``close_size`` defaults to :data:`filament.baseline.DEFAULT_CLOSE_SIZE`
    and is passed to :func:`filament.baseline.label_foreground`. ``None``
    uses that default (0, no closing). Both defaults come from the
    2026-11-01 holdout sweep (``docs/RESULTS.md``).

    The returned array has the same shape as ``semantic``. ``0`` is
    background. Filament ids are ``1..N`` in scan order. This is not a
    list of full-size masks. Grouping does not add the bridged pixels.
    """
    foreground = _foreground(semantic)
    image = _gray_uint8(gray)
    if foreground.shape != image.shape:
        raise ValueError(
            "semantic mask and image must share a shape, "
            f"got {foreground.shape} and {image.shape}"
        )
    area_limit = _resolve_min_area(min_area, foreground.shape)
    gap_kernel = DEFAULT_CLOSE_SIZE if close_size is None else close_size
    # Copy already happened in _foreground. Zeroing stays off the caller.
    foreground[~on_disk(image)] = 0
    labels = label_foreground(foreground, gap_kernel)
    return _drop_small(labels, area_limit)


def labels_for_min_areas(
    semantic: np.ndarray,
    gray: np.ndarray,
    min_areas: Iterable[int],
    *,
    close_size: int | None = None,
    disk: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Return ``{min_area: labels}``, labeling the mask once.

    Each value is identical to ``labels_from_semantic(semantic, gray,
    min_area=area, close_size=close_size)``. The disk gate and the connected
    components do not depend on ``min_area``, so they run once and only the
    small-component drop is repeated. ``disk`` is the boolean result of
    :func:`filament.baseline.on_disk` for ``gray``; pass it to reuse it
    across several masks of the same frame, or leave it ``None`` to compute it.
    """
    foreground = _foreground(semantic)
    image = _gray_uint8(gray)
    if foreground.shape != image.shape:
        raise ValueError(
            "semantic mask and image must share a shape, "
            f"got {foreground.shape} and {image.shape}"
        )
    limits = [_resolve_min_area(area, foreground.shape) for area in min_areas]
    gap_kernel = DEFAULT_CLOSE_SIZE if close_size is None else close_size
    gate = on_disk(image) if disk is None else np.asarray(disk, dtype=bool)
    if gate.shape != foreground.shape:
        raise ValueError(
            f"disk mask {gate.shape} does not match the mask {foreground.shape}"
        )
    foreground[~gate] = 0
    labels = label_foreground(foreground, gap_kernel)
    return {limit: _drop_small(labels, limit) for limit in limits}


def write_instance_submission(
    path: str | Path,
    image_stem: str,
    semantic: np.ndarray,
    gray: np.ndarray,
    *,
    min_area: int | None = None,
    close_size: int | None = None,
) -> None:
    """Write one image's filaments to the submission CSV.

    Columns are ``filament_id`` and ``segmentation_rle``. There is no
    class column: Left, Right, Unidentifiable, and Ambiguous are one
    filament class. ``min_area`` and ``close_size`` are forwarded to
    :func:`labels_from_semantic`. ``None`` keeps that function's
    defaults. Each component is encoded with
    :func:`filament.baseline.iter_filament_counts`, one temporary mask
    at a time. Full-size masks are not stacked.
    """
    labels = labels_from_semantic(
        semantic, gray, min_area=min_area, close_size=close_size
    )
    write_submission(path, _submission_rows(image_stem, labels))


def score_instances(
    predictions: Iterable[np.ndarray],
    ground_truth: Iterable[np.ndarray],
) -> float:
    """Return Kirillov panoptic quality for one image.

    ``predictions`` and ``ground_truth`` are instance masks, one filament
    each, not a label image and not a semantic mask. All four train
    categories are one filament class because the submission CSV has no
    class column. Chirality versus the official metric is [VERIFICAR].

    Average this score over a validation set to compare with the public
    leaderboard without a Kaggle submission. The public score is not
    stored here.
    """
    return panoptic_quality(predictions, ground_truth)


def _submission_rows(image_stem: str, labels: np.ndarray):
    """Yield ``(filament_id, counts)`` without retaining component masks."""
    for number, counts in iter_filament_counts(labels):
        yield filament_id(image_stem, number), counts


def _foreground(semantic: np.ndarray) -> np.ndarray:
    array = np.asarray(semantic)
    if array.ndim != 2:
        raise ValueError(f"semantic mask must be 2D, got shape {array.shape}")
    if array.dtype != np.uint8 and array.dtype != np.dtype(bool):
        raise TypeError(f"semantic mask must be uint8 or bool, got dtype {array.dtype}")
    # New array. Nonzero is foreground, so a 0/255 mask and a bool mask agree.
    return np.ascontiguousarray(array != 0, dtype=np.uint8)


def _gray_uint8(gray: np.ndarray) -> np.ndarray:
    array = np.asarray(gray)
    if array.ndim != 2:
        raise ValueError(f"image must be 2D grayscale, got shape {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(f"image must be uint8 grayscale, got dtype {array.dtype}")
    return np.ascontiguousarray(array)


def _resolve_min_area(min_area: int | None, shape: tuple[int, int]) -> int:
    if min_area is None:
        return starting_min_area(shape[0], shape[1])
    if isinstance(min_area, bool) or not isinstance(min_area, (int, np.integer)):
        raise TypeError("min_area must be an int")
    area = int(min_area)
    if area < 1:
        raise ValueError("min_area must be >= 1")
    return area


def _drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Drop small components and compact the remaining ids to ``1..N``.

    One index map, then one label image. No per-component mask is stored.
    """
    if labels.size == 0:
        return np.zeros(labels.shape, dtype=np.int32)
    max_label = int(labels.max())
    areas = np.bincount(labels.ravel(), minlength=max_label + 1)
    valid = areas >= min_area
    valid[0] = False
    keep = np.cumsum(valid, dtype=np.int32)
    keep[~valid] = 0
    return keep[labels]
