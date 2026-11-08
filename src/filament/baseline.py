"""Classic CPU baseline for dark solar filaments.

Adaptive threshold, then connected components. Off-disk sky is masked
out before labeling so dark specks outside the solar disk are not
filaments. There is no morphological opening: a 3×3 opening removes
thin barbs.

A small closing/gap kernel can join nearby fragments of one filament.
The dilation is used only to assign labels. Pixels that were background
stay background, so the gap itself is not painted in.

The result of :func:`segment_filaments` is one label image. RLE for the
submission is produced one component at a time by
:func:`iter_filament_counts`.

``block_size`` is an untuned starting point for a full-disk image on the
order of 2k pixels. ``min_area`` and ``close_size`` come from the
2026-11-01 holdout sweep (see ``docs/RESULTS.md``): closing hurts the
baseline, so the gap kernel is off. Do not choose them by counting
components on test images.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from filament.rle import encode_mask

_CONNECTIVITY = 8

# Untuned starting point for a full-disk frame on the order of 2k pixels.
# Not fitted on MAGFiLO. A neighborhood this wide keeps a thin filament
# darker than the local mean.
_DEFAULT_BLOCK_SIZE = 151
# Area cut for a 2048×2048 frame. 400 px is the best U-Net value in the
# 2026-11-01 holdout sweep (pooled PQ 0.2886). The classic baseline peaks
# higher (1200 px, 0.0734), but it is far below the U-Net, so one shared
# default is kept. See docs/RESULTS.md.
DEFAULT_MIN_AREA = 400
# 0 turns gap grouping off. In the 2026-11-01 holdout sweep a 3×3 closing
# lowered the baseline at every min_area and changed nothing for the U-Net.
DEFAULT_CLOSE_SIZE = 0


def segment_filaments(
    image: np.ndarray,
    *,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    min_area: int = DEFAULT_MIN_AREA,
    offset: float = 5,
    close_size: int = DEFAULT_CLOSE_SIZE,
) -> np.ndarray:
    """Return one label image of dark on-disk components.

    ``image`` is a 2D grayscale array. Filaments are darker than the local
    disk, so foreground is the low side of an adaptive threshold.
    ``block_size`` is the odd neighborhood used for the local mean. A value
    larger than the frame is clamped to the largest odd integer that is
    still smaller than both sides, so a small image does not crash.
    ``offset`` is the positive gray-level gap below that mean (OpenCV's
    ``C``); a flat disk stays background. ``min_area`` drops components
    smaller than that many pixels after gap grouping.

    ``close_size`` is the side of the closing/gap element passed to
    :func:`label_foreground`. ``0`` (the default) leaves fragments split; the
    2026-11-01 holdout sweep showed closing lowers the baseline PQ. No
    opening is applied.

    The returned array has the same shape as ``image`` and dtype ``int32``.
    ``0`` is background. Filament ids are ``1..N`` in scan order after
    small components are dropped. This is not a list of full-size masks.

    Defaults are ``block_size=151`` (untuned), ``min_area=400`` and
    ``close_size=0``. The last two follow the 2026-11-01 date-holdout sweep
    (``docs/RESULTS.md``). Pass a smaller odd ``block_size``
    and ``min_area`` for a tiny fixture such as the 64×64 smoke image.
    """
    gray = _as_uint8(image)
    block = _resolve_block_size(block_size, gray.shape)
    area_limit = _validate_min_area(min_area)
    gap_kernel = _validate_close_size(close_size)
    gap = _validate_offset(offset)

    foreground = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        gap,
    )
    foreground[~_on_disk(gray)] = 0
    labels = label_foreground(foreground, gap_kernel)
    return _relabel(labels, area_limit)


def label_foreground(foreground: np.ndarray, close_size: int) -> np.ndarray:
    """Label 8-connected foreground, optionally joining a small gap.

    ``foreground`` is a 2D mask. Nonzero pixels are foreground.
    ``close_size`` is the side of a square elliptical structuring element.
    ``0`` and ``1`` leave connectivity unchanged. A larger odd size dilates
    the mask only to decide which original pixels share a label. Pixels
    that were background stay background, including a gap that was bridged.

    The returned array is the int32 image from connected components.
    ``0`` is background. Ids are not compacted and are not filtered by area.

    The caller default is ``close_size`` 0 (see ``DEFAULT_CLOSE_SIZE``).
    """
    array = np.asarray(foreground)
    if array.ndim != 2:
        raise ValueError(f"foreground must be 2D, got shape {array.shape}")
    mask = np.ascontiguousarray(array != 0, dtype=np.uint8)
    size = _validate_close_size(close_size)
    if size <= 1:
        _count, labels = cv2.connectedComponents(mask, connectivity=_CONNECTIVITY)
        return labels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    grown = cv2.dilate(mask, kernel)
    _count, labels = cv2.connectedComponents(grown, connectivity=_CONNECTIVITY)
    # The dilation only chooses labels. Bridged background stays background.
    labeled = labels.copy()
    labeled[mask == 0] = 0
    return labeled


def iter_filament_counts(labels: np.ndarray) -> Iterator[tuple[int, str]]:
    """Yield ``(filament_number, rle_counts)`` for each label, one at a time.

    ``labels`` is one label image from :func:`segment_filaments` (``0`` is
    background). Each yield encodes a single component with
    :func:`filament.rle.encode_mask` into a fresh counts string, using one
    reusable full-size mask. Previous masks are not retained, and the
    return value is not a list of full-size masks. ``filament_number``
    starts at 1 and follows increasing label ids, skipping empty ids.
    """
    array = np.asarray(labels)
    if array.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {array.shape}")
    if array.size == 0:
        return
    max_label = int(array.max())
    if max_label < 1:
        return

    # One temporary mask for the whole loop. encode_mask copies what it
    # needs into the counts string, so the buffer can be reused next time.
    mask = np.zeros(array.shape, dtype=np.uint8)
    number = 0
    for label in range(1, max_label + 1):
        match = array == label
        if not bool(match.any()):
            continue
        mask.fill(0)
        mask[match] = 1
        # Release the boolean index before encoding. encode_mask copies the
        # component, so only the reusable mask stays live with the labels.
        del match
        number += 1
        counts = encode_mask(mask)
        yield number, counts


def on_disk(gray: np.ndarray) -> np.ndarray:
    """Return a boolean mask that is True on the solar disk."""
    return _on_disk(gray)


def _on_disk(gray: np.ndarray) -> np.ndarray:
    """Return a boolean mask that is True on the solar disk.

    Off-disk sky is the dark background around a full-disk image. Otsu
    splits that sky from the bright disk. The mask is the minimum
    enclosing circle of the largest bright contour: a dark filament stays
    inside even when it touches the sky, and a speck outside the limb does
    not. A frame with no bright region is left fully eligible.

    This gate is an untuned starting point, not fitted on MAGFiLO.
    """
    _level, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Copy: findContours is allowed to modify its input.
    contours, _hierarchy = cv2.findContours(
        bright.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.ones(gray.shape, dtype=bool)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return np.ones(gray.shape, dtype=bool)
    (center_x, center_y), radius = cv2.minEnclosingCircle(largest)
    if radius <= 0:
        return np.ones(gray.shape, dtype=bool)
    rows, cols = np.ogrid[: gray.shape[0], : gray.shape[1]]
    return (rows - center_y) ** 2 + (cols - center_x) ** 2 <= radius**2


def _relabel(labels: np.ndarray, min_area: int) -> np.ndarray:
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


def _as_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"image must be 2D grayscale, got shape {array.shape}")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"image must be numeric, got dtype {array.dtype}")
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("image must be finite")
    low = float(values.min())
    high = float(values.max())
    if high == low:
        return np.zeros(array.shape, dtype=np.uint8)
    # A 0–1 float image is scaled up. Values already on 0–255 stay there.
    # Anything wider is stretched so the local contrast is still visible.
    if low >= 0.0 and high <= 1.0:
        scaled = values * 255.0
    elif low >= 0.0 and high <= 255.0:
        scaled = values
    else:
        scaled = (values - low) * (255.0 / (high - low))
    return np.ascontiguousarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8))


def _resolve_block_size(block_size: int, shape: tuple[int, int]) -> int:
    """Return an odd block size strictly smaller than the image.

    The requested size must be an odd integer >= 3. When it does not fit,
    it is clamped down so adaptive threshold can still run.
    """
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise TypeError("block_size must be an int")
    block = int(block_size)
    if block < 3 or block % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")
    height, width = shape
    limit = min(int(height), int(width))
    # Largest odd integer that is still strictly smaller than both sides.
    max_block = limit - 1
    if max_block % 2 == 0:
        max_block -= 1
    if max_block < 3:
        raise ValueError(f"image shape {shape} is too small for an adaptive threshold")
    if block > max_block:
        return max_block
    return block


def _validate_close_size(close_size: int) -> int:
    if isinstance(close_size, bool) or not isinstance(close_size, (int, np.integer)):
        raise TypeError("close_size must be an int")
    size = int(close_size)
    if size == 0:
        return 0
    if size < 1 or size % 2 == 0:
        raise ValueError("close_size must be 0 or an odd integer >= 1")
    return size


def _validate_min_area(min_area: int) -> int:
    if isinstance(min_area, bool) or not isinstance(min_area, (int, np.integer)):
        raise TypeError("min_area must be an int")
    area = int(min_area)
    if area < 1:
        raise ValueError("min_area must be >= 1")
    return area


def _validate_offset(offset: float) -> float:
    if isinstance(offset, bool) or not isinstance(offset, (int, float, np.floating)):
        raise TypeError("offset must be a real number")
    gap = float(offset)
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("offset must be positive so a flat disk stays background")
    return gap
