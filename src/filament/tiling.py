"""Sliding-window inference on a large frame, pure numpy.

A 2048 x 2048 frame does not go through the network at once. It is cut into
overlapping square tiles, a callable predicts each batch of tiles, and the
overlapping predictions are averaged back into one full-size map.

Tile positions start at 0 and advance by ``stride = tile_size - overlap``.
The last tile on each axis is moved back so it ends exactly on the image
edge, so no padding is needed when the image is at least one tile wide. An
axis shorter than ``tile_size`` is padded at the bottom/right with edge
replication (one tile on that axis) and the prediction is cropped back.

Default overlap 128 at tile 512 (stride 384) is a starting choice, not a
tuned one. It puts a 2048 frame on a 5 x 5 grid, 25 tiles, and keeps each
tile's border (where a convolutional network has the least context) out of
the center of most pixels. Overlap 0 is allowed.

By default every tile counts the same (a uniform average). An optional
``window`` of positive per-pixel weights, such as :func:`taper_window`,
down-weights tile borders instead. Torch is not imported here: the model is
a callable that takes a batch of tiles.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

# (B, tile, tile) uint8 tiles in, (B, tile, tile) float scores out.
PredictBatch = Callable[[np.ndarray], np.ndarray]


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return the start offsets of the tiles that cover ``length`` pixels.

    Offsets are ``0, stride, 2 * stride, ...``. When the last of those leaves
    pixels uncovered, one more tile is added at ``length - tile_size`` (flush
    with the edge), so the last two tiles can overlap by more than
    ``overlap``. A ``length`` of at most ``tile_size`` gives ``[0]``.
    """
    _check_tiling(tile_size, overlap)
    if length < 1:
        raise ValueError("length must be >= 1")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] + tile_size < length:
        starts.append(length - tile_size)
    return starts


def tile_boxes(
    height: int, width: int, tile_size: int, overlap: int
) -> list[tuple[int, int]]:
    """Return the ``(top, left)`` corner of every tile, row by row."""
    return [
        (top, left)
        for top in tile_starts(height, tile_size, overlap)
        for left in tile_starts(width, tile_size, overlap)
    ]


def taper_window(tile_size: int, ramp: int) -> np.ndarray:
    """Return a ``(tile, tile)`` float32 weight window that fades at the edges.

    The 1D profile rises linearly over ``ramp`` pixels from each side and is 1
    in between; the window is its outer product. Every weight is positive, so
    no pixel ends up with zero total weight.
    """
    if tile_size < 1:
        raise ValueError("tile_size must be >= 1")
    if ramp < 1:
        raise ValueError("ramp must be >= 1")
    distance = np.minimum(np.arange(tile_size) + 1, tile_size - np.arange(tile_size))
    profile = np.minimum(distance / ramp, 1.0).astype(np.float32)
    return np.outer(profile, profile).astype(np.float32)


def predict_tiled(
    image: np.ndarray,
    predict_batch: PredictBatch,
    *,
    tile_size: int = 512,
    overlap: int = 128,
    batch_size: int = 8,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Predict a full frame tile by tile and average the overlaps.

    ``image`` is a 2D array. ``predict_batch`` receives a ``(B, tile, tile)``
    array of ``image``'s dtype with ``B <= batch_size`` and returns a
    ``(B, tile, tile)`` array of scores. The result is a float32 array of the
    image's shape: the sum of ``window * score`` over the tiles that cover
    a pixel divided by the sum of ``window`` there. ``window=None`` weights
    every tile 1, a uniform average.
    """
    _check_tiling(tile_size, overlap)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"image must be 2D, got shape {array.shape}")
    weights = _check_window(window, tile_size)

    height, width = array.shape
    pad_bottom = max(tile_size - height, 0)
    pad_right = max(tile_size - width, 0)
    if pad_bottom or pad_right:
        array = np.pad(array, ((0, pad_bottom), (0, pad_right)), mode="edge")

    padded_h, padded_w = array.shape
    total = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    boxes = tile_boxes(padded_h, padded_w, tile_size, overlap)

    for batch_boxes in _chunks(boxes, batch_size):
        tiles = np.stack(
            [
                array[top : top + tile_size, left : left + tile_size]
                for top, left in batch_boxes
            ]
        )
        scores = np.asarray(predict_batch(tiles), dtype=np.float32)
        expected = (len(batch_boxes), tile_size, tile_size)
        if scores.shape != expected:
            raise ValueError(
                f"predict_batch returned shape {scores.shape}, expected {expected}"
            )
        for (top, left), score in zip(batch_boxes, scores, strict=True):
            rows = slice(top, top + tile_size)
            cols = slice(left, left + tile_size)
            total[rows, cols] += score * weights
            weight_sum[rows, cols] += weights

    # Every pixel is covered by at least one tile with positive weight.
    return (total / weight_sum)[:height, :width]


def _chunks(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _check_tiling(tile_size: int, overlap: int) -> None:
    if tile_size < 1:
        raise ValueError("tile_size must be >= 1")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"overlap must be in [0, tile_size), got {overlap}")


def _check_window(window: np.ndarray | None, tile_size: int) -> np.ndarray | float:
    if window is None:
        return np.float32(1.0)
    array = np.asarray(window, dtype=np.float32)
    if array.shape != (tile_size, tile_size):
        raise ValueError(f"window shape {array.shape} must be {(tile_size, tile_size)}")
    if not np.all(np.isfinite(array)) or not np.all(array > 0):
        raise ValueError("window weights must be finite and > 0")
    return array
