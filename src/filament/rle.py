"""Encode and decode a 2D binary mask as compressed RLE counts.

The stored value is only the UTF-8 counts string from ``pycocotools.mask``.
It is not a JSON object and it does not include the mask size. Pass height
and width back into :func:`decode_mask`.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util


def encode_mask(mask: np.ndarray) -> str:
    """Return pycocotools compressed RLE counts for a 2D binary mask.

    Nonzero pixels are foreground. The result is a ``str``, not bytes and
    not the ``{"size", "counts"}`` dict.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {array.shape}")
    binary = np.asfortranarray((array != 0).astype(np.uint8))
    counts = mask_util.encode(binary)["counts"]
    if isinstance(counts, bytes):
        return counts.decode("utf-8")
    if isinstance(counts, str):
        return counts
    raise TypeError(f"unexpected RLE counts type: {type(counts).__name__}")


def decode_mask(counts: str, height: int, width: int) -> np.ndarray:
    """Decode compressed RLE counts into a uint8 mask of shape ``(height, width)``.

    ``height`` and ``width`` are the mask size. They are not stored in ``counts``.
    """
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    if not isinstance(counts, str):
        raise TypeError("counts must be a str")
    rle = {
        "size": [int(height), int(width)],
        "counts": counts.encode("utf-8"),
    }
    decoded = mask_util.decode(rle)
    return np.asarray(decoded, dtype=np.uint8)
