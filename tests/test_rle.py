"""Round-trip synthetic binary masks through compressed RLE counts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament.rle import decode_mask, encode_mask


def _sparse_mask() -> np.ndarray:
    mask = np.zeros((5, 9), dtype=np.uint8)
    mask[0, 8] = 1
    mask[4, 0] = 1
    mask[2, 3:6] = 1
    return mask


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((5, 9), dtype=np.uint8),
        np.ones((3, 4), dtype=np.uint8),
        _sparse_mask(),
    ],
    ids=["zeros", "ones", "sparse"],
)
def test_round_trip_equals_original_mask(mask: np.ndarray) -> None:
    height, width = mask.shape
    counts = encode_mask(mask)
    assert isinstance(counts, str)
    assert not counts.startswith("{")
    decoded = decode_mask(counts, height, width)
    assert decoded.shape == (height, width)
    assert np.array_equal(decoded, mask)
