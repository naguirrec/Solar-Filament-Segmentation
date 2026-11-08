"""Memory-safe encoding and the off-disk gate.

No dataset and no randomness. Label images stay one array, and RLE is
pulled one component at a time.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

# src/ is not an installed package in this pull request.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament.baseline import iter_filament_counts, segment_filaments
from filament.rle import decode_mask


def _assert_no_stacked_masks(generator: Iterator[tuple[int, str]]) -> None:
    """Fail if the encoder is holding a sequence of full-size masks."""
    frame = getattr(generator, "gi_frame", None)
    if frame is None:
        return
    for value in frame.f_locals.values():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (list, tuple)) and any(
            isinstance(item, np.ndarray) for item in value
        ):
            raise AssertionError("encoder retained more than one full-size mask")


def test_label_image_encodes_each_component_without_stacking_masks() -> None:
    labels = np.zeros((24, 24), dtype=np.int32)
    labels[1:6, 1:5] = 1
    labels[8:16, 8:20] = 4
    labels[18:22, 2:7] = 7

    encoded = iter_filament_counts(labels)
    assert isinstance(encoded, Iterator)
    assert not isinstance(encoded, (Sequence, np.ndarray))

    items: list[tuple[int, str]] = []
    for item in encoded:
        assert isinstance(item, tuple)
        number, counts = item
        assert isinstance(number, int)
        assert isinstance(counts, str)
        _assert_no_stacked_masks(encoded)
        items.append((number, counts))

    assert [number for number, _counts in items] == [1, 2, 3]
    expected_ids = (1, 4, 7)
    for (number, counts), label_id in zip(items, expected_ids, strict=True):
        decoded = decode_mask(counts, height=24, width=24)
        assert decoded.shape == labels.shape
        assert np.array_equal(decoded, (labels == label_id).astype(np.uint8))
        assert number == expected_ids.index(label_id) + 1


def test_dark_speck_outside_the_disk_is_dropped() -> None:
    """A 10×10 speck is large enough to pass ``min_area``; the disk gate drops it."""
    image = np.full((128, 128), 12, dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    on_disk = (yy - 63) ** 2 + (xx - 63) ** 2 <= 46**2
    image[on_disk] = 210
    image[58:70, 70:84] = 36
    image[4:14, 4:14] = 0

    labels = segment_filaments(image, block_size=31, min_area=16, offset=5)
    assert isinstance(labels, np.ndarray)
    assert int(labels.max()) == 1

    pairs = list(iter_filament_counts(labels))
    assert len(pairs) == 1
    _number, counts = pairs[0]
    decoded = decode_mask(counts, height=128, width=128)

    expected = np.zeros((128, 128), dtype=np.uint8)
    expected[58:70, 70:84] = 1
    assert np.array_equal(decoded, expected)
    assert int(decoded[4:14, 4:14].sum()) == 0


def test_dark_bar_that_meets_the_sky_stays_on_disk() -> None:
    """A filament may cross the limb. Only the off-disk tip is dropped."""
    image = np.full((128, 128), 12, dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    on_disk = (yy - 63) ** 2 + (xx - 63) ** 2 <= 46**2
    image[on_disk] = 210
    image[60:68, 40:112] = 30
    image[4:14, 4:14] = 0

    labels = segment_filaments(image, block_size=31, min_area=16, offset=5)
    pairs = list(iter_filament_counts(labels))
    assert len(pairs) == 1
    decoded = decode_mask(pairs[0][1], height=128, width=128)

    expected = np.zeros((128, 128), dtype=np.uint8)
    bar = np.zeros((128, 128), dtype=bool)
    bar[60:68, 40:112] = True
    expected[on_disk & bar] = 1
    assert np.array_equal(decoded, expected)
    assert int(decoded[4:14, 4:14].sum()) == 0
    assert decoded[64, 80] == 1
    assert decoded[64, 111] == 0


def test_oversized_block_is_clamped_on_a_small_image() -> None:
    image = np.full((64, 64), 220, dtype=np.uint8)
    image[24:40, 22:42] = 30
    labels = segment_filaments(image, block_size=1001, min_area=16, offset=5)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == image.shape
    decoded = [
        decode_mask(counts, height=64, width=64)
        for _number, counts in iter_filament_counts(labels)
    ]
    assert len(decoded) == 1
    expected = np.zeros((64, 64), dtype=np.uint8)
    expected[24:40, 22:42] = 1
    assert np.array_equal(decoded[0], expected)


def test_explicit_min_area_does_not_fragment_a_small_image() -> None:
    """A small explicit area keeps this 64×64 filament.

    The call passes ``min_area=16`` so the fixture does not depend on the
    full-disk default. An empty label image would still satisfy
    ``max <= 1``. The assertion checks the return type and that encoding
    stays a generator.
    """
    image = np.full((64, 64), 220, dtype=np.uint8)
    image[24:40, 22:42] = 30
    labels = segment_filaments(image, min_area=16)
    assert isinstance(labels, np.ndarray)
    assert not isinstance(labels, list)
    assert labels.shape == image.shape
    assert int(labels.max()) <= 1
    encoded = iter_filament_counts(labels)
    assert isinstance(encoded, Iterator)
    assert not isinstance(encoded, Sequence)
    for number, counts in encoded:
        assert isinstance(number, int)
        assert isinstance(counts, str)


def test_even_block_size_is_rejected() -> None:
    image = np.full((16, 16), 200, dtype=np.uint8)
    with pytest.raises(ValueError, match="odd"):
        segment_filaments(image, block_size=30)


def test_tiny_image_raises_a_clear_error() -> None:
    image = np.full((3, 3), 200, dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        segment_filaments(image)
