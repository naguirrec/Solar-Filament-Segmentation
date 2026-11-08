"""Opening is gone, and gap grouping does not paint the bridged pixels.

Synthetic images only. No MAGFiLO frames and no network. These tests do
not import the pooled panoptic-quality function.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament.baseline import DEFAULT_CLOSE_SIZE, DEFAULT_MIN_AREA, segment_filaments
from filament.instances import labels_for_min_areas, labels_from_semantic

_SHAPE = (96, 96)


def _bright(shape: tuple[int, int] = _SHAPE) -> np.ndarray:
    return np.full(shape, 220, dtype=np.uint8)


def _two_rectangles() -> np.ndarray:
    """Two 10×40 blocks with a one-pixel gap on row 40."""
    mask = np.zeros(_SHAPE, dtype=np.uint8)
    mask[30:40, 30:70] = 1
    mask[41:51, 30:70] = 1
    return mask


def _dark_two_rectangles() -> np.ndarray:
    image = _bright()
    dark = _two_rectangles()
    image[dark == 1] = 30
    return image


def test_defaults_follow_the_2026_11_01_holdout_sweep() -> None:
    baseline = inspect.signature(segment_filaments)
    assert baseline.parameters["min_area"].default == DEFAULT_MIN_AREA == 400
    assert baseline.parameters["close_size"].default == DEFAULT_CLOSE_SIZE == 0
    semantic = inspect.signature(labels_from_semantic)
    assert semantic.parameters["min_area"].default is None
    assert semantic.parameters["close_size"].default is None


def test_one_pixel_barb_survives_without_an_opening() -> None:
    image = _bright()
    image[48, 20:80] = 30
    labels = segment_filaments(image, block_size=31, min_area=16, offset=5)
    assert int(labels.max()) == 1
    assert int((labels[48, 20:80] > 0).sum()) == 60
    assert int((labels > 0).sum()) == 60


def test_one_pixel_gap_stays_split_when_closing_is_off() -> None:
    labels = segment_filaments(
        _dark_two_rectangles(),
        block_size=31,
        min_area=16,
        offset=5,
        close_size=0,
    )
    assert int(labels.max()) == 2
    assert int((labels[40] > 0).sum()) == 0
    assert int((labels > 0).sum()) == 800


def test_default_close_size_leaves_a_one_pixel_gap_split() -> None:
    labels = segment_filaments(
        _dark_two_rectangles(),
        block_size=31,
        min_area=16,
        offset=5,
    )
    assert int(labels.max()) == 2
    assert labels[35, 40] != labels[45, 40]


def test_close_size_3_groups_a_one_pixel_gap_without_adding_pixels() -> None:
    labels = segment_filaments(
        _dark_two_rectangles(),
        block_size=31,
        min_area=16,
        offset=5,
        close_size=3,
    )
    assert int(labels.max()) == 1
    assert int((labels[40] > 0).sum()) == 0
    assert int(labels[35, 40] > 0) == 1
    assert int(labels[45, 40] > 0) == 1
    assert labels[35, 40] == labels[45, 40]
    assert int((labels > 0).sum()) == 800


def test_semantic_gap_groups_without_adding_pixels() -> None:
    semantic = _two_rectangles()
    split = labels_from_semantic(semantic, _bright(), min_area=1, close_size=0)
    grouped = labels_from_semantic(semantic, _bright(), min_area=1, close_size=3)
    assert int(split.max()) == 2
    assert int(grouped.max()) == 1
    assert int((grouped[40] > 0).sum()) == 0
    assert int((grouped > 0).sum()) == int(semantic.sum())
    assert grouped[35, 40] == grouped[45, 40]


def test_even_close_size_is_rejected() -> None:
    image = _bright()
    with pytest.raises(ValueError, match="odd"):
        segment_filaments(image, close_size=2)
    with pytest.raises(ValueError, match="odd"):
        labels_from_semantic(_two_rectangles(), image, close_size=4)


@pytest.mark.parametrize("close_size", [0, 3, 5])
def test_labels_for_min_areas_equals_one_call_per_area(close_size: int) -> None:
    rng = np.random.default_rng(0)
    semantic = _two_rectangles() | (rng.random(_SHAPE) < 0.03).astype(np.uint8)
    image = _dark_two_rectangles()
    areas = [1, 5, 50, 400]
    got = labels_for_min_areas(semantic, image, areas, close_size=close_size)
    assert list(got) == areas
    for area in areas:
        expected = labels_from_semantic(
            semantic, image, min_area=area, close_size=close_size
        )
        assert np.array_equal(got[area], expected)
    # A precomputed disk gate gives the same labels.
    from filament.baseline import on_disk

    again = labels_for_min_areas(
        semantic, image, areas, close_size=close_size, disk=on_disk(image)
    )
    assert all(np.array_equal(again[a], got[a]) for a in areas)


def test_labels_for_min_areas_checks_its_inputs() -> None:
    semantic = _two_rectangles()
    with pytest.raises(ValueError, match="share a shape"):
        labels_for_min_areas(semantic, _bright((8, 8)), [10])
    with pytest.raises(ValueError, match="disk mask"):
        labels_for_min_areas(semantic, _bright(), [10], disk=np.ones((8, 8), bool))
    with pytest.raises(ValueError, match="min_area must be >= 1"):
        labels_for_min_areas(semantic, _bright(), [0])
