"""Semantic mask to submission CSV, on synthetic images only.

No MAGFiLO frames and no network. The area cut is the sweep-based
default (400 at 2048, 25 at 512), checked without allocating those frames.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament.instances import (
    labels_from_semantic,
    score_instances,
    starting_min_area,
    write_instance_submission,
)
from filament.rle import decode_mask

_SHAPE = (64, 64)


def _bright_disk(shape: tuple[int, int] = _SHAPE, radius: int = 30) -> np.ndarray:
    image = np.full(shape, 12, dtype=np.uint8)
    rows, cols = shape
    yy, xx = np.ogrid[:rows, :cols]
    center_y = (rows - 1) / 2
    center_x = (cols - 1) / 2
    image[(yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2] = 210
    return image


def _two_blobs(dtype: np.dtype) -> np.ndarray:
    """Two rectangles with a one-pixel gap, so a distance merge would fuse them."""
    mask = np.zeros(_SHAPE, dtype=dtype)
    mask[24:28, 20:30] = 1
    mask[29:33, 20:30] = 1
    return mask


@pytest.mark.parametrize("dtype", [np.uint8, np.dtype(bool)], ids=["uint8", "bool"])
def test_two_blobs_become_two_submission_rows(tmp_path: Path, dtype: np.dtype) -> None:
    image = _bright_disk()
    semantic = _two_blobs(dtype)
    path = tmp_path / "submission.csv"
    write_instance_submission(path, "synthetic64", semantic, image, close_size=0)

    with path.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.reader(handle))

    assert parsed[0] == ["filament_id", "segmentation_rle"]
    assert [row[0] for row in parsed[1:]] == ["synthetic64_1", "synthetic64_2"]
    decoded = [
        decode_mask(row[1], height=_SHAPE[0], width=_SHAPE[1]) for row in parsed[1:]
    ]
    upper = np.zeros(_SHAPE, dtype=np.uint8)
    lower = np.zeros(_SHAPE, dtype=np.uint8)
    upper[24:28, 20:30] = 1
    lower[29:33, 20:30] = 1
    assert np.array_equal(decoded[0], upper)
    assert np.array_equal(decoded[1], lower)


def test_speck_outside_the_disk_is_dropped(tmp_path: Path) -> None:
    image = _bright_disk()
    semantic = np.zeros(_SHAPE, dtype=np.uint8)
    semantic[0:6, 0:6] = 1
    semantic[30:38, 28:36] = 1
    path = tmp_path / "submission.csv"
    write_instance_submission(path, "synthetic64", semantic, image, close_size=0)

    with path.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.reader(handle))

    assert len(parsed) == 2
    decoded = decode_mask(parsed[1][1], height=_SHAPE[0], width=_SHAPE[1])
    expected = np.zeros(_SHAPE, dtype=np.uint8)
    expected[30:38, 28:36] = 1
    assert np.array_equal(decoded, expected)
    assert int(decoded[0:6, 0:6].sum()) == 0


def test_perfect_match_scores_one() -> None:
    shape = (8, 8)
    left = np.zeros(shape, dtype=np.uint8)
    right = np.zeros(shape, dtype=np.uint8)
    left[1:4, 1:4] = 1
    right[1:4, 5:8] = 1
    assert score_instances([left, right], [left, right]) == 1.0


def test_starting_min_area_scales_the_sweep_default() -> None:
    assert starting_min_area(2048, 2048) == 400
    # (512/2048)**2 * 400 = 25
    assert starting_min_area(512, 512) == 25
    # Same area rule on a non-square mask: half the pixels, half the cut.
    assert starting_min_area(2048, 1024) == 200


def test_default_min_area_drops_a_single_on_disk_pixel() -> None:
    # 256×256 scales 400 by (256/2048)**2, which rounds to 6.
    side = 256
    assert starting_min_area(side, side) > 1
    image = _bright_disk((side, side), radius=100)
    semantic = np.zeros((side, side), dtype=np.uint8)
    semantic[128, 128] = 1
    semantic[140:148, 120:128] = 1
    labels = labels_from_semantic(semantic, image, close_size=0)
    assert int(labels.max()) == 1
    assert labels[128, 128] == 0
    assert labels[140, 120] == 1
