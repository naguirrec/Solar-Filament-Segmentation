"""Smoke test for the classic CPU baseline.

A 64×64 bright field with one dark rectangle. No dataset and no randomness.
Block size and minimum area are passed explicitly so the assertion does
not move when the full-disk defaults change.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# src/ is not an installed package in this pull request.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from filament.baseline import iter_filament_counts, segment_filaments
from filament.rle import decode_mask
from filament.submission import filament_id, rows_for_image, write_submission

_RECTANGLE = (slice(24, 40), slice(22, 42))


def _bright_field_with_rectangle() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((64, 64), 220, dtype=np.uint8)
    expected = np.zeros((64, 64), dtype=np.uint8)
    image[_RECTANGLE] = 30
    expected[_RECTANGLE] = 1
    return image, expected


def test_dark_rectangle_round_trips_through_submission(tmp_path: Path) -> None:
    image, expected = _bright_field_with_rectangle()
    labels = segment_filaments(image, block_size=31, min_area=16, offset=5)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == image.shape
    assert int(labels.max()) == 1

    pairs = list(iter_filament_counts(labels))
    assert len(pairs) == 1
    number, counts = pairs[0]
    assert number == 1
    component = decode_mask(counts, height=64, width=64)
    assert component.shape == image.shape
    assert np.array_equal(component, expected)

    # One component, formatted by the existing CSV helper.
    rows = rows_for_image("synthetic64", (component,))
    assert rows == [(filament_id("synthetic64", number), counts)]
    path = tmp_path / "submission.csv"
    write_submission(path, rows)

    with path.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.reader(handle))

    assert parsed[0] == ["filament_id", "segmentation_rle"]
    assert len(parsed) == 2
    assert parsed[1][0] == "synthetic64_1"
    decoded = decode_mask(parsed[1][1], height=64, width=64)
    assert np.array_equal(decoded, expected)
