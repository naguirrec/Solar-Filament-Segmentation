"""Submission CSV: header, one data row, and quoting left intact."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from filament.rle import decode_mask
from filament.submission import rows_for_image, write_submission


def test_submission_csv_has_header_and_one_data_row(tmp_path: Path) -> None:
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1:3, 2:5] = 1
    rows = rows_for_image("20150125172714Mh", [mask])
    path = tmp_path / "submission.csv"
    write_submission(path, rows)

    with path.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.reader(handle))

    counts = rows[0][1]
    assert parsed == [
        ["filament_id", "segmentation_rle"],
        ["20150125172714Mh_1", counts],
    ]
    # This mask's compressed counts contain no comma, so the csv module
    # leaves the field unquoted.
    assert path.read_text(encoding="utf-8") == (
        f"filament_id,segmentation_rle\n20150125172714Mh_1,{counts}\n"
    )
    decoded = decode_mask(parsed[1][1], height=4, width=6)
    assert np.array_equal(decoded, mask)


def test_filament_numbers_start_at_one() -> None:
    masks = [
        np.zeros((2, 2), dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint8),
    ]
    rows = rows_for_image("20150125172714Mh", masks)
    assert [row[0] for row in rows] == [
        "20150125172714Mh_1",
        "20150125172714Mh_2",
    ]


def test_counts_containing_commas_keep_csv_quoting(tmp_path: Path) -> None:
    counts = "ab,cd,ef"
    path = tmp_path / "submission.csv"
    write_submission(path, [("20150125172714Mh_1", counts)])

    raw = path.read_text(encoding="utf-8")
    assert '"ab,cd,ef"' in raw
    with path.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.reader(handle))
    assert parsed[1][1] == counts
