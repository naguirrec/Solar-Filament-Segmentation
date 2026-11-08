"""Submission CSV checks on tiny synthetic rows. One test decodes at 2048."""

from __future__ import annotations

import inspect
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament.rle import encode_mask
from filament.submission import rows_for_image, write_submission
from filament.validate import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    SubmissionValidationError,
    validate_submission,
)


def _pixel(height: int, width: int, row: int, col: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[row, col] = 1
    return mask


def _write(path: Path, rows: list[tuple[str, str]]) -> Path:
    destination = path / "submission.csv"
    write_submission(destination, rows)
    return destination


def test_disjoint_masks_pass_and_other_images_may_reuse_pixels(tmp_path: Path) -> None:
    height, width = 4, 6
    same_image = rows_for_image(
        "20150125172714Mh",
        [_pixel(height, width, 0, 1), _pixel(height, width, 2, 3)],
    )
    other_image = rows_for_image("20150125180000Mh", [_pixel(height, width, 0, 1)])
    path = _write(tmp_path, same_image + other_image)

    assert validate_submission(path, height=height, width=width) is None


def test_header_must_be_exact(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    path.write_text("filament_id,rle\n", encoding="utf-8")

    with pytest.raises(SubmissionValidationError, match="header must be exactly"):
        validate_submission(path, height=4, width=6)


def test_filament_ids_must_be_unique(tmp_path: Path) -> None:
    height, width = 4, 6
    counts_a = encode_mask(_pixel(height, width, 0, 0))
    counts_b = encode_mask(_pixel(height, width, 0, 1))
    path = _write(
        tmp_path,
        [("20150125172714Mh_1", counts_a), ("20150125172714Mh_1", counts_b)],
    )

    with pytest.raises(SubmissionValidationError, match="duplicate filament_id"):
        validate_submission(path, height=height, width=width)


def test_empty_mask_fails(tmp_path: Path) -> None:
    path = _write(tmp_path, rows_for_image("img", [np.zeros((4, 6), dtype=np.uint8)]))

    with pytest.raises(SubmissionValidationError, match="decoded mask is empty"):
        validate_submission(path, height=4, width=6)


def test_overlapping_masks_of_one_image_fail_when_rows_are_interleaved(
    tmp_path: Path,
) -> None:
    height, width = 4, 6
    shared = encode_mask(_pixel(height, width, 1, 1))
    other = encode_mask(_pixel(height, width, 0, 0))
    path = _write(
        tmp_path,
        [
            ("left_1", shared),
            ("right_1", shared),
            ("left_2", shared),
            ("right_2", other),
        ],
    )

    with pytest.raises(SubmissionValidationError, match=r"image left: masks overlap"):
        validate_submission(path, height=height, width=width)


def test_extra_csv_quotes_around_rle_fail(tmp_path: Path) -> None:
    height, width = 4, 6
    path = _write(tmp_path, rows_for_image("img", [_pixel(height, width, 0, 0)]))
    text = path.read_text(encoding="utf-8")
    header, data = text.split("\n", 1)
    filament_id_value, counts = data.strip("\n").split(",", 1)
    path.write_text(
        f'{header}\n{filament_id_value},"{counts}"\n',
        encoding="utf-8",
    )

    with pytest.raises(SubmissionValidationError, match="QUOTE_MINIMAL"):
        validate_submission(path, height=height, width=width)


def test_csv_quotes_around_a_comma_are_not_extra(tmp_path: Path) -> None:
    path = _write(tmp_path, [("img_1", "ab,cd")])

    with pytest.raises(SubmissionValidationError, match="RLE did not decode"):
        validate_submission(path, height=4, width=6)


def test_blank_rle_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, [("img_1", "")])

    with pytest.raises(SubmissionValidationError, match="segmentation_rle is empty"):
        validate_submission(path, height=4, width=6)


def test_quote_inside_rle_counts_fails(tmp_path: Path) -> None:
    path = _write(tmp_path, [("img_1", '91>"')])

    with pytest.raises(SubmissionValidationError, match="must not contain quotes"):
        validate_submission(path, height=4, width=6)


def test_undecodable_rle_fails(tmp_path: Path) -> None:
    path = _write(tmp_path, [("img_1", "abc")])

    with pytest.raises(SubmissionValidationError, match="RLE did not decode"):
        validate_submission(path, height=4, width=6)


def test_more_than_40_instances_warns_and_passes(tmp_path: Path) -> None:
    height, width = 7, 7
    masks = [
        _pixel(height, width, index // width, index % width) for index in range(41)
    ]
    path = _write(tmp_path, rows_for_image("img", masks))

    with pytest.warns(UserWarning, match="more than 40"):
        validate_submission(path, height=height, width=width)


def test_forty_instances_do_not_warn(tmp_path: Path) -> None:
    height, width = 5, 8
    masks = [
        _pixel(height, width, index // width, index % width) for index in range(40)
    ]
    path = _write(tmp_path, rows_for_image("img", masks))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_submission(path, height=height, width=width)

    # pycocotools emits a NumPy DeprecationWarning on decode. That is not
    # the instance-count warning, and it must not fail this check.
    instance_warnings = [
        item for item in caught if issubclass(item.category, UserWarning)
    ]
    assert instance_warnings == []


def test_default_mask_size_is_2048(tmp_path: Path) -> None:
    signature = inspect.signature(validate_submission)
    assert signature.parameters["height"].default == 2048
    assert signature.parameters["width"].default == 2048
    assert DEFAULT_HEIGHT == 2048
    assert DEFAULT_WIDTH == 2048

    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[3, 5] = 1
    path = _write(tmp_path, rows_for_image("img", [mask]))

    assert validate_submission(path) is None
