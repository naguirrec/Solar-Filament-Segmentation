"""Validate a Kaggle submission CSV before upload.

Call this on the file from :func:`filament.submission.write_submission`:

    validate_submission(path)

The inference notebook does that after it writes the CSV. It does not submit
the file.

Checks, in order:

1. The header line is exactly ``filament_id,segmentation_rle``.
2. ``filament_id`` values are unique. Each id is ``{image_stem}_{n}`` with
   ``n`` starting at 1, and masks are grouped by ``image_stem``.
3. Each RLE decodes with :func:`filament.rle.decode_mask` at ``height`` by
   ``width`` (default 2048 by 2048) and the mask is non-empty.
4. Masks of the same image are disjoint. A pixel is foreground in at most
   one of those masks.
5. If one image has more than 40 instances, warn and continue. The warning
   is not an error.
6. A quote character inside the RLE counts is rejected. Quotes that
   ``csv.QUOTE_MINIMAL`` must emit around a field are CSV syntax, not part
   of the counts. pycocotools counts use ASCII 48-111 and do not contain
   quotes.
"""

from __future__ import annotations

import csv
import io
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from filament.rle import decode_mask
from filament.submission import COLUMNS

DEFAULT_HEIGHT = 2048
DEFAULT_WIDTH = 2048
MAX_INSTANCES_PER_IMAGE = 40

_HEADER_LINE = f"{COLUMNS[0]},{COLUMNS[1]}"
_FILAMENT_ID = re.compile(r"^(?P<image>.+)_(?P<number>[1-9]\d*)$")


class SubmissionValidationError(ValueError):
    """Raised when a submission CSV fails validation."""


def validate_submission(
    path: str | Path,
    *,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Raise :class:`SubmissionValidationError` when ``path`` is not uploadable.

    ``height`` and ``width`` are the decoded mask size passed to
    :func:`filament.rle.decode_mask`. They default to 2048 by 2048.

    An image with more than 40 instances emits :class:`UserWarning` and
    does not fail validation.
    """
    rows_h, cols_w = _mask_size(height, width)
    destination = Path(path)
    rows = _read_rows(destination)
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for filament_id_value, counts in rows:
        image = _image_stem(filament_id_value)
        grouped[image].append((filament_id_value, counts))
    for image, instances in grouped.items():
        _validate_image(image, instances, rows_h, cols_w)
        count = len(instances)
        if count > MAX_INSTANCES_PER_IMAGE:
            warnings.warn(
                f"image {image} has {count} instances, "
                f"more than {MAX_INSTANCES_PER_IMAGE}",
                UserWarning,
                stacklevel=2,
            )


def _mask_size(height: int, width: int) -> tuple[int, int]:
    if isinstance(height, bool) or isinstance(width, bool):
        raise TypeError("height and width must be ints")
    if not isinstance(height, (int, np.integer)) or not isinstance(
        width, (int, np.integer)
    ):
        raise TypeError("height and width must be ints")
    rows = int(height)
    cols = int(width)
    if rows < 1 or cols < 1:
        raise ValueError(f"mask shape {(rows, cols)} must be at least 1x1")
    return rows, cols


def _read_rows(path: Path) -> list[tuple[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionValidationError(f"{path} is not UTF-8 text") from exc
    except OSError as exc:
        raise SubmissionValidationError(f"could not read {path}: {exc}") from exc

    first_line, separator, _rest = text.partition("\n")
    if first_line != _HEADER_LINE or separator != "\n":
        raise SubmissionValidationError(
            f"header must be exactly filament_id,segmentation_rle, got {first_line!r}"
        )

    try:
        parsed = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise SubmissionValidationError(f"{path} is not a valid CSV: {exc}") from exc

    if not parsed or parsed[0] != list(COLUMNS):
        raise SubmissionValidationError(
            "header must be exactly filament_id,segmentation_rle, "
            f"got {parsed[0] if parsed else None!r}"
        )

    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, record in enumerate(parsed[1:], start=2):
        if len(record) != 2:
            raise SubmissionValidationError(
                f"row {line_number} must have 2 columns, got {len(record)}"
            )
        filament_id_value, counts = record
        if '"' in filament_id_value or '"' in counts:
            raise SubmissionValidationError(
                f"row {line_number} filament_id {filament_id_value!r} has a "
                "quote inside a field. csv.QUOTE_MINIMAL may quote a whole "
                "field; the RLE counts themselves must not contain quotes"
            )
        if filament_id_value in seen:
            raise SubmissionValidationError(
                f"duplicate filament_id {filament_id_value!r}"
            )
        seen.add(filament_id_value)
        rows.append((filament_id_value, counts))

    if _quote_minimal(rows) != text:
        raise SubmissionValidationError(
            "segmentation_rle quoting must match csv.QUOTE_MINIMAL. "
            "Do not add quotes inside the RLE counts beyond the quotes "
            "the csv module emits around a field"
        )
    return rows


def _quote_minimal(rows: list[tuple[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    return buffer.getvalue()


def _image_stem(filament_id_value: str) -> str:
    match = _FILAMENT_ID.fullmatch(filament_id_value)
    if match is None:
        raise SubmissionValidationError(
            f"filament_id {filament_id_value!r} must be '{{image_stem}}_{{n}}' "
            "with n starting at 1"
        )
    return match.group("image")


def _validate_image(
    image: str,
    instances: list[tuple[str, str]],
    height: int,
    width: int,
) -> None:
    owner = np.zeros((height, width), dtype=np.int32)
    ids = [filament_id_value for filament_id_value, _counts in instances]
    for index, (filament_id_value, counts) in enumerate(instances, start=1):
        if counts == "":
            raise SubmissionValidationError(
                f"{filament_id_value}: segmentation_rle is empty"
            )
        try:
            mask = decode_mask(counts, height, width)
        except (TypeError, ValueError) as exc:
            raise SubmissionValidationError(
                f"{filament_id_value}: RLE did not decode: {exc}"
            ) from exc
        if mask.shape != (height, width):
            raise SubmissionValidationError(
                f"{filament_id_value}: decoded shape {mask.shape}, "
                f"expected {(height, width)}"
            )
        foreground = mask != 0
        if not np.any(foreground):
            raise SubmissionValidationError(
                f"{filament_id_value}: decoded mask is empty"
            )
        overlap = foreground & (owner != 0)
        if np.any(overlap):
            other = ids[int(owner[overlap][0]) - 1]
            raise SubmissionValidationError(
                f"image {image}: masks overlap ({other} and {filament_id_value})"
            )
        owner[foreground] = index
