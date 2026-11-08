"""Write the Kaggle submission CSV.

Columns are exactly ``filament_id`` and ``segmentation_rle``. ``filament_id``
is ``{image_stem}_{n}`` with ``n`` starting at 1. ``segmentation_rle`` is the
pycocotools counts string only.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from filament.rle import encode_mask

COLUMNS = ("filament_id", "segmentation_rle")


def filament_id(image_stem: str, number: int) -> str:
    """Return ``{image_stem}_{number}`` with ``number`` starting at 1."""
    if number < 1:
        raise ValueError("filament number starts at 1")
    return f"{image_stem}_{number}"


def rows_for_image(
    image_stem: str, masks: Sequence[np.ndarray]
) -> list[tuple[str, str]]:
    """Build one ``(filament_id, counts)`` row per mask, numbered from 1."""
    return [
        (filament_id(image_stem, number), encode_mask(mask))
        for number, mask in enumerate(masks, start=1)
    ]


def write_submission(path: str | Path, rows: Iterable[tuple[str, str]]) -> None:
    """Write ``rows`` of ``(filament_id, segmentation_rle)`` with the csv module.

    A counts string that contains commas is quoted by ``csv.QUOTE_MINIMAL``.
    Those quotes are CSV syntax and are not stripped. Concatenate
    :func:`rows_for_image` across images for a full test set.
    """
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(COLUMNS)
        for filament_id_value, counts in rows:
            writer.writerow((filament_id_value, counts))
