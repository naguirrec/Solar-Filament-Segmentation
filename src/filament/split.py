"""Split images into train and validation by observation date.

This is the local split for comparing models without a Kaggle submission.
Pass one mapping per image. ``date_captured`` is a COCO datetime
(``YYYY-MM-DD HH:MM:SS``) and is preferred. When that field is missing
or blank, the calendar date is the ``YYYYMMDD`` prefix of the file stem
(``20220621185332Bh`` in ``20220621185332Bh.jpeg``). Images from the same
calendar date stay together. The newest fraction of distinct dates (default
0.2) is the validation holdout. Dates and file names are sorted, so the
same images always produce the same split.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

_DATE_DIGITS = 8


class DateSplit(NamedTuple):
    """File names for train and validation, oldest dates first on each side."""

    train: tuple[str, ...]
    val: tuple[str, ...]


def observation_date(image: Mapping[str, object]) -> dt.date:
    """Return the calendar date of one image.

    A missing or blank ``date_captured`` falls back to ``file_name``.
    Any other non-string raises ``TypeError``. The calendar date is the
    date written in the timestamp; it is not shifted to another timezone.
    """
    file_name = _require_file_name(image)
    captured = image.get("date_captured")
    if _is_blank(captured):
        return _date_from_file_name(file_name)
    if not isinstance(captured, str):
        raise TypeError(
            "date_captured must be a COCO datetime string, "
            f"got {type(captured).__name__}"
        )
    return _parse_date_captured(captured)


def split_by_date(
    images: Iterable[Mapping[str, object]],
    val_fraction: float = 0.2,
) -> DateSplit:
    """Hold out the most recent fraction of observation dates.

    ``val_fraction`` is the share of distinct calendar dates placed in
    validation. It must lie in the open interval ``(0, 1)``. The number of
    validation dates is the integer closest to ``n_dates * val_fraction``
    (a tie rounds toward the larger holdout) and is clamped so both sides
    keep at least one date. Ten dates at the default 0.2 hold out the two
    newest; two dates hold out the newest one.

    Each image is a mapping with ``file_name`` and, when known,
    ``date_captured``. Other keys are ignored. A missing or blank
    ``date_captured`` falls back to ``file_name``. Returned names are the
    ``file_name`` values the caller passed.

    This is the local split for comparing models without a Kaggle submission.
    """
    fraction = _require_fraction(val_fraction)
    if isinstance(images, Mapping):
        raise TypeError(
            "images must be an iterable of image mappings, not a single mapping"
        )

    resolved: dict[str, dt.date] = {}
    for image in images:
        file_name = _require_file_name(image)
        day = observation_date(image)
        previous = resolved.get(file_name)
        if previous is not None and previous != day:
            earlier, later = sorted((previous, day))
            raise ValueError(
                f"file_name {file_name!r} has observation dates "
                f"{earlier.isoformat()} and {later.isoformat()}"
            )
        resolved[file_name] = day

    by_date: dict[dt.date, list[str]] = {}
    for file_name, day in resolved.items():
        by_date.setdefault(day, []).append(file_name)
    for names in by_date.values():
        names.sort()
    dates = sorted(by_date)
    if len(dates) < 2:
        detail = dates[0].isoformat() if dates else "no images"
        raise ValueError(
            "validation holdout needs at least two distinct observation dates; "
            f"got {len(dates)} ({detail})"
        )

    n_val = _validation_date_count(len(dates), fraction)
    train_days = dates[:-n_val]
    val_days = dates[-n_val:]
    train = tuple(name for day in train_days for name in by_date[day])
    val = tuple(name for day in val_days for name in by_date[day])
    return DateSplit(train, val)


def _require_file_name(image: Mapping[str, object]) -> str:
    if isinstance(image, (str, bytes)) or not isinstance(image, Mapping):
        raise TypeError(
            f"each image must be a mapping with file_name, got {type(image).__name__}"
        )
    file_name = image.get("file_name")
    if not isinstance(file_name, str) or file_name.strip() == "":
        raise ValueError("each image needs a non-empty file_name string")
    return file_name


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip() == ""


def _parse_date_captured(value: str) -> dt.date:
    text = value.strip()
    try:
        return dt.datetime.fromisoformat(text).date()
    except ValueError as exc:
        raise ValueError(
            "date_captured must be a COCO datetime (YYYY-MM-DD HH:MM:SS) "
            f"or an ISO date, got {value!r}"
        ) from exc


def _date_from_file_name(file_name: str) -> dt.date:
    stem = Path(file_name).stem
    prefix = stem[:_DATE_DIGITS]
    if len(prefix) < _DATE_DIGITS or not prefix.isdigit():
        raise ValueError(
            f"file_name must start with a real YYYYMMDD date, got {file_name!r}"
        )
    try:
        return dt.datetime.strptime(prefix, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(
            f"file_name must start with a real YYYYMMDD date, got {file_name!r}"
        ) from exc


def _require_fraction(val_fraction: object) -> float:
    if isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float)):
        raise TypeError(
            "val_fraction must be a real number in (0, 1), "
            f"got {type(val_fraction).__name__}"
        )
    fraction = float(val_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction!r}")
    return fraction


def _validation_date_count(n_dates: int, val_fraction: float) -> int:
    """Nearest date count. A tie rounds toward the larger holdout."""
    nearest = math.floor(n_dates * val_fraction + 0.5)
    return min(n_dates - 1, max(1, nearest))
