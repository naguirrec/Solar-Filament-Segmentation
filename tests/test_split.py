"""Date split from image fields. Strings only: no image files and no network."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# src/ is not an installed package in this pull request.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from filament.split import observation_date, split_by_date


def test_date_captured_overrides_file_name() -> None:
    image = {
        "id": 7,
        "width": 2048,
        "height": 2048,
        "file_name": "20150125172714Mh.jpeg",
        "date_captured": "2022-06-21 18:53:32",
    }
    assert observation_date(image) == dt.date(2022, 6, 21)


@pytest.mark.parametrize("captured", [None, "", "   "])
def test_blank_date_captured_uses_file_name(captured: str | None) -> None:
    image = {
        "file_name": "20220621185332Bh.jpeg",
        "date_captured": captured,
    }
    assert observation_date(image) == dt.date(2022, 6, 21)


def test_absent_date_captured_uses_file_name() -> None:
    assert observation_date({"file_name": "20220621185332Bh.jpeg"}) == dt.date(
        2022, 6, 21
    )
    assert observation_date({"file_name": "gong/20150125172714Mh.jpeg"}) == dt.date(
        2015, 1, 25
    )
    assert observation_date(
        {"file_name": "ignored.jpeg", "date_captured": "2015-01-25"}
    ) == dt.date(2015, 1, 25)


def test_same_calendar_date_stays_on_one_side() -> None:
    images = [
        {"file_name": "20220619120000Bh.jpeg"},
        {"file_name": "20220620120000Bh.jpeg"},
        {
            "file_name": "other-site.jpeg",
            "date_captured": "2022-06-20 19:00:00",
        },
        {"file_name": "20220621185332Bh.jpeg"},
        {"file_name": "20220622120000Bh.jpeg"},
        {
            "file_name": "20220622180000Mh.jpeg",
            "date_captured": "2022-06-22 18:00:00",
        },
    ]
    result = split_by_date(images, val_fraction=0.2)
    assert result.train == (
        "20220619120000Bh.jpeg",
        "20220620120000Bh.jpeg",
        "other-site.jpeg",
        "20220621185332Bh.jpeg",
    )
    assert result.val == (
        "20220622120000Bh.jpeg",
        "20220622180000Mh.jpeg",
    )


def test_most_recent_dates_are_held_out() -> None:
    images: list[dict[str, object]] = []
    for day in range(1, 11):
        ymd = f"202201{day:02d}"
        images.append({"file_name": f"{ymd}120000Bh.jpeg"})
        if day == 3:
            images.append({"file_name": f"{ymd}130000Mh.jpeg"})
    images.append(
        {
            "file_name": "latest-frame.jpeg",
            "date_captured": "2022-01-10 18:53:32",
        }
    )
    result = split_by_date(images)
    train: list[str] = []
    for day in range(1, 9):
        ymd = f"202201{day:02d}"
        train.append(f"{ymd}120000Bh.jpeg")
        if day == 3:
            train.append(f"{ymd}130000Mh.jpeg")
    assert result.train == tuple(train)
    assert result.val == (
        "20220109120000Bh.jpeg",
        "20220110120000Bh.jpeg",
        "latest-frame.jpeg",
    )
    assert split_by_date(images, val_fraction=0.2) == result


@pytest.mark.parametrize(
    ("n_dates", "held_out"),
    [(7, 1), (8, 2), (10, 2)],
)
def test_holdout_count_is_the_nearest_share_of_dates(
    n_dates: int, held_out: int
) -> None:
    images = [
        {"file_name": f"202201{day:02d}120000Bh.jpeg"} for day in range(1, n_dates + 1)
    ]
    result = split_by_date(images, val_fraction=0.2)
    assert [name[:8] for name in result.val] == [
        f"202201{day:02d}" for day in range(n_dates - held_out + 1, n_dates + 1)
    ]
    assert result.train[-1][:8] < result.val[0][:8]


def test_half_rounds_toward_the_larger_holdout() -> None:
    # 5 dates * 0.5 = 2.5. The larger holdout is 3. Banker's round would keep 2.
    images = [{"file_name": f"2022062{day}120000Bh.jpeg"} for day in range(1, 6)]
    result = split_by_date(images, val_fraction=0.5)
    assert result.train == (
        "20220621120000Bh.jpeg",
        "20220622120000Bh.jpeg",
    )
    assert result.val == (
        "20220623120000Bh.jpeg",
        "20220624120000Bh.jpeg",
        "20220625120000Bh.jpeg",
    )


def test_newest_of_two_example_names_is_validation() -> None:
    older = {"file_name": "20150125172714Mh.jpeg"}
    newer = {"file_name": "20220621185332Bh.jpeg"}
    held_out = split_by_date([newer, older])
    assert held_out.train == ("20150125172714Mh.jpeg",)
    assert held_out.val == ("20220621185332Bh.jpeg",)
    assert split_by_date([older, newer], val_fraction=0.8) == held_out


def test_split_does_not_depend_on_input_order() -> None:
    images = [
        {"file_name": "20220622120000Bh.jpeg"},
        {"file_name": "20150125172714Mh.jpeg"},
        {"file_name": "20150125010101Bh.jpeg"},
        {
            "file_name": "20220621185332Bh.jpeg",
            "date_captured": "2022-06-21 18:53:32",
        },
        {"file_name": "20220621190000Mh.jpeg"},
    ]
    permuted = [images[3], images[0], images[4], images[1], images[2]]
    assert split_by_date(permuted) == split_by_date(list(reversed(images)))
    duplicated = images + [images[1], images[3]]
    assert split_by_date(duplicated) == split_by_date(images)


@pytest.mark.parametrize(
    "images",
    [
        [],
        [{"file_name": "20150125172714Mh.jpeg"}],
        [
            {"file_name": "20150125172714Mh.jpeg"},
            {"file_name": "20150125010101Bh.jpeg"},
            {
                "file_name": "also-that-day.jpeg",
                "date_captured": "2015-01-25 17:27:14",
            },
        ],
    ],
)
def test_fewer_than_two_dates_raises(images: list[dict[str, object]]) -> None:
    with pytest.raises(ValueError, match="two distinct observation dates"):
        split_by_date(images)


@pytest.mark.parametrize("fraction", [0, 0.0, 1, 1.0, -0.1, 1.2, float("nan")])
def test_fraction_outside_open_interval_raises(fraction: float) -> None:
    images = [
        {"file_name": "20150125172714Mh.jpeg"},
        {"file_name": "20220621185332Bh.jpeg"},
    ]
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        split_by_date(images, fraction)


def test_unparseable_date_captured_is_not_taken_from_file_name() -> None:
    images = [
        {
            "file_name": "20220621185332Bh.jpeg",
            "date_captured": "not-a-date",
        },
        {"file_name": "20150125172714Mh.jpeg"},
    ]
    with pytest.raises(ValueError, match="date_captured"):
        split_by_date(images)


@pytest.mark.parametrize("file_name", ["filament.jpeg", "20220230120000Bh.jpeg"])
def test_file_name_without_a_date_raises_when_captured_is_missing(
    file_name: str,
) -> None:
    images = [
        {"file_name": file_name},
        {"file_name": "20150125172714Mh.jpeg"},
    ]
    with pytest.raises(ValueError, match="YYYYMMDD"):
        split_by_date(images)


def test_fraction_must_be_a_real_number() -> None:
    images = [
        {"file_name": "20150125172714Mh.jpeg"},
        {"file_name": "20220621185332Bh.jpeg"},
    ]
    with pytest.raises(TypeError, match="val_fraction"):
        split_by_date(images, "0.2")
    with pytest.raises(TypeError, match="val_fraction"):
        split_by_date(images, True)


def test_non_string_date_captured_raises() -> None:
    image = {"file_name": "20220621185332Bh.jpeg", "date_captured": 20220621}
    with pytest.raises(TypeError, match="date_captured"):
        observation_date(image)


def test_bare_stem_is_rejected() -> None:
    with pytest.raises(TypeError, match="mapping"):
        split_by_date(["20220621185332Bh.jpeg"])


def test_same_file_name_with_two_dates_raises() -> None:
    images = [
        {
            "file_name": "20220621185332Bh.jpeg",
            "date_captured": "2022-06-21 18:53:32",
        },
        {
            "file_name": "20220621185332Bh.jpeg",
            "date_captured": "2015-01-25 17:27:14",
        },
    ]
    with pytest.raises(ValueError, match="20220621185332Bh.jpeg"):
        split_by_date(images)
