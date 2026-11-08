"""The H1 and H2 analysis on synthetic frames and counts."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.gap_analysis import (
    CROSSFIT_FOLDS,
    CROSSFIT_SEEDS,
    CURRENT_POINT,
    GAP_CLOSE_SIZES,
    GAP_MIN_AREAS,
    GAP_THRESHOLDS,
    PER_FILE_COLUMNS,
    SUMMARY_COLUMNS,
    format_summary,
    gap_points,
    gap_summary_rows,
    holdout_dates,
    per_file_rows,
    read_per_file_counts,
    repeated_crossfit,
    write_per_file_rows,
    write_summary_rows,
)
from filament.metrics import PqCounts
from filament.stats import weighted_pq
from filament.sweep import GridPoint, PerFileCounts, per_file_counts

GOOD = GridPoint(0.45, 11, 400)
BAD = GridPoint(0.5, 0, 400)


def _per_file(
    good_pqs: list[float], bad_pqs: list[float], views: list[int]
) -> PerFileCounts:
    def counts(pqs: list[float]) -> tuple[PqCounts, ...]:
        return tuple(PqCounts(tp=10, iou_sum=10 * pq) for pq in pqs)

    return PerFileCounts(
        file_names=tuple(f"f{i}.jpeg" for i in range(len(views))),
        views=tuple(views),
        counts={GOOD: counts(good_pqs), BAD: counts(bad_pqs)},
    )


def _days(count: int) -> list[dt.date]:
    return [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(count)]


def _rows_by_name(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["quantity"]: row for row in rows}


def test_the_grid_holds_the_current_setting() -> None:
    points = gap_points()
    assert CURRENT_POINT == GridPoint(0.45, 11, 400)
    assert CURRENT_POINT in points
    assert len(points) == len(GAP_THRESHOLDS) * len(GAP_CLOSE_SIZES) * len(
        GAP_MIN_AREAS
    )
    assert len(CROSSFIT_SEEDS) == 5 and CROSSFIT_FOLDS == 2


def test_strata_split_files_by_their_number_of_views() -> None:
    per_file = _per_file([0.6] * 4 + [0.3] * 4, [0.1] * 8, [1, 1, 1, 1, 2, 2, 3, 3])
    rows = _rows_by_name(gap_summary_rows(per_file, _days(8), resamples=100))
    assert float(rows["h2_stratum_one_view"]["value"]) == pytest.approx(0.6)
    assert rows["h2_stratum_one_view"]["files"] == "4"
    assert float(rows["h2_stratum_several_views"]["value"]) == pytest.approx(0.3)
    assert rows["h2_stratum_several_views"]["files"] == "4"


def test_reweighting_toward_more_views_moves_pq_toward_those_files() -> None:
    views = [1] * 6 + [2] * 3 + [3] * 3
    per_file = _per_file([0.6] * 6 + [0.3] * 6, [0.1] * 12, views)
    rows = _rows_by_name(
        gap_summary_rows(per_file, _days(12), target_views=2.4, resamples=100)
    )
    plain = float(rows["current_setting"]["value"])
    tilted = float(rows["h2_reweighted_views"]["value"])
    assert plain == pytest.approx(weighted_pq(per_file.counts[GOOD]))
    assert 0.3 < tilted < plain
    assert "2.4 views per file" in rows["h2_reweighted_views"]["note"]
    assert "effective files" in rows["h2_reweighted_views"]["note"]


def test_a_target_these_files_cannot_reach_is_reported_not_raised() -> None:
    # A holdout with at most 2 views per file cannot reach 2.44. The scoring took
    # a long time by then, so the rest of the table has to survive.
    per_file = _per_file([0.6] * 4, [0.1] * 4, [1, 1, 2, 2])
    rows = _rows_by_name(
        gap_summary_rows(per_file, _days(4), target_views=2.44, resamples=50)
    )
    assert rows["h2_reweighted_views"]["value"] == "nan"
    assert "outside the range" in rows["h2_reweighted_views"]["note"]
    assert float(rows["h1_crossfit_mean"]["value"]) == pytest.approx(0.6)


def test_each_exact_view_count_gets_its_own_stratum() -> None:
    per_file = _per_file([0.6, 0.6, 0.4, 0.4, 0.2, 0.2], [0.1] * 6, [1, 1, 2, 2, 3, 3])
    rows = _rows_by_name(gap_summary_rows(per_file, _days(6), resamples=50))
    assert float(rows["h2_stratum_views_2"]["value"]) == pytest.approx(0.4)
    assert float(rows["h2_stratum_views_3"]["value"]) == pytest.approx(0.2)
    assert "h2_stratum_views_1" not in rows


def test_a_holdout_without_one_stratum_says_so() -> None:
    per_file = _per_file([0.5] * 4, [0.1] * 4, [2, 2, 3, 3])
    rows = _rows_by_name(gap_summary_rows(per_file, _days(4), resamples=50))
    assert rows["h2_stratum_one_view"]["value"] == "nan"
    assert rows["h2_stratum_one_view"]["note"] == "no such files"


def test_a_real_setting_has_no_selection_optimism() -> None:
    per_file = _per_file([0.6] * 12, [0.1] * 12, [1] * 6 + [2] * 6)
    rows = _rows_by_name(
        gap_summary_rows(per_file, _days(12), target_views=1.8, resamples=50)
    )
    assert float(rows["h1_sweep_maximum"]["value"]) == pytest.approx(0.6)
    assert float(rows["h1_crossfit_mean"]["value"]) == pytest.approx(0.6)
    assert float(rows["h1_sweep_maximum_minus_crossfit"]["value"]) == pytest.approx(0.0)
    for seed in CROSSFIT_SEEDS:
        assert f"h1_crossfit_seed_{seed}" in rows


def test_selection_noise_shows_up_as_optimism() -> None:
    rng = np.random.default_rng(0)
    counts = {
        GridPoint(0.5, 0, area): tuple(
            PqCounts(tp=10, iou_sum=10 * float(v))
            for v in np.clip(rng.normal(0.4, 0.15, 142), 0, 1)
        )
        for area in range(100, 700, 10)
    }
    per_file = PerFileCounts(
        file_names=tuple(f"f{i}" for i in range(142)),
        views=tuple([1] * 71 + [2] * 71),
        counts=counts,
    )
    point = next(iter(counts))
    rows = _rows_by_name(
        gap_summary_rows(per_file, _days(142), point=point, resamples=50)
    )
    assert float(rows["h1_sweep_maximum_minus_crossfit"]["value"]) > 0.01
    assert abs(float(rows["h1_crossfit_mean"]["value"]) - 0.40) < 0.02


def test_partitions_differ_between_seeds() -> None:
    per_file = _per_file([0.6] * 20, [0.1] * 20, [1] * 20)
    results = repeated_crossfit(per_file, _days(20), seeds=(0, 1))
    assert len(results) == 2


def test_dates_have_to_match_the_files() -> None:
    per_file = _per_file([0.6] * 4, [0.1] * 4, [1, 1, 2, 2])
    with pytest.raises(ValueError, match="must match"):
        repeated_crossfit(per_file, _days(3))


def test_a_setting_that_was_not_scored_is_refused() -> None:
    per_file = _per_file([0.6] * 4, [0.1] * 4, [1, 1, 2, 2])
    other = GridPoint(0.3, 0, 400)
    with pytest.raises(ValueError, match="not scored"):
        gap_summary_rows(per_file, _days(4), point=other)
    with pytest.raises(ValueError, match="not scored"):
        per_file_rows(per_file, other)


def test_holdout_dates_follow_the_split_rule() -> None:
    image = TrainImage(
        id=1,
        file_name="20150125172714Mh.jpeg",
        date_captured=None,
        height=8,
        width=8,
        annotations=(),
    )
    dated = TrainImage(
        id=2,
        file_name="20150125172714Mh.jpeg",
        date_captured="2016-03-04 10:00:00",
        height=8,
        width=8,
        annotations=(),
    )
    assert holdout_dates([image, dated]) == [dt.date(2015, 1, 25), dt.date(2016, 3, 4)]


def test_summary_csv_and_text(tmp_path: Path) -> None:
    per_file = _per_file([0.6] * 6, [0.1] * 6, [1, 1, 1, 2, 2, 3])
    rows = gap_summary_rows(per_file, _days(6), resamples=50)
    path = tmp_path / "gap.csv"
    write_summary_rows(path, rows)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == SUMMARY_COLUMNS
        assert list(reader) == rows
    text = format_summary(rows)
    assert "current_setting" in text and "h1_crossfit_mean" in text


def test_per_file_csv_round_trips_for_a_later_paired_comparison(tmp_path: Path) -> None:
    per_file = PerFileCounts(
        file_names=("a.jpeg", "b.jpeg"),
        views=(1, 2),
        counts={GOOD: (PqCounts(3, 1, 2, 2.125), PqCounts(0, 4, 1, 0.0))},
    )
    rows = per_file_rows(per_file, GOOD)
    path = tmp_path / "per_file.csv"
    write_per_file_rows(path, rows)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == PER_FILE_COLUMNS
        first = next(reader)
    assert first["file_name"] == "a.jpeg" and first["views"] == "1"
    assert read_per_file_counts(path) == list(per_file.counts[GOOD])


def test_the_analysis_runs_on_real_scoring_of_synthetic_frames() -> None:
    side = 128
    rows, cols = np.ogrid[:side, :side]
    gray = np.where((rows - 64) ** 2 + (cols - 64) ** 2 <= 58**2, 200, 5).astype(
        np.uint8
    )

    def box(x0: int, y0: int, x1: int, y1: int) -> tuple[float, ...]:
        return (x0, y0, x1, y0, x1, y1, x0, y1)

    images = []
    maps = {}
    for index in range(4):
        annotation = FilamentAnnotation(
            id=index,
            image_id=index,
            category_id=1,
            spine=None,
            segmentation=(box(40, 40 + index, 80, 60 + index),),
        )
        name = f"2020010{index + 1}120000Aa.jpeg"
        images.append(
            TrainImage(
                id=index,
                file_name=name,
                date_captured=None,
                height=side,
                width=side,
                annotations=(annotation,),
                views=((annotation,), ()) if index % 2 else (),
            )
        )
        maps[name] = (np.asarray(annotation.rasterize(side, side)) != 0).astype(
            np.float32
        ) * 0.9
    points = [GridPoint(0.45, 0, 100), GridPoint(0.5, 0, 100), GridPoint(0.5, 0, 5000)]
    per_file = per_file_counts(
        images, lambda _image: gray, lambda image: maps[image.file_name], points
    )
    assert per_file.views == (1, 2, 1, 2)
    summary = _rows_by_name(
        gap_summary_rows(
            per_file,
            holdout_dates(images),
            point=points[0],
            target_views=1.8,
            resamples=50,
        )
    )
    # One view: a perfect match. Two views, one empty: TP in one, FP in the other.
    assert float(summary["h2_stratum_one_view"]["value"]) > 0.9
    several = float(summary["h2_stratum_several_views"]["value"])
    assert several == pytest.approx(1.0 / 1.5, abs=0.05)
