"""Bootstrap, reweighting and cross-fitting on synthetic per-file counts."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from filament.metrics import PqCounts
from filament.stats import (
    PairedResult,
    adopts_change,
    bootstrap_pq,
    crossfit_select,
    date_folds,
    effective_files,
    paired_bootstrap,
    plateau_center,
    tilt_to_mean_views,
    weighted_pq,
)


def _files(pqs: list[float], matched: int = 10) -> list[PqCounts]:
    """Files with ``matched`` true positives whose pooled PQ is the given value."""
    return [PqCounts(tp=matched, iou_sum=matched * pq) for pq in pqs]


def _noisy_files(rng: np.random.Generator, mean: float, n: int) -> list[PqCounts]:
    return _files(list(np.clip(rng.normal(mean, 0.15, n), 0.0, 1.0)))


# --- weighted_pq -----------------------------------------------------------------


def test_equal_weights_give_the_pooled_pq_of_the_added_counts() -> None:
    counts = [
        PqCounts(tp=3, fp=1, fn=2, iou_sum=2.1),
        PqCounts(tp=1, fp=4, fn=0, iou_sum=0.7),
        PqCounts(tp=0, fp=2, fn=5, iou_sum=0.0),
    ]
    total = counts[0] + counts[1] + counts[2]
    assert weighted_pq(counts) == pytest.approx(total.pq)
    assert weighted_pq(counts, [1.0, 1.0, 1.0]) == pytest.approx(total.pq)
    assert weighted_pq(counts, [2.0, 2.0, 2.0]) == pytest.approx(total.pq)


def test_a_weight_counts_a_file_that_many_times() -> None:
    a, b = PqCounts(tp=2, fn=1, iou_sum=1.5), PqCounts(tp=1, fp=3, iou_sum=0.6)
    assert weighted_pq([a, b], [2.0, 1.0]) == pytest.approx((a + a + b).pq)
    assert weighted_pq([a, b], [0.0, 1.0]) == pytest.approx(b.pq)


def test_weighted_pq_is_not_the_mean_of_per_file_pq() -> None:
    a, b = PqCounts(tp=1, iou_sum=1.0), PqCounts(fp=100)
    assert weighted_pq([a, b]) == pytest.approx(1.0 / 51.0)
    assert weighted_pq([a, b]) != pytest.approx((a.pq + b.pq) / 2)


def test_weighted_pq_with_nothing_to_score_is_nan() -> None:
    assert np.isnan(weighted_pq([PqCounts()]))


@pytest.mark.parametrize("weights", [[1.0], [1.0, -1.0], [1.0, np.nan]])
def test_weighted_pq_refuses_bad_weights(weights: list[float]) -> None:
    with pytest.raises(ValueError):
        weighted_pq([PqCounts(tp=1, iou_sum=1.0), PqCounts()], weights)


def test_no_files_is_an_error() -> None:
    with pytest.raises(ValueError, match="no files"):
        weighted_pq([])


# --- bootstrap_pq ----------------------------------------------------------------


def test_bootstrap_interval_contains_the_point_and_is_seeded() -> None:
    counts = _noisy_files(np.random.default_rng(0), 0.4, 142)
    result = bootstrap_pq(counts, n=500, seed=1)
    assert result.point == pytest.approx(weighted_pq(counts))
    assert result.low < result.point < result.high
    assert result.resamples == 500
    assert bootstrap_pq(counts, n=500, seed=1) == result
    assert bootstrap_pq(counts, n=500, seed=2) != result


def test_bootstrap_interval_covers_the_truth_about_as_often_as_it_says() -> None:
    rng = np.random.default_rng(0)
    covered = 0
    for trial in range(200):
        counts = _noisy_files(rng, 0.4, 142)
        result = bootstrap_pq(counts, n=200, seed=trial)
        covered += result.low <= 0.4 <= result.high
    assert 0.88 <= covered / 200 <= 0.99


def test_identical_files_have_no_spread() -> None:
    result = bootstrap_pq(_files([0.5] * 20), n=100)
    assert result.low == pytest.approx(0.5)
    assert result.high == pytest.approx(0.5)


def test_more_files_narrow_the_interval() -> None:
    rng = np.random.default_rng(1)
    few = bootstrap_pq(_noisy_files(rng, 0.4, 30), n=500)
    many = bootstrap_pq(_noisy_files(rng, 0.4, 600), n=500)
    assert many.high - many.low < few.high - few.low


def test_bootstrap_ignores_resamples_with_nothing_to_score() -> None:
    counts = [PqCounts(), PqCounts(tp=1, iou_sum=0.5)]
    result = bootstrap_pq(counts, n=200)
    assert result.low == pytest.approx(0.5)
    assert result.high == pytest.approx(0.5)


def test_bootstrap_with_weights_centers_on_the_weighted_pq() -> None:
    counts = _files([0.2] * 20 + [0.8] * 20)
    weights = [3.0] * 20 + [1.0] * 20
    result = bootstrap_pq(counts, n=500, weights=weights)
    assert result.point == pytest.approx(weighted_pq(counts, weights))
    assert result.point == pytest.approx(0.35)
    assert result.low < result.point < result.high


def test_bootstrap_refuses_no_resamples() -> None:
    with pytest.raises(ValueError):
        bootstrap_pq(_files([0.5]), n=0)


# --- paired_bootstrap ------------------------------------------------------------


def test_a_setting_against_itself_is_no_difference() -> None:
    counts = _noisy_files(np.random.default_rng(0), 0.4, 50)
    result = paired_bootstrap(counts, counts, n=200)
    assert result.point == 0.0 and result.low == 0.0 and result.high == 0.0
    assert result.fraction_better == 0.0


def test_a_uniformly_better_setting_wins_every_resample() -> None:
    a = _noisy_files(np.random.default_rng(0), 0.4, 50)
    b = [PqCounts(tp=c.tp, iou_sum=c.iou_sum + 0.5) for c in a]
    result = paired_bootstrap(a, b, n=200)
    assert result.fraction_better == 1.0
    assert result.low > 0 and result.point > 0


def test_direction_is_b_minus_a() -> None:
    a = _files([0.5] * 10)
    b = _files([0.3] * 10)
    assert paired_bootstrap(a, b, n=50).point == pytest.approx(-0.2)
    assert paired_bootstrap(b, a, n=50).point == pytest.approx(0.2)


def test_pairing_removes_what_the_two_settings_share() -> None:
    # Files differ a lot, while b beats a by the same small amount on each.
    rng = np.random.default_rng(3)
    a = _noisy_files(rng, 0.4, 142)
    b = [PqCounts(tp=c.tp, iou_sum=c.iou_sum + 0.05) for c in a]
    paired = paired_bootstrap(a, b, n=500)
    alone = bootstrap_pq(a, n=500)
    assert paired.high - paired.low < (alone.high - alone.low) / 5
    assert paired.fraction_better == 1.0


def test_paired_needs_the_same_files() -> None:
    with pytest.raises(ValueError, match="files"):
        paired_bootstrap(_files([0.5]), _files([0.5, 0.5]))


def test_paired_with_nothing_to_score_is_an_error() -> None:
    with pytest.raises(ValueError, match="zero denominator"):
        paired_bootstrap([PqCounts()], [PqCounts()], n=10)


@pytest.mark.parametrize(
    ("point", "fraction", "expected"),
    [
        (0.005, 0.90, True),
        (0.02, 0.99, True),
        (0.004, 0.99, False),  # too small
        (0.02, 0.89, False),  # not consistent
    ],
)
def test_adoption_rule(point: float, fraction: float, expected: bool) -> None:
    result = PairedResult(point, point, 0.0, 0.0, fraction, 100)
    assert adopts_change(result) is expected


# --- tilt_to_mean_views ----------------------------------------------------------


def test_tilt_reaches_the_target_mix() -> None:
    views = [1] * 80 + [2] * 40 + [3] * 22
    weights = tilt_to_mean_views(views, 2.4)
    assert np.sum(weights * views) / np.sum(weights) == pytest.approx(2.4, abs=1e-6)
    assert np.sum(weights) == pytest.approx(len(views))
    # Files with the same number of views keep the same weight.
    assert len(set(np.round(weights[:80], 9))) == 1
    assert weights[0] < weights[80] < weights[120]


def test_tilt_at_the_current_mean_changes_nothing() -> None:
    views = [1, 1, 2, 2, 3, 3]
    assert np.allclose(tilt_to_mean_views(views, 2.0), 1.0, atol=1e-6)


def test_tilt_can_also_go_down() -> None:
    views = [1, 2, 3, 3]
    weights = tilt_to_mean_views(views, 1.5)
    assert np.sum(weights * np.array(views)) / np.sum(weights) == pytest.approx(1.5)


@pytest.mark.parametrize("target", [0.5, 1.0, 3.0, 4.0])
def test_tilt_refuses_a_mix_these_files_cannot_reach(target: float) -> None:
    with pytest.raises(ValueError, match="outside the range"):
        tilt_to_mean_views([1, 2, 3], target)


def test_tilt_with_one_view_count() -> None:
    assert np.array_equal(tilt_to_mean_views([2, 2], 2.0), [1.0, 1.0])
    with pytest.raises(ValueError, match="every file"):
        tilt_to_mean_views([2, 2], 3.0)


def test_tilt_refuses_no_files_or_a_file_without_views() -> None:
    with pytest.raises(ValueError):
        tilt_to_mean_views([], 2.0)
    with pytest.raises(ValueError):
        tilt_to_mean_views([0, 2], 1.0)


def test_effective_files() -> None:
    assert effective_files([1.0] * 10) == pytest.approx(10.0)
    assert effective_files([5.0] * 10) == pytest.approx(10.0)
    assert effective_files([1.0, 0.0, 0.0]) == pytest.approx(1.0)


# --- date_folds ------------------------------------------------------------------


def _dates(*days: int) -> list[dt.date]:
    return [dt.date(2020, 1, day) for day in days]


def test_files_of_one_date_stay_together_and_folds_balance() -> None:
    dates = _dates(1, 1, 1, 2, 3, 3, 4, 5, 6, 6)
    for seed in (None, 0, 1, 2):
        folds = date_folds(dates, 2, seed=seed)
        assert set(folds.tolist()) == {0, 1}
        for day in set(dates):
            assert len({f for f, d in zip(folds, dates, strict=True) if d == day}) == 1
        sizes = np.bincount(folds)
        assert abs(int(sizes[0]) - int(sizes[1])) <= 3


def test_folds_are_deterministic_and_seeds_differ() -> None:
    dates = _dates(*range(1, 21))
    assert np.array_equal(date_folds(dates, 2), date_folds(dates, 2))
    assert np.array_equal(date_folds(dates, 2, seed=5), date_folds(dates, 2, seed=5))
    partitions = {tuple(date_folds(dates, 2, seed=s).tolist()) for s in range(5)}
    assert len(partitions) > 1


def test_folds_refuse_too_few_dates_or_folds() -> None:
    with pytest.raises(ValueError, match="cannot fill"):
        date_folds(_dates(1, 1), 2)
    with pytest.raises(ValueError, match="n_folds"):
        date_folds(_dates(1, 2), 1)


# --- crossfit_select -------------------------------------------------------------


def test_crossfit_scores_the_chosen_setting_on_the_other_fold() -> None:
    # Setting "a" is better on fold 0 and "b" on fold 1, so each fold is scored
    # with the setting that won the other: the worse one.
    grid = {
        "a": _files([0.8, 0.8, 0.2, 0.2]),
        "b": _files([0.3, 0.3, 0.9, 0.9]),
    }
    result = crossfit_select(grid, [0, 0, 1, 1])
    assert result.selected == ("b", "a")
    assert result.fold_pqs == pytest.approx((0.3, 0.2))
    assert result.pooled_pq == pytest.approx(0.25)
    assert result.mean_fold_pq == pytest.approx(0.25)
    assert result.best_key == "b"
    assert result.best_pq == pytest.approx(0.6)
    assert result.optimism == pytest.approx(0.35)


def test_a_peak_that_is_noise_does_not_survive_crossfitting() -> None:
    # 60 settings with the same true PQ. Each file's score carries noise that
    # belongs to the setting, so the sweep's maximum is pure selection.
    rng = np.random.default_rng(0)
    grid = {key: _noisy_files(rng, 0.40, 142) for key in range(60)}
    folds = date_folds(
        [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(142)], 2, seed=0
    )
    result = crossfit_select(grid, folds)
    assert result.best_pq > 0.42  # the sweep's peak looks like a real gain
    assert abs(result.pooled_pq - 0.40) < 0.02  # cross-fitting reports the truth
    assert result.optimism > 0.01


def test_a_real_difference_survives_crossfitting() -> None:
    rng = np.random.default_rng(1)
    grid = {key: _noisy_files(rng, 0.35, 142) for key in range(20)}
    grid["real"] = _noisy_files(rng, 0.50, 142)
    folds = date_folds(
        [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(142)], 2, seed=0
    )
    result = crossfit_select(grid, folds)
    assert result.selected == ("real", "real")
    assert result.best_key == "real"
    assert result.pooled_pq == pytest.approx(result.best_pq)
    assert result.optimism == pytest.approx(0.0)


def test_crossfit_pools_counts_rather_than_averaging_folds() -> None:
    grid = {"a": [PqCounts(tp=1, iou_sum=1.0)] + [PqCounts(fp=2)] * 3}
    result = crossfit_select(grid, [0, 1, 1, 1])
    # Fold 0 scores 1.0 and fold 1 scores 0; pooled is one match over 4.
    assert result.mean_fold_pq == pytest.approx(0.5)
    assert result.pooled_pq == pytest.approx(1.0 / 4.0)


def test_crossfit_tie_goes_to_the_first_setting() -> None:
    same = _files([0.5, 0.5])
    result = crossfit_select({"first": same, "second": same}, [0, 1])
    assert result.selected == ("first", "first")
    assert result.best_key == "first"


def test_crossfit_refuses_bad_input() -> None:
    with pytest.raises(ValueError, match="no settings"):
        crossfit_select({}, [0, 1])
    with pytest.raises(ValueError, match="two folds"):
        crossfit_select({"a": _files([0.5, 0.5])}, [0, 0])
    with pytest.raises(ValueError, match="files"):
        crossfit_select({"a": _files([0.5])}, [0, 1])
    with pytest.raises(ValueError, match="zero denominator"):
        crossfit_select({"a": [PqCounts(), PqCounts()]}, [0, 1])


def test_out_of_fold_counts_come_from_the_setting_chosen_without_the_file() -> None:
    grid = {
        "a": _files([0.8, 0.8, 0.2, 0.2]),
        "b": _files([0.3, 0.3, 0.9, 0.9]),
    }
    result = crossfit_select(grid, [0, 0, 1, 1])
    assert result.selected == ("b", "a")
    assert list(result.out_of_fold) == [
        grid["b"][0],
        grid["b"][1],
        grid["a"][2],
        grid["a"][3],
    ]
    assert weighted_pq(result.out_of_fold) == pytest.approx(result.pooled_pq)


def test_out_of_fold_counts_go_into_a_paired_comparison() -> None:
    rng = np.random.default_rng(1)
    base = _noisy_files(rng, 0.35, 142)
    better = [PqCounts(tp=c.tp, iou_sum=c.iou_sum + 0.5) for c in base]
    folds = date_folds(
        [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(142)], 2, seed=0
    )
    result = crossfit_select({"base": base, "better": better}, folds)
    assert result.selected == ("better", "better")
    comparison = paired_bootstrap(base, result.out_of_fold, n=200)
    assert comparison.fraction_better == 1.0


# --- plateau_center ---------------------------------------------------------------


def test_plateau_center_prefers_the_middle_over_the_peak() -> None:
    values = [0.30, 0.399, 0.400, 0.399, 0.399, 0.399, 0.30]
    assert plateau_center(values, tolerance=0.002) == 3


def test_a_lone_peak_is_its_own_plateau() -> None:
    assert plateau_center([0.1, 0.5, 0.1], tolerance=0.002) == 1


def test_plateau_stops_at_a_gap_even_if_later_values_are_close() -> None:
    assert plateau_center([0.40, 0.30, 0.40, 0.40, 0.40], tolerance=0.002) == 0


def test_plateau_edges_and_ties() -> None:
    assert plateau_center([0.4, 0.4]) == 0
    assert plateau_center([0.1, 0.4, 0.4, 0.4]) == 2


def test_nan_breaks_a_plateau_and_all_nan_is_an_error() -> None:
    assert plateau_center([0.4, float("nan"), 0.4], tolerance=0.1) == 0
    with pytest.raises(ValueError):
        plateau_center([float("nan")])
    with pytest.raises(ValueError):
        plateau_center([])
    with pytest.raises(ValueError):
        plateau_center([0.1], tolerance=-1.0)
