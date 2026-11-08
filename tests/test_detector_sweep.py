"""The detector sweep on synthetic frames. Every expected number is worked out by hand."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from filament.axis_report import MIN_PARTITIONS_ADOPTED, crossfit_rows
from filament.components import ViewHits
from filament.detector_fusion import (
    DEFAULT_CONFIG,
    Detection,
    FusionConfig,
    dedup_detections,
    merge_across_windows,
)
from filament.detector_sweep import (
    CROSSFIT_FOLDS,
    CROSSFIT_SEEDS,
    ClaimHits,
    CrossfitSummary,
    DedupStats,
    Setting,
    axis_settings,
    calibrate_crossfit,
    claim_file,
    counts_for,
    crossfit_summary,
    dedup_stats,
    submit_without_asking,
    unique_settings,
    verdict_line,
)
from filament.metrics import PqCounts
from filament.stats import crossfit_select, date_folds, paired_bootstrap, weighted_pq
from filament.sweep import format_pq

SIDE = 64
WINDOWS = [(0, 0, SIDE, 40), (0, 24, SIDE, SIDE)]
# Three filaments of the U-Net. The first two are true, the third is not.
A, B, C = (5, 5, 15, 20), (25, 5, 35, 20), (45, 5, 55, 20)


def rect(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = True
    return mask


def det(mask: np.ndarray, score: float, window: int = 0) -> Detection:
    return Detection.from_mask(mask, score, window)


def _scene():
    unet = rect(A, B, C)
    detections = [det(rect(A), 0.9), det(rect(B), 0.2), det(rect(C), 0.1)]
    views = [[rect(A), rect(B)]]
    disk = np.ones((SIDE, SIDE), dtype=bool)
    return detections, unet, disk, views


# ------------------------------------------------------------------ settings


def test_axes_change_one_thing_at_a_time_from_the_defaults() -> None:
    axes = axis_settings()
    assert list(axes) == ["score_raw", "score_calibrated", "dilation", "emit_unclaimed"]
    for setting in axes["score_raw"] + axes["score_calibrated"]:
        assert setting.config == DEFAULT_CONFIG and setting.cut is not None
    assert all(s.calibrated for s in axes["score_calibrated"])
    for setting in axes["dilation"]:
        assert setting.cut is None
        assert setting.config.emit_unclaimed is False
        assert setting.config.min_area == DEFAULT_CONFIG.min_area
    assert {s.config.dilation for s in axes["dilation"]} >= {0, DEFAULT_CONFIG.dilation}
    assert [s.config.emit_unclaimed for s in axes["emit_unclaimed"]] == [False, True]


def test_the_default_appears_once_in_the_unique_settings() -> None:
    axes = axis_settings()
    unique = unique_settings(axes)
    assert len(unique) == len(set(unique))
    plain = [s for s in unique if s.cut is None and s.config == DEFAULT_CONFIG]
    assert len(plain) == 1 and plain[0].label == "default"
    # Fewer than the sum of the axes: the default sits on two of them.
    assert len(unique) == sum(len(v) for v in axes.values()) - 1


def test_labels_name_the_change() -> None:
    assert Setting("a", DEFAULT_CONFIG, 0.3).label == "raw>=0.3"
    assert Setting("a", DEFAULT_CONFIG, 0.3, calibrated=True).label == "cal>=0.3"
    changed = FusionConfig(dilation=9, emit_unclaimed=True)
    assert Setting("a", changed).label == "dilation=9, emit_unclaimed"


def test_a_setting_refuses_a_cut_in_the_config_or_a_calibrated_unscored_claim() -> None:
    with pytest.raises(ValueError, match="min_score"):
        Setting("a", FusionConfig(min_score=0.5))
    with pytest.raises(ValueError, match="needs a cut"):
        Setting("a", DEFAULT_CONFIG, None, calibrated=True)
    with pytest.raises(ValueError, match="calibrate"):
        Setting("a", FusionConfig(emit_unclaimed=True), 0.3, calibrated=True)


# ------------------------------------------------------------- claim and cuts


def test_one_claim_serves_every_cut() -> None:
    detections, unet, disk, views = _scene()
    config = FusionConfig(dilation=0, min_area=1)
    claimed = claim_file(detections, unet, disk, views, WINDOWS, [config])
    claim = claimed.claims[config]
    assert claim.scores.tolist() == [0.9, 0.2, 0.1]
    everything = counts_for(claim)
    assert (everything.tp, everything.fp, everything.fn) == (2, 1, 0)
    assert everything.iou_sum == pytest.approx(2.0)
    assert everything.pq == pytest.approx(2 / 2.5)
    for cut, expected in ((0.15, (2, 0, 0)), (0.5, (1, 0, 1)), (0.95, (0, 0, 2))):
        counts = counts_for(claim, cut)
        assert (counts.tp, counts.fp, counts.fn) == expected, cut


def test_a_cut_matches_cutting_the_label_image_with_the_fusion_module() -> None:
    from filament.detector_fusion import FusionFile, per_file_detector_counts

    detections, unet, disk, views = _scene()
    claimed = claim_file(
        detections, unet, disk, views, WINDOWS, [FusionConfig(dilation=0, min_area=1)]
    )
    claim = claimed.claims[FusionConfig(dilation=0, min_area=1)]
    for cut in (0.05, 0.15, 0.5):
        config = FusionConfig(dilation=0, min_area=1, min_score=cut)
        (reference,) = per_file_detector_counts(
            [FusionFile("a", detections, unet, views, disk)], WINDOWS, config
        )
        assert counts_for(claim, cut) == reference


def test_nan_scores_need_keep_unscored() -> None:
    hits = ViewHits(
        candidate=np.array([0, 1]),
        ground_truth=np.array([0, 1]),
        iou=np.array([0.9, 0.8]),
        ground_truths=2,
    )
    claim = ClaimHits(hits=(hits,), scores=np.array([0.9, np.nan]))
    assert counts_for(claim, 0.5).tp == 1
    assert counts_for(claim, 0.5, keep_unscored=True).tp == 2
    with pytest.raises(ValueError, match="scores"):
        counts_for(claim, 0.5, scores=np.array([0.9]))


def test_configs_may_differ_only_in_dilation_and_emit_unclaimed() -> None:
    detections, unet, disk, views = _scene()
    with pytest.raises(ValueError, match="only in dilation"):
        claim_file(
            detections, unet, disk, views, WINDOWS,
            [FusionConfig(), FusionConfig(dedup_iou=0.5)],
        )  # fmt: skip
    with pytest.raises(ValueError, match="min_score"):
        claim_file(
            detections, unet, disk, views, WINDOWS, [FusionConfig(min_score=0.3)]
        )


def test_the_disk_gate_is_applied_in_every_claim() -> None:
    detections, unet, _, views = _scene()
    gate = np.ones((SIDE, SIDE), dtype=bool)
    gate[40:] = False  # C is off the disk
    config = FusionConfig(dilation=0, min_area=1)
    claimed = claim_file(detections, unet, gate, views, WINDOWS, [config])
    assert claimed.claims[config].scores.tolist() == [0.9, 0.2]
    only = claimed.detector_only
    assert only.scores.tolist() == [0.9, 0.2]


def test_detector_only_ignores_the_unet_and_uses_the_detector_masks() -> None:
    # The U-Net sees nothing at all; the detector still claims its own pixels.
    detections, _, disk, views = _scene()
    empty = np.zeros((SIDE, SIDE), dtype=bool)
    config = FusionConfig(dilation=0, min_area=1)
    claimed = claim_file(detections, empty, disk, views, WINDOWS, [config])
    assert claimed.claims[config].scores.size == 0
    counts = counts_for(claimed.detector_only)
    assert (counts.tp, counts.fp, counts.fn) == (2, 1, 0)


def test_emit_unclaimed_adds_unscored_pieces() -> None:
    detections, unet, disk, views = _scene()
    detections = detections[:2]  # nobody claims C
    plain = FusionConfig(dilation=0, min_area=1)
    loud = FusionConfig(dilation=0, min_area=1, emit_unclaimed=True)
    claimed = claim_file(detections, unet, disk, views, WINDOWS, [plain, loud])
    assert claimed.claims[plain].scores.tolist() == [0.9, 0.2]
    assert claimed.claims[loud].scores[:2].tolist() == [0.9, 0.2]
    assert np.isnan(claimed.claims[loud].scores[2])
    kept = counts_for(claimed.claims[loud], 0.5, keep_unscored=True)
    assert (kept.tp, kept.fp) == (1, 1)


# --------------------------------------------------------------------- dedup


def test_dedup_stats_counts_drops_and_how_many_were_across_windows() -> None:
    first = det(rect((10, 0, 20, 30)), 0.9, window=0)
    other_window = det(rect((10, 20, 20, 50)), 0.6, window=1)
    same_window = det(rect((10, 0, 20, 25)), 0.3, window=0)
    # Seen by two windows over the columns 24..39: IoU 6/16 there, under 0.5, so
    # the merge keeps them apart. Over the whole frame they overlap at 0.2.
    merged = merge_across_windows([first, other_window, same_window], WINDOWS, 0.5)
    assert len(merged) == 3
    kept = dedup_detections(merged, 0.05)
    assert [d.score for d in kept] == [0.9]
    stats = dedup_stats(merged, kept, 0.05)
    assert (stats.merged, stats.kept, stats.dropped) == (3, 1, 2)
    assert stats.dropped_cross_window == 1


def test_dedup_stats_add_up() -> None:
    one = DedupStats(10, 8, 5, 3, 1)
    assert one + one == DedupStats(20, 16, 10, 6, 2)
    assert one.__add__(1) is NotImplemented


def test_claim_file_records_the_input_count_and_the_drops() -> None:
    detections, unet, disk, views = _scene()
    extra = det(rect((6, 6, 14, 19)), 0.05)  # inside A, lower score: a duplicate
    claimed = claim_file(
        [*detections, extra], unet, disk, views, WINDOWS,
        [FusionConfig(dilation=0, min_area=1)],
    )  # fmt: skip
    stats = claimed.dedup
    assert (stats.detections, stats.dropped, stats.dropped_cross_window) == (4, 1, 0)


# --------------------------------------------------------------- calibration


def _frame_claim() -> ClaimHits:
    hits = ViewHits(
        candidate=np.array([0, 1]),
        ground_truth=np.array([0, 1]),
        iou=np.array([0.9, 0.8]),
        ground_truths=2,
    )
    return ClaimHits(hits=(hits,), scores=np.array([0.9, 0.5, 0.1]))


def test_calibration_is_out_of_fold_by_date_and_keeps_the_shapes() -> None:
    claims = [_frame_claim() for _ in range(6)]
    dates = [dt.date(2020, 1, day) for day in range(1, 7)]
    calibrated = calibrate_crossfit(claims, dates, n_splits=2)
    assert [c.shape for c in calibrated] == [(3,)] * 6
    for values in calibrated:
        # The target is the matched IoU: 0.9, 0.8 and 0 for the unmatched id.
        assert values == pytest.approx([0.9, 0.8, 0.0])
    with pytest.raises(ValueError, match="dates"):
        calibrate_crossfit(claims, dates[:2])


def test_calibration_handles_a_frame_without_ids() -> None:
    empty = ClaimHits(hits=(), scores=np.zeros(0))
    claims = [_frame_claim(), empty, _frame_claim(), _frame_claim()]
    dates = [dt.date(2020, 1, day) for day in range(1, 5)]
    calibrated = calibrate_crossfit(claims, dates, n_splits=2)
    assert [c.size for c in calibrated] == [3, 0, 3, 3]
    with pytest.raises(ValueError, match="no claimed ids"):
        calibrate_crossfit([empty, empty], dates[:2], n_splits=2)


# ----------------------------------------------------------------- crossfit


def _dates(files: int = 20) -> list[dt.date]:
    return [dt.date(2020, 1, 1 + index // 2) for index in range(files)]


def _counts(tp: int, fp: int, fn: int, iou: float, files: int = 20) -> list[PqCounts]:
    return [PqCounts(tp, fp, fn, iou) for _ in range(files)]


def test_the_summary_names_the_setting_chosen_most_often() -> None:
    base = _counts(1, 1, 1, 0.8)
    better = _counts(2, 0, 0, 1.8)
    summary = crossfit_summary(base, {"base": base, "better": better}, _dates())
    assert summary.config == "better"
    assert summary.picks == {"better": 10}
    assert summary.adopted == summary.partitions == 5
    assert summary.fraction_better == 1.0
    expected = PqCounts(40, 0, 0, 36.0).pq - PqCounts(20, 20, 20, 16.0).pq
    assert summary.diff == pytest.approx(expected)
    assert submit_without_asking(summary)
    line = verdict_line(summary)
    assert line.startswith("VERDICT better pq=")
    assert " fraction_better=1.000 adopted=5/5 base_fixed_in_sample=" in line
    assert line.endswith(format_pq(weighted_pq(base)))
    assert f"diff=+{expected:.6f}" in line


def test_keeping_the_base_is_no_gain_and_no_submission() -> None:
    base = _counts(2, 0, 0, 1.8)
    worse = _counts(1, 1, 1, 0.8)
    summary = crossfit_summary(base, {"base": base, "worse": worse}, _dates())
    assert summary.config == "base"
    assert summary.diff == pytest.approx(0.0)
    assert summary.adopted == 0
    assert not submit_without_asking(summary)
    assert " diff=+0.000000 " in verdict_line(summary) or " diff=-0.000000 " in (
        verdict_line(summary)
    )


def test_a_gain_of_exactly_the_threshold_is_not_enough() -> None:
    base = _counts(1, 0, 0, 0.5)
    even = _counts(1, 0, 0, 0.51)
    summary = crossfit_summary(base, {"even": even}, _dates())
    assert summary.diff == pytest.approx(0.01)
    assert not submit_without_asking(summary)
    assert submit_without_asking(summary, gain=0.009)


def test_the_summary_agrees_with_the_axis_report_verdict_row() -> None:
    base = _counts(1, 1, 1, 0.8)
    settings = {
        "base": base,
        "a": _counts(2, 0, 1, 1.5),
        "b": _counts(1, 0, 1, 0.9),
    }
    summary = crossfit_summary(base, settings, _dates())
    rows, adopted = crossfit_rows("all", base, settings, _dates())
    assert format_pq(summary.pooled_pq) == rows[-1]["pooled_pq"]
    assert summary.adopted == sum(row["note"].startswith("passes") for row in rows[:-1])
    assert (summary.adopted >= 4) == adopted


# ------------------------------------------------------- the submission gate


def _summary(diff: float, adopted: int, pooled_pq: float = 0.4311) -> CrossfitSummary:
    return CrossfitSummary(
        config="s2",
        pooled_pq=pooled_pq,
        diff=diff,
        fraction_better=0.933,
        adopted=adopted,
        partitions=5,
        picks={"s2": 10},
    )


def test_the_gate_also_needs_the_adoption_rule() -> None:
    # The reviewer's input: diff above the bar, only 3 of 5 partitions adopted.
    assert not submit_without_asking(_summary(0.010090, adopted=3))
    assert not submit_without_asking(_summary(0.010090, adopted=0))
    assert submit_without_asking(_summary(0.010090, adopted=MIN_PARTITIONS_ADOPTED))
    assert submit_without_asking(_summary(0.010090, adopted=5))
    # Enough partitions do not rescue a gain at or under the bar.
    assert not submit_without_asking(_summary(0.010000, adopted=5))


def test_a_pilot_never_submits_without_asking() -> None:
    assert submit_without_asking(_summary(0.02, adopted=5))
    assert not submit_without_asking(_summary(0.02, adopted=5), pilot=True)


def test_the_verdict_line_shows_adopted_and_the_base_diff_is_against() -> None:
    line = verdict_line(_summary(0.0056, adopted=3, pooled_pq=0.4311))
    assert line == (
        "VERDICT s2 pq=0.431100 diff=+0.005600 fraction_better=0.933 "
        "adopted=3/5 base_fixed_in_sample=0.425500"
    )


# ------------------------------------- the verdict is out of fold, not the peak


def _noisy_grid(
    files: int = 40, settings: int = 12, seed: int = 7
) -> tuple[list[PqCounts], dict[str, list[PqCounts]]]:
    """Per-file counts that differ file to file; no setting is truly better."""
    rng = np.random.default_rng(seed)

    def draw() -> list[PqCounts]:
        found = []
        for _ in range(files):
            tp = int(rng.integers(1, 7))
            found.append(
                PqCounts(
                    tp,
                    int(rng.integers(0, 5)),
                    int(rng.integers(0, 5)),
                    float(tp * rng.uniform(0.55, 0.85)),
                )
            )
        return found

    base = draw()
    grid = {"base": base}
    for index in range(settings):
        grid["s" + str(index)] = draw()
    return base, grid


def _noisy_dates(files: int) -> list[dt.date]:
    return [dt.date(2020, 1, 1 + index // 2) for index in range(files)]


def test_the_summary_pools_the_out_of_fold_counts_and_not_the_sweep_peak() -> None:
    base, grid = _noisy_grid()
    dates = _noisy_dates(len(base))
    summary = crossfit_summary(base, grid, dates)

    # By hand: the same partitions and the same bootstrap seeds, one call each.
    results = [
        crossfit_select(grid, date_folds(dates, CROSSFIT_FOLDS, seed=seed))
        for seed in CROSSFIT_SEEDS
    ]
    by_hand = float(np.mean([result.pooled_pq for result in results]))
    peak = max(result.best_pq for result in results)
    fractions = [
        paired_bootstrap(base, result.out_of_fold, n=2000, seed=seed).fraction_better
        for seed, result in zip(CROSSFIT_SEEDS, results, strict=True)
    ]

    # The test is only worth anything if the peak and the out-of-fold number differ.
    assert peak - by_hand > 0.01
    assert summary.pooled_pq == pytest.approx(by_hand)
    assert summary.pooled_pq < peak - 0.01
    assert summary.diff == pytest.approx(by_hand - weighted_pq(base))
    assert summary.fraction_better == pytest.approx(float(np.mean(fractions)))
    assert sum(summary.picks.values()) == len(CROSSFIT_SEEDS) * CROSSFIT_FOLDS


def test_the_partitions_use_different_seeds() -> None:
    base, grid = _noisy_grid()
    dates = _noisy_dates(len(base))
    per_seed = [
        crossfit_select(grid, date_folds(dates, CROSSFIT_FOLDS, seed=seed)).pooled_pq
        for seed in CROSSFIT_SEEDS
    ]
    # Five different partitions give five different out-of-fold numbers here, so a
    # summary that reused one partition cannot match the mean over the seeds.
    assert len({round(value, 9) for value in per_seed}) == len(CROSSFIT_SEEDS)
    summary = crossfit_summary(base, grid, dates)
    assert summary.pooled_pq == pytest.approx(float(np.mean(per_seed)))
    assert summary.pooled_pq != pytest.approx(per_seed[0], abs=1e-9)
