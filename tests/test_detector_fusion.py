"""Detector fusion on synthetic frames. Every expected number is worked out by hand."""

from __future__ import annotations

import datetime as dt
import itertools

import numpy as np
import pytest

from filament import detector_fusion
from filament.components import ViewHits, selection_counts
from filament.detector_fusion import (
    Detection,
    FusionConfig,
    FusionFile,
    claim_pixels,
    crossfit_calibrated_scores,
    dedup_detections,
    emit_by_pooled_pq,
    fusion_targets,
    isotonic_calibrator,
    keep_scored,
    merge_across_windows,
    per_file_detector_counts,
)
from filament.metrics import PqCounts, fast_view_counts
from filament.stats import crossfit_select, date_folds, paired_bootstrap

SIDE = 64
# Two windows over a 64 x 64 frame. They share the columns 24..39.
WINDOWS = [(0, 0, SIDE, 40), (0, 24, SIDE, SIDE)]


def rect(shape: tuple[int, int], top: int, left: int, bottom: int, right: int):
    """A boolean mask with ones in rows top..bottom-1 and columns left..right-1."""
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def det(mask: np.ndarray, score: float, window: int = 0) -> Detection:
    return Detection.from_mask(mask, score, window)


def frame(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = True
    return mask


# ---------------------------------------------------------------- Detection


def test_detection_round_trips_and_keeps_its_box() -> None:
    mask = frame((10, 5, 20, 45))
    detection = det(mask, 0.9, window=1)
    assert np.array_equal(detection.decode(), mask)
    assert detection.bbox == (10, 5, 20, 45)
    assert detection.crop().shape == (10, 40)
    assert detection.window == 1


def test_detection_is_compact_next_to_a_frame_sized_array() -> None:
    big = rect((2048, 2048), 100, 100, 160, 900)
    detection = det(big, 0.5)
    # A frame-sized bool array is 4 MiB. The RLE of this long, thin mask (one run
    # per column, the worst orientation for a column-major encoding) is 1.6 KB.
    assert big.nbytes == 2048 * 2048
    assert len(detection.mask) < big.nbytes // 1000


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_detection_rejects_a_score_outside_the_unit_interval(score: float) -> None:
    with pytest.raises(ValueError, match="score"):
        det(frame((1, 1, 5, 5)), score)


def test_detection_rejects_an_empty_mask_and_a_bad_box() -> None:
    with pytest.raises(ValueError, match="at least one pixel"):
        det(np.zeros((8, 8), bool), 0.5)
    with pytest.raises(ValueError, match="bbox"):
        Detection(0.5, "x", (8, 8), 0, (0, 0, 9, 4))


# ---------------------------------------------------------------- merging


def test_a_filament_across_a_window_edge_is_merged_despite_clipped_views() -> None:
    full = frame((10, 5, 20, 45))
    # Window 0 sees columns 0..39 and window 1 columns 24..63: each view is clipped.
    seen_by_0 = full & frame((0, 0, SIDE, 40))
    seen_by_1 = full & frame((0, 24, SIDE, SIDE))
    plain_iou = (seen_by_0 & seen_by_1).sum() / (seen_by_0 | seen_by_1).sum()
    # 160 shared pixels over 400 in the union: a plain IoU says "different filaments".
    assert plain_iou == pytest.approx(0.4) and plain_iou < 0.5

    merged = merge_across_windows(
        [det(seen_by_0, 0.7, 0), det(seen_by_1, 0.9, 1)], WINDOWS
    )
    assert len(merged) == 1
    assert np.array_equal(merged[0].decode(), full)
    assert merged[0].score == 0.9 and merged[0].window == 1


def test_two_distinct_filaments_side_by_side_do_not_merge() -> None:
    # Both filaments sit in the strip the two windows share, and both windows see both.
    upper = frame((10, 26, 15, 38))
    lower = frame((20, 26, 25, 38))
    detections = [
        det(upper, 0.9, 0),
        det(lower, 0.8, 0),
        det(upper, 0.7, 1),
        det(lower, 0.6, 1),
    ]
    merged = merge_across_windows(detections, WINDOWS)
    assert len(merged) == 2
    boxes = sorted(d.bbox for d in merged)
    assert boxes == [(10, 26, 15, 38), (20, 26, 25, 38)]
    assert sorted(d.score for d in merged) == [0.8, 0.9]


def test_merge_depends_on_the_iou_over_the_shared_area_and_its_threshold() -> None:
    first = frame((10, 26, 15, 38))
    shifted = frame((12, 26, 17, 38))  # rows 12..14 overlap: 3 of 7 rows = 3/7
    pair = [det(first, 0.9, 0), det(shifted, 0.8, 1)]
    assert len(merge_across_windows(pair, WINDOWS)) == 2
    assert len(merge_across_windows(pair, WINDOWS, windows_iou=0.4)) == 1


def test_detections_of_one_window_and_of_windows_without_overlap_never_merge() -> None:
    same = [det(frame((10, 5, 20, 15)), 0.9, 0), det(frame((10, 5, 20, 15)), 0.8, 0)]
    assert len(merge_across_windows(same, WINDOWS)) == 2
    apart = [(0, 0, SIDE, 20), (0, 40, SIDE, SIDE)]
    distinct = [
        det(frame((10, 5, 20, 15)), 0.9, 0),
        det(frame((10, 45, 20, 55)), 0.8, 1),
    ]
    assert len(merge_across_windows(distinct, apart)) == 2


def test_merge_is_transitive_and_does_not_depend_on_the_input_order() -> None:
    windows = [(0, 0, SIDE, 30), (0, 20, SIDE, 50), (0, 40, SIDE, SIDE)]
    long = frame((10, 2, 18, 62))
    parts = [
        det(long & frame((0, 0, SIDE, 30)), 0.5, 0),
        det(long & frame((0, 20, SIDE, 50)), 0.8, 1),
        det(long & frame((0, 40, SIDE, SIDE)), 0.6, 2),
    ]
    results = []
    for order in itertools.permutations(parts):
        merged = merge_across_windows(list(order), windows)
        assert len(merged) == 1
        results.append((merged[0].mask, merged[0].score, merged[0].window))
    assert len(set(results)) == 1
    assert np.array_equal(merge_across_windows(parts, windows)[0].decode(), long)


def test_merge_checks_its_inputs() -> None:
    with pytest.raises(ValueError, match="no rectangle"):
        merge_across_windows([det(frame((1, 1, 5, 5)), 0.5, 3)], WINDOWS)
    with pytest.raises(ValueError, match="windows_iou"):
        merge_across_windows([], WINDOWS, windows_iou=1.0)
    other = Detection.from_mask(np.ones((8, 8), bool), 0.5, 0)
    with pytest.raises(ValueError, match="frame shapes"):
        merge_across_windows([det(frame((1, 1, 5, 5)), 0.5), other], WINDOWS)
    assert merge_across_windows([], WINDOWS) == []


# ---------------------------------------------------------------- dedup


def test_dedup_keeps_the_best_score_and_drops_overlaps_above_the_threshold() -> None:
    best = det(rect((40, 40), 0, 0, 10, 10), 0.9)
    # Shares row 9 over 10 columns: inter 10, union 190, IoU 0.0526 > 0.05.
    touching = det(rect((40, 40), 9, 0, 19, 10), 0.8)
    # Shares 8 pixels: inter 8, union 192, IoU 0.0417 < 0.05.
    grazing = det(rect((40, 40), 9, 2, 19, 12), 0.7)
    far = det(rect((40, 40), 30, 30, 38, 38), 0.6)
    kept = dedup_detections([far, touching, best, grazing])
    assert [d.score for d in kept] == [0.9, 0.7, 0.6]
    assert len(dedup_detections([best, touching], iou=0.06)) == 2


def test_dedup_breaks_ties_in_a_fixed_order() -> None:
    mask = rect((40, 40), 5, 5, 15, 15)
    first, second = det(mask, 0.5, window=0), det(mask, 0.5, window=1)
    for order in ([first, second], [second, first]):
        kept = dedup_detections(order)
        assert [d.window for d in kept] == [0]


def test_dedup_of_nothing_and_a_bad_threshold() -> None:
    assert dedup_detections([]) == []
    with pytest.raises(ValueError, match="iou"):
        dedup_detections([], iou=-0.1)


# ---------------------------------------------------------------- claiming

CLAIM_SHAPE = (40, 40)


def _claim_scene():
    """Two U-Net blobs, a detection over both, one over the right blob, one tie."""
    unet = rect(CLAIM_SHAPE, 5, 5, 15, 25) | rect(CLAIM_SHAPE, 5, 30, 15, 40)
    # Covers the left blob and the first five columns of the right one: 200 + 50.
    wide = det(rect(CLAIM_SHAPE, 5, 5, 15, 35), 0.9, 0)
    # Covers columns 28..39 of the right blob; loses the columns 30..34 to ``wide``.
    right = det(rect(CLAIM_SHAPE, 5, 28, 15, 40), 0.5, 0)
    # Same score as ``right``, later window, all inside what ``right`` takes: gets 0.
    tied = det(rect(CLAIM_SHAPE, 5, 36, 15, 40), 0.5, 1)
    return unet, [wide, right, tied]


def test_claims_are_disjoint_ordered_by_score_and_ids_have_no_gaps() -> None:
    unet, detections = _claim_scene()
    labels, scores = claim_pixels(detections, unet, dilation=0, min_area=10)
    assert labels.dtype == np.int32
    # ``tied`` claims nothing (``right`` took it first), so it is dropped and the
    # ids stay 1, 2.
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
    assert scores.tolist() == [0.9, 0.5]
    # The wide detection keeps both blobs' pixels in one id: two pieces, one id.
    first = labels == 1
    assert first.sum() == 250
    assert np.array_equal(first, unet & rect(CLAIM_SHAPE, 5, 5, 15, 35))
    assert np.array_equal(labels == 2, rect(CLAIM_SHAPE, 5, 35, 15, 40))
    # Nothing outside the U-Net mask is labeled, and nothing is labeled twice.
    assert not np.any((labels > 0) & ~unet)
    assert np.array_equal(labels > 0, unet)


def test_claims_do_not_depend_on_the_input_order() -> None:
    unet, detections = _claim_scene()
    reference = claim_pixels(detections, unet, dilation=2, min_area=10)
    for order in itertools.permutations(detections):
        labels, scores = claim_pixels(list(order), unet, dilation=2, min_area=10)
        assert np.array_equal(labels, reference[0])
        assert np.array_equal(scores, reference[1])


def test_the_dilation_reaches_unet_pixels_beyond_the_detector_mask() -> None:
    unet = rect(CLAIM_SHAPE, 10, 10, 20, 20)
    detection = det(rect(CLAIM_SHAPE, 12, 12, 18, 18), 0.8)
    inside, _ = claim_pixels([detection], unet, dilation=0, min_area=1)
    assert (inside > 0).sum() == 36
    # A 2 px ellipse reaches the rows 10..19 only partly at the corners, but the
    # straight edges are fully reached: 6 + 2 * 2 = 10 pixels per side minus corners.
    reached, _ = claim_pixels([detection], unet, dilation=2, min_area=1)
    assert 36 < (reached > 0).sum() <= 100
    assert reached[15, 10] == 1 and reached[10, 15] == 1


def test_min_area_applies_to_the_whole_claim_and_compacts_the_ids() -> None:
    unet = rect(CLAIM_SHAPE, 5, 5, 15, 25) | rect(CLAIM_SHAPE, 30, 30, 33, 33)
    big = det(rect(CLAIM_SHAPE, 5, 5, 15, 25), 0.9)
    tiny = det(rect(CLAIM_SHAPE, 30, 30, 33, 33), 0.8)  # 9 pixels
    last = det(rect(CLAIM_SHAPE, 20, 20, 25, 25), 0.3)  # claims nothing at all
    labels, scores = claim_pixels([big, tiny, last], unet, dilation=0, min_area=10)
    assert sorted(np.unique(labels).tolist()) == [0, 1]
    assert scores.tolist() == [0.9]
    assert not labels[30:33, 30:33].any()


def test_unclaimed_unet_pixels_are_emitted_only_when_asked() -> None:
    unet = (
        rect(CLAIM_SHAPE, 5, 5, 15, 25)
        | rect(CLAIM_SHAPE, 25, 5, 35, 25)  # 200 px, nobody claims it
        | rect(CLAIM_SHAPE, 20, 30, 22, 32)  # 4 px, below min_area
    )
    claimed = det(rect(CLAIM_SHAPE, 5, 5, 15, 25), 0.9)
    labels, scores = claim_pixels([claimed], unet, dilation=0, min_area=10)
    assert labels.max() == 1 and scores.tolist() == [0.9]

    labels, scores = claim_pixels(
        [claimed], unet, dilation=0, min_area=10, emit_unclaimed=True
    )
    assert labels.max() == 2
    assert scores[0] == 0.9 and np.isnan(scores[1])
    assert np.array_equal(labels == 2, rect(CLAIM_SHAPE, 25, 5, 35, 25))
    assert not labels[20:22, 30:32].any()


def test_a_detection_with_no_unet_pixels_and_an_empty_frame() -> None:
    labels, scores = claim_pixels(
        [det(rect(CLAIM_SHAPE, 5, 5, 15, 15), 0.9)],
        np.zeros(CLAIM_SHAPE, bool),
        dilation=0,
        min_area=1,
    )
    assert labels.shape == CLAIM_SHAPE and not labels.any() and scores.size == 0
    labels, scores = claim_pixels([], np.zeros(CLAIM_SHAPE, bool))
    assert not labels.any() and scores.size == 0


def test_claim_checks_its_inputs() -> None:
    unet, detections = _claim_scene()
    with pytest.raises(ValueError, match="differ"):
        claim_pixels(detections, np.zeros((10, 10), bool))
    with pytest.raises(ValueError, match="dilation"):
        claim_pixels(detections, unet, dilation=-1)
    with pytest.raises(ValueError, match="2D"):
        claim_pixels([], np.zeros(5, bool))


# ---------------------------------------------------------------- score cut


def test_keep_scored_cuts_and_compacts_without_running_the_claim_again() -> None:
    labels = np.zeros((4, 12), dtype=np.int32)
    labels[0, 0:3], labels[1, 0:3], labels[2, 0:3], labels[3, 0:3] = 1, 2, 3, 4
    scores = np.array([0.9, 0.4, np.nan, 0.6])
    kept, kept_scores = keep_scored(labels, scores, 0.5)
    assert kept_scores.tolist() == [0.9, 0.6]
    assert sorted(np.unique(kept).tolist()) == [0, 1, 2]
    assert (kept[0] == 1).sum() == 3 and (kept[3] == 2).sum() == 3
    assert not kept[1].any() and not kept[2].any()
    kept, kept_scores = keep_scored(labels, scores, 0.5, keep_unscored=True)
    assert np.isnan(kept_scores).sum() == 1 and kept.max() == 3
    # A cut of 0 keeps every scored id and leaves the pixels as they were.
    same, _ = keep_scored(labels, np.array([0.9, 0.4, 0.5, 0.6]), 0.0)
    assert np.array_equal(same, labels)
    with pytest.raises(ValueError, match="scores for"):
        keep_scored(labels, scores[:2], 0.5)


# ---------------------------------------------------------------- calibration


def test_isotonic_calibrator_pools_violations_and_clamps() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    targets = np.array([0.0, 0.0, 0.2, 0.1, 0.6, 0.7])
    calibrate = isotonic_calibrator(scores, targets)
    # 0.2 then 0.1 break the order, so they pool to 0.15.
    assert calibrate(scores).tolist() == pytest.approx([0, 0, 0.15, 0.15, 0.6, 0.7])
    assert calibrate(np.array([0.0, 0.95])).tolist() == pytest.approx([0.0, 0.7])
    grid = calibrate(np.linspace(0, 1, 50))
    assert np.all(np.diff(grid) >= 0)
    assert np.isnan(calibrate(np.array([np.nan]))[0])


def test_isotonic_calibrator_checks_its_inputs() -> None:
    with pytest.raises(ValueError, match="no pairs"):
        isotonic_calibrator(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="finite"):
        isotonic_calibrator(np.array([np.nan, 0.5]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match=">= 0"):
        isotonic_calibrator(np.array([0.1, 0.2]), np.array([0.1, -0.5]))
    # A target above 1 is valid: see the test on summed IoU over overlapping masks.
    calibrator = isotonic_calibrator(np.array([0.1, 0.2]), np.array([0.1, 1.5]))
    assert calibrator(np.array([0.2])).tolist() == [1.5]


def _dated(count: int = 40, per_day: int = 4):
    rng = np.random.default_rng(0)
    scores = np.linspace(0.01, 0.99, count)
    targets = np.clip(scores + rng.normal(0, 0.2, count), 0, 1)
    days = [dt.date(2020, 1, 1) + dt.timedelta(days=i // per_day) for i in range(count)]
    return scores, targets, days


def test_crossfit_never_fits_and_predicts_the_same_date(monkeypatch) -> None:
    scores, targets, days = _dated()
    day_of = dict(zip(scores.tolist(), days, strict=True))
    trained_on: list[set[dt.date]] = []
    real = detector_fusion.isotonic_calibrator

    def spy(x, y):
        trained_on.append({day_of[float(v)] for v in x})
        return real(x, y)

    monkeypatch.setattr(detector_fusion, "isotonic_calibrator", spy)
    out = crossfit_calibrated_scores(scores, targets, days, n_splits=5, seed=3)
    assert out.shape == scores.shape and np.all(np.isfinite(out))
    all_days = set(days)
    held_out = [all_days - fitted for fitted in trained_on]
    assert len(trained_on) == 5
    assert all(held_out)  # every fit leaves some dates out
    # The held-out dates of the five fits are disjoint and cover every date once.
    assert sum(len(h) for h in held_out) == len(all_days)
    assert set().union(*held_out) == all_days


def test_a_calibrated_score_does_not_depend_on_its_own_dates_targets() -> None:
    scores, targets, days = _dated()
    base = crossfit_calibrated_scores(scores, targets, days, n_splits=5, seed=1)
    target_day = days[10]
    same_day = np.array([d == target_day for d in days])
    changed = targets.copy()
    changed[same_day] = 1.0 - changed[same_day]
    again = crossfit_calibrated_scores(scores, changed, days, n_splits=5, seed=1)
    assert np.array_equal(base[same_day], again[same_day])
    assert not np.array_equal(base[~same_day], again[~same_day])


def test_crossfit_checks_its_inputs() -> None:
    scores, targets, days = _dated()
    with pytest.raises(ValueError, match="groups"):
        crossfit_calibrated_scores(scores, targets, days[:-1])
    with pytest.raises(ValueError, match="cannot fill"):
        crossfit_calibrated_scores(scores, targets, days, n_splits=11)


# ---------------------------------------------------------------- emission rule


def _hits() -> list[ViewHits]:
    """Four candidates. 0 and 1 match a ground truth (IoU 0.8, 0.6) of three."""
    return [
        ViewHits(
            candidate=np.array([0, 1]),
            ground_truth=np.array([0, 1]),
            iou=np.array([0.8, 0.6]),
            ground_truths=3,
        )
    ]


def test_emission_iterates_lambda_to_its_fixed_point() -> None:
    scores = [np.array([0.8, 0.6, 0.21, 0.1])]
    # lambda 0 emits all four: tp 2, fp 2, fn 1, iou 1.4 -> 1.4 / 3.5 = 0.4.
    # lambda 0.4 (cut 0.2) emits three: 1.4 / 3.0 = 0.4667.
    # lambda 0.4667 (cut 0.2333) drops the 0.21: tp 2, fp 0, fn 1 -> 1.4 / 2.5 = 0.56.
    # lambda 0.56 (cut 0.28) emits the same two, so it has converged.
    result = emit_by_pooled_pq([_hits()], scores)
    assert result.lam == pytest.approx(0.56)
    assert result.emit[0].tolist() == [True, True, False, False]
    assert result.counts[0].tp == 2 and result.counts[0].fp == 0
    assert result.counts[0].fn == 1 and result.counts[0].iou_sum == pytest.approx(1.4)
    # With a single round it stops at 0.4 and still emits the 0.21.
    once = emit_by_pooled_pq([_hits()], scores, rounds=1)
    assert once.lam == pytest.approx(0.4)
    assert once.emit[0].tolist() == [True, True, True, False]
    assert once.counts[0].fp == 1


def test_emission_checks_its_inputs() -> None:
    with pytest.raises(ValueError, match="files"):
        emit_by_pooled_pq([_hits()], [])


# ---------------------------------------------------------------- per-file counts


def _scene():
    """The scene the hand counts below are made for. 64 x 64, two windows.

    U-Net: A (rows 10..19, cols 5..44, 400 px), B (rows 30..39, cols 5..24, 200 px)
    and a false blob C (rows 50..55, cols 40..59, 120 px). Window 0 sees A up to
    column 39 and window 1 sees it from column 24 on, so A comes as two clipped
    detections that must be merged. B is seen by window 0, C by window 1.
    """
    box_a, box_b, box_c = (10, 5, 20, 45), (30, 5, 40, 25), (50, 40, 56, 60)
    unet = frame(box_a, box_b, box_c)
    detections = [
        det(frame(box_a) & frame((0, 0, SIDE, 40)), 0.9, 0),
        det(frame(box_a) & frame((0, 24, SIDE, SIDE)), 0.85, 1),
        det(frame(box_b), 0.7, 0),
        det(frame(box_c), 0.4, 1),
    ]
    view_one = [
        frame(box_a),  # matches A with IoU 1
        frame((30, 5, 40, 20)),  # 150 px inside B (200): IoU 0.75
        frame((55, 5, 61, 15)),  # nothing predicts it: a false negative
    ]
    view_two = [frame((10, 5, 20, 35))]  # 300 px inside A (400): IoU 0.75
    return FusionFile("scene.jpeg", detections, unet, [view_one, view_two])


CONFIG = FusionConfig(dilation=2, min_area=50)


def test_per_file_counts_match_the_hand_count() -> None:
    # View one: preds A, B, C. A-A_gt 1.0 and B-B_gt 0.75 match, C does not, and the
    # third ground truth is missed: tp 2, fp 1, fn 1, iou 1.75.
    # View two: only A matches (300 / 400 = 0.75): tp 1, fp 2, fn 0, iou 0.75.
    (counts,) = per_file_detector_counts([_scene()], WINDOWS, CONFIG)
    assert (counts.tp, counts.fp, counts.fn) == (3, 3, 1)
    assert counts.iou_sum == pytest.approx(2.5)
    assert counts.pq == pytest.approx(2.5 / 5.0)


def test_the_per_file_scene_is_not_vacuous() -> None:
    # A guard against the failure PROJECT_REPORT.md records for #35: the check
    # must see matches, and the pipeline must really merge A and keep C.
    scene = _scene()
    labels, scores = detector_fusion.fuse(
        scene.detections, scene.unet_mask, WINDOWS, CONFIG
    )
    assert labels.max() == 3 and scores.tolist() == [0.9, 0.7, 0.4]
    assert (labels == 1).sum() == 400 and (labels == 2).sum() == 200
    assert (labels == 3).sum() == 120
    totals = [fast_view_counts(labels, masks) for masks in scene.views]
    assert sum(c.tp for c in totals) > 0 and sum(c.iou_sum for c in totals) > 0


def test_a_score_cut_changes_the_counts_as_expected() -> None:
    # The cut 0.5 removes C (0.4): view one tp 2, fp 0, fn 1; view two tp 1, fp 1.
    cut = FusionConfig(dilation=2, min_area=50, min_score=0.5)
    (counts,) = per_file_detector_counts([_scene()], WINDOWS, cut)
    assert (counts.tp, counts.fp, counts.fn) == (3, 1, 1)
    assert counts.iou_sum == pytest.approx(2.5)


def test_replacement_scores_are_used_for_the_cut_but_not_for_the_claim() -> None:
    # Ids are A, B, C. Scores 0.1, 0.9, 0.9 with a cut at 0.5 drop A: B and C remain.
    # View one: B matches (0.75); C is fp; A_gt and the third are missed: tp 1, fp 1,
    # fn 2. View two: both preds are fp, its one ground truth is missed: tp 0, fp 2,
    # fn 1. Total tp 1, fp 3, fn 3, iou 0.75.
    cut = FusionConfig(dilation=2, min_area=50, min_score=0.5)
    (counts,) = per_file_detector_counts(
        [_scene()], WINDOWS, cut, scores=[np.array([0.1, 0.9, 0.9])]
    )
    assert (counts.tp, counts.fp, counts.fn) == (1, 3, 3)
    assert counts.iou_sum == pytest.approx(0.75)
    with pytest.raises(ValueError, match="scores for"):
        per_file_detector_counts([_scene()], WINDOWS, cut, scores=[np.array([0.1])])
    with pytest.raises(ValueError, match="no scores"):
        per_file_detector_counts([_scene()], WINDOWS, cut, scores=[])
    with pytest.raises(ValueError, match="score arrays"):
        per_file_detector_counts(
            [_scene()], WINDOWS, cut, scores=[np.array([1.0, 1.0, 1.0])] * 2
        )


def test_counts_are_one_entry_per_file_in_order() -> None:
    empty = FusionFile(
        "empty.jpeg", [], np.zeros((SIDE, SIDE), bool), [[frame((5, 5, 15, 15))]]
    )
    counts = per_file_detector_counts([_scene(), empty], WINDOWS, CONFIG)
    assert len(counts) == 2
    assert counts[1] == PqCounts(tp=0, fp=0, fn=1, iou_sum=0.0)
    # A generator works too: files need not be held in memory together.
    lazy = per_file_detector_counts(iter([_scene(), empty]), WINDOWS, CONFIG)
    assert lazy == counts


def test_fusion_targets_match_the_counts_of_the_metric() -> None:
    scene = _scene()
    labels, _ = detector_fusion.fuse(scene.detections, scene.unet_mask, WINDOWS, CONFIG)
    hits, target = fusion_targets(labels, scene.views)
    # A: (1.0 + 0.75) / 2, B: 0.75 / 2, C: no match.
    assert target.tolist() == pytest.approx([0.875, 0.375, 0.0])
    pooled = selection_counts(hits, np.ones(3, bool))
    expected = sum(
        (fast_view_counts(labels, masks) for masks in scene.views), PqCounts()
    )
    assert pooled.tp == expected.tp and pooled.fp == expected.fp
    assert pooled.fn == expected.fn
    assert pooled.iou_sum == pytest.approx(expected.iou_sum)


def test_the_output_feeds_the_stats_module() -> None:
    config_a = CONFIG
    config_b = FusionConfig(dilation=2, min_area=50, min_score=0.5)
    files = [_scene() for _ in range(4)]
    by_config = {
        config: per_file_detector_counts(files, WINDOWS, config)
        for config in (config_a, config_b)
    }
    # Per file: config A is 2.5 / 5.0 = 0.5, config B is 2.5 / 4.0 = 0.625.
    paired = paired_bootstrap(by_config[config_a], by_config[config_b], n=50, seed=0)
    assert paired.point == pytest.approx(0.125)
    assert paired.fraction_better == 1.0
    folds = date_folds(
        [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(4)], 2
    )
    result = crossfit_select(by_config, folds)
    assert result.pooled_pq == pytest.approx(0.625)
    assert set(result.selected) == {config_b}


# ------------------------------------------- review of #63: targets, crop, wiring


def test_a_target_above_one_is_a_valid_calibration_target() -> None:
    # One id that matches two overlapping masks of a single view: the roadmap target
    # sums IoU over every matched pair, so it is 1.0 + 0.975 = 1.975.
    shape = (SIDE, SIDE)
    first = rect(shape, 5, 5, 15, 45)
    second = rect(shape, 5, 5, 15, 44)  # the same mask minus one column: IoU 0.975
    labels = first.astype(np.int32)
    _, target = fusion_targets(labels, [[first, second]])
    assert target.tolist() == pytest.approx([1.975])

    scores, targets, days = _dated()
    targets = targets.copy()
    targets[0] = float(target[0])
    out = crossfit_calibrated_scores(scores, targets, days, n_splits=5, seed=1)
    assert out.shape == scores.shape and np.all(np.isfinite(out))
    # The unclipped value is what the fit sees: clipping it to 1 would change the fit.
    clipped = targets.copy()
    clipped[0] = 1.0
    other = crossfit_calibrated_scores(scores, clipped, days, n_splits=5, seed=1)
    assert not np.array_equal(out, other)
    # A negative target is still invalid.
    targets[1] = -0.1
    with pytest.raises(ValueError, match="targets"):
        isotonic_calibrator(scores, targets)
    with pytest.raises(ValueError, match="targets"):
        crossfit_calibrated_scores(scores, targets, days)


def test_a_crop_does_not_keep_the_frame_it_was_cut_from_alive() -> None:
    detection = det(rect((2048, 2048), 100, 100, 160, 900), 0.5)
    crop = detection.crop()
    retained = crop if crop.base is None else crop.base
    # The crop is 60 x 800 = 48,000 bytes. A view would retain the 4 MiB frame.
    assert crop.shape == (60, 800)
    assert retained.nbytes == crop.nbytes == 48_000


def test_claim_scores_stay_aligned_with_ids_when_a_middle_id_is_dropped() -> None:
    unet = frame((2, 2, 12, 22), (15, 2, 17, 4), (25, 2, 35, 22))
    detections = [
        det(frame((2, 2, 12, 22)), 0.9),
        det(frame((15, 2, 17, 4)), 0.8),  # 4 px: under min_area, dropped
        det(frame((25, 2, 35, 22)), 0.7),
    ]
    labels, scores = claim_pixels(detections, unet, dilation=0, min_area=10)
    assert labels.max() == 2 and scores.tolist() == [0.9, 0.7]
    assert (labels == 2).sum() == 200 and labels[30, 10] == 2


def _overlapping_pair():
    """Two detections of one window that overlap: plain IoU 30 / 370 = 0.081."""
    first, second = frame((5, 5, 15, 25)), frame((5, 18, 15, 40))
    return first | second, [det(first, 0.9, 0), det(second, 0.5, 0)]


def test_fuse_applies_dedup_with_the_configured_threshold() -> None:
    unet, detections = _overlapping_pair()
    strict = FusionConfig(dilation=0, min_area=10, dedup_iou=0.05)
    loose = FusionConfig(dilation=0, min_area=10, dedup_iou=0.5)
    labels, scores = detector_fusion.fuse(detections, unet, WINDOWS, strict)
    assert labels.max() == 1 and scores.tolist() == [0.9]
    labels, scores = detector_fusion.fuse(detections, unet, WINDOWS, loose)
    assert labels.max() == 2 and scores.tolist() == [0.9, 0.5]


def test_fuse_merges_across_windows_with_the_configured_threshold() -> None:
    # Shared-strip IoU 32 / 96 = 0.333, plain IoU 0.127. Dedup is off (0.9) so only
    # the merge decides.
    first, second = frame((10, 5, 14, 40)), frame((12, 24, 16, 60))
    unet = first | second
    detections = [det(first, 0.9, 0), det(second, 0.8, 1)]
    low = FusionConfig(dilation=0, min_area=1, dedup_iou=0.9, windows_iou=0.2)
    high = FusionConfig(dilation=0, min_area=1, dedup_iou=0.9, windows_iou=0.5)
    labels, scores = detector_fusion.fuse(detections, unet, WINDOWS, low)
    assert labels.max() == 1 and scores.tolist() == [0.9]
    assert np.array_equal(labels > 0, unet)
    labels, scores = detector_fusion.fuse(detections, unet, WINDOWS, high)
    assert labels.max() == 2 and scores.tolist() == [0.9, 0.8]


def _unclaimed_file() -> FusionFile:
    claimed, unclaimed = (5, 5, 15, 25), (30, 5, 40, 25)
    return FusionFile(
        "unclaimed.jpeg",
        [det(frame(claimed), 0.9)],
        frame(claimed, unclaimed),
        [[frame(claimed), frame(unclaimed)]],
    )


def test_unclaimed_unet_pixels_follow_emit_unclaimed_through_fuse_and_counts() -> None:
    item = _unclaimed_file()
    off = FusionConfig(dilation=0, min_area=10, min_score=0.5)
    on = FusionConfig(dilation=0, min_area=10, min_score=0.5, emit_unclaimed=True)
    labels, scores = detector_fusion.fuse(item.detections, item.unet_mask, WINDOWS, off)
    assert labels.max() == 1 and scores.tolist() == [0.9]
    # The unscored piece survives the cut because ``emit_unclaimed`` keeps it.
    labels, scores = detector_fusion.fuse(item.detections, item.unet_mask, WINDOWS, on)
    assert labels.max() == 2 and scores[0] == 0.9 and np.isnan(scores[1])
    (counts_off,) = per_file_detector_counts([item], WINDOWS, off)
    (counts_on,) = per_file_detector_counts([item], WINDOWS, on)
    assert (counts_off.tp, counts_off.fp, counts_off.fn) == (1, 0, 1)
    assert (counts_on.tp, counts_on.fp, counts_on.fn) == (2, 0, 0)
    assert counts_on.iou_sum == pytest.approx(2.0)


def test_crossfit_passes_its_seed_to_the_partition(monkeypatch) -> None:
    scores, targets, days = _dated()
    seen: list[int] = []
    real = detector_fusion.date_folds

    def spy(groups, n_splits, *, seed):
        seen.append(seed)
        return real(groups, n_splits, seed=seed)

    monkeypatch.setattr(detector_fusion, "date_folds", spy)
    first = crossfit_calibrated_scores(scores, targets, days, seed=7)
    second = crossfit_calibrated_scores(scores, targets, days, seed=8)
    assert seen == [7, 8]
    # The two seeds give different partitions of the dates, hence different scores.
    assert not np.array_equal(first, second)
