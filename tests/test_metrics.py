"""Panoptic quality on tiny synthetic instance masks."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament.metrics import (
    PqCounts,
    fast_view_counts,
    panoptic_quality,
    reference_view_counts,
)


def _mask(shape: tuple[int, int], rows: slice, cols: slice) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rows, cols] = 1
    return mask


def test_perfect_match_scores_one() -> None:
    # Two disjoint filaments, each copied exactly, ground truth reversed.
    # sum(IoU) = 2, TP = 2, PQ = 1.
    shape = (2, 4)
    left = _mask(shape, slice(0, 2), slice(0, 2))
    right = _mask(shape, slice(0, 2), slice(2, 4))
    assert panoptic_quality([left, right], [right, left]) == 1.0


def test_total_miss_scores_zero() -> None:
    shape = (4, 4)
    predicted = _mask(shape, slice(0, 2), slice(0, 2))
    actual = _mask(shape, slice(2, 4), slice(2, 4))
    assert panoptic_quality([predicted], [actual]) == 0.0


def test_one_false_positive() -> None:
    # One exact match and one extra prediction.
    # PQ = 1 / (1 + 0.5 * 1 + 0.5 * 0) = 2/3.
    shape = (4, 4)
    matched = _mask(shape, slice(0, 2), slice(0, 2))
    extra = _mask(shape, slice(2, 4), slice(2, 4))
    score = panoptic_quality([matched, extra], [matched])
    assert score == pytest.approx(2 / 3)


def test_one_false_negative() -> None:
    # One exact match and one missed ground-truth instance.
    # PQ = 1 / (1 + 0.5 * 0 + 0.5 * 1) = 2/3.
    shape = (4, 4)
    matched = _mask(shape, slice(0, 2), slice(0, 2))
    missed = _mask(shape, slice(2, 4), slice(2, 4))
    score = panoptic_quality([matched], [matched, missed])
    assert score == pytest.approx(2 / 3)


def test_overlap_above_half_keeps_iou() -> None:
    # 3 of 4 pixels agree. IoU = 3/4. One match and no leftovers, so PQ = 3/4.
    ground_truth = np.ones((2, 2), dtype=np.uint8)
    predicted = ground_truth.copy()
    predicted[1, 1] = 0
    assert panoptic_quality([predicted], [ground_truth]) == pytest.approx(0.75)


def test_overlap_below_half_scores_zero() -> None:
    # 1 shared pixel and union 4. IoU = 1/4, so the pair does not match.
    ground_truth = np.ones((1, 4), dtype=np.uint8)
    predicted = np.zeros((1, 4), dtype=np.uint8)
    predicted[0, 0] = 1
    assert panoptic_quality([predicted], [ground_truth]) == 0.0


def test_overlap_of_exactly_half_does_not_match() -> None:
    # Intersection 2, union 4, IoU = 0.5. The match rule is IoU > 0.5.
    ground_truth = np.array([[1, 1, 1, 0]], dtype=np.uint8)
    predicted = np.array([[0, 1, 1, 1]], dtype=np.uint8)
    assert panoptic_quality([predicted], [ground_truth]) == 0.0


def test_kirillov_empty_prediction_and_empty_ground_truth_score_one() -> None:
    # The greedy helper keeps this definition. The pooled score is NaN.
    assert panoptic_quality([], []) == 1.0


def test_greedy_match_is_one_to_one() -> None:
    # One prediction covers 6/8 of each overlapping ground-truth mask.
    # IoU is 0.75 with both, and only one pair may match.
    # PQ = 0.75 / (1 + 0.5 * 1) = 0.5.
    predicted = np.ones((2, 4), dtype=np.uint8)
    left = np.zeros((2, 4), dtype=np.uint8)
    right = np.zeros((2, 4), dtype=np.uint8)
    left[:, :3] = 1
    right[:, 1:] = 1
    assert panoptic_quality([predicted], [left, right]) == pytest.approx(0.5)


def test_empty_instance_mask_is_rejected() -> None:
    blank = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        panoptic_quality([blank], [])


def test_pooled_empty_view_is_nan_not_one() -> None:
    labels = np.zeros((2, 2), dtype=np.int32)
    fast = fast_view_counts(labels, [])
    reference = reference_view_counts([], [])
    assert (fast.tp, fast.fp, fast.fn, fast.iou_sum) == (0, 0, 0, 0.0)
    assert (reference.tp, reference.fp, reference.fn, reference.iou_sum) == (
        0,
        0,
        0,
        0.0,
    )
    assert math.isnan(fast.pq)
    assert math.isnan(reference.pq)
    assert math.isnan((fast + reference).pq)


def test_fast_and_reference_agree_on_synthetic_views() -> None:
    """Two views, an overlap, a miss, and an empty view.

    [VERIFICAR] against the live organizer notebook. It was not pulled.
    The overlap is one prediction matching two ground-truth masks. Every
    pair with IoU > 0.5 counts, so this is two true positives, not the
    greedy one-to-one match.
    """
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[:2, :2] = 1
    prediction = labels == 1
    exact = prediction.copy()
    overlap = np.zeros((4, 4), dtype=bool)
    overlap[:2, :3] = True
    missed = np.zeros((4, 4), dtype=bool)
    missed[2:, 2:] = True
    views = (
        [exact, overlap],
        [missed],
        [],
    )

    fast = PqCounts()
    reference = PqCounts()
    for view in views:
        fast += fast_view_counts(labels, view)
        reference += reference_view_counts([prediction], view)

    assert (fast.tp, fast.fp, fast.fn) == (reference.tp, reference.fp, reference.fn)
    assert fast.iou_sum == reference.iou_sum
    assert (reference.tp, reference.fp, reference.fn) == (2, 2, 1)
    assert reference.iou_sum == pytest.approx(1.0 + 4.0 / 6.0)

    overlap_view = reference_view_counts([prediction], [exact, overlap])
    assert (overlap_view.tp, overlap_view.fp, overlap_view.fn) == (2, 0, 0)
    greedy = panoptic_quality([prediction], [exact, overlap])
    assert greedy == pytest.approx(1.0 / 1.5)
    assert overlap_view.pq == pytest.approx((1.0 + 4.0 / 6.0) / 2.0)
    assert overlap_view.pq != pytest.approx(greedy)


def test_overlapping_predictions_inflate_true_positives() -> None:
    # Two copies of one mask against one ground-truth mask. Greedy
    # matching would keep a single TP and call the other an FP.
    full = np.ones((2, 2), dtype=np.uint8)
    counts = reference_view_counts([full, full], [full])
    assert (counts.tp, counts.fp, counts.fn) == (2, 0, 0)
    assert counts.iou_sum == 2.0
    assert counts.pq == 1.0
    assert panoptic_quality([full, full], [full]) == pytest.approx(2 / 3)


def test_iou_of_exactly_half_is_not_a_pair() -> None:
    predicted = np.array([[1, 1, 1, 0]], dtype=np.uint8)
    ground_truth = np.array([[0, 1, 1, 1]], dtype=np.uint8)
    labels = predicted.astype(np.int32)
    reference = reference_view_counts([predicted], [ground_truth])
    fast = fast_view_counts(labels, [ground_truth])
    assert (reference.tp, reference.fp, reference.fn) == (0, 1, 1)
    assert (fast.tp, fast.fp, fast.fn) == (reference.tp, reference.fp, reference.fn)
    assert fast.iou_sum == reference.iou_sum


def test_pooled_pq_is_not_the_mean_of_view_scores() -> None:
    perfect = PqCounts(tp=1, fp=0, fn=0, iou_sum=1.0)
    partial = PqCounts(tp=1, fp=1, fn=0, iou_sum=0.6)
    pooled = perfect + partial
    assert perfect.pq == 1.0
    assert partial.pq == pytest.approx(0.4)
    assert pooled.pq == pytest.approx(1.6 / 2.5)
    assert pooled.pq != pytest.approx((perfect.pq + partial.pq) / 2)


def test_shape_mismatch_is_rejected() -> None:
    small = np.ones((2, 2), dtype=np.uint8)
    wide = np.ones((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        panoptic_quality([small], [wide])
    with pytest.raises(ValueError, match="shape"):
        panoptic_quality([small, wide], [small])
