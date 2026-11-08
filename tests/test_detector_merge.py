"""Merge-only detector fusion on synthetic frames. Expected numbers are worked out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from filament.detector_fusion import Detection, merge_components
from filament.detector_merge_sweep import (
    MergeSetting,
    merge_axis_settings,
    merge_outcomes,
    pooled_diagnostics,
)
from filament.detector_sweep import unique_settings
from filament.metrics import PqCounts

SIDE = 64


def boxes(*specs: tuple[int, int, int, int]) -> np.ndarray:
    """Labels: box ``i`` (top, left, bottom, right) gets id ``i + 1``."""
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    for number, (top, left, bottom, right) in enumerate(specs, start=1):
        labels[top:bottom, left:right] = number
    return labels


def detection(
    top: int, left: int, bottom: int, right: int, score: float = 0.9
) -> Detection:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    mask[top:bottom, left:right] = True
    return Detection.from_mask(mask, score, 0)


A = (10, 5, 20, 15)  # 100 px
B = (10, 25, 20, 35)  # 100 px
C = (40, 25, 50, 35)  # 100 px, far from A


def test_a_detection_covering_two_components_joins_them_and_changes_no_pixel() -> None:
    labels = boxes(A, B, C)
    out = merge_components(labels, [detection(8, 3, 22, 37)], dilation=0)
    assert out.dtype == np.int32
    assert np.array_equal(out > 0, labels > 0)
    assert set(np.unique(out)) == {0, 1, 2}
    assert out[15, 10] == out[15, 30] == 1
    assert out[45, 30] == 2


def test_no_detections_returns_the_same_partition() -> None:
    labels = boxes(A, B, C)
    assert np.array_equal(merge_components(labels, []), labels)
    empty = np.zeros((SIDE, SIDE), dtype=np.int32)
    assert np.array_equal(merge_components(empty, [detection(0, 0, 5, 5)]), empty)


def test_a_detection_below_the_cut_does_nothing() -> None:
    labels = boxes(A, B)
    low = detection(8, 3, 22, 37, score=0.4)
    assert np.array_equal(merge_components(labels, [low], cut=0.5, dilation=0), labels)
    # The cut is inclusive.
    out = merge_components(labels, [detection(8, 3, 22, 37, 0.5)], cut=0.5, dilation=0)
    assert out.max() == 1


def test_a_component_covered_below_coverage_is_not_joined() -> None:
    labels = boxes(A, B)
    # Covers all of A and columns 25..28 of B: 40% of B.
    partial = detection(8, 3, 22, 29)
    apart = merge_components(labels, [partial], coverage=0.5, dilation=0)
    assert np.array_equal(apart, labels)
    joined = merge_components(labels, [partial], coverage=0.4, dilation=0)
    assert joined.max() == 1


def test_coverage_is_the_share_of_the_components_own_area() -> None:
    # B is big, so a detection inside it covers a small share of B, however large it is.
    labels = boxes(A, (5, 20, 60, 60))
    inside = detection(8, 3, 22, 40)
    assert merge_components(labels, [inside], dilation=0).max() == 2


def test_the_dilation_lets_a_detection_reach_a_neighbour() -> None:
    labels = boxes(A, B)
    short = detection(8, 3, 22, 27)  # B's columns 25 and 26 only: 20%
    assert merge_components(labels, [short], dilation=0).max() == 2
    assert merge_components(labels, [short], dilation=3).max() == 1  # now 50%


def test_joins_are_transitive_across_detections() -> None:
    labels = boxes(A, B, C, (55, 5, 60, 15))
    # One detection joins A and B, another B and C; the fourth component stays alone.
    first = detection(8, 3, 22, 37)
    second = detection(8, 23, 52, 37, score=0.8)
    out = merge_components(labels, [first, second], dilation=0)
    assert set(np.unique(out)) == {0, 1, 2}
    assert out[15, 10] == out[15, 30] == out[45, 30] == 1
    assert out[57, 10] == 2


def test_it_never_splits_drops_or_moves_a_pixel() -> None:
    rng = np.random.default_rng(0)
    for _ in range(30):
        labels = np.zeros((SIDE, SIDE), dtype=np.int32)
        for number in range(1, int(rng.integers(2, 7))):
            top, left = rng.integers(0, SIDE - 12, size=2)
            height, width = rng.integers(3, 12, size=2)
            region = labels[top : top + height, left : left + width]
            region[region == 0] = number
        present = np.unique(labels[labels > 0])
        labels = np.searchsorted(present, labels) + (labels > 0)  # compact ids
        labels = labels.astype(np.int32)
        detections = []
        for _ in range(int(rng.integers(0, 5))):
            top, left = rng.integers(0, SIDE - 20, size=2)
            height, width = rng.integers(2, 20, size=2)
            detections.append(
                detection(top, left, top + height, left + width, float(rng.random()))
            )
        out = merge_components(
            labels,
            detections,
            cut=float(rng.random()),
            coverage=float(rng.uniform(0.1, 1.0)),
            dilation=int(rng.integers(0, 6)),
        )
        assert np.array_equal(out > 0, labels > 0)
        # Every input component sits whole inside one output component.
        for old in np.unique(labels[labels > 0]):
            assert np.unique(out[labels == old]).size == 1
        assert out.max() <= labels.max()
        assert set(np.unique(out[out > 0])) == set(range(1, int(out.max()) + 1))


def test_bad_arguments_are_rejected() -> None:
    labels = boxes(A)
    with pytest.raises(ValueError, match="cut"):
        merge_components(labels, [], cut=1.5)
    with pytest.raises(ValueError, match="coverage"):
        merge_components(labels, [], coverage=0.0)
    with pytest.raises(ValueError, match="dilation"):
        merge_components(labels, [], dilation=-1)
    with pytest.raises(ValueError, match="2D"):
        merge_components(labels[0], [])
    with pytest.raises(ValueError, match="integer"):
        merge_components(labels.astype(float), [])
    with pytest.raises(ValueError, match="differ"):
        merge_components(labels[:32], [detection(0, 0, 5, 5)])


# ------------------------------------------------------------------ the sweep module


def test_the_axes_are_seven_unique_settings_one_change_from_the_defaults() -> None:
    axes = merge_axis_settings()
    assert [len(v) for v in axes.values()] == [5, 2, 2]
    settings = unique_settings(axes)
    assert len(settings) == 7
    default = MergeSetting("merge_cut")
    assert (default.cut, default.coverage, default.dedup) == (0.5, 0.5, False)
    for setting in settings:
        changed = [
            setting.cut != default.cut,
            setting.coverage != default.coverage,
            setting.dedup != default.dedup,
        ]
        assert sum(changed) <= 1
    assert {s.cut for s in settings} == {0.2, 0.3, 0.5, 0.7, 0.9}
    assert {s.coverage for s in settings} == {0.5, 0.8}
    assert any(s.dedup for s in settings)


def test_outcomes_count_joins_and_use_deduplicated_detections_only_when_asked() -> None:
    labels = boxes(A, B, C)
    joining = detection(8, 3, 22, 37)
    merged = [joining]
    unique: list[Detection] = []  # as if deduplication had dropped it
    plain = MergeSetting("x")
    deduped = MergeSetting("y", dedup=True)
    high_cut = MergeSetting("z", cut=0.95)
    outcomes = merge_outcomes(
        labels, merged, unique, [plain, deduped, high_cut], dilation=0
    )
    assert outcomes[plain].joins == 1
    assert outcomes[plain].labels.max() == 2
    assert outcomes[deduped].joins == 0
    assert np.array_equal(outcomes[deduped].labels, labels)
    assert outcomes[high_cut].joins == 0


def test_diagnostics_split_pq_into_sq_times_rq() -> None:
    counts = [
        PqCounts(tp=3, fp=1, fn=2, iou_sum=2.4),
        PqCounts(tp=1, fp=0, fn=0, iou_sum=0.8),
    ]
    diag = pooled_diagnostics(counts)
    assert (diag.tp, diag.fp, diag.fn) == (4, 1, 2)
    assert diag.sq == pytest.approx(3.2 / 4)
    assert diag.rq == pytest.approx(4 / (4 + 0.5 + 1))
    assert diag.pq == pytest.approx(diag.sq * diag.rq)
    assert "sq=0.8000" in diag.text()
    empty = pooled_diagnostics([PqCounts()])
    assert np.isnan(empty.sq) and np.isnan(empty.rq)
