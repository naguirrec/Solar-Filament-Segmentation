"""Final-evaluation metrics against hand-computed values.

Frames are 20x20. A box is ``(top, left, bottom, right)``. Every expected number
below is worked out by hand in the comment next to it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament.final_eval import (
    EXAMPLE_KINDS,
    Example,
    candidates,
    select_examples,
)
from filament.final_figures import (
    ATTRIBUTION,
    distribution_figure,
    overlay_figure,
    relations_figure,
    save_figure,
)
from filament.final_metrics import (
    ViewResult,
    file_counts,
    matched_dice,
    pooled_counts,
    pooled_pq,
    positive_scores,
    relation_counts,
    summary_rows,
    view_overlap,
    view_result,
    write_summary,
)
from filament.metrics import PqCounts, fast_view_counts
from filament.stats import weighted_pq

SIDE = 20


def _mask(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = True
    return mask


def _labels(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    for number, (top, left, bottom, right) in enumerate(boxes, start=1):
        labels[top:bottom, left:right] = number
    return labels


def _rows(results: list[ViewResult]) -> dict[str, object]:
    return {str(r["metric"]): r["value"] for r in summary_rows(results)}


def test_a_perfect_match() -> None:
    box = (0, 0, 10, 10)
    result = view_result("a", 0, _labels(box), [_mask(box)])
    assert result.iou.tolist() == [[1.0]]
    assert result.dice.tolist() == [[1.0]]
    assert result.counts == PqCounts(tp=1, fp=0, fn=0, iou_sum=1.0)
    rows = _rows([result])
    assert rows["pq"] == 1.0 and rows["sq"] == 1.0 and rows["rq"] == 1.0
    assert rows["mean_dice"] == 1.0 and rows["mean_dice_tp"] == 1.0
    assert (
        rows["gt_1to1"] == 1 and rows["one_to_many"] == 0 and rows["many_to_one"] == 0
    )


def test_an_empty_prediction_misses_every_filament() -> None:
    gt = [_mask((0, 0, 10, 10)), _mask((12, 12, 18, 18))]
    result = view_result("a", 0, np.zeros((SIDE, SIDE), dtype=np.int32), gt)
    assert result.iou.shape == (2, 0)
    assert result.counts == PqCounts(tp=0, fp=0, fn=2, iou_sum=0.0)
    rows = _rows([result])
    assert rows["pq"] == 0.0  # 0 / (0 + 0 + 0.5 * 2)
    assert math.isnan(rows["sq"])  # no true positive
    assert rows["rq"] == 0.0
    assert math.isnan(rows["mean_dice"]) and math.isnan(rows["iou_q50"])
    assert rows["gt_miss_1to0"] == 2 and rows["pred_total"] == 0
    assert rows["gt_degree_0"] == 2


def test_an_empty_view_makes_every_prediction_a_false_positive() -> None:
    result = view_result("a", 0, _labels((0, 0, 5, 5), (10, 10, 15, 15)), [])
    assert result.iou.shape == (0, 2)
    assert result.counts == PqCounts(tp=0, fp=2, fn=0, iou_sum=0.0)
    rows = _rows([result])
    assert rows["pq"] == 0.0 and rows["pred_miss_0to1"] == 2 and rows["gt_total"] == 0


def test_nothing_on_either_side_has_no_score() -> None:
    result = view_result("a", 0, np.zeros((SIDE, SIDE), dtype=np.int32), [])
    assert result.iou.shape == (0, 0)
    assert math.isnan(_rows([result])["pq"])
    assert math.isnan(pooled_pq([]))


def test_a_single_fragmentation() -> None:
    # GT is 10x10 = 100 px. Prediction 1 takes 10x6 = 60 px, prediction 2 takes 10x4 = 40 px.
    gt = [_mask((0, 0, 10, 10))]
    labels = _labels((0, 0, 10, 6), (0, 6, 10, 10))
    result = view_result("a", 0, labels, gt)
    # IoU: 60 / 100 = 0.6 and 40 / 100 = 0.4. Dice: 120 / 160 = 0.75 and 80 / 140 = 4/7.
    assert result.iou[0] == pytest.approx([0.6, 0.4])
    assert result.dice[0] == pytest.approx([0.75, 4 / 7])
    # Only the first pair has IoU > 0.5: TP 1; prediction 2 is a FP; GT is hit so FN 0.
    assert result.counts == PqCounts(
        tp=1,
        fp=1,
        fn=0,
        iou_sum=pytest.approx(0.6),  # type: ignore[arg-type]
    )
    rows = _rows([result])
    assert rows["pq"] == pytest.approx(0.6 / 1.5)  # 0.4
    assert rows["sq"] == pytest.approx(0.6) and rows["rq"] == pytest.approx(1 / 1.5)
    assert rows["pq"] == pytest.approx(rows["sq"] * rows["rq"])  # type: ignore[operator]
    assert rows["mean_dice"] == pytest.approx((0.75 + 4 / 7) / 2)
    assert rows["mean_dice_tp"] == pytest.approx(0.75)
    assert rows["mean_iou"] == pytest.approx(0.5)
    # The ground truth touches 2 predictions: one 1:2 relation, nothing else.
    assert (
        rows["one_to_many"] == 1 and rows["gt_1to1"] == 0 and rows["many_to_one"] == 0
    )
    assert rows["gt_degree_2"] == 1 and rows["pred_degree_1"] == 2
    assert rows["pred_1to1"] == 2 and rows["pred_miss_0to1"] == 0


def test_a_single_merge() -> None:
    # Two GT of 10x10 = 100 px side by side and one prediction over both (200 px).
    gt = [_mask((0, 0, 10, 10)), _mask((0, 10, 10, 20))]
    result = view_result("a", 0, _labels((0, 0, 10, 20)), gt)
    # IoU 100 / 200 = 0.5 for both, which is not > 0.5. Dice 200 / 300 = 2/3.
    assert result.iou[:, 0] == pytest.approx([0.5, 0.5])
    assert result.dice[:, 0] == pytest.approx([2 / 3, 2 / 3])
    assert result.counts == PqCounts(tp=0, fp=1, fn=2, iou_sum=0.0)
    rows = _rows([result])
    assert rows["pq"] == 0.0
    assert rows["many_to_one"] == 1 and rows["one_to_many"] == 0
    assert rows["pred_degree_2"] == 1 and rows["gt_degree_1"] == 2


def test_a_merge_that_still_matches_the_larger_filament() -> None:
    # GT 1 is 10x10 = 100 px, GT 2 is 10x6 = 60 px, the prediction covers both (160 px).
    gt = [_mask((0, 0, 10, 10)), _mask((0, 10, 10, 16))]
    result = view_result("a", 0, _labels((0, 0, 10, 16)), gt)
    assert result.iou[:, 0] == pytest.approx([100 / 160, 60 / 160])
    # GT 1 hits (0.625 > 0.5), GT 2 does not: TP 1, FN 1, FP 0.
    assert result.counts == PqCounts(tp=1, fp=0, fn=1, iou_sum=pytest.approx(0.625))  # type: ignore[arg-type]
    rows = _rows([result])
    assert rows["pq"] == pytest.approx(0.625 / 1.5)
    assert rows["many_to_one"] == 1


def test_a_touching_but_unmatched_pair_is_in_the_distributions() -> None:
    # Prediction shifted by 9 columns: intersection 10 x 1 = 10 px, union 190 px.
    gt = [_mask((0, 0, 10, 10))]
    result = view_result("a", 0, _labels((0, 9, 10, 19)), gt)
    assert result.iou[0, 0] == pytest.approx(10 / 190)
    assert result.dice[0, 0] == pytest.approx(20 / 200)
    assert result.counts == PqCounts(tp=0, fp=1, fn=1, iou_sum=0.0)
    assert positive_scores([result], "iou").tolist() == pytest.approx([10 / 190])
    assert matched_dice([result]).size == 0
    rows = _rows([result])
    assert rows["mean_dice"] == pytest.approx(0.1)
    assert math.isnan(rows["mean_dice_tp"])  # type: ignore[arg-type]
    # The official relation counts any overlap: this is a 1:1 pair.
    assert rows["gt_1to1"] == 1 and rows["pred_1to1"] == 1


def test_pairs_that_do_not_touch_are_left_out_of_the_distributions() -> None:
    gt = [_mask((0, 0, 4, 4))]
    result = view_result("a", 0, _labels((10, 10, 14, 14)), gt)
    assert result.iou.tolist() == [[0.0]] and result.dice.tolist() == [[0.0]]
    assert positive_scores([result], "dice").size == 0
    with pytest.raises(ValueError):
        positive_scores([result], "f1")


def test_quantiles_of_the_positive_scores() -> None:
    # Four views, each a GT of 100 px and a prediction with IoU 0.5, 0.6, 0.7, 0.8
    # (pred area = 100 * iou, fully inside the GT): Dice = 2 * iou / (1 + iou).
    results = []
    for number, iou in enumerate((0.5, 0.6, 0.7, 0.8)):
        rows_ = round(iou * 10)
        results.append(
            view_result(
                f"f{number}", 0, _labels((0, 0, rows_, 10)), [_mask((0, 0, 10, 10))]
            )
        )
    values = _rows(results)
    assert values["iou_q50"] == pytest.approx(0.65)
    assert values["iou_q25"] == pytest.approx(0.575)
    assert values["iou_q05"] == pytest.approx(0.515)
    assert values["iou_pairs"] == 4
    assert values["mean_iou"] == pytest.approx(0.65)
    assert values["dice_q50"] == pytest.approx((2 * 0.6 / 1.6 + 2 * 0.7 / 1.7) / 2)
    # Only IoU > 0.5 matches: 3 of the 4 pairs (0.5 itself is not a hit).
    assert values["tp"] == 3 and values["fp"] == 1 and values["fn"] == 1


def test_the_matrix_is_the_same_as_a_direct_computation_of_every_pair() -> None:
    rng = np.random.default_rng(0)
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    for ident in range(1, 6):
        top, left = rng.integers(0, SIDE - 6, size=2)
        labels[top : top + 6, left : left + 6] = ident
    gts = [rng.random((SIDE, SIDE)) > 0.7 for _ in range(4)]
    iou, dice = view_overlap(labels, gts)
    assert iou.shape == (4, int(labels.max()))
    for i, gt in enumerate(gts):
        for j in range(iou.shape[1]):
            pred = labels == j + 1
            inter = np.count_nonzero(pred & gt)
            union = np.count_nonzero(pred | gt)
            assert iou[i, j] == pytest.approx(inter / union if union else 0.0)
            total = np.count_nonzero(pred) + np.count_nonzero(gt)
            assert dice[i, j] == pytest.approx(2 * inter / total if total else 0.0)


def test_pooled_pq_equals_the_repo_pq_on_random_input() -> None:
    rng = np.random.default_rng(1)
    results: list[ViewResult] = []
    repo_counts: list[PqCounts] = []
    for frame in range(6):
        labels = np.zeros((SIDE, SIDE), dtype=np.int32)
        for ident in range(1, int(rng.integers(0, 5)) + 1):
            top, left = rng.integers(0, SIDE - 8, size=2)
            height, width = rng.integers(3, 9, size=2)
            labels[top : top + height, left : left + width] = ident
        frame_total = PqCounts()
        for view in range(int(rng.integers(1, 4))):
            gts = []
            for _ in range(int(rng.integers(0, 4))):
                top, left = rng.integers(0, SIDE - 8, size=2)
                height, width = rng.integers(3, 9, size=2)
                gts.append(_mask((top, left, top + height, left + width)))
            results.append(view_result(f"frame{frame}", view, labels, gts))
            frame_total = frame_total + fast_view_counts(labels, gts)
            assert results[-1].counts.tp == fast_view_counts(labels, gts).tp
            assert results[-1].counts.fp == fast_view_counts(labels, gts).fp
            assert results[-1].counts.fn == fast_view_counts(labels, gts).fn
        repo_counts.append(frame_total)
    ours = file_counts(results)
    assert len(ours) == len(repo_counts) == 6
    for mine, theirs in zip(ours, repo_counts, strict=True):
        assert (mine.tp, mine.fp, mine.fn) == (theirs.tp, theirs.fp, theirs.fn)
        assert mine.iou_sum == pytest.approx(theirs.iou_sum, abs=1e-12)
    assert pooled_pq(results) == pytest.approx(weighted_pq(repo_counts), abs=1e-12)
    total = pooled_counts(results)
    assert total.pq == pytest.approx(weighted_pq(repo_counts), abs=1e-12)
    assert _rows(results)["pq"] == pytest.approx(weighted_pq(repo_counts), abs=1e-12)


def test_pooling_sums_counts_and_does_not_average_views() -> None:
    perfect = view_result("a", 0, _labels((0, 0, 10, 10)), [_mask((0, 0, 10, 10))])
    empty = view_result(
        "b", 0, np.zeros((SIDE, SIDE), dtype=np.int32), [_mask((0, 0, 5, 5))]
    )
    # Pooled: iou_sum 1 / (1 + 0.5 * 1) = 2/3; the mean of the two PQs would be 0.5.
    assert pooled_pq([perfect, empty]) == pytest.approx(2 / 3)


def test_view_overlap_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        view_overlap(np.zeros((SIDE, SIDE), dtype=np.float32), [])
    with pytest.raises(ValueError):
        view_overlap(np.zeros(SIDE, dtype=np.int32), [])
    with pytest.raises(ValueError):
        view_overlap(-np.ones((SIDE, SIDE), dtype=np.int32), [])
    with pytest.raises(ValueError):
        view_overlap(np.zeros((SIDE, SIDE), dtype=np.int32), [np.zeros((3, 3), bool)])


def test_the_summary_csv_is_small_and_has_the_requested_columns(tmp_path: Path) -> None:
    result = view_result(
        "a", 0, _labels((0, 0, 10, 6), (0, 6, 10, 10)), [_mask((0, 0, 10, 10))]
    )
    path = tmp_path / "out" / "holdout_final_metrics.csv"
    write_summary(path, summary_rows([result]))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "metric,value,source"
    names = {line.split(",")[0] for line in lines[1:]}
    for name in (
        "pq", "sq", "rq", "tp", "fp", "fn", "mean_dice", "iou_q50", "dice_q95",
        "one_to_many", "many_to_one",
    ):  # fmt: skip
        assert name in names, name
    assert "pq,0.400000,official" in lines
    assert len(lines) < 60


# ------------------------------------------------------------- relations, examples


def _mixed_results() -> list[ViewResult]:
    """One view with a good match, a fragmentation, a merge, a miss and a false positive."""
    side = 60
    gts, labels = [], np.zeros((side, side), dtype=np.int32)

    def gt(top: int, left: int, bottom: int, right: int) -> None:
        mask = np.zeros((side, side), dtype=bool)
        mask[top:bottom, left:right] = True
        gts.append(mask)

    gt(0, 0, 10, 10)  # 0: good match
    labels[0:10, 0:10] = 1
    gt(0, 20, 10, 30)  # 1: fragmented into ids 2 and 3
    labels[0:10, 20:26] = 2
    labels[0:10, 26:30] = 3
    gt(20, 0, 30, 10)  # 2: merged with 3 by id 4
    gt(20, 10, 30, 20)  # 3
    labels[20:30, 0:20] = 4
    gt(40, 40, 50, 50)  # 4: miss
    labels[50:55, 0:5] = 5  # false positive
    return [view_result("frame", 0, labels, gts)]


def test_relation_counts_and_candidates_on_a_mixed_view() -> None:
    results = _mixed_results()
    relations = relation_counts(results)
    assert relations.gt_degrees == {0: 1, 1: 3, 2: 1}
    assert relations.pred_degrees == {0: 1, 1: 3, 2: 1}
    assert (relations.one_to_many, relations.many_to_one) == (1, 1)
    assert (relations.gt_miss, relations.pred_miss) == (1, 1)
    found = {kind: candidates(results[0], kind) for kind in EXAMPLE_KINDS}
    assert [e.gt_indices for e in found["good_match"]] == [(0,)]
    assert [e.pred_ids for e in found["fragmentation"]] == [(2, 3)]
    assert [(e.gt_indices, e.pred_ids) for e in found["over_merge"]] == [((2, 3), (4,))]
    assert [e.gt_indices for e in found["miss"]] == [(4,)]
    assert [e.pred_ids for e in found["false_positive"]] == [(5,)]
    with pytest.raises(ValueError):
        candidates(results[0], "other")


def test_select_examples_is_fixed_by_the_seed_and_covers_every_kind() -> None:
    results = _mixed_results()
    first = select_examples(results, per_kind=2, seed=7)
    again = select_examples(results, per_kind=2, seed=7)
    assert first == again
    assert {e.kind for e in first} == set(EXAMPLE_KINDS)
    # With one candidate per kind, nothing is repeated.
    assert len(first) == len(EXAMPLE_KINDS)
    with pytest.raises(ValueError):
        select_examples(results, per_kind=0)


def test_select_examples_prefers_different_frames_and_changes_with_the_seed() -> None:
    results = []
    for number in range(8):
        for result in _mixed_results():
            results.append(
                ViewResult(f"frame{number}", 0, result.iou, result.dice, result.counts)
            )
    picks = [
        e for e in select_examples(results, per_kind=3, seed=0) if e.kind == "miss"
    ]
    assert len({e.file_name for e in picks}) == 3
    other = [
        e for e in select_examples(results, per_kind=3, seed=1) if e.kind == "miss"
    ]
    assert [e.file_name for e in picks] != [e.file_name for e in other]


# ------------------------------------------------------------------------ figures


def test_figures_are_written_small_with_the_attribution(tmp_path: Path) -> None:
    results = _mixed_results()
    gray = np.full((60, 60), 120, dtype=np.uint8)
    scores = positive_scores(results, "iou")
    paths = [
        save_figure(distribution_figure(scores, "IoU", "IoU"), tmp_path / "iou.png"),
        save_figure(
            relations_figure(relation_counts(results), "Relations"),
            tmp_path / "rel.png",
        ),
        save_figure(
            distribution_figure(np.zeros(0), "empty", "x"), tmp_path / "empty.png"
        ),
    ]
    # An overlay needs the label image and masks, rebuilt the way the notebook does.
    side = 60
    labels = np.zeros((side, side), dtype=np.int32)
    labels[0:10, 20:26] = 1
    labels[0:10, 26:30] = 2
    gt_mask = np.zeros((side, side), dtype=bool)
    gt_mask[0:10, 20:30] = True
    example = Example(
        "fragmentation", "frame", 0, (0,), (1, 2), "1 filament in 2 predictions"
    )
    paths.append(
        save_figure(
            overlay_figure("fragmentation", [(example, gray, labels, [gt_mask])]),
            tmp_path / "overlay.png",
        )
    )
    for path in paths:
        assert path.is_file() and 1_000 < path.stat().st_size < 400_000
    assert ATTRIBUTION == "GONG data, NSO/AURA/NSF"
