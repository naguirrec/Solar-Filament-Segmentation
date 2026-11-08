"""Final-evaluation metrics: IoU and Dice distributions and match relations.

The organizers score the final submission on PQ, mean Dice, the distributions
of Dice and IoU, and the distribution of one-to-many and many-to-one
relations. Their self-evaluation notebook is the definition (Kaggle:
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook; the copy
read for this module is the one downloaded on 2026-11-07). Where this module
takes a rule from it, the comment says which function of that notebook
(called "the official notebook" below).

What the official notebook defines, and what this module does the same way:

* ``get_overlap_matrices``: for each annotator view, ``iou[i, j]`` and
  ``dice[i, j]`` of ground-truth mask ``i`` and predicted mask ``j``. IoU is
  ``inter / union`` and Dice is ``2 * inter / (area_gt + area_pred)``; a pair
  with an empty union is 0. Every pair is computed, not only the matched
  ones.
* ``get_pq_score``: a pair is a true positive when ``iou > 0.5``. Every such
  pair counts (no one-to-one assignment). A prediction with no hit is a false
  positive, a ground-truth mask with no hit is a false negative, and
  ``PQ = sum(IoU of hits) / (TP + FP / 2 + FN / 2)``, pooled over all views.
  This is the rule of :mod:`filament.metrics`, and :func:`counts_from_overlap`
  reproduces it from the matrices.
* ``plot_distribution``: the IoU and Dice distributions are made of the
  pairs with a score above 0 (pairs that do not touch are dropped), and the
  mean line of the plot is the mean of those pairs. That is the only "mean
  Dice" the official notebook computes, so it is ``mean_dice`` here.
* ``plot_m2n_counts``: the degree of a ground-truth mask is the number of
  predictions with ``iou > 0``; the degree of a prediction is the number of
  ground-truth masks with ``iou > 0``. Degree 0 is a miss (1:0, 0:1).
  Degree 1 is a plain 1:1 match. A ground-truth mask with degree n >= 2 is a
  one-to-many relation (the filament is fragmented into n predictions), a
  prediction with degree n >= 2 is a many-to-one relation (the prediction
  merges n filaments). Any overlap counts, however small.

What the official notebook does not define, and the definition used here:

* SQ and RQ. The Kirillov split of PQ is used: ``SQ = sum(IoU of hits) / TP``
  and ``RQ = TP / (TP + FP / 2 + FN / 2)``, so ``PQ = SQ * RQ`` with the
  pooled counts.
* A mean Dice over matched pairs only. ``mean_dice_tp`` is the mean Dice of
  the pairs with ``iou > 0.5``. It is reported next to ``mean_dice`` because
  the competition text says "mean Dice" and the notebook's mean includes
  every touching pair. It is not an official number.
* Quantiles of the two distributions (5, 25, 50, 75, 95 %), over the same
  pairs as the official histograms.
* Views with no ground-truth mask. The official notebook only visits views
  that have rows in the ground-truth table, so such a view cannot occur
  there. Here every view of :func:`filament.annotations.annotator_views`
  counts, as in :func:`filament.metrics.fast_view_counts`: an empty view
  makes every prediction a false positive. A frame with no view at all
  counts nothing, as in the official notebook.
* A label id in ``1..max`` with no pixel is an unmatched prediction, as in
  :func:`filament.metrics.fast_view_counts`.

Pooling is the repo's: counts are summed over the views of a frame, and the
pooled PQ is :func:`filament.stats.weighted_pq` over the frames. There is no
per-image average.

This module needs numpy only (no torch).
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from filament.metrics import PqCounts
from filament.stats import weighted_pq

# get_pq_score: iou_threshold = 0.5, a hit is iou strictly above it.
OFFICIAL_IOU_THRESHOLD = 0.5
# Quantiles reported for the IoU and Dice distributions (not in the official notebook).
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

SOURCE_OFFICIAL = "official"
SOURCE_DERIVED = "derived"
SOURCE_OURS = "ours"


@dataclass(frozen=True, slots=True, eq=False)
class ViewResult:
    """Overlap of one prediction with one annotator view of one frame.

    ``iou`` and ``dice`` have shape ``(n_gt, n_pred)``: rows are the
    ground-truth masks of the view in the order given, columns are the label
    ids ``1..max`` of the prediction (column ``j`` is id ``j + 1``).
    """

    file_name: str
    view_index: int
    iou: np.ndarray
    dice: np.ndarray
    counts: PqCounts

    @property
    def n_gt(self) -> int:
        return int(self.iou.shape[0])

    @property
    def n_pred(self) -> int:
        return int(self.iou.shape[1])

    @property
    def gt_degree(self) -> np.ndarray:
        """Predictions that touch each ground-truth mask (``plot_m2n_counts``)."""
        return (self.iou > 0).sum(axis=1)

    @property
    def pred_degree(self) -> np.ndarray:
        """Ground-truth masks that touch each prediction (``plot_m2n_counts``)."""
        return (self.iou > 0).sum(axis=0)


def view_overlap(
    pred_labels: np.ndarray, gt_masks: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """IoU and Dice matrices of one view, ``get_overlap_matrices`` on a label image.

    ``pred_labels`` is a 2D non-negative integer image, ``0`` is background and
    the ids are ``1..max``. Returns two ``(n_gt, max)`` float64 arrays.
    """
    labels = np.asarray(pred_labels)
    if labels.ndim != 2:
        raise ValueError(f"pred_labels must be 2D, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            f"pred_labels must be an integer label image, got dtype {labels.dtype}"
        )
    if labels.size and int(labels.min()) < 0:
        raise ValueError("pred_labels must be non-negative")

    n = int(labels.max()) if labels.size else 0
    pred_area = np.bincount(labels.ravel(), minlength=n + 1).astype(np.float64)
    iou = np.zeros((len(gt_masks), n), dtype=np.float64)
    dice = np.zeros((len(gt_masks), n), dtype=np.float64)
    for index, gt_mask in enumerate(gt_masks):
        mask = np.asarray(gt_mask)
        if mask.shape != labels.shape:
            raise ValueError(
                f"ground_truth[{index}] has shape {mask.shape}, expected {labels.shape}"
            )
        ys, xs = np.nonzero(mask)
        inter = np.bincount(labels[ys, xs], minlength=n + 1).astype(np.float64)
        inter[0] = 0.0
        gt_area = np.float64(ys.size)
        union = pred_area + gt_area - inter
        total = pred_area + gt_area
        # An empty union gives 0, as in get_overlap_matrices (torch.where).
        iou_row = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        dice_row = np.divide(
            2.0 * inter, total, out=np.zeros_like(inter), where=union > 0
        )
        iou[index] = iou_row[1:]
        dice[index] = dice_row[1:]
    return iou, dice


def counts_from_overlap(iou: np.ndarray) -> PqCounts:
    """TP, FP, FN and matched IoU of one view, ``get_pq_score`` for one entry.

    With no ground-truth mask every prediction is a false positive, with no
    prediction every ground-truth mask is a false negative.
    """
    n_gt, n_pred = iou.shape
    hit = iou > OFFICIAL_IOU_THRESHOLD
    return PqCounts(
        tp=int(hit.sum()),
        fp=int((hit.sum(axis=0) == 0).sum()) if n_gt else n_pred,
        fn=int((hit.sum(axis=1) == 0).sum()) if n_pred else n_gt,
        iou_sum=float(iou[hit].sum()),
    )


def view_result(
    file_name: str,
    view_index: int,
    pred_labels: np.ndarray,
    gt_masks: Sequence[np.ndarray],
) -> ViewResult:
    """Overlap matrices and PQ counts of one prediction against one view."""
    iou, dice = view_overlap(pred_labels, gt_masks)
    return ViewResult(file_name, view_index, iou, dice, counts_from_overlap(iou))


def file_counts(results: Sequence[ViewResult]) -> list[PqCounts]:
    """Counts summed over the views of each frame, in order of first appearance.

    This is the per-file form that :func:`filament.stats.weighted_pq` pools.
    """
    by_file: dict[str, PqCounts] = {}
    for result in results:
        by_file[result.file_name] = (
            by_file.get(result.file_name, PqCounts()) + result.counts
        )
    return list(by_file.values())


def pooled_counts(results: Sequence[ViewResult]) -> PqCounts:
    """Counts summed over every view of every frame."""
    total = PqCounts()
    for result in results:
        total = total + result.counts
    return total


def pooled_pq(results: Sequence[ViewResult]) -> float:
    """Pooled PQ through :func:`filament.stats.weighted_pq`; NaN with nothing to score."""
    counts = file_counts(results)
    if not counts:
        return float("nan")
    return weighted_pq(counts)


def positive_scores(results: Sequence[ViewResult], kind: str) -> np.ndarray:
    """Every ``iou`` or ``dice`` value above 0, as ``plot_distribution`` selects them."""
    if kind not in ("iou", "dice"):
        raise ValueError(f"kind must be 'iou' or 'dice', got {kind!r}")
    parts = [np.asarray(getattr(r, kind), dtype=np.float64).ravel() for r in results]
    scores = np.concatenate(parts) if parts else np.zeros(0)
    return scores[scores > 0]


def matched_dice(results: Sequence[ViewResult]) -> np.ndarray:
    """Dice of the pairs with ``iou > 0.5`` (the true-positive pairs)."""
    parts = [r.dice[r.iou > OFFICIAL_IOU_THRESHOLD] for r in results]
    return np.concatenate(parts) if parts else np.zeros(0)


def _degree_histogram(results: Sequence[ViewResult], attribute: str) -> dict[int, int]:
    counter: Counter[int] = Counter()
    for result in results:
        counter.update(int(d) for d in getattr(result, attribute))
    return dict(sorted(counter.items()))


@dataclass(frozen=True, slots=True)
class RelationCounts:
    """How many ground-truth masks and predictions have each match degree."""

    gt_degrees: dict[int, int]
    pred_degrees: dict[int, int]

    @property
    def gt_total(self) -> int:
        return sum(self.gt_degrees.values())

    @property
    def pred_total(self) -> int:
        return sum(self.pred_degrees.values())

    @property
    def gt_miss(self) -> int:
        """1:0, a ground-truth mask no prediction touches."""
        return self.gt_degrees.get(0, 0)

    @property
    def gt_one_to_one(self) -> int:
        return self.gt_degrees.get(1, 0)

    @property
    def one_to_many(self) -> int:
        """1:n with n >= 2, a ground-truth mask split over several predictions."""
        return sum(c for d, c in self.gt_degrees.items() if d >= 2)

    @property
    def pred_miss(self) -> int:
        """0:1, a prediction no ground-truth mask touches."""
        return self.pred_degrees.get(0, 0)

    @property
    def pred_one_to_one(self) -> int:
        return self.pred_degrees.get(1, 0)

    @property
    def many_to_one(self) -> int:
        """n:1 with n >= 2, a prediction that covers several ground-truth masks."""
        return sum(c for d, c in self.pred_degrees.items() if d >= 2)


def relation_counts(results: Sequence[ViewResult]) -> RelationCounts:
    """Degree histograms of ground-truth masks and predictions (``plot_m2n_counts``)."""
    return RelationCounts(
        gt_degrees=_degree_histogram(results, "gt_degree"),
        pred_degrees=_degree_histogram(results, "pred_degree"),
    )


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def summary_rows(results: Sequence[ViewResult]) -> list[dict[str, object]]:
    """The rows of ``holdout_final_metrics.csv``: ``metric``, ``value``, ``source``.

    ``source`` is ``official`` (the self-evaluation notebook computes or plots
    it), ``derived`` (a function of official counts) or ``ours`` (not in the
    notebook, see the module docstring).
    """
    total = pooled_counts(results)
    pq = pooled_pq(results)
    sq = total.iou_sum / total.tp if total.tp else float("nan")
    denominator = total.tp + 0.5 * total.fp + 0.5 * total.fn
    rq = total.tp / denominator if denominator else float("nan")
    iou = positive_scores(results, "iou")
    dice = positive_scores(results, "dice")
    relations = relation_counts(results)

    rows: list[tuple[str, object, str]] = [
        ("views", len(results), SOURCE_DERIVED),
        ("frames", len({r.file_name for r in results}), SOURCE_DERIVED),
        ("pq", pq, SOURCE_OFFICIAL),
        ("sq", sq, SOURCE_DERIVED),
        ("rq", rq, SOURCE_DERIVED),
        ("tp", total.tp, SOURCE_OFFICIAL),
        ("fp", total.fp, SOURCE_OFFICIAL),
        ("fn", total.fn, SOURCE_OFFICIAL),
        ("mean_iou", _mean(iou), SOURCE_OFFICIAL),
        ("mean_dice", _mean(dice), SOURCE_OFFICIAL),
        ("mean_dice_tp", _mean(matched_dice(results)), SOURCE_OURS),
        ("iou_pairs", int(iou.size), SOURCE_OFFICIAL),
        ("dice_pairs", int(dice.size), SOURCE_OFFICIAL),
    ]
    for name, scores in (("iou", iou), ("dice", dice)):
        for q in QUANTILES:
            value = float(np.quantile(scores, q)) if scores.size else float("nan")
            rows.append((f"{name}_q{round(q * 100):02d}", value, SOURCE_OURS))
    rows += [
        ("gt_total", relations.gt_total, SOURCE_OFFICIAL),
        ("gt_miss_1to0", relations.gt_miss, SOURCE_OFFICIAL),
        ("gt_1to1", relations.gt_one_to_one, SOURCE_OFFICIAL),
        ("one_to_many", relations.one_to_many, SOURCE_OFFICIAL),
        ("pred_total", relations.pred_total, SOURCE_OFFICIAL),
        ("pred_miss_0to1", relations.pred_miss, SOURCE_OFFICIAL),
        ("pred_1to1", relations.pred_one_to_one, SOURCE_OFFICIAL),
        ("many_to_one", relations.many_to_one, SOURCE_OFFICIAL),
    ]
    rows += [
        (f"gt_degree_{degree}", count, SOURCE_OFFICIAL)
        for degree, count in relations.gt_degrees.items()
    ]
    rows += [
        (f"pred_degree_{degree}", count, SOURCE_OFFICIAL)
        for degree, count in relations.pred_degrees.items()
    ]
    return [{"metric": m, "value": v, "source": s} for m, v, s in rows]


def format_value(value: object) -> str:
    """Integers as they are, floats with 6 decimals, NaN as ``nan``."""
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return str(value)


def write_summary(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write the summary rows as CSV. The file holds numbers only, no image data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "source"])
        for row in rows:
            writer.writerow([row["metric"], format_value(row["value"]), row["source"]])


def format_summary(rows: Sequence[dict[str, object]]) -> str:
    """The summary as aligned text for the notebook log."""
    width = max((len(str(row["metric"])) for row in rows), default=0)
    return "\n".join(
        f"{row['metric']!s:<{width}}  {format_value(row['value']):>10}  {row['source']}"
        for row in rows
    )
