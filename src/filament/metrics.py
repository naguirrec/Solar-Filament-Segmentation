"""Panoptic quality for filament instance masks.

The holdout score is the pooled count in :class:`PqCounts`. One
prediction is scored against each annotator view on its own. A view is
that annotator's instance masks: not their union and not a consensus.
TP, FP, FN, and matched IoU are summed over every view of every image,
then

    PQ = iou_sum / (TP + 0.5 * FP + 0.5 * FN)

There is no per-image average. A zero denominator is NaN, including
when the prediction and every view are empty. It is not 1.

A pair matches when IoU is strictly greater than 0.5. Every such pair
counts. This is not the greedy one-to-one match from Kirillov et al.
One ground-truth mask can match several predictions, and one prediction
can match several ground-truth masks. Overlapping predictions therefore
inflate TP. This module does not hide that overlap. A separate
submission validator rejects overlapping masks.

:func:`fast_view_counts` is the label-image form of that rule.
:func:`reference_view_counts` is the same rule on a list of masks.
[VERIFICAR] that either one matches the live organizer notebook byte
for byte. That notebook was not downloaded:
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook.

:func:`panoptic_quality` is the Kirillov helper (greedy, one-to-one).
The holdout does not call it. Its empty-empty score stays 1, which is
not the pooled rule above.

Train labels have four filament categories (Left, Right, Unidentifiable,
Ambiguous). Both scores treat every instance as one filament class,
because the submission CSV has no class column. Whether the official
metric ignores chirality is [VERIFICAR] against the competition
Evaluation page.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


def panoptic_quality(
    predictions: Iterable[np.ndarray],
    ground_truth: Iterable[np.ndarray],
) -> float:
    """Return Kirillov PQ for the filament instances in one image.

    ``predictions`` and ``ground_truth`` are sequences of 2D masks, one
    filament each. Nonzero pixels are foreground. Masks in one call
    share a single shape. An empty sequence means there is no filament.
    An all-zero mask is rejected because background is not an instance.

    Raises ``ValueError`` when a mask is not 2D, has no foreground, or
    has a different shape from the others.
    """
    predicted = _binary_instances(predictions, "predictions")
    actual = _binary_instances(ground_truth, "ground_truth")
    _require_same_shape(predicted, actual)

    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred_mask in enumerate(predicted):
        for gt_index, gt_mask in enumerate(actual):
            intersection = int(np.count_nonzero(pred_mask & gt_mask))
            union = int(np.count_nonzero(pred_mask | gt_mask))
            # IoU > 1/2 iff 2 * intersection > union. Integer math keeps
            # the Kirillov cutoff exact for pixel counts.
            if intersection * 2 <= union:
                continue
            candidates.append((intersection / union, pred_index, gt_index))

    # Equal IoU: lower prediction index, then lower ground-truth index.
    candidates.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()
    iou_sum = 0.0
    for iou, pred_index, gt_index in candidates:
        if pred_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
        iou_sum += iou

    true_positives = len(matched_predictions)
    false_positives = len(predicted) - true_positives
    false_negatives = len(actual) - true_positives
    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    # Kirillov helper only. The pooled score in PqCounts.pq is NaN here.
    if denominator == 0:
        return 1.0
    return iou_sum / denominator


@dataclass(frozen=True, slots=True)
class PqCounts:
    """TP, FP, FN, and matched IoU for one or more annotator views.

    ``pq`` is ``iou_sum / (tp + 0.5 * fp + 0.5 * fn)``. A zero
    denominator is NaN, not 1. Add counts to pool views. Do not average
    the ``pq`` values.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0

    def __add__(self, other: object) -> PqCounts:
        if not isinstance(other, PqCounts):
            return NotImplemented
        return PqCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            iou_sum=self.iou_sum + other.iou_sum,
        )

    @property
    def pq(self) -> float:
        """One pooled PQ. NaN when ``tp + 0.5 * fp + 0.5 * fn`` is 0."""
        denominator = self.tp + 0.5 * self.fp + 0.5 * self.fn
        if denominator == 0:
            return float("nan")
        return self.iou_sum / denominator


def reference_view_counts(
    predictions: Sequence[np.ndarray],
    ground_truth: Sequence[np.ndarray],
) -> PqCounts:
    """Count every prediction/ground-truth pair with IoU > 0.5.

    This is the mask-list form of the pooled match. It is not a greedy
    one-to-one assignment. A prediction that covers two ground-truth
    masks adds two true positives. Two predictions that cover one
    ground-truth mask also add two. Overlap inflates TP on purpose.

    Masks in one call share a shape. A zero denominator is left for
    :attr:`PqCounts.pq` and is not turned into 1 here.
    """
    predicted = [
        _as_bool_mask(mask, "predictions", index)
        for index, mask in enumerate(predictions)
    ]
    actual = [
        _as_bool_mask(mask, "ground_truth", index)
        for index, mask in enumerate(ground_truth)
    ]
    _require_mask_shape(predicted, actual)

    pred_hit = [False] * len(predicted)
    true_positives = 0
    false_negatives = 0
    matched_iou: list[np.float64] = []
    for gt_mask in actual:
        gt_area = int(np.count_nonzero(gt_mask))
        hits: list[np.float64] = []
        for pred_index, pred_mask in enumerate(predicted):
            inter = int(np.count_nonzero(pred_mask & gt_mask))
            pred_area = int(np.count_nonzero(pred_mask))
            union = pred_area + gt_area - inter
            if union <= 0:
                continue
            iou = np.float64(inter) / np.float64(union)
            if iou <= 0.5:
                continue
            hits.append(iou)
            pred_hit[pred_index] = True
        if hits:
            true_positives += len(hits)
            matched_iou.extend(hits)
        else:
            false_negatives += 1
    false_positives = sum(not hit for hit in pred_hit)
    iou_sum = float(np.sum(np.asarray(matched_iou, dtype=np.float64), dtype=np.float64))
    return PqCounts(
        tp=true_positives,
        fp=false_positives,
        fn=false_negatives,
        iou_sum=iou_sum,
    )


def fast_view_counts(
    pred_labels: np.ndarray,
    gt_masks: Sequence[np.ndarray],
) -> PqCounts:
    """Count every IoU > 0.5 pair from a disjoint label image.

    ``pred_labels`` is an integer image. ``0`` is background. Ids are
    ``1..max`` and are disjoint by construction, which is what
    connected components produce. A ground-truth mask may match several
    ids, and one id may match several ground-truth masks. Each pair
    counts. This is the same rule as :func:`reference_view_counts`.

    A label image cannot store two predictions on one pixel. Pass
    overlapping prediction masks to :func:`reference_view_counts`; do
    not collapse them before counting.

    An id in ``1..max`` with no pixels is an unmatched prediction. The
    audit snippet counts it, and so does this function.
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
    pred_area = np.bincount(labels.ravel(), minlength=n + 1)
    pred_hit = np.zeros(n + 1, dtype=bool)
    true_positives = 0
    false_negatives = 0
    iou_sum = 0.0
    for index, gt_mask in enumerate(gt_masks):
        mask = np.asarray(gt_mask)
        if mask.shape != labels.shape:
            raise ValueError(
                f"ground_truth[{index}] has shape {mask.shape}, expected {labels.shape}"
            )
        ys, xs = np.nonzero(mask)
        inter = np.bincount(labels[ys, xs], minlength=n + 1).astype(np.float64)
        inter[0] = 0.0
        union = pred_area.astype(np.float64) + np.float64(ys.size) - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        hits = np.flatnonzero(iou > 0.5)
        if hits.size:
            true_positives += int(hits.size)
            iou_sum += float(iou[hits].sum())
            pred_hit[hits] = True
        else:
            false_negatives += 1
    false_positives = int((~pred_hit[1:]).sum())
    return PqCounts(
        tp=true_positives,
        fp=false_positives,
        fn=false_negatives,
        iou_sum=iou_sum,
    )


def _binary_instances(
    masks: Iterable[np.ndarray],
    name: str,
) -> list[np.ndarray]:
    binary_masks: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for index, mask in enumerate(masks):
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(f"{name}[{index}] must be 2D, got shape {array.shape}")
        binary = array != 0
        if not binary.any():
            raise ValueError(f"{name}[{index}] is empty; background is not an instance")
        mask_shape = (int(binary.shape[0]), int(binary.shape[1]))
        if shape is None:
            shape = mask_shape
        elif mask_shape != shape:
            raise ValueError(
                f"{name}[{index}] has shape {mask_shape}, expected {shape}"
            )
        binary_masks.append(binary)
    return binary_masks


def _as_bool_mask(mask: np.ndarray, name: str, index: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name}[{index}] must be 2D, got shape {array.shape}")
    return array != 0


def _require_mask_shape(
    predictions: Sequence[np.ndarray],
    ground_truth: Sequence[np.ndarray],
) -> None:
    shapes = [mask.shape for mask in (*predictions, *ground_truth)]
    if not shapes:
        return
    expected = shapes[0]
    if any(shape != expected for shape in shapes[1:]):
        raise ValueError(
            f"prediction and ground-truth masks must share a shape, got {shapes}"
        )


def _require_same_shape(
    predictions: list[np.ndarray],
    ground_truth: list[np.ndarray],
) -> None:
    if not predictions or not ground_truth:
        return
    predicted_shape = predictions[0].shape
    actual_shape = ground_truth[0].shape
    if predicted_shape != actual_shape:
        raise ValueError(
            "prediction and ground-truth masks must share a shape, "
            f"got {predicted_shape} and {actual_shape}"
        )
