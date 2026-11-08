"""The fixed final pipeline on the holdout, and the choice of figure examples.

The pipeline is the one of ``docs/PROJECT_REPORT.md``: U-Net r34 (80 epochs)
with flip TTA, pixel threshold 0.45, closing 11, ``min_area`` 400, solar-disk
gate, object filter ``mean >= 0.8``. The probability map comes from the cache
of the export notebook; this module does the rest, so the evaluation notebook
and the sweep notebooks share one definition instead of a copy each.

Examples for the overlay figures are picked by a fixed rule from the overlap
matrices and a seed, never by hand. The rule is in :func:`candidates`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from filament.annotations import TrainImage, annotator_views
from filament.components import (
    ADOPTED_OBJECT_FILTER,
    filter_components,
    object_cut_from_config,
)
from filament.final_metrics import ViewResult, view_result
from filament.gap_analysis import CURRENT_POINT
from filament.instances import labels_for_min_areas
from filament.sweep import view_masks

FINAL_POINT = CURRENT_POINT
FINAL_OBJECT_CUT = object_cut_from_config(ADOPTED_OBJECT_FILTER)

EXAMPLE_KINDS = (
    "good_match",
    "fragmentation",
    "over_merge",
    "miss",
    "false_positive",
)
# A "good match" is a 1:1 pair with at least this IoU. Only figure selection uses it.
GOOD_MATCH_IOU = 0.8


def final_labels(
    gray: np.ndarray, probability: np.ndarray, disk: np.ndarray | None = None
) -> np.ndarray:
    """Instance labels of one frame: threshold, closing, min area, disk gate, object filter."""
    semantic = (probability >= FINAL_POINT.threshold).astype(np.uint8)
    labels = labels_for_min_areas(
        semantic,
        gray,
        [FINAL_POINT.min_area],
        close_size=FINAL_POINT.close_size,
        disk=disk,
    )[FINAL_POINT.min_area]
    return filter_components(labels, probability, FINAL_OBJECT_CUT)


def frame_results(image: TrainImage, labels: np.ndarray) -> list[ViewResult]:
    """One :class:`ViewResult` per annotator view of the frame."""
    return [
        view_result(image.file_name, index, labels, view_masks(image, view))
        for index, view in enumerate(annotator_views(image))
    ]


@dataclass(frozen=True, slots=True)
class Example:
    """One object to draw. ``gt_indices`` index the view's masks, ``pred_ids`` are label ids."""

    kind: str
    file_name: str
    view_index: int
    gt_indices: tuple[int, ...]
    pred_ids: tuple[int, ...]
    note: str


def candidates(result: ViewResult, kind: str) -> list[Example]:
    """Every object of ``kind`` in one view, from the official match degrees.

    * ``good_match``: a ground-truth mask touched by one prediction that touches
      only it, with IoU >= :data:`GOOD_MATCH_IOU`.
    * ``fragmentation``: a ground-truth mask touched by n >= 2 predictions.
    * ``over_merge``: a prediction touching n >= 2 ground-truth masks.
    * ``miss``: a ground-truth mask no prediction touches.
    * ``false_positive``: a prediction no ground-truth mask touches.
    """
    if kind not in EXAMPLE_KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {EXAMPLE_KINDS}")
    found: list[Example] = []
    gt_degree = result.gt_degree
    pred_degree = result.pred_degree
    where = (result.file_name, result.view_index)
    if kind in ("good_match", "fragmentation", "miss"):
        for i in range(result.n_gt):
            touching = np.flatnonzero(result.iou[i] > 0)
            if kind == "miss" and gt_degree[i] == 0:
                found.append(
                    Example(
                        kind, *where, (i,), (), "no prediction touches this filament"
                    )
                )
            elif kind == "fragmentation" and gt_degree[i] >= 2:
                ids = tuple(int(j) + 1 for j in touching)
                found.append(
                    Example(
                        kind, *where, (i,), ids, f"1 filament in {len(ids)} predictions"
                    )
                )
            elif kind == "good_match" and gt_degree[i] == 1:
                j = int(touching[0])
                if pred_degree[j] == 1 and result.iou[i, j] >= GOOD_MATCH_IOU:
                    found.append(
                        Example(
                            kind,
                            *where,
                            (i,),
                            (j + 1,),
                            f"IoU {result.iou[i, j]:.2f}, Dice {result.dice[i, j]:.2f}",
                        )
                    )
    else:
        for j in range(result.n_pred):
            touching = np.flatnonzero(result.iou[:, j] > 0)
            if kind == "false_positive" and pred_degree[j] == 0:
                found.append(
                    Example(
                        kind, *where, (), (j + 1,), "prediction touches no filament"
                    )
                )
            elif kind == "over_merge" and pred_degree[j] >= 2:
                found.append(
                    Example(
                        kind,
                        *where,
                        tuple(int(i) for i in touching),
                        (j + 1,),
                        f"1 prediction over {len(touching)} filaments",
                    )
                )
    return found


def select_examples(
    results: Sequence[ViewResult], *, per_kind: int = 3, seed: int = 0
) -> list[Example]:
    """Up to ``per_kind`` examples of each kind, seeded and on different frames first.

    The candidates are listed in the order of ``results``; a generator seeded
    with ``(seed, kind position)`` shuffles them, and the first pass keeps one
    per frame before a second pass fills what is missing. The same results and
    seed give the same examples.
    """
    if per_kind < 1:
        raise ValueError("per_kind must be >= 1")
    chosen: list[Example] = []
    for position, kind in enumerate(EXAMPLE_KINDS):
        pool = [e for result in results for e in candidates(result, kind)]
        order = np.random.default_rng([seed, position]).permutation(len(pool))
        shuffled = [pool[int(k)] for k in order]
        picked: list[Example] = []
        seen: set[str] = set()
        for example in shuffled:
            if len(picked) < per_kind and example.file_name not in seen:
                picked.append(example)
                seen.add(example.file_name)
        for example in shuffled:
            if len(picked) < per_kind and example not in picked:
                picked.append(example)
        chosen.extend(picked)
    return chosen
