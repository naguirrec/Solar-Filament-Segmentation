"""Decide which candidate components to emit, with a small learned model.

Step B2 of ``docs/ROADMAP_V2.md``. The candidates come from a permissive
segmentation (a low threshold and a low ``min_area``). A regressor predicts each
candidate's expected matched IoU per annotator view from its features
(:func:`filament.components.component_features`), and a candidate is emitted when
the prediction is above ``lambda / 2``, where ``lambda`` is the pooled PQ that the
emitted set itself reaches. Section 4 of the roadmap derives that rule, and
:func:`dinkelbach_lambda` finds the fixed point.

Every score here is out of fold: a file is scored by a model that did not see its
date, under a ``lambda`` computed without that fold. :func:`evaluate_selector` then
applies the roadmap's rule against a fixed setting. With about 142 files a large
model overfits, so the models are deliberately small, and they are not tuned.

The selector only holds for the probability maps it was fitted on: the same
checkpoint, flips and post-processing.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from filament.annotations import TrainImage, annotator_views
from filament.axis_report import MIN_PARTITIONS_ADOPTED
from filament.baseline import on_disk
from filament.components import (
    FEATURE_NAMES,
    ViewHits,
    component_features,
    component_view_hits,
    expected_matched_iou,
    selection_counts,
)
from filament.instances import labels_for_min_areas
from filament.metrics import PqCounts
from filament.stats import adopts_change, date_folds, paired_bootstrap, weighted_pq
from filament.sweep import GridPoint, format_pq, view_masks

# From below: at 0 every candidate with a positive score is emitted, which is a set
# with a real PQ. A start above the optimum can emit nothing and stay there.
DINKELBACH_START = 0.0
DINKELBACH_ITERATIONS = 10
DINKELBACH_TOLERANCE = 1e-4
SELECTOR_FOLDS = 5
SELECTOR_SEEDS = (0, 1, 2, 3, 4)

# Permissive candidates, from which the selector chooses. Untuned.
CANDIDATE_POINTS = (GridPoint(0.45, 11, 100), GridPoint(0.35, 11, 100))


class Regressor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def shallow_tree(seed: int) -> Regressor:
    """The baseline: three levels, untuned."""
    return DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=seed)


def small_gbm(seed: int) -> Regressor:
    """A deliberately small gradient-boosted model, untuned."""
    return HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=100,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )


SELECTOR_MODELS: dict[str, Callable[[int], Regressor]] = {
    "shallow_tree": shallow_tree,
    "small_gbm": small_gbm,
}


@dataclass(frozen=True, slots=True)
class FileCandidates:
    """One file's candidates: what a model sees, and what the metric needs."""

    file_name: str
    day: dt.date
    features: np.ndarray
    views: tuple[ViewHits, ...]
    target: np.ndarray

    def __post_init__(self) -> None:
        if self.features.shape != (self.target.size, len(FEATURE_NAMES)):
            raise ValueError(
                f"{self.file_name}: features {self.features.shape} do not fit "
                f"{self.target.size} candidates"
            )

    @property
    def size(self) -> int:
        return int(self.target.size)


def build_file_candidates(
    image: TrainImage,
    day: dt.date,
    gray: np.ndarray,
    mean: np.ndarray,
    flip_std: np.ndarray,
    point: GridPoint,
) -> FileCandidates:
    """Label one file at ``point`` and record its features, matches and target."""
    disk = on_disk(gray)
    semantic = (np.asarray(mean) >= point.threshold).astype(np.uint8)
    labels = labels_for_min_areas(
        semantic, gray, [point.min_area], close_size=point.close_size, disk=disk
    )[point.min_area]
    features = component_features(labels, mean, flip_std, gray, disk)
    views = tuple(
        component_view_hits(labels, view_masks(image, view))
        for view in annotator_views(image)
    )
    return FileCandidates(
        file_name=image.file_name,
        day=day,
        features=features,
        views=views,
        target=expected_matched_iou(views, features.shape[0]),
    )


def out_of_fold_scores(
    files: Sequence[FileCandidates],
    folds: np.ndarray,
    make_model: Callable[[int], Regressor],
    seed: int = 0,
) -> list[np.ndarray]:
    """Each file's candidate scores, from a model fitted on the other folds."""
    assignment = np.asarray(folds)
    if assignment.size != len(files):
        raise ValueError(f"{assignment.size} folds for {len(files)} files")
    scores = [np.zeros(item.size, dtype=np.float64) for item in files]
    for label in sorted(set(assignment.tolist())):
        train = [f for f, fold in zip(files, assignment, strict=True) if fold != label]
        inputs = np.concatenate([item.features for item in train])
        outputs = np.concatenate([item.target for item in train])
        model = None
        if inputs.shape[0] > 0:
            model = make_model(seed)
            model.fit(inputs, outputs)
        for index in np.flatnonzero(assignment == label):
            if model is not None and files[index].size > 0:
                scores[index] = np.asarray(model.predict(files[index].features))
    return scores


def _counts_at(
    files: Sequence[FileCandidates], scores: Sequence[np.ndarray], lam: float
) -> list[PqCounts]:
    return [
        selection_counts(item.views, score > lam / 2.0)
        for item, score in zip(files, scores, strict=True)
    ]


def dinkelbach_lambda(
    files: Sequence[FileCandidates],
    scores: Sequence[np.ndarray],
    *,
    start: float = DINKELBACH_START,
    iterations: int = DINKELBACH_ITERATIONS,
) -> float:
    """Iterate ``lambda = pooled PQ of emitting score > lambda / 2`` from ``start``.

    The emission rule depends on the PQ, and the PQ on what is emitted. A few rounds
    settle it. A round whose set has no usable PQ (nothing emitted, or nothing that
    matches) keeps the last usable value.
    """
    lam = float(start)
    for _ in range(iterations):
        value = weighted_pq(_counts_at(files, scores, lam))
        if np.isnan(value) or value <= 0:
            break
        converged = abs(value - lam) < DINKELBACH_TOLERANCE
        lam = value
        if converged:
            break
    return lam


def crossfit_emission(
    files: Sequence[FileCandidates],
    scores: Sequence[np.ndarray],
    folds: np.ndarray,
) -> tuple[list[PqCounts], tuple[float, ...]]:
    """Per-file counts where each fold's ``lambda`` comes from the other folds."""
    assignment = np.asarray(folds)
    counts: list[PqCounts] = [PqCounts()] * len(files)
    lambdas = []
    for label in sorted(set(assignment.tolist())):
        others = [i for i, fold in enumerate(assignment.tolist()) if fold != label]
        lam = dinkelbach_lambda([files[i] for i in others], [scores[i] for i in others])
        lambdas.append(lam)
        for index in np.flatnonzero(assignment == label):
            counts[index] = selection_counts(
                files[index].views, scores[index] > lam / 2.0
            )
    return counts, tuple(lambdas)


SELECTOR_COLUMNS = (
    "candidates",
    "model",
    "partition",
    "pooled_pq",
    "diff_vs_base",
    "diff_low",
    "diff_high",
    "fraction_better",
    "emitted",
    "note",
)


def evaluate_selector(
    files: Sequence[FileCandidates],
    base: Sequence[PqCounts],
    make_model: Callable[[int], Regressor],
    *,
    model_name: str = "model",
    candidates_name: str = "candidates",
    seeds: Sequence[int] = SELECTOR_SEEDS,
    n_folds: int = SELECTOR_FOLDS,
    resamples: int = 2000,
) -> tuple[list[dict[str, str]], bool]:
    """Score the selector against ``base`` on several partitions of the dates.

    ``base`` is the per-file counts of the fixed setting, in the order of ``files``.
    It is adopted when the out-of-fold counts pass the roadmap's rule against ``base``
    (a gain of at least 0.005 in at least 90% of resamples) in at least 4 partitions.
    Returns one row per partition, a verdict row, and the verdict.
    """
    if len(files) != len(base):
        raise ValueError(f"{len(files)} files against {len(base)} base counts")
    days = [item.day for item in files]
    total = sum(item.size for item in files)
    rows = []
    passes = 0
    pooled = []
    for seed in seeds:
        folds = date_folds(days, n_folds, seed=seed)
        scores = out_of_fold_scores(files, folds, make_model, seed=seed)
        counts, lambdas = crossfit_emission(files, scores, folds)
        comparison = paired_bootstrap(base, counts, n=resamples, seed=seed)
        passed = adopts_change(comparison)
        passes += passed
        value = weighted_pq(counts)
        pooled.append(value)
        emitted = 0
        for index, label in enumerate(folds.tolist()):
            lam = lambdas[sorted(set(folds.tolist())).index(label)]
            emitted += int(np.count_nonzero(scores[index] > lam / 2.0))
        rows.append(
            {
                "candidates": candidates_name,
                "model": model_name,
                "partition": str(seed),
                "pooled_pq": format_pq(value),
                "diff_vs_base": format_pq(comparison.point),
                "diff_low": format_pq(comparison.low),
                "diff_high": format_pq(comparison.high),
                "fraction_better": f"{comparison.fraction_better:.3f}",
                "emitted": f"{emitted} of {total}",
                "note": (
                    ("passes; " if passed else "does not pass; ")
                    + "lambda "
                    + " ".join(f"{lam:.3f}" for lam in lambdas)
                ),
            }
        )
    verdict = passes >= MIN_PARTITIONS_ADOPTED
    mean_pq = float(np.mean(pooled))
    rows.append(
        {
            "candidates": candidates_name,
            "model": model_name,
            "partition": "verdict",
            "pooled_pq": format_pq(mean_pq),
            "diff_vs_base": format_pq(mean_pq - weighted_pq(base)),
            "diff_low": "",
            "diff_high": "",
            "fraction_better": "",
            "emitted": "",
            "note": (
                f"{'ADOPT' if verdict else 'DO NOT ADOPT'}: {passes} of {len(seeds)} "
                f"partitions pass, {MIN_PARTITIONS_ADOPTED} needed"
            ),
        }
    )
    return rows, verdict
