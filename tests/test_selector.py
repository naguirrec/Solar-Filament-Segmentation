"""The component selector on synthetic candidates, and on one synthetic frame."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.components import (
    FEATURE_NAMES,
    ViewHits,
    component_view_hits,
    keep_components,
    selection_counts,
)
from filament.instances import labels_for_min_areas
from filament.metrics import PqCounts, fast_view_counts
from filament.selector import (
    CANDIDATE_POINTS,
    SELECTOR_COLUMNS,
    SELECTOR_MODELS,
    FileCandidates,
    build_file_candidates,
    crossfit_emission,
    dinkelbach_lambda,
    evaluate_selector,
    out_of_fold_scores,
    shallow_tree,
    small_gbm,
)
from filament.stats import date_folds, weighted_pq
from filament.sweep import GridPoint, view_masks

SIDE = 128
START = dt.date(2020, 1, 1)
DIMENSIONS = len(FEATURE_NAMES)


def _file(
    rng: np.random.Generator, index: int, *, informative: bool = True
) -> FileCandidates:
    """Six real candidates (one view, IoU 0.7) and twelve false ones.

    With ``informative`` the first feature says which is which; otherwise every
    feature is noise, and no model can tell them apart.
    """
    real, false = 6, 12
    size = real + false
    features = rng.normal(0.0, 1.0, (size, DIMENSIONS)).astype(np.float32)
    if informative:
        features[:real, 0] += 4.0
    order = rng.permutation(size)
    features = features[order]
    is_real = order < real
    positions = np.flatnonzero(is_real)
    view = ViewHits(
        candidate=positions.astype(np.int64),
        ground_truth=np.arange(real, dtype=np.int64),
        iou=np.full(real, 0.7),
        ground_truths=real,
    )
    target = np.zeros(size)
    target[positions] = 0.7
    return FileCandidates(
        file_name=f"f{index}.jpeg",
        day=START + dt.timedelta(days=index),
        features=features,
        views=(view,),
        target=target,
    )


def _dataset(count: int = 60, *, informative: bool = True) -> list[FileCandidates]:
    rng = np.random.default_rng(0)
    return [_file(rng, i, informative=informative) for i in range(count)]


def _emit_all(files: list[FileCandidates]) -> list[PqCounts]:
    return [selection_counts(f.views, np.ones(f.size, bool)) for f in files]


def test_candidate_table_checks_its_shape() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        FileCandidates("a", START, np.zeros((2, DIMENSIONS)), (), np.zeros(3))


def test_candidate_points_are_more_permissive_than_the_current_setting() -> None:
    for point in CANDIDATE_POINTS:
        assert point.threshold <= 0.45 and point.min_area < 400


@pytest.mark.parametrize("name", sorted(SELECTOR_MODELS))
def test_scores_are_out_of_fold(name: str) -> None:
    files = _dataset(20)
    folds = date_folds([f.day for f in files], 5, seed=0)
    seen: list[int] = []

    class Spy:
        def __init__(self) -> None:
            self.inner = SELECTOR_MODELS[name](0)

        def fit(self, X, y):
            seen.append(X.shape[0])
            return self.inner.fit(X, y)

        def predict(self, X):
            return self.inner.predict(X)

    scores = out_of_fold_scores(files, folds, lambda _seed: Spy())
    assert len(seen) == 5
    # Each model is fitted on the other folds only: never on all 20 files.
    for fold, fitted in zip(range(5), seen, strict=True):
        assert fitted == 18 * int(np.sum(folds != fold))
    assert all(s.shape == (18,) for s in scores)


def test_scores_refuse_mismatched_folds() -> None:
    with pytest.raises(ValueError, match="folds"):
        out_of_fold_scores(_dataset(4), np.array([0, 1]), small_gbm)


def test_dinkelbach_reaches_a_fixed_point_of_its_own_rule() -> None:
    files = _dataset(10)
    scores = [f.target.copy() for f in files]  # a perfect score
    lam = dinkelbach_lambda(files, scores)
    assert lam == pytest.approx(0.7)
    emit = [s > lam / 2 for s in scores]
    counts = [selection_counts(f.views, e) for f, e in zip(files, emit, strict=True)]
    assert weighted_pq(counts) == pytest.approx(lam)


def test_dinkelbach_keeps_a_usable_value_when_nothing_matches() -> None:
    files = _dataset(4)
    zero = [np.zeros(f.size) for f in files]
    assert dinkelbach_lambda(files, zero, start=0.4) == pytest.approx(0.4)


def _low_score_files() -> tuple[list[FileCandidates], list[np.ndarray]]:
    """Real candidates that match in 1 view of 5, so their target is only 0.14."""
    rng = np.random.default_rng(0)
    files = []
    scores = []
    for index in range(6):
        real = _file(rng, index)
        empty = ViewHits(
            np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), ground_truths=0
        )
        views = (real.views[0], empty, empty, empty, empty)
        target = real.target / 5.0
        files.append(
            FileCandidates(real.file_name, real.day, real.features, views, target)
        )
        scores.append(target.copy())
    return files, scores


def test_dinkelbach_from_the_default_start_finds_a_low_optimum() -> None:
    # A start of 0.4 would ask for a score above 0.2, emit nothing and stay there.
    files, scores = _low_score_files()
    assert max(float(s.max()) for s in scores) < 0.2
    assert dinkelbach_lambda(files, scores, start=0.4) == pytest.approx(0.4)
    lam = dinkelbach_lambda(files, scores)
    assert 0 < lam < 0.28  # the rule keeps emitting: 0.14 > lam / 2
    emitted = sum(int(np.count_nonzero(s > lam / 2)) for s in scores)
    assert emitted == 6 * 6


def test_dinkelbach_from_below_prunes_the_candidates_with_low_scores() -> None:
    files = _dataset(10)
    # Real candidates score 0.7, false ones a small positive value.
    scores = [f.target + 0.01 for f in files]
    lam = dinkelbach_lambda(files, scores)
    assert lam == pytest.approx(0.7)
    assert all(int(np.count_nonzero(s > lam / 2)) == 6 for s in scores)


def test_each_fold_gets_a_lambda_from_the_other_folds() -> None:
    files = _dataset(10)
    scores = [f.target.copy() for f in files]
    folds = date_folds([f.day for f in files], 5, seed=0)
    counts, lambdas = crossfit_emission(files, scores, folds)
    assert len(lambdas) == 5 and len(counts) == 10
    assert all(lam == pytest.approx(0.7) for lam in lambdas)
    assert weighted_pq(counts) == pytest.approx(0.7)


@pytest.mark.parametrize("make_model", [shallow_tree, small_gbm])
def test_a_selector_with_real_signal_is_adopted(make_model) -> None:
    files = _dataset(60)
    base = _emit_all(files)
    rows, verdict = evaluate_selector(files, base, make_model, resamples=200)
    assert verdict is True
    assert rows[-1]["note"].startswith("ADOPT: 5 of 5")
    # Emitting 18 candidates per file with 6 real ones: TP 6, FP 12, so 4.2 / 12.
    assert weighted_pq(base) == pytest.approx(0.35)
    assert float(rows[-1]["pooled_pq"]) > 0.6
    assert tuple(rows[0]) == SELECTOR_COLUMNS


def test_a_selector_on_noise_is_not_adopted() -> None:
    files = _dataset(60, informative=False)
    base = _emit_all(files)
    rows, verdict = evaluate_selector(files, base, small_gbm, resamples=200)
    assert verdict is False
    assert rows[-1]["note"].startswith("DO NOT ADOPT")


def test_a_selector_is_not_scored_on_what_it_fitted() -> None:
    # In-sample, a large model on noise looks useful. Out of fold it must not.
    files = _dataset(60, informative=False)
    inputs = np.concatenate([f.features for f in files])
    outputs = np.concatenate([f.target for f in files])
    deep = SELECTOR_MODELS["small_gbm"](0)
    deep.fit(inputs, outputs)
    in_sample = [deep.predict(f.features) for f in files]
    in_sample_pq = weighted_pq(
        [
            selection_counts(f.views, s > 0.2)
            for f, s in zip(files, in_sample, strict=True)
        ]
    )
    folds = date_folds([f.day for f in files], 5, seed=0)
    scores = out_of_fold_scores(files, folds, small_gbm)
    out_of_fold_pq = weighted_pq(
        [selection_counts(f.views, s > 0.2) for f, s in zip(files, scores, strict=True)]
    )
    assert in_sample_pq > out_of_fold_pq + 0.02


def test_evaluate_needs_matching_base_counts() -> None:
    with pytest.raises(ValueError, match="base counts"):
        evaluate_selector(_dataset(4), [], small_gbm)


def test_a_fold_with_no_candidates_is_harmless() -> None:
    files = _dataset(10)
    empty = FileCandidates(
        file_name="empty.jpeg",
        day=START + dt.timedelta(days=100),
        features=np.zeros((0, DIMENSIONS), dtype=np.float32),
        views=(
            ViewHits(
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
                np.zeros(0),
                ground_truths=2,
            ),
        ),
        target=np.zeros(0),
    )
    files = [*files, empty]
    folds = date_folds([f.day for f in files], 5, seed=0)
    scores = out_of_fold_scores(files, folds, small_gbm)
    counts, _ = crossfit_emission(files, scores, folds)
    assert counts[-1] == PqCounts(tp=0, fp=0, fn=2, iou_sum=0.0)


# --- one real frame through the pipeline --------------------------------------------


def _gray() -> np.ndarray:
    rows, cols = np.ogrid[:SIDE, :SIDE]
    return np.where((rows - 64) ** 2 + (cols - 64) ** 2 <= 58**2, 200, 5).astype(
        np.uint8
    )


def _box(x0: int, y0: int, x1: int, y1: int) -> tuple[float, ...]:
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def _annotation(number: int, box: tuple[float, ...]) -> FilamentAnnotation:
    return FilamentAnnotation(
        id=number, image_id=1, category_id=1, spine=None, segmentation=(box,)
    )


def test_a_frame_goes_from_maps_to_the_same_counts_as_the_metric() -> None:
    real_a, real_b = (
        _annotation(1, _box(40, 40, 80, 60)),
        _annotation(2, _box(40, 70, 70, 85)),
    )
    image = TrainImage(
        id=1,
        file_name="a.jpeg",
        date_captured=None,
        height=SIDE,
        width=SIDE,
        annotations=(real_a, real_b),
        views=((real_a, real_b), (real_a,)),
    )
    mean = np.zeros((SIDE, SIDE), dtype=np.float32)
    for annotation in (real_a, real_b):
        mean[np.asarray(annotation.rasterize(SIDE, SIDE)) != 0] = 0.9
    mean[90:100, 50:70] = 0.5  # a false blob
    spread = np.zeros((SIDE, SIDE), dtype=np.float32)
    gray = _gray()
    point = GridPoint(0.45, 0, 20)
    item = build_file_candidates(image, START, gray, mean, spread, point)
    assert item.size == 3
    assert item.features.shape == (3, DIMENSIONS)
    assert len(item.views) == 2

    labels = labels_for_min_areas(
        (mean >= 0.45).astype(np.uint8), gray, [20], close_size=0
    )[20]
    masks = [view_masks(image, view) for view in image.views]
    expected = sum((fast_view_counts(labels, m) for m in masks), PqCounts())
    found = selection_counts(item.views, np.ones(3, bool))
    assert (found.tp, found.fp, found.fn) == (expected.tp, expected.fp, expected.fn)
    assert found.iou_sum == pytest.approx(expected.iou_sum)

    # The target: a in both views, b in one of two, the blob in none.
    assert sorted(np.round(item.target, 3).tolist()) == pytest.approx([0.0, 0.5, 1.0])
    drop_blob = item.target > 0
    kept = keep_components(labels, drop_blob)
    expected = sum((fast_view_counts(kept, m) for m in masks), PqCounts())
    found = selection_counts(item.views, drop_blob)
    assert (found.tp, found.fp, found.fn) == (expected.tp, expected.fp, expected.fn)
    assert component_view_hits(labels, masks[0]).ground_truths == 2
