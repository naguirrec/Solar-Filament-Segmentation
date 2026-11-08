"""How much to trust a holdout PQ. Pure numpy.

The holdout has 142 files and the sweeps compare many settings on it, so a
difference of a few thousandths can be noise or selection. Everything here
works on one :class:`filament.metrics.PqCounts` per file and resamples or splits
whole files, never views or pixels, because the views of a file share a frame.

* :func:`bootstrap_pq`: a confidence interval for one setting.
* :func:`paired_bootstrap`: the difference between two settings on the same
  resampled files, which removes what the two have in common.
* :func:`adopts_change`: the adoption rule of ``docs/ROADMAP_V2.md`` section 6.
* :func:`weighted_pq` and :func:`tilt_to_mean_views`: PQ under another mix of
  files, for example more annotator views per file.
* :func:`date_folds` and :func:`crossfit_select`: choose a setting on some dates
  and score it on the others, so the reported number is not the maximum of the
  sweep.

Every PQ is pooled, as in :class:`filament.metrics.PqCounts`: the numerators and
the denominators are added first, and the ratio is taken once. Per-file PQ values
are never averaged. A zero denominator is NaN.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from filament.metrics import PqCounts

CI_LEVELS = (2.5, 97.5)

# docs/ROADMAP_V2.md section 6.
ADOPT_MIN_GAIN = 0.005
ADOPT_MIN_FRACTION = 0.90


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """``point`` is the PQ of the files as they are. The rest is over resamples."""

    point: float
    mean: float
    low: float
    high: float
    resamples: int


@dataclass(frozen=True, slots=True)
class PairedResult:
    """``b`` minus ``a``. ``fraction_better`` counts resamples with ``b > a``."""

    point: float
    mean: float
    low: float
    high: float
    fraction_better: float
    resamples: int


def _parts(counts: Sequence[PqCounts]) -> tuple[np.ndarray, np.ndarray]:
    """Per-file numerators and denominators of the pooled PQ."""
    if len(counts) == 0:
        raise ValueError("no files to score")
    numerator = np.array([c.iou_sum for c in counts], dtype=np.float64)
    denominator = np.array(
        [c.tp + 0.5 * c.fp + 0.5 * c.fn for c in counts], dtype=np.float64
    )
    return numerator, denominator


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=out, where=denominator > 0)


def _resample_index(files: int, resamples: int, seed: int) -> np.ndarray:
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    return np.random.default_rng(seed).integers(0, files, size=(resamples, files))


def weighted_pq(
    counts_by_file: Sequence[PqCounts], weights: Sequence[float] | None = None
) -> float:
    """Pooled PQ with a weight per file: ``sum(w * iou) / sum(w * denominator)``.

    Equal weights give the pooled PQ. Weights must be finite and non-negative.
    """
    numerator, denominator = _parts(counts_by_file)
    if weights is None:
        factor = np.ones_like(numerator)
    else:
        factor = np.asarray(weights, dtype=np.float64)
        if factor.shape != numerator.shape:
            raise ValueError(
                f"{factor.size} weights for {numerator.size} files must match"
            )
        if not np.all(np.isfinite(factor)) or np.any(factor < 0):
            raise ValueError("weights must be finite and non-negative")
    return float(_ratio(np.sum(factor * numerator), np.sum(factor * denominator)))


def bootstrap_pq(
    counts_by_file: Sequence[PqCounts],
    n: int = 2000,
    seed: int = 0,
    weights: Sequence[float] | None = None,
) -> BootstrapResult:
    """Percentile 95% interval of the pooled PQ, resampling files with replacement.

    With ``weights``, as in :func:`weighted_pq`, each file keeps its weight when
    it is drawn.
    """
    numerator, denominator = _parts(counts_by_file)
    point = weighted_pq(counts_by_file, weights)
    if weights is not None:
        factor = np.asarray(weights, dtype=np.float64)
        numerator, denominator = factor * numerator, factor * denominator
    index = _resample_index(numerator.size, n, seed)
    values = _ratio(numerator[index].sum(axis=1), denominator[index].sum(axis=1))
    low, high = np.nanpercentile(values, CI_LEVELS)
    return BootstrapResult(
        point=point,
        mean=float(np.nanmean(values)),
        low=float(low),
        high=float(high),
        resamples=n,
    )


def paired_bootstrap(
    a: Sequence[PqCounts],
    b: Sequence[PqCounts],
    n: int = 2000,
    seed: int = 0,
) -> PairedResult:
    """Compare two settings on the same files: ``b`` minus ``a``.

    Each resample draws the files once and scores both settings on them. The
    two lists have to be per-file counts in the same file order.
    """
    if len(a) != len(b):
        raise ValueError(f"{len(a)} files against {len(b)} files")
    num_a, den_a = _parts(a)
    num_b, den_b = _parts(b)
    index = _resample_index(num_a.size, n, seed)
    pq_a = _ratio(num_a[index].sum(axis=1), den_a[index].sum(axis=1))
    pq_b = _ratio(num_b[index].sum(axis=1), den_b[index].sum(axis=1))
    difference = pq_b - pq_a
    valid = difference[~np.isnan(difference)]
    if valid.size == 0:
        raise ValueError("every resample has a zero denominator")
    low, high = np.percentile(valid, CI_LEVELS)
    return PairedResult(
        point=weighted_pq(b) - weighted_pq(a),
        mean=float(valid.mean()),
        low=float(low),
        high=float(high),
        fraction_better=float(np.mean(valid > 0)),
        resamples=n,
    )


def adopts_change(
    result: PairedResult,
    *,
    min_gain: float = ADOPT_MIN_GAIN,
    min_fraction: float = ADOPT_MIN_FRACTION,
) -> bool:
    """The roadmap's rule: a gain of at least 0.005 in at least 90% of resamples."""
    return result.point >= min_gain and result.fraction_better >= min_fraction


def tilt_to_mean_views(
    views_per_file: Sequence[int],
    target: float,
    *,
    iterations: int = 100,
) -> np.ndarray:
    """Weights ``exp(theta * views)`` whose weighted mean of views is ``target``.

    This is the smallest change of the file mix that reaches the target (it
    keeps the relative weight of files with the same number of views). The
    weights sum to the number of files, so equal weights mean no change. It
    raises when no mix of these files can reach the target.
    """
    views = np.asarray(views_per_file, dtype=np.float64)
    if views.size == 0:
        raise ValueError("no files")
    if np.any(views < 1):
        raise ValueError("a file has at least one view")
    low_views, high_views = float(views.min()), float(views.max())
    if low_views == high_views:
        if target != low_views:
            raise ValueError(f"every file has {low_views:g} views, target {target:g}")
        return np.ones_like(views)
    if not low_views < target < high_views:
        raise ValueError(
            f"target {target:g} is outside the range of views "
            f"({low_views:g}, {high_views:g}) of these files"
        )

    def _weights(theta: float) -> np.ndarray:
        # Centered on the smallest count so a large theta cannot overflow early.
        return np.exp(theta * (views - low_views))

    def _mean(theta: float) -> float:
        weights = _weights(theta)
        return float(np.sum(weights * views) / np.sum(weights))

    low, high = -1.0, 1.0
    while _mean(low) > target:
        low *= 2.0
    while _mean(high) < target:
        high *= 2.0
    for _ in range(iterations):
        middle = (low + high) / 2.0
        if _mean(middle) < target:
            low = middle
        else:
            high = middle
    weights = _weights((low + high) / 2.0)
    return weights * (views.size / np.sum(weights))


def effective_files(weights: Sequence[float]) -> float:
    """Kish's effective sample size of a weighting: how many equal files it is worth."""
    values = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values) ** 2 / np.sum(values**2))


def date_folds(
    dates: Sequence[dt.date], n_folds: int = 2, seed: int | None = None
) -> np.ndarray:
    """Return a fold number per file. Files of one date always stay together.

    Dates are dealt in order, or in a seeded random order, each to the fold with
    the fewest files so far, so the folds have about the same number of files.
    Without a seed the partition is deterministic. Different seeds give the
    different partitions a repeated cross-fit needs.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    unique = sorted(set(dates))
    if len(unique) < n_folds:
        raise ValueError(f"{len(unique)} dates cannot fill {n_folds} folds")
    if seed is not None:
        order = np.random.default_rng(seed).permutation(len(unique))
        unique = [unique[int(i)] for i in order]
    files_on = {day: 0 for day in unique}
    for day in dates:
        files_on[day] += 1
    fold_of: dict[dt.date, int] = {}
    sizes = [0] * n_folds
    for day in unique:
        fold = int(np.argmin(sizes))
        fold_of[day] = fold
        sizes[fold] += files_on[day]
    return np.array([fold_of[day] for day in dates], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class CrossfitResult:
    """A setting chosen on some files and scored on the others.

    ``pooled_pq`` pools every file, each scored with the setting chosen without
    it, and ``fold_pqs`` holds the score of each fold. ``best_pq`` is the maximum
    of the sweep over all files, ``best_key`` its setting, and ``optimism`` is
    ``best_pq - pooled_pq``, an estimate of what selection adds. ``out_of_fold`` is
    each file's counts under the setting chosen without it, in file order, for a
    paired comparison with a fixed setting.
    """

    pooled_pq: float
    mean_fold_pq: float
    fold_pqs: tuple[float, ...]
    selected: tuple[Hashable, ...]
    best_key: Hashable
    best_pq: float
    optimism: float
    out_of_fold: tuple[PqCounts, ...]


def _best_key(
    grid: Mapping[Hashable, np.ndarray], mask: np.ndarray, denominators: Mapping
) -> Hashable:
    best, best_value = None, -np.inf
    for key, numerator in grid.items():
        value = _ratio(np.sum(numerator[mask]), np.sum(denominators[key][mask]))
        if not np.isnan(value) and value > best_value:
            best, best_value = key, float(value)
    if best is None:
        raise ValueError("every setting has a zero denominator on these files")
    return best


def plateau_center(values: Sequence[float], tolerance: float = 0.002) -> int:
    """Index at the middle of the plateau around the maximum of an ordered axis.

    The plateau is the unbroken run of settings within ``tolerance`` of the maximum
    that holds it. ``docs/ROADMAP_V2.md`` section 6 picks its center and not the
    peak, which is more likely to be noise. NaN breaks a run. On a tie between two
    middles the lower index wins.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.all(np.isnan(array)):
        raise ValueError("no values to choose from")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    peak = int(np.nanargmax(array))
    limit = array[peak] - tolerance
    low = high = peak
    while low > 0 and array[low - 1] >= limit:
        low -= 1
    while high < array.size - 1 and array[high + 1] >= limit:
        high += 1
    return (low + high) // 2


def crossfit_select(
    grid_counts: Mapping[Hashable, Sequence[PqCounts]],
    folds: Sequence[int],
) -> CrossfitResult:
    """Choose on the other folds, score on this one, for every fold.

    ``grid_counts`` maps a setting to its per-file counts, all in the file order
    of ``folds``. On ties the first setting in ``grid_counts`` wins. The number
    to report is ``pooled_pq`` (or ``mean_fold_pq``), not ``best_pq``.
    """
    if not grid_counts:
        raise ValueError("no settings to choose from")
    numerators: dict[Hashable, np.ndarray] = {}
    denominators: dict[Hashable, np.ndarray] = {}
    assignment = np.asarray(folds)
    for key, counts in grid_counts.items():
        if len(counts) != assignment.size:
            raise ValueError(
                f"setting {key!r} has {len(counts)} files, folds have {assignment.size}"
            )
        numerators[key], denominators[key] = _parts(counts)
    labels = sorted(set(assignment.tolist()))
    if len(labels) < 2:
        raise ValueError("cross-fitting needs at least two folds")

    selected: list[Hashable] = []
    fold_pqs: list[float] = []
    out_of_fold: list[PqCounts] = list(next(iter(grid_counts.values())))
    total_numerator = 0.0
    total_denominator = 0.0
    for label in labels:
        inside = assignment == label
        key = _best_key(numerators, ~inside, denominators)
        selected.append(key)
        for position in np.flatnonzero(inside):
            out_of_fold[position] = grid_counts[key][position]
        numerator = float(np.sum(numerators[key][inside]))
        denominator = float(np.sum(denominators[key][inside]))
        fold_pqs.append(float(_ratio(numerator, denominator)))
        total_numerator += numerator
        total_denominator += denominator

    everywhere = np.ones(assignment.size, dtype=bool)
    best_key = _best_key(numerators, everywhere, denominators)
    best_pq = float(
        _ratio(np.sum(numerators[best_key]), np.sum(denominators[best_key]))
    )
    pooled = float(_ratio(total_numerator, total_denominator))
    return CrossfitResult(
        pooled_pq=pooled,
        mean_fold_pq=float(np.nanmean(fold_pqs)),
        fold_pqs=tuple(fold_pqs),
        selected=tuple(selected),
        best_key=best_key,
        best_pq=best_pq,
        optimism=best_pq - pooled,
        out_of_fold=tuple(out_of_fold),
    )
