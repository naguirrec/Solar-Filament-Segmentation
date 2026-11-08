"""Pooled holdout PQ for several ``min_area`` values from one prediction.

This is the scoring of ``notebooks/sweep_postprocess.ipynb`` with the loops
swapped: each holdout image is read and predicted once, its ground-truth
views are rasterized once, and every ``min_area`` is scored on that same
semantic mask. The per-``min_area`` totals are the same sums the notebook
makes, added in the same order (images in holdout order, views in
``annotator_views`` order), so the numbers do not change.

The metric is not reimplemented. Counts come from
:func:`filament.metrics.fast_view_counts`, are added as
:class:`filament.metrics.PqCounts`, and ``.pq`` is read once per
``min_area``. Per-image PQ is never averaged. A zero denominator is NaN.
Ground truth is :func:`filament.annotations.annotator_views`, one view per
original annotator entry.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from filament.annotations import TrainImage, annotator_views
from filament.baseline import on_disk
from filament.ensemble import EvalConfig, combine, flipped_predictions, needed_flips
from filament.instances import labels_for_min_areas, labels_from_semantic
from filament.metrics import PqCounts, fast_view_counts

# Same columns as results/postprocess_sweep_2026-11-01.csv.
COLUMNS = (
    "predictor",
    "min_area",
    "close_size",
    "pooled_pq",
    "holdout_files",
    "annotator_views",
)

# The grid of the 2026-11-01 sweep.
DEFAULT_MIN_AREAS = (80, 140, 240, 400, 700, 1200, 2000, 3000)

# The post-processing grid for the r34 crops model (see ``sweep_postproc_grid``).
# The current setting is threshold 0.5, close_size 0, min_area 400. close_size
# cannot go below 0, so the grid goes up through odd kernel sizes.
DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
DEFAULT_CLOSE_SIZES = (0, 3, 7, 15)
DEFAULT_GRID_MIN_AREAS = (240, 320, 400, 500, 600, 700)

# The TTA / ensemble sweep: the grid below is the one around threshold 0.45,
# close_size 15, min_area 400. Even kernel sizes are not accepted by
# ``label_foreground``, so 10 and 20 become the odd neighbours 11 and 21.
TTA_THRESHOLDS = (0.40, 0.45, 0.50)
TTA_CLOSE_SIZES = (11, 15, 21, 25)
TTA_MIN_AREAS = (400,)

TTA_COLUMNS = (
    "config",
    "models",
    "tta",
    "threshold",
    "close_size",
    "min_area",
    "pooled_pq",
    "holdout_files",
    "annotator_views",
)

POSTPROC_COLUMNS = (
    "predictor",
    "threshold",
    "close_size",
    "min_area",
    "pooled_pq",
    "holdout_files",
    "annotator_views",
)


def view_masks(image: TrainImage, view: tuple) -> list[np.ndarray]:
    """One mask per annotation in this annotator view. Not a union.

    Empty rasterizations are omitted. ``category_id`` is not read.
    ``semantic_only`` is not drawn. ``image.annotations`` is not read.
    """
    expected = (image.height, image.width)
    masks: list[np.ndarray] = []
    for annotation in view:
        mask = np.asarray(annotation.rasterize(image.height, image.width))
        if mask.shape != expected:
            raise ValueError(
                f"{image.file_name} ground truth has shape {mask.shape}, "
                f"expected {expected}"
            )
        if np.any(mask):
            masks.append(np.ascontiguousarray(mask))
    return masks


def count_views(holdout: Sequence[TrainImage]) -> int:
    """Number of annotator views over the whole holdout."""
    return sum(len(annotator_views(image)) for image in holdout)


def sweep_min_area(
    holdout: Sequence[TrainImage],
    read_gray: Callable[[TrainImage], np.ndarray],
    predict_semantic: Callable[[np.ndarray], np.ndarray],
    min_areas: Sequence[int],
    *,
    close_size: int = 0,
    progress: Callable[[int, int, TrainImage], None] | None = None,
) -> dict[int, PqCounts]:
    """Return the pooled :class:`PqCounts` for each ``min_area``.

    ``read_gray(image)`` returns the uint8 frame. ``predict_semantic(gray)``
    returns the binary mask at the frame's size and is called **once per
    image**. Each ``min_area`` then runs
    :func:`filament.instances.labels_from_semantic` on that mask.
    """
    if len(holdout) == 0:
        raise ValueError("holdout is empty")
    areas = [int(area) for area in min_areas]
    if not areas or len(set(areas)) != len(areas):
        raise ValueError(f"min_areas must be non-empty and distinct, got {areas}")
    totals = {area: PqCounts() for area in areas}
    total_images = len(holdout)
    for index, image in enumerate(holdout, start=1):
        gray = read_gray(image)
        semantic = predict_semantic(gray)
        if semantic.shape != gray.shape:
            raise ValueError(
                f"{image.file_name} semantic {semantic.shape} does not match "
                f"the frame {gray.shape}"
            )
        # All views of one frame are held at once (about 4 MB per filament mask).
        views = [view_masks(image, view) for view in annotator_views(image)]
        for area, running in totals.items():
            labels = labels_from_semantic(
                semantic, gray, min_area=area, close_size=close_size
            )
            if labels.shape != gray.shape:
                raise ValueError(
                    f"{image.file_name} labels {labels.shape} do not match {gray.shape}"
                )
            for masks in views:
                running = running + fast_view_counts(labels, masks)
            totals[area] = running
        if progress is not None:
            progress(index, total_images, image)
    return totals


def format_pq(value: float) -> str:
    """Six decimals, or ``nan``."""
    number = float(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.6f}"


def sweep_rows(
    predictor: str,
    totals: dict[int, PqCounts],
    *,
    close_size: int,
    holdout_files: int,
    views: int,
) -> list[dict[str, str]]:
    """One CSV row per ``min_area``, in the order of ``totals``."""
    return [
        {
            "predictor": predictor,
            "min_area": str(area),
            "close_size": str(close_size),
            "pooled_pq": format_pq(counts.pq),
            "holdout_files": str(holdout_files),
            "annotator_views": str(views),
        }
        for area, counts in totals.items()
    ]


def best_row(rows: Sequence[dict[str, str]]) -> dict[str, str] | None:
    """The row with the highest ``pooled_pq``, first on ties. None if all NaN."""
    best: dict[str, str] | None = None
    best_value = -math.inf
    for row in rows:
        value = float(row["pooled_pq"])
        if not math.isnan(value) and value > best_value:
            best, best_value = row, value
    return best


def write_rows(path: str | Path, rows: Sequence[dict[str, str]]) -> None:
    """Write ``rows`` with the header ``COLUMNS``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: Sequence[dict[str, str]]) -> str:
    """The rows as comma-separated lines under the header, as the sweep prints."""
    lines = [",".join(COLUMNS)]
    lines.extend(",".join(row[column] for column in COLUMNS) for row in rows)
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One post-processing setting: probability cut, closing kernel, area cut."""

    threshold: float
    close_size: int
    min_area: int


def postproc_grid(
    thresholds: Sequence[float],
    close_sizes: Sequence[int],
    min_areas: Sequence[int],
) -> list[GridPoint]:
    """Every combination, ordered threshold, then close_size, then min_area.

    Each axis must be non-empty and free of repeats. A threshold is in (0, 1),
    a ``close_size`` is 0 or an odd integer (the rule of
    :func:`filament.baseline.label_foreground`), a ``min_area`` is at least 1.
    """
    cuts = [_unit_threshold(value) for value in thresholds]
    kernels = [_close_size(value) for value in close_sizes]
    areas = [_min_area(value) for value in min_areas]
    for name, values in (
        ("thresholds", cuts),
        ("close_sizes", kernels),
        ("min_areas", areas),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be non-empty and distinct, got {values}")
    return [
        GridPoint(cut, kernel, area)
        for cut in cuts
        for kernel in kernels
        for area in areas
    ]


def sweep_postproc_grid(
    holdout: Sequence[TrainImage],
    read_gray: Callable[[TrainImage], np.ndarray],
    predict_probability: Callable[[np.ndarray], np.ndarray],
    thresholds: Sequence[float],
    close_sizes: Sequence[int],
    min_areas: Sequence[int],
    *,
    progress: Callable[[int, int, TrainImage], None] | None = None,
) -> dict[GridPoint, PqCounts]:
    """Return the pooled :class:`PqCounts` for each grid point.

    ``predict_probability(gray)`` is called **once per image** and its
    probability map is reused for every threshold, so the network never runs
    again for the grid. Per threshold the mask is ``probability >= threshold``
    (as ``crop_training.semantic_from_probability``); per ``close_size`` the
    components are labeled once and every ``min_area`` is cut from them
    (:func:`filament.instances.labels_for_min_areas`). Counts are added in the
    order of :func:`sweep_min_area` (images in holdout order, views in
    ``annotator_views`` order), so the point (0.5, 0, a) reproduces that
    sweep's ``min_area`` ``a``.
    """
    if len(holdout) == 0:
        raise ValueError("holdout is empty")
    points = postproc_grid(thresholds, close_sizes, min_areas)
    totals = {point: PqCounts() for point in points}
    total_images = len(holdout)
    for index, image in enumerate(holdout, start=1):
        gray = read_gray(image)
        probability = _checked_probability(image, gray, predict_probability(gray))
        _accumulate_image(totals, image, gray, probability)
        if progress is not None:
            progress(index, total_images, image)
    return totals


def sweep_postproc_configs(
    holdout: Sequence[TrainImage],
    read_gray: Callable[[TrainImage], np.ndarray],
    predict_probabilities: Callable[[np.ndarray], dict[str, np.ndarray]],
    thresholds: Sequence[float],
    close_sizes: Sequence[int],
    min_areas: Sequence[int],
    *,
    progress: Callable[[int, int, TrainImage], None] | None = None,
) -> dict[str, dict[GridPoint, PqCounts]]:
    """:func:`sweep_postproc_grid` for several probability maps of the same frame.

    ``predict_probabilities(gray)`` is called once per image and returns
    ``{config name: probability}`` (flip TTA and model ensembles built from one
    shared set of network passes, see :mod:`filament.ensemble`). The same grid
    is scored on each map, and every config's totals are the ones
    :func:`sweep_postproc_grid` gives for that map alone.
    """
    if len(holdout) == 0:
        raise ValueError("holdout is empty")
    points = postproc_grid(thresholds, close_sizes, min_areas)
    totals: dict[str, dict[GridPoint, PqCounts]] = {}
    total_images = len(holdout)
    for index, image in enumerate(holdout, start=1):
        gray = read_gray(image)
        maps = predict_probabilities(gray)
        if not maps:
            raise ValueError(f"{image.file_name} produced no probability maps")
        for name in maps:
            totals.setdefault(name, {point: PqCounts() for point in points})
        if set(maps) != set(totals):
            raise ValueError(
                f"{image.file_name} returned configs {sorted(maps)}, "
                f"expected {sorted(totals)}"
            )
        # The views of one frame and its disk gate are shared by every config.
        disk = on_disk(gray)
        views = [view_masks(image, view) for view in annotator_views(image)]
        for name, probability in maps.items():
            checked = _checked_probability(image, gray, probability)
            _accumulate_image(totals[name], image, gray, checked, disk, views)
        if progress is not None:
            progress(index, total_images, image)
    return totals


def _checked_probability(
    image: TrainImage, gray: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    array = np.asarray(probability)
    if array.shape != gray.shape:
        raise ValueError(
            f"{image.file_name} probability {array.shape} does not "
            f"match the frame {gray.shape}"
        )
    return array


def _accumulate_image(
    totals: dict[GridPoint, PqCounts],
    image: TrainImage,
    gray: np.ndarray,
    probability: np.ndarray,
    disk: np.ndarray | None = None,
    views: list[list[np.ndarray]] | None = None,
) -> None:
    """Add this image's counts to every grid point in ``totals``, in place."""
    cuts = list(dict.fromkeys(point.threshold for point in totals))
    kernels = list(dict.fromkeys(point.close_size for point in totals))
    areas = list(dict.fromkeys(point.min_area for point in totals))
    if disk is None:
        disk = on_disk(gray)
    if views is None:
        views = [view_masks(image, view) for view in annotator_views(image)]
    for cut in cuts:
        semantic = (probability >= cut).astype(np.uint8)
        for kernel in kernels:
            by_area = labels_for_min_areas(
                semantic, gray, areas, close_size=kernel, disk=disk
            )
            for area in areas:
                point = GridPoint(cut, kernel, area)
                if point not in totals:
                    continue
                running = totals[point]
                for masks in views:
                    running = running + fast_view_counts(by_area[area], masks)
                totals[point] = running


@dataclass(frozen=True, slots=True)
class PerFileCounts:
    """One :class:`PqCounts` per file for each grid point, in holdout order.

    ``counts`` adds up to the pooled totals of :func:`sweep_postproc_grid`.
    ``views`` is the number of annotator views of each file. These are the
    inputs of :mod:`filament.stats`.
    """

    file_names: tuple[str, ...]
    views: tuple[int, ...]
    counts: dict[GridPoint, tuple[PqCounts, ...]]

    def pooled(self, point: GridPoint) -> PqCounts:
        """The counts of every file added together."""
        return sum(self.counts[point], PqCounts())


def per_file_counts(
    holdout: Sequence[TrainImage],
    read_gray: Callable[[TrainImage], np.ndarray],
    load_probability: Callable[[TrainImage], np.ndarray],
    points: Sequence[GridPoint],
    *,
    progress: Callable[[int, int, TrainImage], None] | None = None,
) -> PerFileCounts:
    """Score ``points`` file by file from probability maps that already exist.

    ``load_probability(image)`` returns the map of that file, for example
    ``cache.mean(image.file_name)`` from :mod:`filament.probcache`. The scoring
    is the one of :func:`sweep_postproc_grid`, kept per file instead of pooled.
    ``points`` need not be a full grid.
    """
    if len(holdout) == 0:
        raise ValueError("holdout is empty")
    wanted = list(points)
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("points must be non-empty and distinct")
    rows: dict[GridPoint, list[PqCounts]] = {point: [] for point in wanted}
    total_images = len(holdout)
    for index, image in enumerate(holdout, start=1):
        gray = read_gray(image)
        probability = _checked_probability(image, gray, load_probability(image))
        totals = {point: PqCounts() for point in wanted}
        _accumulate_image(totals, image, gray, probability)
        for point in wanted:
            rows[point].append(totals[point])
        if progress is not None:
            progress(index, total_images, image)
    return PerFileCounts(
        file_names=tuple(image.file_name for image in holdout),
        views=tuple(len(annotator_views(image)) for image in holdout),
        counts={point: tuple(values) for point, values in rows.items()},
    )


def postproc_rows(
    predictor: str,
    totals: dict[GridPoint, PqCounts],
    *,
    holdout_files: int,
    views: int,
) -> list[dict[str, str]]:
    """One CSV row per grid point, highest ``pooled_pq`` first, NaN last.

    Ties keep the order of ``totals`` (the grid order).
    """
    scored = [(point, counts.pq) for point, counts in totals.items()]
    scored.sort(
        key=lambda item: (math.isnan(item[1]), 0.0 if math.isnan(item[1]) else -item[1])
    )
    return [
        {
            "predictor": predictor,
            "threshold": f"{point.threshold:g}",
            "close_size": str(point.close_size),
            "min_area": str(point.min_area),
            "pooled_pq": format_pq(value),
            "holdout_files": str(holdout_files),
            "annotator_views": str(views),
        }
        for point, value in scored
    ]


def write_postproc_rows(path: str | Path, rows: Sequence[dict[str, str]]) -> None:
    """Write ``rows`` with the header ``POSTPROC_COLUMNS``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSTPROC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_postproc_table(
    rows: Sequence[dict[str, str]], limit: int | None = None
) -> str:
    """The first ``limit`` rows (all by default) as comma-separated lines."""
    shown = rows if limit is None else rows[:limit]
    lines = [",".join(POSTPROC_COLUMNS)]
    lines.extend(",".join(row[column] for column in POSTPROC_COLUMNS) for row in shown)
    return "\n".join(lines)


def _unit_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError("threshold must be a number")
    number = float(value)
    if not 0.0 < number < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {value!r}")
    return number


def _close_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("close_size must be an int")
    size = int(value)
    if size != 0 and (size < 1 or size % 2 == 0):
        raise ValueError(f"close_size must be 0 or an odd integer, got {value!r}")
    return size


def _min_area(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("min_area must be an int")
    area = int(value)
    if area < 1:
        raise ValueError(f"min_area must be >= 1, got {value!r}")
    return area


@dataclass(frozen=True, slots=True)
class ConfigSweep:
    totals: dict[str, dict[GridPoint, PqCounts]]
    inference_seconds: float
    total_seconds: float


def sweep_configs(
    holdout: Sequence[TrainImage],
    read_gray: Callable[[TrainImage], np.ndarray],
    predictors: dict[str, Callable[[np.ndarray], np.ndarray]],
    configs: Sequence[EvalConfig],
    thresholds: Sequence[float],
    close_sizes: Sequence[int],
    min_areas: Sequence[int],
    *,
    progress: Callable[[int, int, TrainImage], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ConfigSweep:
    """Score flip-TTA and ensemble configs on the holdout, one network pass each.

    ``predictors[name](gray)`` returns a probability map for the frame.
    For every image and every model, each flip that any config needs is
    predicted **once** and cached (:func:`filament.ensemble.flipped_predictions`);
    every config is then built from that cache
    (:func:`filament.ensemble.combine`) and scored on the same grid with
    :func:`sweep_postproc_configs`. A config therefore costs no extra network
    passes. ``inference_seconds`` is the time spent inside the predictors.
    """
    names = [config.name for config in configs]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"config names must be non-empty and distinct, got {names}")
    wanted = needed_flips(configs)
    unknown = sorted(set(wanted) - set(predictors))
    if unknown:
        raise KeyError(f"no predictor for model(s) {unknown}")
    inference = 0.0

    def _predict_all(gray: np.ndarray) -> dict[str, np.ndarray]:
        nonlocal inference
        begin = clock()
        cache = {
            model: flipped_predictions(predictors[model], gray, flips)
            for model, flips in wanted.items()
        }
        inference += clock() - begin
        return {config.name: combine(cache, config) for config in configs}

    begin = clock()
    totals = sweep_postproc_configs(
        holdout,
        read_gray,
        _predict_all,
        thresholds,
        close_sizes,
        min_areas,
        progress=progress,
    )
    return ConfigSweep(
        totals=totals,
        inference_seconds=inference,
        total_seconds=clock() - begin,
    )


def config_rows(
    configs: Sequence[EvalConfig],
    totals: dict[str, dict[GridPoint, PqCounts]],
    *,
    holdout_files: int,
    views: int,
) -> list[dict[str, str]]:
    """One CSV row per (config, grid point), highest ``pooled_pq`` first, NaN last.

    ``models`` lists each model with its normalized weight, for example
    ``r18:0.3+r34:0.7``. Ties keep the order of ``configs`` and then of the grid.
    """
    entries = []
    for config in configs:
        total = sum(float(weight) for weight in config.weights.values())
        models = "+".join(
            f"{name}:{float(weight) / total:g}"
            for name, weight in config.weights.items()
        )
        for point, counts in totals[config.name].items():
            entries.append((config, models, point, counts.pq))
    entries.sort(key=lambda e: (math.isnan(e[3]), 0.0 if math.isnan(e[3]) else -e[3]))
    return [
        {
            "config": config.name,
            "models": models,
            "tta": "yes" if config.tta else "no",
            "threshold": f"{point.threshold:g}",
            "close_size": str(point.close_size),
            "min_area": str(point.min_area),
            "pooled_pq": format_pq(value),
            "holdout_files": str(holdout_files),
            "annotator_views": str(views),
        }
        for config, models, point, value in entries
    ]


def write_config_rows(path: str | Path, rows: Sequence[dict[str, str]]) -> None:
    """Write ``rows`` with the header ``TTA_COLUMNS``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TTA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def format_config_table(
    rows: Sequence[dict[str, str]], limit: int | None = None
) -> str:
    """The first ``limit`` rows (all by default) as comma-separated lines."""
    shown = rows if limit is None else rows[:limit]
    lines = [",".join(TTA_COLUMNS)]
    lines.extend(",".join(row[column] for column in TTA_COLUMNS) for row in shown)
    return "\n".join(lines)
