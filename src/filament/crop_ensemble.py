"""Average the probabilities of several crop U-Nets, for inference and the holdout.

Each member predicts the frame with its own tiled inference and, with TTA, its own
four flips. Its flips are averaged first, then the members are averaged with weights
normalized to sum to 1 (:func:`filament.ensemble.combine`). The threshold, the
components and the object filter only see that one averaged map.

A member is a name of :data:`filament.checkpoint.CROP_CHECKPOINTS`, so its weights
file goes through the same guard as a single model. An ensemble is only as trusted
as its least checked member, so :func:`unpinned_members` lets a notebook refuse a
member whose hash is not pinned.

With no ensemble (``members=None``) the weights are ``{model: 1.0}``: the map is the
single model's, bit for bit.

Pure numpy: the predictors are callables the notebooks build with torch.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from filament.annotations import TrainImage, annotator_views
from filament.baseline import on_disk
from filament.checkpoint import CROP_CHECKPOINTS
from filament.components import ObjectCut, filter_components
from filament.ensemble import EvalConfig, combine, flipped_predictions, needed_flips
from filament.instances import labels_for_min_areas
from filament.metrics import PqCounts, fast_view_counts
from filament.stats import paired_bootstrap, weighted_pq
from filament.sweep import GridPoint, PerFileCounts, format_pq, view_masks

# The setting of the best leaderboard submission (docs/RESULTS.md): flip TTA,
# threshold 0.45, close_size 11, min_area 400, and the adopted object filter.
ENSEMBLE_POINT = GridPoint(0.45, 11, 400)

# r34 alone first: every other row is compared with it.
HOLDOUT_ENSEMBLE_CONFIGS = (
    EvalConfig("r34_tta", {"r34": 1.0}, tta=True),
    EvalConfig("r34_v2_agree_tta", {"r34_v2_agree": 1.0}, tta=True),
    EvalConfig("r34+agree_50_50_tta", {"r34": 0.5, "r34_v2_agree": 0.5}, tta=True),
    EvalConfig("r34+agree_70_30_tta", {"r34": 0.7, "r34_v2_agree": 0.3}, tta=True),
)

ENSEMBLE_COLUMNS = (
    "config",
    "models",
    "tta",
    "threshold",
    "close_size",
    "min_area",
    "object_filter",
    "pooled_pq",
    "diff_vs_first",
    "diff_low",
    "diff_high",
    "fraction_better",
    "holdout_files",
    "annotator_views",
)


def ensemble_weights(
    members: Sequence[tuple[str, float]] | None, model: str
) -> dict[str, float]:
    """Normalized ``{name: weight}``, in the given order. ``None`` is ``model`` alone.

    Names must be crop models of :data:`filament.checkpoint.CROP_CHECKPOINTS`, each
    once, with finite positive weights.
    """
    if members is None:
        return {model: 1.0}
    pairs = [(str(name), float(weight)) for name, weight in members]
    if not pairs:
        raise ValueError("an ensemble needs at least one member")
    names = [name for name, _ in pairs]
    if len(set(names)) != len(names):
        raise ValueError(f"ensemble members must be distinct, got {names}")
    unknown = [name for name in names if name not in CROP_CHECKPOINTS]
    if unknown:
        raise ValueError(
            f"unknown crop model(s) {unknown}, expected names of {sorted(CROP_CHECKPOINTS)}"
        )
    if any(not math.isfinite(w) or w <= 0 for _, w in pairs):
        raise ValueError(f"ensemble weights must be finite and > 0, got {pairs}")
    total = sum(weight for _, weight in pairs)
    return {name: weight / total for name, weight in pairs}


def unpinned_members(weights: Mapping[str, float]) -> list[str]:
    """Members whose ``tensor_sha256`` is not pinned in :mod:`filament.checkpoint`."""
    return [name for name in weights if CROP_CHECKPOINTS[name].tensor_sha256 is None]


def require_pinned(names: Sequence[str], *, allow_unverified: bool = False) -> None:
    """Stop when a member has no pinned hash, unless the notebook allows it."""
    unpinned = unpinned_members({name: 1.0 for name in names})
    if unpinned and not allow_unverified:
        raise RuntimeError(
            f"Ensemble member(s) {unpinned} have no pinned tensor_sha256 in "
            "filament.checkpoint. Pin them first."
        )


def submission_model_weights(
    model: str,
    crop_ensemble: Sequence[tuple[str, float]] | None,
    *,
    use_r18_ensemble: bool = False,
    r18_weights: tuple[float, float] = (0.5, 0.5),
    allow_unverified: bool = False,
) -> dict[str, float]:
    """The model weights the inference notebook combines, checked before any loading.

    Without ``crop_ensemble`` this is what the notebook always used: ``{model: 1.0}``,
    or ``{"r18": w18, model: w}`` with the ResNet-18 ensemble (weights as given;
    :func:`filament.ensemble.combine` normalizes them). With ``crop_ensemble`` the
    weights are normalized, ``model`` has to be a member, every member needs a
    pinned hash, and the ResNet-18 ensemble cannot be on too.
    """
    if crop_ensemble is None:
        if use_r18_ensemble:
            w18, w_model = (float(w) for w in r18_weights)
            if not (math.isfinite(w18) and math.isfinite(w_model)):
                raise ValueError(f"ENSEMBLE_WEIGHTS must be finite, got {r18_weights}")
            if w18 < 0 or w_model < 0 or w18 + w_model <= 0:
                raise ValueError(
                    "ENSEMBLE_WEIGHTS must be non-negative with a positive sum, "
                    f"got {r18_weights}"
                )
            return {"r18": r18_weights[0], model: r18_weights[1]}
        return {model: 1.0}
    if use_r18_ensemble:
        raise ValueError("Set CROP_ENSEMBLE or USE_ENSEMBLE, not both.")
    weights = ensemble_weights(crop_ensemble, model)
    if model not in weights:
        raise ValueError(f"MODEL {model!r} has to be a member of CROP_ENSEMBLE.")
    require_pinned(list(weights), allow_unverified=allow_unverified)
    return weights


def members_to_load(weights: Mapping[str, float], model: str) -> list[str]:
    """Crop models to load besides ``model``. The ResNet-18 loads on its own path."""
    return [name for name in weights if name not in (model, "r18")]


def describe_models(config: EvalConfig) -> str:
    """``r34:0.7+r34_v2_agree:0.3``, with the weights normalized."""
    total = sum(float(weight) for weight in config.weights.values())
    return "+".join(
        f"{name}:{float(weight) / total:g}" for name, weight in config.weights.items()
    )


def per_file_ensemble_counts(
    holdout: Sequence[TrainImage],
    read_gray: Callable[[TrainImage], np.ndarray],
    predictors: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    configs: Sequence[EvalConfig],
    *,
    point: GridPoint = ENSEMBLE_POINT,
    cut: ObjectCut | None = None,
    progress: Callable[[int, int, TrainImage], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[PerFileCounts, float]:
    """Score each config file by file at ``point``, with the object filter ``cut``.

    Each model predicts each flip it needs once per frame, and every config is built
    from those maps, so a config costs no extra network pass. Returns the per-file
    counts keyed by config name, and the seconds spent in the predictors.
    """
    if len(holdout) == 0:
        raise ValueError("holdout is empty")
    names = [config.name for config in configs]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"config names must be non-empty and distinct, got {names}")
    wanted = needed_flips(configs)
    missing = sorted(set(wanted) - set(predictors))
    if missing:
        raise KeyError(f"no predictor for model(s) {missing}")
    rows: dict[str, list[PqCounts]] = {name: [] for name in names}
    inference = 0.0
    for index, image in enumerate(holdout, start=1):
        gray = read_gray(image)
        begin = clock()
        cache = {
            model: flipped_predictions(predictors[model], gray, flips)
            for model, flips in wanted.items()
        }
        inference += clock() - begin
        disk = on_disk(gray)
        views = [view_masks(image, view) for view in annotator_views(image)]
        for config in configs:
            probability = combine(cache, config)
            semantic = (probability >= point.threshold).astype(np.uint8)
            labels = labels_for_min_areas(
                semantic,
                gray,
                [point.min_area],
                close_size=point.close_size,
                disk=disk,
            )[point.min_area]
            labels = filter_components(labels, probability, cut)
            counts = PqCounts()
            for masks in views:
                counts = counts + fast_view_counts(labels, masks)
            rows[config.name].append(counts)
        if progress is not None:
            progress(index, len(holdout), image)
    per_file = PerFileCounts(
        file_names=tuple(image.file_name for image in holdout),
        views=tuple(len(annotator_views(image)) for image in holdout),
        counts={name: tuple(values) for name, values in rows.items()},
    )
    return per_file, inference


def ensemble_rows(
    per_file: PerFileCounts,
    configs: Sequence[EvalConfig],
    *,
    point: GridPoint = ENSEMBLE_POINT,
    cut: ObjectCut | None = None,
    resamples: int = 2000,
    seed: int = 0,
) -> list[dict[str, str]]:
    """One CSV row per config, in config order, each compared with the first.

    The difference is a paired bootstrap over files: the same resampled files score
    both configs. Pick a config by that difference and its interval, not by the
    point PQ alone.
    """
    first = per_file.counts[configs[0].name]
    rows = []
    for config in configs:
        counts = per_file.counts[config.name]
        comparison = paired_bootstrap(first, counts, n=resamples, seed=seed)
        rows.append(
            {
                "config": config.name,
                "models": describe_models(config),
                "tta": "yes" if config.tta else "no",
                "threshold": f"{point.threshold:g}",
                "close_size": str(point.close_size),
                "min_area": str(point.min_area),
                "object_filter": "none" if cut is None else str(cut),
                "pooled_pq": format_pq(weighted_pq(counts)),
                "diff_vs_first": format_pq(comparison.point),
                "diff_low": format_pq(comparison.low),
                "diff_high": format_pq(comparison.high),
                "fraction_better": f"{comparison.fraction_better:.3f}",
                "holdout_files": str(len(per_file.file_names)),
                "annotator_views": str(sum(per_file.views)),
            }
        )
    return rows


def write_ensemble_rows(path: str | Path, rows: Sequence[dict[str, str]]) -> None:
    """Write the rows with the header ``ENSEMBLE_COLUMNS``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENSEMBLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
