"""Test-time flips and probability averaging across models, pure numpy.

Two independent, optional steps on a foreground probability map:

* **Flip TTA.** The frame is flipped, predicted, and the prediction is flipped
  back. Flips are their own inverse, so :func:`unflip` is :func:`flip_image`.
  The maps of the chosen flips are averaged with equal weight.
* **Ensemble.** The maps of several models are averaged with fixed weights
  that are normalized to sum to 1.

Torch is not imported here. A model is a callable ``gray (H, W) uint8 ->
probability (H, W) float``; ``filament.crop_training`` builds those callables.
The evaluation notebook predicts every needed (model, flip) map once per frame
and builds each configuration from that cache with :func:`combine`, so the
extra configurations cost no extra network passes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

# The identity, a horizontal flip (columns reversed), a vertical flip (rows
# reversed) and both. TTA averages all four.
FLIP_NAMES = ("none", "h", "v", "hv")

# name -> axes reversed
_AXES: dict[str, tuple[int, ...]] = {"none": (), "h": (1,), "v": (0,), "hv": (0, 1)}


def flip_image(array: np.ndarray, name: str) -> np.ndarray:
    """Return ``array`` flipped as ``name`` says, as a new C-contiguous array.

    ``array`` is 2D (``H × W``). ``"h"`` reverses the columns, ``"v"`` the rows.
    """
    if name not in _AXES:
        raise ValueError(f"unknown flip {name!r}, expected one of {FLIP_NAMES}")
    data = np.asarray(array)
    if data.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {data.shape}")
    axes = _AXES[name]
    flipped = np.flip(data, axis=axes) if axes else data
    return np.array(flipped, order="C", copy=True)


def unflip(array: np.ndarray, name: str) -> np.ndarray:
    """Undo :func:`flip_image`. A flip is its own inverse."""
    return flip_image(array, name)


def flipped_predictions(
    predict: Callable[[np.ndarray], np.ndarray],
    gray: np.ndarray,
    flips: Sequence[str] = FLIP_NAMES,
) -> dict[str, np.ndarray]:
    """Predict each flip of ``gray`` and put every map back in the frame's orientation.

    Returns ``{flip: probability}`` as float32, one network pass per flip.
    Each map has the shape of ``gray``.
    """
    names = list(flips)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"flips must be non-empty and distinct, got {names}")
    maps: dict[str, np.ndarray] = {}
    for name in names:
        probability = np.asarray(predict(flip_image(gray, name)))
        if probability.shape != gray.shape:
            raise ValueError(
                f"prediction for flip {name!r} has shape {probability.shape}, "
                f"expected {gray.shape}"
            )
        maps[name] = unflip(probability, name).astype(np.float32, copy=False)
    return maps


def mean_maps(maps: Sequence[np.ndarray]) -> np.ndarray:
    """Equal-weight mean of same-shaped probability maps, float32."""
    return weighted_mean(maps, [1.0] * len(maps))


def weighted_mean(maps: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted mean of same-shaped maps. ``weights`` are normalized to sum to 1.

    Weights must be finite and non-negative with a positive sum. A zero weight
    is allowed and the map still has to have the right shape. The result is
    float32 and one input map is never modified.
    """
    if len(maps) == 0:
        raise ValueError("no maps to average")
    if len(maps) != len(weights):
        raise ValueError(f"{len(maps)} maps but {len(weights)} weights")
    values = [float(w) for w in weights]
    if any(not np.isfinite(w) or w < 0 for w in values) or sum(values) <= 0:
        raise ValueError(
            f"weights must be finite, non-negative and sum to > 0, got {values}"
        )
    total = sum(values)
    shape = np.shape(maps[0])
    result = np.zeros(shape, dtype=np.float32)
    for probability, weight in zip(maps, values, strict=True):
        if np.shape(probability) != shape:
            raise ValueError(f"map shapes differ: {np.shape(probability)} and {shape}")
        if weight > 0:
            result += np.float32(weight / total) * np.asarray(
                probability, dtype=np.float32
            )
    return result


def resize_probability(probability: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a probability map to ``shape`` (rows, columns), float32.

    Used to bring a model that predicts a downscaled frame (the 512 px
    ResNet-18) onto the 2048 px frame of the others. Values stay inside the
    input's range.
    """
    data = np.ascontiguousarray(probability, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected a 2D map, got shape {data.shape}")
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"target shape {shape} must be at least 1×1")
    if data.shape == (rows, cols):
        return data.copy()
    return cv2.resize(data, (cols, rows), interpolation=cv2.INTER_LINEAR)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """One way of turning cached model maps into a probability map.

    ``weights`` maps a model name to its weight (normalized when combined);
    ``tta`` averages the four flips of each model first.
    """

    name: str
    weights: Mapping[str, float] = field(default_factory=dict)
    tta: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a config needs a name")
        if not self.weights:
            raise ValueError(f"config {self.name!r} has no models")

    @property
    def flips(self) -> tuple[str, ...]:
        return FLIP_NAMES if self.tta else ("none",)


def needed_flips(configs: Sequence[EvalConfig]) -> dict[str, tuple[str, ...]]:
    """For each model name, the flips any config needs (in ``FLIP_NAMES`` order)."""
    wanted: dict[str, set[str]] = {}
    for config in configs:
        for model in config.weights:
            wanted.setdefault(model, set()).update(config.flips)
    return {
        model: tuple(flip for flip in FLIP_NAMES if flip in flips)
        for model, flips in wanted.items()
    }


def combine(
    cache: Mapping[str, Mapping[str, np.ndarray]], config: EvalConfig
) -> np.ndarray:
    """Build the config's probability map from ``cache[model][flip]``.

    Per model: the mean of its flips (just ``"none"`` without TTA). Across
    models: the weighted mean with the config's weights. Models are combined
    in the order of ``config.weights``.
    """
    per_model = []
    for model in config.weights:
        if model not in cache:
            raise KeyError(f"config {config.name!r} needs model {model!r}")
        missing = [flip for flip in config.flips if flip not in cache[model]]
        if missing:
            raise KeyError(f"model {model!r} has no cached flips {missing}")
        per_model.append(mean_maps([cache[model][flip] for flip in config.flips]))
    return weighted_mean(per_model, list(config.weights.values()))


def ensemble_predictor(
    predictors: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    config: EvalConfig,
) -> Callable[[np.ndarray], np.ndarray]:
    """``fn(gray)`` that runs a config end to end, for the inference notebook.

    Only the models in ``config.weights`` run, each once per flip the config
    needs. Evaluation uses :func:`combine` on a shared cache instead.
    """
    for model in config.weights:
        if model not in predictors:
            raise KeyError(f"config {config.name!r} needs model {model!r}")

    def predict(gray: np.ndarray) -> np.ndarray:
        cache = {
            model: flipped_predictions(predictors[model], gray, config.flips)
            for model in config.weights
        }
        return combine(cache, config)

    return predict
