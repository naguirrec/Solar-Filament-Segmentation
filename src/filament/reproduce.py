"""The torch-free half of ``notebooks/reproduce.ipynb``.

The notebook loads the weights, builds a ``predict(gray) -> probability`` function
(tiled inference with flip TTA, which needs torch) and hands it to the functions
here. Everything else is plain numpy: the post-processing of a frame, the rows
of the submission CSV, the pooled holdout PQ, the check of that PQ against the
documented value, the timing and the overlays. It runs in CI on synthetic data
with a stand-in ``predict``.

The pipeline is fixed (``docs/RESULTS.md``): U-Net ResNet-34 crops 80 epochs, tiled
inference with flip TTA, threshold 0.45, ``close_size`` 11, ``min_area`` 400, the
solar-disk gate, and the object filter ``mean >= 0.8``. :data:`FINAL_CONFIG`
holds the numbers; the notebook's settings cell repeats them as literals and a test
keeps the two equal.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from filament.annotations import TrainImage
from filament.baseline import iter_filament_counts
from filament.components import (
    ADOPTED_OBJECT_FILTER,
    NO_OBJECT_FILTER,
    ObjectCut,
    filter_components,
    object_cut_from_config,
    per_file_object_counts,
)
from filament.crop_data import read_gray
from filament.gap_analysis import CURRENT_POINT
from filament.instances import labels_from_semantic
from filament.stats import weighted_pq
from filament.submission import filament_id
from filament.sweep import GridPoint, count_views

FULL_SIDE = 2048
# <repo>/src/filament/reproduce.py -> <repo>. Local data is looked up here and not in
# the working directory, which is notebooks/ when Jupyter opens the notebook.
REPO_ROOT = Path(__file__).resolve().parents[2]
# The pooled per-view PQ of the final pipeline on the date holdout (142 files, 231
# views), as in docs/RESULTS.md (object filter row, and the crop ensemble row).
EXPECTED_HOLDOUT_PQ = 0.4255
PQ_TOLERANCE = 0.002
EXPECTED_HOLDOUT = (142, 231)
# Under the Kaggle data layout, relative to the competition folder.
DATA_FOLDER = "MAGFiLO_1.0_Kaggle_2026"
KAGGLE_DATA_ROOT = Path("/kaggle/input/competitions/filament-segmentation-2026") / (
    DATA_FOLDER
)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Post-processing and TTA of the final pipeline."""

    threshold: float
    close_size: int
    min_area: int
    object_filter: Mapping[str, object] | None
    tta: bool

    @property
    def object_cut(self) -> ObjectCut | None:
        return object_cut_from_config(self.object_filter)

    @property
    def grid_point(self) -> GridPoint:
        return GridPoint(self.threshold, self.close_size, self.min_area)


FINAL_CONFIG = PipelineConfig(
    threshold=CURRENT_POINT.threshold,
    close_size=CURRENT_POINT.close_size,
    min_area=CURRENT_POINT.min_area,
    object_filter=ADOPTED_OBJECT_FILTER,
    tta=True,
)


class HoldoutPqError(RuntimeError):
    """The re-scored holdout PQ is not the documented one."""


# ------------------------------------------------------------- environment


def check_accelerator(cuda_available: bool, allow_cpu: bool) -> str:
    """``"cuda"`` or ``"cpu"``. Stops with a clear message on a CPU unless allowed.

    Efficiency is measured on a GPU, and 180 test images of tiled inference with
    four flips are very slow on a CPU.
    """
    if cuda_available:
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise RuntimeError(
        "No GPU found. This notebook is meant for a GPU (Kaggle: Settings > "
        "Accelerator > GPU T4), where efficiency is measured, and 180 images of tiled "
        "inference with four flips each are very slow on a CPU. Set ALLOW_CPU = True "
        "in the settings cell to run on the CPU anyway (a PILOT run is practical)."
    )


def resolve_data_root(
    setting: str | Path | None,
    *,
    environ: Mapping[str, str] | None = None,
    kaggle_root: Path = KAGGLE_DATA_ROOT,
    repo_root: Path = REPO_ROOT,
    local_roots: Sequence[Path] | None = None,
) -> Path:
    """The folder that holds ``train/`` and ``test/``.

    Order: the ``setting``, ``FILAMENT_DATA_ROOT``, the Kaggle path, then
    ``data/MAGFiLO_1.0_Kaggle_2026`` and ``data`` under ``repo_root`` (never
    under the working directory, which is ``notebooks/`` in Jupyter). A relative
    ``setting`` or ``FILAMENT_DATA_ROOT`` is the caller's own path, taken as is.
    A folder without ``test/test_images`` is skipped; none found is an error that
    lists what was tried.
    """
    if local_roots is None:
        local_roots = (repo_root / "data" / DATA_FOLDER, repo_root / "data")
    env = os.environ if environ is None else environ
    candidates: list[Path] = []
    if setting is not None:
        candidates.append(Path(setting))
    if env.get("FILAMENT_DATA_ROOT", "").strip():
        candidates.append(Path(env["FILAMENT_DATA_ROOT"].strip()))
    candidates += [kaggle_root, *local_roots]
    for candidate in candidates:
        if (candidate / "test" / "test_images").is_dir():
            return candidate
    tried = "\n".join(f"  {c}" for c in candidates)
    raise FileNotFoundError(
        "Could not find the competition data (a folder with test/test_images and "
        f"train/). Tried:\n{tried}\nOn Kaggle attach the competition "
        "filament-segmentation-2026. Locally set DATA_ROOT in the settings cell, or "
        "FILAMENT_DATA_ROOT."
    )


# ----------------------------------------------------------- one frame


def labels_for_frame(
    gray: np.ndarray, probability: np.ndarray, config: PipelineConfig = FINAL_CONFIG
) -> np.ndarray:
    """Instance labels of one frame: threshold, close, disk gate, area cut, object filter."""
    if probability.shape != gray.shape:
        raise ValueError(
            f"probability {probability.shape} does not match the frame {gray.shape}"
        )
    semantic = (probability >= config.threshold).astype(np.uint8)
    labels = labels_from_semantic(
        semantic, gray, min_area=config.min_area, close_size=config.close_size
    )
    return filter_components(labels, probability, config.object_cut)


# ------------------------------------------------------------------ timing


@dataclass(slots=True)
class Timing:
    """Seconds spent in ``predict`` and in the whole loop, for ``images`` frames."""

    images: int = 0
    inference_seconds: float = 0.0
    total_seconds: float = 0.0
    started: float = field(default_factory=time.monotonic, repr=False)

    @property
    def inference_per_image(self) -> float:
        return self.inference_seconds / self.images if self.images else float("nan")

    @property
    def total_per_image(self) -> float:
        return self.total_seconds / self.images if self.images else float("nan")

    def summary(self, label: str) -> dict[str, object]:
        return {
            "label": label,
            "images": self.images,
            "inference_seconds": round(self.inference_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "inference_seconds_per_image": round(self.inference_per_image, 4),
            "total_seconds_per_image": round(self.total_per_image, 4),
        }


def write_timing(path: str | Path, summaries: Sequence[Mapping[str, object]]) -> None:
    Path(path).write_text(
        json.dumps(list(summaries), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _timed_predict(
    predict: Callable[[np.ndarray], np.ndarray], gray: np.ndarray, timing: Timing
) -> np.ndarray:
    begin = time.monotonic()
    probability = predict(gray)
    timing.inference_seconds += time.monotonic() - begin
    return probability


# --------------------------------------------------------------- submission


def iter_submission_rows(
    paths: Sequence[Path],
    predict: Callable[[np.ndarray], np.ndarray],
    config: PipelineConfig,
    timing: Timing,
    *,
    reader: Callable[[Path], np.ndarray] = lambda p: read_gray(
        p, (FULL_SIDE, FULL_SIDE)
    ),
    progress: Callable[[int, int, Path, int], None] | None = None,
) -> Iterator[tuple[str, str]]:
    """``(filament_id, rle counts)`` for every filament of every image.

    Fills ``timing`` as it goes (the total is set when the iterator is
    exhausted, so consume it fully, for example with ``write_submission``).
    """
    begin = time.monotonic()
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        gray = reader(path)
        probability = _timed_predict(predict, gray, timing)
        labels = labels_for_frame(gray, probability, config)
        count = 0
        for number, counts in iter_filament_counts(labels):
            count += 1
            yield filament_id(path.stem, number), counts
        timing.images = index
        timing.total_seconds = time.monotonic() - begin
        if progress is not None:
            progress(index, total, path, count)
        del gray, probability, labels
    timing.total_seconds = time.monotonic() - begin


# ----------------------------------------------------------------- holdout


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    pq: float
    files: int
    views: int
    probabilities: dict[str, np.ndarray]


def score_holdout(
    holdout: Sequence[TrainImage],
    image_dir: Path,
    predict: Callable[[np.ndarray], np.ndarray],
    config: PipelineConfig,
    timing: Timing,
    *,
    keep: Sequence[str] = (),
    progress: Callable[[int, int, TrainImage], None] | None = None,
) -> HoldoutResult:
    """Pooled per-view PQ of the pipeline on ``holdout``, with the existing scorer.

    This is :func:`filament.components.per_file_object_counts` and
    :func:`filament.stats.weighted_pq`, the code behind the 0.4255 of
    ``docs/RESULTS.md``, fed with maps predicted now instead of cached ones.
    ``keep`` names the frames whose probability map is returned (for overlays).
    """
    kept: dict[str, np.ndarray] = {}
    wanted = set(keep)
    last: dict[str, np.ndarray] = {}
    begin = time.monotonic()

    def read(image: TrainImage) -> np.ndarray:
        gray = read_gray(image_dir / image.file_name, (image.height, image.width))
        last["gray"] = gray
        return gray

    def load(image: TrainImage) -> np.ndarray:
        probability = _timed_predict(predict, last["gray"], timing)
        if image.file_name in wanted:
            kept[image.file_name] = probability
        return probability

    def tick(index: int, total: int, image: TrainImage) -> None:
        timing.images = index
        timing.total_seconds = time.monotonic() - begin
        if progress is not None:
            progress(index, total, image)

    cut = config.object_cut or NO_OBJECT_FILTER
    result = per_file_object_counts(
        holdout, read, load, config.grid_point, [cut], progress=tick
    )
    return HoldoutResult(
        pq=weighted_pq(result.counts[cut]),
        files=len(holdout),
        views=count_views(holdout),
        probabilities=kept,
    )


def check_holdout_pq(
    measured: float,
    *,
    expected: float = EXPECTED_HOLDOUT_PQ,
    tolerance: float = PQ_TOLERANCE,
    pilot: bool = False,
) -> str:
    """The line to print. Raises :class:`HoldoutPqError` outside ``expected +- tolerance``.

    A pilot scores a few frames only, so its PQ is not comparable: it never raises
    and the line says it is not a result.
    """
    if pilot:
        return (
            f"PILOT: holdout PQ {measured:.4f} on a few frames. This is not a "
            f"result and is not compared with {expected:.4f}."
        )
    difference = measured - expected
    if not abs(difference) <= tolerance:
        raise HoldoutPqError(
            f"Holdout PQ {measured:.4f} is {difference:+.4f} from the documented "
            f"{expected:.4f} (allowed +-{tolerance:g}). The weights, the code or the "
            "settings are not the documented pipeline. Stopping."
        )
    return (
        f"holdout PQ {measured:.4f} vs documented {expected:.4f} "
        f"({difference:+.4f}, within +-{tolerance:g}): OK"
    )


# ---------------------------------------------------------------- overlays


def pick_overlay_names(
    names: Sequence[str], count: int = 6, seed: int = 0
) -> list[str]:
    """``count`` names chosen with a fixed seed, in the order of ``names``."""
    ordered = sorted(names)
    if count >= len(ordered):
        return ordered
    chosen = set(random.Random(seed).sample(ordered, count))
    return [name for name in ordered if name in chosen]


_COLORS = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
)


def overlay(gray: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """An RGB uint8 image: the frame with each instance tinted and outlined."""
    if labels.shape != gray.shape:
        raise ValueError(f"labels {labels.shape} do not match the frame {gray.shape}")
    rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)
    outlines: list[tuple[list[np.ndarray], tuple[int, int, int]]] = []
    for number in range(1, int(labels.max()) + 1):
        mask = labels == number
        if not mask.any():
            continue
        color = _COLORS[(number - 1) % len(_COLORS)]
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(color, np.float32)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        outlines.append((list(contours), color))
    image = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    for contours, color in outlines:
        cv2.drawContours(image, contours, -1, color, 2)
    return image
