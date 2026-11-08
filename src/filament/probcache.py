"""Foreground probability maps on disk, so a sweep does not need the GPU again.

For one model and one split the cache is a directory with three files::

    <root>/<model>/<split>/mean.npy    (N, H, W) float16, the mean of the flips
    <root>/<model>/<split>/std.npy     (N, H, W) float16, their standard deviation
    <root>/<model>/<split>/files.json  the N file names, in the order of the arrays

``<root>/<model>/manifest.json`` says how the maps were made (:func:`write_manifest`).

The maps are stacked in one array per split instead of one file per frame, so an
export leaves a handful of files in the Kaggle output, whatever the file limit
there is, and a reader maps the array and only touches the frames it asks for.

``mean`` is the map that flip TTA scores: :func:`filament.ensemble.mean_maps` of
the four flips. ``std`` is how much the flips disagree, per pixel. float16 keeps
2 bytes per pixel: 8 MiB per map at 2048 x 2048. On [0.25, 1] a stored value is
off by at most 2**-12, so a threshold decision can change only for a pixel that
close to it. Maps are returned as float32. A float16 array would compare against
a Python float rounded to float16 too, which moves a threshold such as 0.45.

``files.json`` is written last, so a directory without it is an interrupted
export and :class:`ProbCache` refuses it.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np

from filament.crop_data import read_gray
from filament.ensemble import FLIP_NAMES, flipped_predictions, mean_maps

MEAN_FILE = "mean.npy"
STD_FILE = "std.npy"
INDEX_FILE = "files.json"
MANIFEST_FILE = "manifest.json"

# A stored probability has to lie in [0, 1], give or take float32 rounding. This
# catches logits stored by mistake.
_RANGE_SLACK = 1e-4


def model_dir(root: str | Path, model: str) -> Path:
    """Where one model's splits and manifest live."""
    return Path(root) / model


def split_dir(root: str | Path, model: str, split: str) -> Path:
    """Where one model's maps for one split live."""
    return model_dir(root, model) / split


def flip_statistics(maps: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return the float32 ``(mean, std)`` of the flips of one frame.

    ``maps`` is the output of :func:`filament.ensemble.flipped_predictions`. The
    mean is :func:`filament.ensemble.mean_maps`, the map flip TTA scores. The
    standard deviation is the population one, over the flips.
    """
    if not maps:
        raise ValueError("no maps to summarize")
    values = list(maps.values())
    return mean_maps(values), np.std(np.stack(values), axis=0, dtype=np.float32)


class ProbCacheWriter:
    """Write the maps of one split. Use it as a context manager.

    Every name in ``file_names`` has to be written once. On a clean exit the
    index is written. On an exception it is not, so the directory cannot be read
    as a finished cache.
    """

    def __init__(
        self,
        directory: str | Path,
        file_names: Sequence[str],
        shape: tuple[int, int],
    ) -> None:
        names = list(file_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("file_names must be non-empty and distinct")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # A previous index would make a half-written export look finished.
        (self.directory / INDEX_FILE).unlink(missing_ok=True)
        self.file_names = names
        self.shape = (int(shape[0]), int(shape[1]))
        self._position = {name: index for index, name in enumerate(names)}
        self._written: set[str] = set()
        full = (len(names), *self.shape)
        self._mean = np.lib.format.open_memmap(
            self.directory / MEAN_FILE, mode="w+", dtype=np.float16, shape=full
        )
        self._std = np.lib.format.open_memmap(
            self.directory / STD_FILE, mode="w+", dtype=np.float16, shape=full
        )

    def write(self, file_name: str, mean: np.ndarray, std: np.ndarray) -> None:
        """Store one frame. Raises on a bad shape, name or value."""
        if file_name not in self._position:
            raise KeyError(f"{file_name!r} is not in this split")
        if file_name in self._written:
            raise ValueError(f"{file_name!r} was already written")
        index = self._position[file_name]
        self._mean[index] = self._checked(file_name, "mean", mean)
        self._std[index] = self._checked(file_name, "std", std)
        self._written.add(file_name)

    def close(self) -> None:
        """Flush the arrays and write the index. Raises if a frame is missing."""
        missing = [name for name in self.file_names if name not in self._written]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(self.file_names)} frames were not written, "
                f"for example {missing[0]!r}"
            )
        self._flush()
        temporary = self.directory / (INDEX_FILE + ".tmp")
        temporary.write_text(json.dumps(self.file_names), encoding="utf-8")
        os.replace(temporary, self.directory / INDEX_FILE)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self._flush()

    def _flush(self) -> None:
        self._mean.flush()
        self._std.flush()

    def _checked(self, file_name: str, label: str, array: np.ndarray) -> np.ndarray:
        data = np.asarray(array, dtype=np.float32)
        if data.shape != self.shape:
            raise ValueError(
                f"{file_name} {label} has shape {data.shape}, expected {self.shape}"
            )
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{file_name} {label} has non-finite values")
        low, high = float(data.min()), float(data.max())
        if low < -_RANGE_SLACK or high > 1.0 + _RANGE_SLACK:
            raise ValueError(
                f"{file_name} {label} is not a probability: range [{low}, {high}]"
            )
        return data.astype(np.float16)


class ProbCache:
    """Read the maps of one split. Frames are read from disk only when asked for."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        index = self.directory / INDEX_FILE
        if not index.is_file():
            raise FileNotFoundError(
                f"{index} does not exist: this is not a finished export."
            )
        self.file_names: list[str] = json.loads(index.read_text(encoding="utf-8"))
        self._position = {name: i for i, name in enumerate(self.file_names)}
        self._mean = np.load(self.directory / MEAN_FILE, mmap_mode="r")
        self._std = np.load(self.directory / STD_FILE, mmap_mode="r")
        for label, array in (("mean", self._mean), ("std", self._std)):
            if array.dtype != np.float16 or array.ndim != 3:
                raise ValueError(
                    f"{label} is {array.dtype} with shape {array.shape}, "
                    "expected (N, H, W) float16"
                )
            if array.shape[0] != len(self.file_names):
                raise ValueError(
                    f"{label} has {array.shape[0]} frames, the index has "
                    f"{len(self.file_names)}"
                )

    def __len__(self) -> int:
        return len(self.file_names)

    def __contains__(self, file_name: object) -> bool:
        return file_name in self._position

    def mean(self, file_name: str) -> np.ndarray:
        """The mean of the flips, float32, as a new array."""
        return self._frame(self._mean, file_name)

    def std(self, file_name: str) -> np.ndarray:
        """The standard deviation between flips, float32, as a new array."""
        return self._frame(self._std, file_name)

    def _frame(self, stack: np.ndarray, file_name: str) -> np.ndarray:
        if file_name not in self._position:
            raise KeyError(f"{file_name!r} is not in {self.directory}")
        return np.array(stack[self._position[file_name]], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ExportTiming:
    """Inference time of one split: network passes and tiling, not reading or writing."""

    files: int
    inference_seconds: float
    total_seconds: float

    @property
    def seconds_per_image(self) -> float:
        return self.inference_seconds / self.files


def export_prob_maps(
    paths: Sequence[Path],
    predict: Callable[[np.ndarray], np.ndarray],
    directory: str | Path,
    *,
    shape: tuple[int, int],
    flips: Sequence[str] = FLIP_NAMES,
    progress: Callable[[int, int, Path], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> ExportTiming:
    """Predict every frame once per flip and write one finished cache.

    ``predict(gray)`` returns the probability map of a uint8 frame. ``paths``
    are the frames, and each file's name is its key in the cache. A frame of
    another size stops the export.
    """
    frames = [Path(path) for path in paths]
    inference = 0.0
    begin = clock()
    with ProbCacheWriter(directory, [p.name for p in frames], shape) as writer:
        for index, path in enumerate(frames, start=1):
            gray = read_gray(path, shape)
            started = clock()
            maps = flipped_predictions(predict, gray, flips)
            inference += clock() - started
            mean, std = flip_statistics(maps)
            writer.write(path.name, mean, std)
            if progress is not None:
                progress(index, len(frames), path)
    return ExportTiming(
        files=len(frames),
        inference_seconds=inference,
        total_seconds=clock() - begin,
    )


def write_manifest(
    root: str | Path, model: str, manifest: Mapping[str, object]
) -> Path:
    """Write how this model's maps were made. Returns the path."""
    path = model_dir(root, model) / MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(root: str | Path, model: str) -> dict[str, object]:
    """Read what :func:`write_manifest` wrote."""
    path = model_dir(root, model) / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_fingerprint(manifest: Mapping[str, object]) -> dict[str, object]:
    """The fields :mod:`filament.checkpoint` compares, as the export recorded them."""
    return {
        "epochs": manifest.get("epochs"),
        "encoder": manifest.get("encoder"),
        "tensor_sha256": manifest.get("tensor_sha256"),
    }


def find_cache_root(model: str, search_root: str | Path) -> Path:
    """Find the root of this model's cache under ``search_root``, at any depth.

    A Kaggle dataset can land at several depths under ``/kaggle/input``. The root
    is the parent of the directory ``<model>`` that holds a manifest. Directories
    named ``competitions`` are not walked. It raises when there is no such
    directory, and when there are several, since the wrong maps would silently
    change every result.
    """
    start = Path(search_root)
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(start, topdown=True):
        dirnames[:] = [name for name in dirnames if name != "competitions"]
        current = Path(dirpath)
        if current.name == model and MANIFEST_FILE in filenames:
            matches.append(current.parent)
            dirnames.clear()
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No {model}/{MANIFEST_FILE} under {start}. Attach the dataset made "
            "from the output of notebooks/export_probs.ipynb for this model."
        )
    listed = "\n".join(f"  {path}" for path in matches)
    raise RuntimeError(f"Several caches of {model} were found:\n{listed}")
