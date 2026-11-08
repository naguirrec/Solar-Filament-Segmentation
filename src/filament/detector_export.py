"""Detector output on disk: frame-coordinate masks as RLE, one file per split.

``notebooks/export_detections.ipynb`` predicts every window of a frame and keeps
every detection above a low score (0.01, untuned), so later score cuts cost
nothing. This module is the part that does not need the GPU:

* :func:`frame_detections` maps the masks of one window to the frame, with the
  window geometry of :mod:`filament.detector_data`, and returns
  :class:`filament.detector_fusion.Detection` objects.
* :class:`DetectionWriter` and :func:`read_detections` store them. One gzip file
  per split, one JSON line per frame, so a reader can stream it and a frame is
  never held with its neighbors. A detection is ``[window, score, bbox, rle]``,
  with the pycocotools compressed counts string as ``rle``
  (:mod:`filament.rle`).
* :func:`window_rects` gives the rectangles :func:`filament.detector_fusion.fuse`
  needs, and :func:`window_input` builds the detector's input for a window the
  way the training files were written.

Numpy only. The detector's masks come in as arrays, whatever produced them.
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np

from filament.detector_data import (
    JPEG_QUALITY,
    TARGET,
    Window,
    paste_window_mask,
    resize_window,
)
from filament.detector_fusion import Detection, Rect

EXPORT_MANIFEST = "detector_export_manifest.json"
FORMAT_VERSION = 1


def detections_name(split: str) -> str:
    """The file name of one split's detections."""
    if not split or "/" in split or split.startswith("."):
        raise ValueError(f"bad split name {split!r}")
    return f"detections_{split}.jsonl.gz"


def window_rects(windows: Sequence[Window]) -> list[Rect]:
    """Windows as ``(top, left, bottom, right)`` rectangles, in the same order."""
    return [(w.y0, w.x0, w.y1, w.x1) for w in windows]


def window_input(
    gray: np.ndarray,
    window: Window,
    target: int = TARGET,
    quality: int = JPEG_QUALITY,
) -> np.ndarray:
    """The detector's input for one window: ``(target, target, 3)`` RGB ``uint8``.

    The window is cut, resized with :func:`filament.detector_data.resize_window`
    and repeated into 3 channels, then passed through the JPEG encoder at
    ``quality`` and back, because that is how
    :func:`filament.detector_data.write_detector_dataset` stores the training
    windows. Predicting on the raw resize would show the detector a picture
    without the compression artefacts it was trained on.
    """
    frame = np.asarray(gray)
    if frame.ndim != 2 or frame.dtype != np.uint8:
        raise ValueError(f"gray must be 2-D uint8, got {frame.dtype} {frame.shape}")
    if window.x1 > frame.shape[1] or window.y1 > frame.shape[0]:
        raise ValueError(f"{window} does not fit in a frame of {frame.shape}")
    small = resize_window(frame[window.y0 : window.y1, window.x0 : window.x1], target)
    rgb = np.repeat(small[:, :, None], 3, axis=2)
    ok, encoded = cv2.imencode(
        ".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("could not JPEG-encode the window")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(decoded[:, :, ::-1])


def frame_detections(
    masks: np.ndarray,
    scores: Sequence[float] | np.ndarray,
    window_index: int,
    window: Window,
    frame_shape: tuple[int, int],
    min_score: float = 0.0,
) -> tuple[list[Detection], int]:
    """Detections of one window, as frame-sized RLE masks.

    ``masks`` is ``(n, h, w)``, boolean or 0/1, the detector's output for the
    window: ``h`` and ``w`` are the resized window or the window's own size, and
    :func:`filament.detector_data.paste_window_mask` resizes (nearest) into the
    frame. ``scores[i]`` belongs to ``masks[i]``. Detections under ``min_score``
    are skipped. Returns the detections, in input order, and how many were
    skipped because the mask was empty (before or after the resize): an empty mask
    is not a detection.
    """
    array = np.asarray(masks)
    values = np.asarray(scores, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"masks must be (n, h, w), got shape {array.shape}")
    if values.shape != (array.shape[0],):
        raise ValueError(f"{values.size} scores for {array.shape[0]} masks")
    found: list[Detection] = []
    empty = 0
    for mask, score in zip(array, values, strict=True):
        if score < min_score:
            continue
        if not mask.any():
            empty += 1
            continue
        frame = paste_window_mask(mask != 0, window, frame_shape)
        if not frame.any():
            empty += 1
            continue
        found.append(Detection.from_mask(frame != 0, float(score), window_index))
    return found, empty


class DetectionWriter:
    """Write the detections of many frames to one ``.jsonl.gz`` file.

    Use it as a context manager. Each :meth:`write_frame` is one line. Frames are
    written as they come, so a long export keeps what it has done if it is cut
    short, once the gzip stream is closed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Closed by close(), which the context manager calls.
        self._handle = gzip.open(  # noqa: SIM115
            self.path, "wt", encoding="utf-8", compresslevel=6
        )
        self.frames = 0
        self.detections = 0
        self._names: set[str] = set()

    def write_frame(
        self,
        file_name: str,
        shape: tuple[int, int],
        detections: Sequence[Detection],
    ) -> None:
        if file_name in self._names:
            raise ValueError(f"frame {file_name!r} written twice")
        for detection in detections:
            if detection.shape != tuple(shape):
                raise ValueError(
                    f"{file_name}: a detection has shape {detection.shape}, "
                    f"the frame {tuple(shape)}"
                )
        self._names.add(file_name)
        line = {
            "file": file_name,
            "shape": [int(shape[0]), int(shape[1])],
            "detections": [
                [d.window, round(d.score, 6), list(d.bbox), d.mask] for d in detections
            ],
        }
        self._handle.write(json.dumps(line, separators=(",", ":")) + "\n")
        self.frames += 1
        self.detections += len(detections)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def read_detections(
    path: str | Path,
) -> Iterator[tuple[str, tuple[int, int], list[Detection]]]:
    """Stream ``(file_name, frame_shape, detections)`` from a written file."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            shape = (int(row["shape"][0]), int(row["shape"][1]))
            detections = [
                Detection(
                    score=float(score),
                    mask=str(rle),
                    shape=shape,
                    window=int(window),
                    bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                )
                for window, score, box, rle in row["detections"]
            ]
            yield row["file"], shape, detections


def find_detection_export(search_root: str | Path, split: str) -> Path:
    """The directory under ``search_root`` that holds this split's detections.

    Found at any depth, since a Kaggle input can land at several. Directories
    named ``competitions`` are not walked. It raises when there is none, and when
    there are several, because the wrong detections would silently change every
    result.
    """
    start = Path(search_root)
    name = detections_name(split)
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(start, topdown=True):
        dirnames[:] = [d for d in dirnames if d != "competitions"]
        if name in filenames and EXPORT_MANIFEST in filenames:
            matches.append(Path(dirpath))
            dirnames.clear()
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No {name} next to {EXPORT_MANIFEST} under {start}. Attach the "
            "output of notebooks/export_detections.ipynb."
        )
    listed = "\n".join(f"  {path}" for path in sorted(matches))
    raise RuntimeError(f"Several detection exports with {name} were found:\n{listed}")
