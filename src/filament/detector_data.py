"""Training data for the instance detector: windows, polygons, COCO folders.

Why windows exist. The detector (RF-DETR-Seg, ``rfdetr``, Apache 2.0) takes
inputs of at most about 768 px, and a frame is 2048 x 2048. Resizing the whole
frame to 768 divides every length by 2.7 and loses the thin barbs. So the
detector trains on overlapping windows of the frame, each resized to 768. The
default 1152 px window is resized by 0.667. The detector only groups and
scores the U-Net pixels (``docs/ROADMAP_V2.md``, phase F), so it does not need
full resolution.

Every default below is untuned. None of them was chosen on the holdout or on
the leaderboard:

* ``target=768`` is the largest input the detector accepts, not an optimum.
  With "4 quadrants" it is all that comes from the public Kaggle write-up
  cited in the roadmap (``docs/ROADMAP_V2.md``, section on phase F).
* ``window=1152`` and ``stride=896`` are this PR's choice. They give four
  quadrants of a 2048 frame with 256 px of overlap. The write-up gives only
  the four quadrants and 768. The 1152 px window, the 896 px stride and the
  256 px overlap come from neither it nor anything else in ``docs/``.
* ``min_visible=0.25`` drops a filament whose visible part is a sliver.
* ``min_area=DEFAULT_MIN_LABEL_AREA`` (100) counts pixels inside the window
  at native resolution, before the resize. It only drops clipped slivers. It
  is a detector-only value and not the U-Net's
  ``filament.baseline.DEFAULT_MIN_AREA``, which was chosen on the holdout
  sweep and so may not decide which labels reach detector training. 100 was
  not read from any sweep or holdout number.
* ``detector_val_fraction=0.1`` leaves the newest tenth of the remaining
  dates for detector validation.

Nothing in this module ran on real data. Tests use small synthetic frames.

Splits. The holdout is the date split the U-Net already uses
(:func:`filament.train_data.split_training_images`). It is never an input to a
detector training run or to checkpoint selection. Detector validation is the
newest dates of what remains, so no file and no date is in two splits.

Examples. Each annotator view of a file is one example per window: same
pixels, different labels (:func:`filament.annotations.annotator_views`). A
view with nothing in a window is a negative example. ``category_id`` is not
read: all filaments are one category.

Files. RF-DETR reads ``train``, ``valid`` and ``test`` folders, each with
``_annotations.coco.json`` next to its images. Image names are unique per
(file, window, view), so a file with three views writes three JPEG files per
window. The names require that. The window is cropped, resized and encoded
once, and the same bytes are written under each view's name. A hard link would
save disk, but not on every file system, so it is not used. The ``test``
folder is a copy of ``valid``: the holdout is never written to disk by this
module.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from filament.annotations import (
    FilamentAnnotation,
    TrainAnnotations,
    TrainImage,
    annotator_views,
    rasterize_polygons,
)
from filament.split import observation_date, split_by_date
from filament.train_data import DEFAULT_HOLDOUT_FRACTION, split_training_images

# Untuned. See the module docstring.
FRAME_SIZE = 2048
WINDOW = 1152
STRIDE = 896
TARGET = 768
MIN_VISIBLE = 0.25
# Pixels inside the window at native resolution. Untuned: it only drops
# clipped slivers. Not read from any holdout sweep, and not the U-Net's
# ``filament.baseline.DEFAULT_MIN_AREA``, which was.
DEFAULT_MIN_LABEL_AREA = 100
# The U-Net's holdout share, defined once in ``filament.train_data``.
HOLDOUT_VAL_FRACTION = DEFAULT_HOLDOUT_FRACTION
DETECTOR_VAL_FRACTION = 0.1
JPEG_QUALITY = 95

# One category for every filament. The detector does not classify chirality.
FILAMENT_CATEGORY_ID = 1
FILAMENT_CATEGORY_NAME = "filament"

ANNOTATIONS_NAME = "_annotations.coco.json"
TRAIN_DIR = "train"
VALID_DIR = "valid"
TEST_DIR = "test"

_Polygon = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Window:
    """A pixel box of a frame: columns ``x0:x1``, rows ``y0:y1``."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {value!r}")
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError(f"window origin must be >= 0, got {self}")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"window must have a positive size, got {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def frame_windows(
    frame_size: int = FRAME_SIZE,
    window: int = WINDOW,
    stride: int = STRIDE,
) -> tuple[Window, ...]:
    """Square windows that cover a square frame, row by row.

    Starts advance by ``stride`` along each axis. The last start along an
    axis is moved back so its window ends exactly on the frame edge, so
    nothing is padded and every pixel is covered. ``stride`` may not exceed
    ``window``, which would leave gaps. The defaults (2048, 1152, 896) give
    four quadrants that overlap by 256 px. They are this module's untuned
    choice (the module docstring says what the roadmap does give).
    """
    for name, value in (
        ("frame_size", frame_size),
        ("window", window),
        ("stride", stride),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if window > frame_size:
        raise ValueError(f"window {window} is larger than frame_size {frame_size}")
    if stride > window:
        raise ValueError(f"stride {stride} is larger than window {window}: gaps")
    last = frame_size - window
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return tuple(Window(x, y, x + window, y + window) for y in starts for x in starts)


def _scale(window: Window, target: int) -> tuple[float, float]:
    _require_target(target)
    return target / window.width, target / window.height


def _require_target(target: int) -> None:
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError(f"target must be a positive int, got {target!r}")


def frame_to_window_points(
    points: np.ndarray | Sequence[Sequence[float]],
    window: Window,
    target: int = TARGET,
) -> np.ndarray:
    """Map ``(n, 2)`` ``x, y`` frame coordinates into the resized window.

    Coordinates are continuous, pixel edges on integers, the convention of
    COCO polygons and of ``cv2.resize``. A window resized to ``target`` x
    ``target`` is scaled by ``target / width`` and ``target / height``.
    """
    array = _as_points(points)
    scale_x, scale_y = _scale(window, target)
    out = np.empty_like(array)
    out[:, 0] = (array[:, 0] - window.x0) * scale_x
    out[:, 1] = (array[:, 1] - window.y0) * scale_y
    return out


def window_to_frame_points(
    points: np.ndarray | Sequence[Sequence[float]],
    window: Window,
    target: int = TARGET,
) -> np.ndarray:
    """Inverse of :func:`frame_to_window_points`."""
    array = _as_points(points)
    scale_x, scale_y = _scale(window, target)
    out = np.empty_like(array)
    out[:, 0] = array[:, 0] / scale_x + window.x0
    out[:, 1] = array[:, 1] / scale_y + window.y0
    return out


def _as_points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"points must have shape (n, 2), got {array.shape}")
    return array


def resize_window(crop: np.ndarray, target: int = TARGET) -> np.ndarray:
    """Resize a window crop to ``target`` x ``target``.

    ``INTER_AREA`` when it shrinks, bilinear otherwise. Not for masks: use
    :func:`paste_window_mask`, which is nearest by default.
    """
    _require_target(target)
    height, width = crop.shape[:2]
    shrinking = target < max(height, width)
    method = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(crop, (target, target), interpolation=method)


def paste_window_mask(
    mask: np.ndarray,
    window: Window,
    frame_shape: tuple[int, int],
    out: np.ndarray | None = None,
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """Resize a window's mask to the window and paste it into a frame array.

    ``mask`` is the detector's output for the window, shape ``(t, t)``, or
    already the window's own ``(height, width)``. It is resized to the
    window (nearest by default; pass ``cv2.INTER_LINEAR`` for a probability
    map). Where ``out`` already holds values the larger one is kept, so
    overlapping windows combine by maximum, which is the union for a binary
    mask. With ``out=None`` a zero array of ``frame_shape`` is made, with the
    mask's dtype. A float mask cannot go into an integer or bool ``out``
    (``TypeError``): the cast would truncate probabilities to 0.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if window.x1 > width or window.y1 > height:
        raise ValueError(f"{window} does not fit in a frame of {frame_shape}")
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {array.shape}")
    was_bool = array.dtype == np.bool_
    if was_bool:
        array = array.astype(np.uint8)
    if array.shape != (window.height, window.width):
        array = cv2.resize(
            array, (window.width, window.height), interpolation=interpolation
        )
    if out is None:
        out = np.zeros((height, width), dtype=array.dtype)
    elif out.shape != (height, width):
        raise ValueError(f"out has shape {out.shape}, expected {(height, width)}")
    elif array.dtype.kind == "f" and out.dtype.kind in "iub":
        # astype would truncate a probability such as 0.9 to 0, silently.
        raise TypeError(
            f"a {array.dtype} mask cannot be pasted into an {out.dtype} out: "
            "values would be truncated; threshold the mask or pass a float out"
        )
    region = out[window.y0 : window.y1, window.x0 : window.x1]
    np.maximum(region, array.astype(out.dtype, copy=False), out=region)
    return out


@dataclass(frozen=True, slots=True)
class WindowInstance:
    """One filament inside one window, in resized window coordinates.

    ``polygons`` are COCO flat ``x, y`` lists, one per connected piece that
    survived the clip. The pieces are not split into several instances.
    ``area`` is in resized window pixels. ``visible`` is the share of the
    filament's frame area that lies inside the window.
    """

    annotation_id: int | str
    polygons: tuple[_Polygon, ...]
    bbox: tuple[float, float, float, float]
    area: float
    visible: float


def window_instances(
    annotations: Sequence[FilamentAnnotation],
    window: Window,
    target: int = TARGET,
    min_visible: float = MIN_VISIBLE,
    min_area: int = DEFAULT_MIN_LABEL_AREA,
    *,
    frame_shape: tuple[int, int] = (FRAME_SIZE, FRAME_SIZE),
) -> tuple[WindowInstance, ...]:
    """Filaments of one annotator view, clipped to ``window``.

    Each annotation is rasterized with
    :func:`filament.annotations.rasterize_polygons` on a ``frame_shape``
    canvas and cropped to the window. It is dropped when under
    ``min_visible`` of its pixels are inside the window, or when fewer than
    ``min_area`` pixels are (native-resolution pixels inside the window,
    counted before the resize; the default is untuned and is not the U-Net's
    cut, see the module docstring). The polygons
    come from ``cv2.findContours`` (outer contours only, so a hole is
    filled) and are mapped to the resized window. A clipped filament in
    several pieces stays one instance with several polygons. A contour with
    under 3 points is dropped, and so is an instance left with none.

    Contours run through the centers of the boundary pixels, so a polygon is
    up to half a pixel narrower than the mask on every side, and a piece one
    pixel wide has no polygon. ``frame_shape`` is ``(height, width)``.
    """
    _check_clip_args(window, target, min_visible, min_area, frame_shape)
    return _clip_instances(
        _rasterize_view(annotations, frame_shape),
        window,
        target,
        min_visible,
        min_area,
    )


class _Rasterized(NamedTuple):
    annotation_id: int | str
    mask: np.ndarray
    area: int


def _rasterize_view(
    annotations: Sequence[FilamentAnnotation],
    frame_shape: tuple[int, int],
) -> list[_Rasterized]:
    height, width = frame_shape
    rasterized = []
    for annotation in annotations:
        mask = rasterize_polygons(annotation.segmentation, height, width)
        rasterized.append(_Rasterized(annotation.id, mask, int(np.count_nonzero(mask))))
    return rasterized


def _clip_instances(
    rasterized: Sequence[_Rasterized],
    window: Window,
    target: int,
    min_visible: float,
    min_area: int,
) -> tuple[WindowInstance, ...]:
    scale_x, scale_y = _scale(window, target)
    found = []
    for item in rasterized:
        if item.area == 0:
            continue
        crop = item.mask[window.y0 : window.y1, window.x0 : window.x1]
        inside = int(np.count_nonzero(crop))
        visible = inside / item.area
        if inside == 0 or visible < min_visible or inside < min_area:
            continue
        polygons = _polygons(crop, window, target)
        if not polygons:
            continue
        points = np.concatenate([np.asarray(p).reshape(-1, 2) for p in polygons])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        found.append(
            WindowInstance(
                annotation_id=item.annotation_id,
                polygons=polygons,
                bbox=_rounded(x_min, y_min, x_max - x_min, y_max - y_min),
                area=round(inside * scale_x * scale_y, 2),
                visible=visible,
            )
        )
    return tuple(found)


def _polygons(crop: np.ndarray, window: Window, target: int) -> tuple[_Polygon, ...]:
    contours, _ = cv2.findContours(
        np.ascontiguousarray(crop), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    scale_x, scale_y = _scale(window, target)
    polygons = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float64)
        if len(points) < 3:
            continue
        # Contour points are boundary pixel indices of the crop, which starts
        # at the window origin. A pixel covers [i, i + 1), so its center is
        # i + 0.5. Scale into the resized window.
        mapped = (points + 0.5) * (scale_x, scale_y)
        polygons.append(tuple(round(float(v), 2) for v in mapped.reshape(-1)))
    return tuple(polygons)


def _rounded(*values: float) -> tuple[float, float, float, float]:
    x, y, w, h = (round(float(v), 2) for v in values)
    return x, y, w, h


def _check_clip_args(
    window: Window,
    target: int,
    min_visible: float,
    min_area: int,
    frame_shape: tuple[int, int],
) -> None:
    _require_target(target)
    if isinstance(min_visible, bool) or not 0.0 <= float(min_visible) <= 1.0:
        raise ValueError(f"min_visible must be in [0, 1], got {min_visible!r}")
    if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 0:
        raise ValueError(f"min_area must be a non-negative int, got {min_area!r}")
    height, width = frame_shape
    if window.x1 > width or window.y1 > height:
        raise ValueError(f"{window} does not fit in a frame of {frame_shape}")


@dataclass(frozen=True, slots=True)
class DetectorExample:
    """One (file, window, annotator view): a picture name and its labels.

    No pixels are stored. ``image_name`` is the file the example is written
    under, ``<stem>__w<window_index>__v<view_index>.jpg``.
    """

    file_name: str
    window_index: int
    view_index: int
    window: Window
    target: int
    instances: tuple[WindowInstance, ...]

    @property
    def image_name(self) -> str:
        stem = Path(self.file_name).stem
        return f"{stem}__w{self.window_index}__v{self.view_index}.jpg"


def image_examples(
    image: TrainImage,
    windows: Sequence[Window],
    target: int = TARGET,
    min_visible: float = MIN_VISIBLE,
    min_area: int = DEFAULT_MIN_LABEL_AREA,
) -> tuple[DetectorExample, ...]:
    """Every (window, view) example of one file, window by window.

    One example per annotator view, as :func:`annotator_views` defines them.
    A file with three views gives three examples per window with the same
    pixels and different labels. A view with no filament in a window is a
    negative example, not skipped. ``semantic_only`` polygons are not used:
    they belong to the semantic union, not to an annotator.
    """
    frame_shape = (image.height, image.width)
    views = annotator_views(image)
    if not windows:
        raise ValueError("windows is empty")
    for window in windows:
        _check_clip_args(window, target, min_visible, min_area, frame_shape)
    rasterized = [_rasterize_view(view, frame_shape) for view in views]
    examples = []
    for window_index, window in enumerate(windows):
        for view_index, view in enumerate(rasterized):
            instances = _clip_instances(view, window, target, min_visible, min_area)
            examples.append(
                DetectorExample(
                    file_name=image.file_name,
                    window_index=window_index,
                    view_index=view_index,
                    window=window,
                    target=target,
                    instances=instances,
                )
            )
    return tuple(examples)


class DetectorSplits(NamedTuple):
    """File-level splits. ``holdout`` is not for detector training or tuning."""

    train: tuple[TrainImage, ...]
    val: tuple[TrainImage, ...]
    holdout: tuple[TrainImage, ...]


def build_detector_splits(
    train_annotations: TrainAnnotations | Sequence[TrainImage],
    val_fraction: float = HOLDOUT_VAL_FRACTION,
    detector_val_fraction: float = DETECTOR_VAL_FRACTION,
    *,
    allow_nondefault_holdout: bool = False,
) -> DetectorSplits:
    """Holdout, detector validation and detector train, split by date.

    The holdout is :func:`filament.train_data.split_training_images` with
    ``val_fraction``, the split the U-Net uses. It is not a second
    implementation. ``val_fraction`` must equal the repo's default
    (:data:`HOLDOUT_VAL_FRACTION`, 0.2): any other value raises
    ``ValueError`` unless the caller passes ``allow_nondefault_holdout=True``,
    because a smaller holdout would put dates that the U-Net's holdout holds
    out into detector train and val. The override exists for experiments;
    then the detector holdout is not the U-Net's.

    The detector validation set is the newest ``detector_val_fraction`` of the
    dates that remain (:func:`filament.split.split_by_date`) and the rest is
    detector train. Every date is in one split only, so are files. Repeated
    file names are one file, as in the U-Net split. Order is oldest date
    first. Nothing here reads annotations beyond what that split reads.

    Neither the holdout nor anything derived from it may feed detector
    training or checkpoint choice: select checkpoints on ``val``.
    :func:`write_detector_dataset` refuses a split whose holdout shares a file
    or a date with ``train`` or ``val``.
    """
    if val_fraction != HOLDOUT_VAL_FRACTION and not allow_nondefault_holdout:
        raise ValueError(
            f"val_fraction {val_fraction!r} is not the repo's holdout "
            f"({HOLDOUT_VAL_FRACTION}); pass allow_nondefault_holdout=True to "
            "build a different split on purpose"
        )
    images = getattr(train_annotations, "images", train_annotations)
    outer = split_training_images(tuple(images), val_fraction=val_fraction)
    by_name = {image.file_name: image for image in outer.train}
    payload = [
        {"file_name": image.file_name, "date_captured": image.date_captured}
        for image in outer.train
    ]
    inner = split_by_date(payload, val_fraction=detector_val_fraction)
    return DetectorSplits(
        train=tuple(by_name[name] for name in inner.train),
        val=tuple(by_name[name] for name in inner.val),
        holdout=outer.holdout,
    )


def split_dates(images: Sequence[TrainImage]) -> frozenset[str]:
    """Calendar dates (ISO strings) that appear in ``images``."""
    return frozenset(
        observation_date(
            {"file_name": image.file_name, "date_captured": image.date_captured}
        ).isoformat()
        for image in images
    )


def coco_payload(examples: Sequence[DetectorExample]) -> dict:
    """A COCO detection/segmentation dict for these examples.

    One category (:data:`FILAMENT_CATEGORY_ID`). Image ids and annotation ids
    count from 1 in the order of ``examples``. Each image is ``target`` x
    ``target``. ``image_name`` values must be unique. Extra image keys
    (``frame_file_name``, ``window``, ``view``) trace an image back to its
    frame, and COCO readers ignore them.
    """
    images = []
    annotations = []
    seen: set[str] = set()
    for image_id, example in enumerate(examples, start=1):
        name = example.image_name
        if name in seen:
            raise ValueError(f"duplicate image name {name!r}")
        seen.add(name)
        window = example.window
        images.append(
            {
                "id": image_id,
                "file_name": name,
                "width": example.target,
                "height": example.target,
                "frame_file_name": example.file_name,
                "window": [window.x0, window.y0, window.x1, window.y1],
                "view": example.view_index,
            }
        )
        for instance in example.instances:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": FILAMENT_CATEGORY_ID,
                    "segmentation": [list(polygon) for polygon in instance.polygons],
                    "bbox": list(instance.bbox),
                    "area": instance.area,
                    "iscrowd": 0,
                }
            )
    return {
        "info": {"description": "Solar filament detector windows"},
        "licenses": [],
        "categories": [
            {
                "id": FILAMENT_CATEGORY_ID,
                "name": FILAMENT_CATEGORY_NAME,
                "supercategory": "filament",
            }
        ],
        "images": images,
        "annotations": annotations,
    }


class FolderCounts(NamedTuple):
    """Files and instances written to one folder."""

    files: int
    instances: int


def _check_holdout_is_apart(splits: DetectorSplits) -> None:
    """Refuse a holdout file or date that is also in train or val."""
    holdout_files = {i.file_name for i in splits.holdout}
    holdout_dates = split_dates(splits.holdout)
    for name, images in (("train", splits.train), ("val", splits.val)):
        shared = holdout_files & {i.file_name for i in images}
        if shared:
            raise ValueError(f"holdout files in {name}: {sorted(shared)[:3]}")
        shared_dates = holdout_dates & split_dates(images)
        if shared_dates:
            raise ValueError(f"holdout dates in {name}: {sorted(shared_dates)[:3]}")


def _check_coverage(windows: Sequence[Window], height: int, width: int) -> None:
    """Raise ``ValueError`` unless ``windows`` cover every pixel of the image.

    The union of the boxes, clipped to ``height`` x ``width``, is checked on
    the grid their edges make, so the cost does not depend on the image size.
    """
    xs = sorted({0, width, *(min(w.x0, width) for w in windows)}
                | {min(w.x1, width) for w in windows})  # fmt: skip
    ys = sorted({0, height, *(min(w.y0, height) for w in windows)}
                | {min(w.y1, height) for w in windows})  # fmt: skip
    for y_low, y_high in pairwise(ys):
        for x_low, x_high in pairwise(xs):
            if not any(
                w.x0 <= x_low and x_high <= w.x1 and w.y0 <= y_low and y_high <= w.y1
                for w in windows
            ):
                raise ValueError(
                    f"windows do not cover a {height} x {width} image: pixels "
                    f"x {x_low}:{x_high}, y {y_low}:{y_high} are in no window"
                )


def write_detector_dataset(
    root: str | Path,
    splits: DetectorSplits,
    load_frame: Callable[[TrainImage], np.ndarray],
    windows: Sequence[Window] | None = None,
    target: int = TARGET,
    min_visible: float = MIN_VISIBLE,
    min_area: int = DEFAULT_MIN_LABEL_AREA,
    jpeg_quality: int = JPEG_QUALITY,
) -> Mapping[str, FolderCounts]:
    """Write the RF-DETR folder layout and return counts per folder.

    ``root/train`` holds ``splits.train`` and ``root/valid`` holds
    ``splits.val``. ``root/test`` is a copy of ``valid``: RF-DETR wants the
    folder. Only the names and dates of ``splits.holdout`` are read, to refuse an
    overlap; no holdout frame or label is loaded, so none reaches a training
    run. Each folder gets ``_annotations.coco.json`` and
    one JPEG per example (see the module docstring on why a file with three
    views writes three files per window, from one encoding).

    ``load_frame(image)`` returns the full frame as 2-D ``uint8``. It is
    called once per file. Images are 8-bit, 3 channels, the gray frame
    repeated. ``windows`` defaults to :func:`frame_windows`. Raises
    ``FileExistsError`` when a target folder already has files, and
    ``ValueError`` when

    * a file is in both ``train`` and ``val``,
    * a holdout file, or a holdout calendar date, is in ``train`` or ``val``
      (nothing is written then), or
    * the windows do not cover every pixel of an image of ``train`` or
      ``val``, judged against that image's own height and width.
    """
    root_path = Path(root)
    used = tuple(windows) if windows is not None else frame_windows()
    overlap = {i.file_name for i in splits.train} & {i.file_name for i in splits.val}
    if overlap:
        raise ValueError(f"files in both train and val: {sorted(overlap)[:3]}")
    _check_holdout_is_apart(splits)
    for height, width in sorted(
        {(i.height, i.width) for i in splits.train + splits.val}
    ):
        _check_coverage(used, height, width)
    for folder in (TRAIN_DIR, VALID_DIR, TEST_DIR):
        existing = root_path / folder
        if existing.exists() and any(existing.iterdir()):
            raise FileExistsError(f"{existing} is not empty")

    counts = {}
    for folder, images in ((TRAIN_DIR, splits.train), (VALID_DIR, splits.val)):
        counts[folder] = _write_folder(
            root_path / folder,
            images,
            load_frame,
            used,
            target,
            min_visible,
            min_area,
            jpeg_quality,
        )
    shutil.copytree(root_path / VALID_DIR, root_path / TEST_DIR, dirs_exist_ok=True)
    counts[TEST_DIR] = counts[VALID_DIR]
    return counts


def _write_folder(
    folder: Path,
    images: Sequence[TrainImage],
    load_frame: Callable[[TrainImage], np.ndarray],
    windows: Sequence[Window],
    target: int,
    min_visible: float,
    min_area: int,
    jpeg_quality: int,
) -> FolderCounts:
    folder.mkdir(parents=True, exist_ok=True)
    options = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    all_examples: list[DetectorExample] = []
    for image in images:
        examples = image_examples(image, windows, target, min_visible, min_area)
        frame = np.asarray(load_frame(image))
        if frame.dtype != np.uint8 or frame.ndim != 2:
            raise ValueError(
                f"{image.file_name}: frame must be 2-D uint8, got "
                f"{frame.dtype} with shape {frame.shape}"
            )
        if frame.shape != (image.height, image.width):
            raise ValueError(
                f"{image.file_name}: frame shape {frame.shape} differs from "
                f"{(image.height, image.width)}"
            )
        by_window: dict[int, list[DetectorExample]] = {}
        for example in examples:
            by_window.setdefault(example.window_index, []).append(example)
        for window_index, window in enumerate(windows):
            crop = frame[window.y0 : window.y1, window.x0 : window.x1]
            small = resize_window(crop, target)
            rgb = np.repeat(small[:, :, None], 3, axis=2)
            ok, encoded = cv2.imencode(".jpg", rgb, options)
            if not ok:
                raise RuntimeError(f"could not encode window {window_index}")
            data = encoded.tobytes()
            for example in by_window[window_index]:
                (folder / example.image_name).write_bytes(data)
        all_examples.extend(examples)
    payload = coco_payload(all_examples)
    (folder / ANNOTATIONS_NAME).write_text(json.dumps(payload), encoding="utf-8")
    return FolderCounts(len(payload["images"]), len(payload["annotations"]))
