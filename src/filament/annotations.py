"""Load Kaggle train annotations and rasterize filament masks.

The train set has no mask folder. ``load_train_annotations`` reads the
COCO-style JSON. ``rasterize_polygons`` is the only place that turns a
polygon segmentation into a mask, and it does that in memory.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_util

# Verified on Kaggle with find. There is no mask directory.
TRAIN_IMAGES_DIR = (
    "/kaggle/input/competitions/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026/train/train_images"
)
TRAIN_ANNOTATIONS_PATH = (
    "/kaggle/input/competitions/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026/train/"
    "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)

# Key names verified on the train JSON. The rasterizer below is the one
# place that interprets ``segmentation``.
TOP_LEVEL_KEYS = ("info", "licenses", "categories", "images", "annotations")
CATEGORY_KEYS = ("supercategory", "id", "name")
IMAGE_KEYS = (
    "license",
    "file_name",
    "url",
    "height",
    "width",
    "date_captured",
    "id",
)
ANNOTATION_KEYS = (
    "segmentation",
    "area",
    "iscrowd",
    "spine",
    "image_id",
    "bbox",
    "category_id",
    "id",
)

# Verified category table. Every id is one filament class for a mask.
# Nothing in this tuple is dropped. The submission CSV has no class column.
CATEGORIES = (
    {"id": 1, "name": "Left", "supercategory": "filament"},
    {"id": 2, "name": "Right", "supercategory": "filament"},
    {"id": 3, "name": "Unidentifiable", "supercategory": "filament"},
    {"id": 4, "name": "Ambiguous", "supercategory": "filament"},
)
FILAMENT_CATEGORY_IDS = tuple(category["id"] for category in CATEGORIES)


class AnnotationSchemaError(ValueError):
    """The JSON is not the verified train annotation object."""


@dataclass(frozen=True, slots=True)
class Category:
    """One chirality label from the train JSON."""

    id: int
    name: str
    supercategory: str


@dataclass(frozen=True, slots=True)
class FilamentAnnotation:
    """One filament. ``spine`` is the JSON value, unparsed.

    ``segmentation`` is a tuple of COCO polygons (flat ``x, y`` lists).
    Call :meth:`rasterize` for the mask. The mask is not stored.

    ``semantic_only`` holds other segmentations from the same file that
    belong in the semantic union and not in the stand-in panoptic mask.
    It is empty unless duplicate file names were collapsed.
    :meth:`rasterize` ignores it. Scoring does not use that stand-in
    mask; each annotator entry is a view on :class:`TrainImage`.
    """

    id: int | str
    image_id: int | str
    category_id: int
    spine: Any
    segmentation: tuple[tuple[float, ...], ...]
    semantic_only: tuple[tuple[tuple[float, ...], ...], ...] = ()

    def rasterize(self, height: int, width: int) -> np.ndarray:
        """Return this filament's uint8 mask of shape ``(height, width)``.

        ``semantic_only`` is not drawn. Those polygons are for the
        semantic union, not for this instance.
        """
        try:
            return rasterize_polygons(self.segmentation, height, width)
        except AnnotationSchemaError as exc:
            raise AnnotationSchemaError(f"annotation {self.id!r}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class TrainImage:
    """One train observation. Masks are rasterized when asked for.

    ``annotations`` on a collapsed file is the stand-in panoptic list:
    exact duplicate polygons are dropped, and cross-entry instances with
    IoU greater than 0.5 keep one mask. ``views`` keeps every original
    annotator entry. Scoring reads ``views``. The semantic union still
    reads ``annotations`` plus ``semantic_only``.
    """

    id: int | str
    file_name: str
    date_captured: str | None
    height: int
    width: int
    annotations: tuple[FilamentAnnotation, ...]
    views: tuple[tuple[FilamentAnnotation, ...], ...] = ()

    def split_key(self) -> str:
        """Date used to group this image, or ``file_name`` when it is missing."""
        return date_split_key(self.date_captured, self.file_name)

    def masks(self) -> tuple[tuple[FilamentAnnotation, np.ndarray], ...]:
        """Rasterize every annotation. One mask each. None are filtered."""
        return tuple(
            (annotation, annotation.rasterize(self.height, self.width))
            for annotation in self.annotations
        )


@dataclass(frozen=True, slots=True)
class TrainAnnotations:
    """Parsed train JSON. Polygons stay polygons until :meth:`TrainImage.masks`."""

    categories: tuple[Category, ...]
    images: tuple[TrainImage, ...]


def annotator_views(
    image: TrainImage,
) -> tuple[tuple[FilamentAnnotation, ...], ...]:
    """Return one annotation tuple per annotator entry.

    A collapsed file stores those entries on ``views``. An image that
    was not rebuilt has an empty ``views`` and its ``annotations`` are
    the single view. An empty tuple inside ``views`` is an annotator
    who drew nothing. It is still a view.
    """
    stored = getattr(image, "views", ())
    if stored:
        return tuple(tuple(view) for view in stored)
    return (tuple(getattr(image, "annotations", ())),)


def date_split_key(
    date_captured: str | None,
    file_name: str | None = None,
) -> str:
    """Return the observation label a date split should group on.

    Pass ``date_captured`` when the image has one. Pass ``file_name`` when
    ``date_captured`` is missing or blank. A non-blank date wins.
    """
    if date_captured is not None and not isinstance(date_captured, str):
        raise TypeError("date_captured must be a str or None")
    if isinstance(date_captured, str) and date_captured.strip():
        return date_captured
    if isinstance(file_name, str) and file_name:
        return file_name
    raise ValueError(
        "date_captured is missing; pass file_name to identify the observation"
    )


def load_train_annotations(path: str | Path) -> TrainAnnotations:
    """Read the train JSON. Does not rasterize masks and does not write files."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    dataset = _require_object(payload)
    categories = _categories(dataset["categories"])
    images = _images(dataset["images"])
    annotations = _annotations(dataset["annotations"], images)
    return TrainAnnotations(
        categories=categories,
        images=tuple(_with_annotations(image, annotations) for image in images),
    )


def rasterize_polygons(
    segmentation: Sequence[Sequence[float]],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize one annotation's polygons into a single uint8 mask.

    This is the only decoder. ``segmentation`` is a list of COCO polygons,
    each a flat sequence of ``x, y`` pixel coordinates. Pieces are unioned.
    Compressed RLE is rejected. The array is not written to disk.
    """
    polygons = [list(piece) for piece in _polygons(segmentation)]
    _require_size(height, width)
    try:
        encoded = mask_util.frPyObjects(polygons, height, width)
        if isinstance(encoded, list):
            decoded = mask_util.decode(mask_util.merge(encoded))
        else:
            decoded = mask_util.decode(encoded)
    except Exception as exc:
        raise AnnotationSchemaError(f"could not rasterize polygons: {exc}") from exc
    mask = np.asarray(decoded, dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != (height, width):
        raise AnnotationSchemaError(
            f"rasterized mask has shape {mask.shape}, expected {(height, width)}"
        )
    return np.ascontiguousarray(mask)


def _require_object(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AnnotationSchemaError(
            f"train annotations must be a JSON object, got {type(payload).__name__}"
        )
    missing = [key for key in TOP_LEVEL_KEYS if key not in payload]
    if missing:
        raise AnnotationSchemaError(
            "train annotations JSON is missing "
            f"{missing}; keys present: {sorted(payload)}"
        )
    return payload


def _require_keys(
    record: Mapping[str, Any], required: Sequence[str], kind: str
) -> None:
    missing = [key for key in required if key not in record]
    if missing:
        raise AnnotationSchemaError(
            f"{kind} is missing {missing}; keys present: {sorted(record)}"
        )


def _require_id(value: object, label: str) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise AnnotationSchemaError(f"{label} must be an int or a str")
    if isinstance(value, str) and not value:
        raise AnnotationSchemaError(f"{label} must be non-empty")
    return value


def _require_size(height: object, width: object) -> tuple[int, int]:
    for label, value in (("height", height), ("width", width)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AnnotationSchemaError(f"{label} must be a positive int")
    return int(height), int(width)


def _categories(payload: object) -> tuple[Category, ...]:
    if not isinstance(payload, list):
        raise AnnotationSchemaError(
            f"categories must be a list, got {type(payload).__name__}"
        )
    found: list[tuple[int, str, str]] = []
    categories: list[Category] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise AnnotationSchemaError(
                f"categories[{index}] must be an object, got {type(item).__name__}"
            )
        _require_keys(item, CATEGORY_KEYS, f"categories[{index}]")
        category_id = item["id"]
        name = item["name"]
        supercategory = item["supercategory"]
        if isinstance(category_id, bool) or not isinstance(category_id, int):
            raise AnnotationSchemaError(f"categories[{index}].id must be an int")
        if not isinstance(name, str) or not isinstance(supercategory, str):
            raise AnnotationSchemaError(
                f"categories[{index}] name and supercategory must be strings"
            )
        found.append((category_id, name, supercategory))
        categories.append(
            Category(id=category_id, name=name, supercategory=supercategory)
        )
    expected = tuple(
        (category["id"], category["name"], category["supercategory"])
        for category in CATEGORIES
    )
    if tuple(sorted(found)) != tuple(sorted(expected)):
        raise AnnotationSchemaError(
            f"categories do not match the verified Kaggle names; found {found}"
        )
    return tuple(categories)


def _images(payload: object) -> tuple[TrainImage, ...]:
    if not isinstance(payload, list):
        raise AnnotationSchemaError(
            f"images must be a list, got {type(payload).__name__}"
        )
    images: list[TrainImage] = []
    seen: set[int | str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise AnnotationSchemaError(
                f"images[{index}] must be an object, got {type(item).__name__}"
            )
        _require_keys(item, IMAGE_KEYS, f"images[{index}]")
        image_id = _require_id(item["id"], f"images[{index}].id")
        if image_id in seen:
            raise AnnotationSchemaError(f"duplicate image id {image_id!r}")
        seen.add(image_id)
        file_name = item["file_name"]
        if not isinstance(file_name, str) or not file_name:
            raise AnnotationSchemaError(
                f"images[{index}].file_name must be a non-empty str"
            )
        date_captured = item["date_captured"]
        if date_captured is not None and not isinstance(date_captured, str):
            raise AnnotationSchemaError(
                f"images[{index}].date_captured must be a str or null"
            )
        height, width = _require_size(item["height"], item["width"])
        images.append(
            TrainImage(
                id=image_id,
                file_name=file_name,
                date_captured=date_captured,
                height=height,
                width=width,
                annotations=(),
            )
        )
    return tuple(images)


def _annotations(
    payload: object,
    images: Sequence[TrainImage],
) -> dict[int | str, list[FilamentAnnotation]]:
    if not isinstance(payload, list):
        raise AnnotationSchemaError(
            f"annotations must be a list, got {type(payload).__name__}"
        )
    known = {image.id for image in images}
    grouped: dict[int | str, list[FilamentAnnotation]] = {
        image.id: [] for image in images
    }
    seen: set[int | str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise AnnotationSchemaError(
                f"annotations[{index}] must be an object, got {type(item).__name__}"
            )
        _require_keys(item, ANNOTATION_KEYS, f"annotations[{index}]")
        annotation_id = _require_id(item["id"], f"annotations[{index}].id")
        if annotation_id in seen:
            raise AnnotationSchemaError(f"duplicate annotation id {annotation_id!r}")
        seen.add(annotation_id)
        image_id = _require_id(item["image_id"], f"annotations[{index}].image_id")
        if image_id not in known:
            raise AnnotationSchemaError(
                f"annotations[{index}].image_id {image_id!r} does not match an image"
            )
        category_id = item["category_id"]
        if isinstance(category_id, bool) or category_id not in FILAMENT_CATEGORY_IDS:
            raise AnnotationSchemaError(
                f"annotations[{index}].category_id {category_id!r} "
                f"is outside {FILAMENT_CATEGORY_IDS}"
            )
        grouped[image_id].append(
            FilamentAnnotation(
                id=annotation_id,
                image_id=image_id,
                category_id=category_id,
                spine=item["spine"],
                segmentation=_polygons(item["segmentation"]),
            )
        )
    return grouped


def _with_annotations(
    image: TrainImage,
    grouped: Mapping[int | str, Sequence[FilamentAnnotation]],
) -> TrainImage:
    return TrainImage(
        id=image.id,
        file_name=image.file_name,
        date_captured=image.date_captured,
        height=image.height,
        width=image.width,
        annotations=tuple(grouped.get(image.id, ())),
    )


def _polygons(segmentation: object) -> tuple[tuple[float, ...], ...]:
    if isinstance(segmentation, dict):
        raise AnnotationSchemaError(
            "segmentation must be a list of polygons; compressed RLE is not "
            f"the train format (keys {sorted(segmentation)})"
        )
    if not isinstance(segmentation, (list, tuple)) or not segmentation:
        raise AnnotationSchemaError("segmentation must be a non-empty list of polygons")
    if isinstance(segmentation[0], (int, float)) and not isinstance(
        segmentation[0], bool
    ):
        raise AnnotationSchemaError(
            "segmentation must be a list of polygons, not one flat coordinate list"
        )
    polygons: list[tuple[float, ...]] = []
    for index, piece in enumerate(segmentation):
        if not isinstance(piece, (list, tuple)):
            raise AnnotationSchemaError(
                f"segmentation[{index}] must be a polygon list, "
                f"got {type(piece).__name__}"
            )
        if len(piece) < 6 or len(piece) % 2 != 0:
            raise AnnotationSchemaError(
                f"segmentation[{index}] must be an even-length list of x, y "
                f"coordinates (at least 3 points), got length {len(piece)}"
            )
        coords: list[float] = []
        for value in piece:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AnnotationSchemaError(
                    f"segmentation[{index}] coordinates must be numbers"
                )
            number = float(value)
            if not np.isfinite(number):
                raise AnnotationSchemaError(
                    f"segmentation[{index}] coordinates must be finite"
                )
            coords.append(number)
        polygons.append(tuple(coords))
    return tuple(polygons)
