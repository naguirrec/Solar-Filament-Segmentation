"""Train JSON loader: synthetic polygons in tmp, no real MAGFiLO files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament.annotations import (
    CATEGORIES,
    AnnotationSchemaError,
    date_split_key,
    load_train_annotations,
)

# Axis-aligned rectangle (1, 1)-(4, 3) on a 6×8 frame. pycocotools fills
# the half-open interior: rows 1-2, columns 1-3.
_RECTANGLE = [[1, 1, 4, 1, 4, 3, 1, 3]]
_RECTANGLE_MASK = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)

# Two overlapping pieces on a 5×5 frame. Union, not a sum.
_UNION = [
    [0, 0, 3, 0, 3, 2, 0, 2],
    [2, 1, 4, 1, 4, 3, 2, 3],
]
_UNION_MASK = np.array(
    [
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def _category_list() -> list[dict[str, object]]:
    return [dict(category) for category in CATEGORIES]


def _image(
    image_id: int,
    file_name: str,
    *,
    date_captured: str | None,
    height: int,
    width: int,
) -> dict[str, object]:
    return {
        "license": 1,
        "file_name": file_name,
        "url": "https://example.invalid/not-a-real-image",
        "height": height,
        "width": width,
        "date_captured": date_captured,
        "id": image_id,
    }


def _annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    segmentation: list[list[int]],
    spine: object,
) -> dict[str, object]:
    return {
        "segmentation": segmentation,
        "area": 1,
        "iscrowd": 0,
        "spine": spine,
        "image_id": image_id,
        "bbox": [0, 0, 1, 1],
        "category_id": category_id,
        "id": annotation_id,
    }


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dataset(
    images: list[dict[str, object]], annotations: list[dict[str, object]]
) -> dict:
    return {
        "info": {"version": "synthetic"},
        "licenses": [],
        "categories": _category_list(),
        "images": images,
        "annotations": annotations,
    }


def test_rectangle_polygon_rasterizes_to_known_mask(tmp_path: Path) -> None:
    spine = [0, 1, "unparsed"]
    path = _write(
        tmp_path,
        _dataset(
            [
                _image(
                    7,
                    "20220621185332Bh.jpeg",
                    date_captured="2022-06-21 18:53:32",
                    height=6,
                    width=8,
                )
            ],
            [_annotation(11, 7, 4, _RECTANGLE, spine)],
        ),
    )
    before = sorted(path.name for path in tmp_path.iterdir())

    loaded = load_train_annotations(path)

    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert loaded.categories[3].name == "Ambiguous"
    image = loaded.images[0]
    assert image.file_name == "20220621185332Bh.jpeg"
    assert image.date_captured == "2022-06-21 18:53:32"
    assert image.split_key() == "2022-06-21 18:53:32"
    assert len(image.annotations) == 1
    annotation = image.annotations[0]
    assert annotation.category_id == 4
    assert annotation.spine == spine
    pairs = image.masks()
    assert len(pairs) == 1
    assert pairs[0][0].category_id == 4
    assert pairs[0][1].dtype == np.uint8
    assert pairs[0][1].shape == (6, 8)
    assert np.array_equal(pairs[0][1], _RECTANGLE_MASK)


def test_all_four_categories_are_kept_and_spines_stay_raw(tmp_path: Path) -> None:
    spines = (
        [1, 2, 3],
        {"raw": True},
        "not-a-polyline",
        None,
    )
    path = _write(
        tmp_path,
        _dataset(
            [_image(1, "a.jpeg", date_captured=None, height=5, width=5)],
            [
                _annotation(index, 1, index, _UNION, spine)
                for index, spine in enumerate(spines, start=1)
            ],
        ),
    )

    image = load_train_annotations(path).images[0]

    assert image.split_key() == "a.jpeg"
    assert [item.category_id for item in image.annotations] == [1, 2, 3, 4]
    assert [item.spine for item in image.annotations] == list(spines)
    masks = image.masks()
    assert len(masks) == 4
    for _, mask in masks:
        assert np.array_equal(mask, _UNION_MASK)
        assert int(mask.max()) == 1


def test_date_split_key_uses_file_name_when_date_is_blank() -> None:
    assert date_split_key("2021-06-07 14:23:50", "ignored.jpeg") == (
        "2021-06-07 14:23:50"
    )
    assert date_split_key(None, "20220621185332Bh.jpeg") == "20220621185332Bh.jpeg"
    assert date_split_key("  ", "fallback.jpeg") == "fallback.jpeg"
    with pytest.raises(ValueError, match="file_name"):
        date_split_key(None)
    with pytest.raises(TypeError):
        date_split_key(20210607, "fallback.jpeg")  # type: ignore[arg-type]


def test_wrong_top_level_names_the_keys_present(tmp_path: Path) -> None:
    path = _write(tmp_path, {"info": {}, "notes": 1})
    with pytest.raises(AnnotationSchemaError, match="notes") as raised:
        load_train_annotations(path)
    message = str(raised.value)
    assert "images" in message
    assert "annotations" in message


def test_compressed_rle_is_rejected(tmp_path: Path) -> None:
    annotation = _annotation(1, 1, 1, _RECTANGLE, None)
    annotation["segmentation"] = {"size": [6, 8], "counts": "abc"}
    path = _write(
        tmp_path,
        _dataset(
            [_image(1, "a.jpeg", date_captured="2020-01-01", height=6, width=8)],
            [annotation],
        ),
    )
    with pytest.raises(AnnotationSchemaError, match="RLE"):
        load_train_annotations(path)


def test_category_names_must_match_the_verified_table(tmp_path: Path) -> None:
    payload = _dataset(
        [_image(1, "a.jpeg", date_captured="2020-01-01", height=6, width=8)],
        [],
    )
    payload["categories"] = [
        {"supercategory": "filament", "id": 1, "name": "Left"},
        {"supercategory": "filament", "id": 2, "name": "Right"},
        {"supercategory": "filament", "id": 3, "name": "Unidentifiable"},
        {"supercategory": "filament", "id": 4, "name": "Something else"},
    ]
    path = _write(tmp_path, payload)
    with pytest.raises(AnnotationSchemaError, match="Something else"):
        load_train_annotations(path)
