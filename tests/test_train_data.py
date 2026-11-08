"""Semantic targets and the date holdout. Synthetic polygons, no images."""

from __future__ import annotations

import datetime as dt
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament import train_data
from filament.annotations import FilamentAnnotation, TrainImage, rasterize_polygons
from filament.instances import labels_from_semantic, starting_min_area
from filament.split import observation_date, split_by_date
from filament.train_data import (
    FULL_DISK_SIDE,
    TRAIN_SIDE,
    downscale_pair,
    instance_pq,
    semantic_target,
    semantic_target_from_annotations,
    split_training_images,
)

# Same rectangle the loader tests rasterize: rows 1-2, columns 1-3 on 6×8.
_LEFT = ((1.0, 1.0, 4.0, 1.0, 4.0, 3.0, 1.0, 3.0),)
# Overlaps _LEFT on column 3.
_RIGHT = ((3.0, 1.0, 6.0, 1.0, 6.0, 3.0, 3.0, 3.0),)

# 12×12 frame. Counts below are what rasterize_polygons produces.
# 6×6 at the origin, 36 pixels.
_BLOCK = ((0.0, 0.0, 6.0, 0.0, 6.0, 6.0, 0.0, 6.0),)
# One pixel down and right. IoU with _BLOCK is above one half, and each
# mask has foreground the other lacks.
_BLOCK_SHIFT = ((1.0, 1.0, 7.0, 1.0, 7.0, 7.0, 1.0, 7.0),)
# Disjoint from _BLOCK.
_CORNER = ((8.0, 8.0, 11.0, 8.0, 11.0, 11.0, 8.0, 11.0),)
# Shifted two pixels. Raster IoU with _BLOCK is exactly 1/2.
_HALF = ((2.0, 0.0, 8.0, 0.0, 8.0, 6.0, 2.0, 6.0),)
# Shifted three pixels. Raster IoU with _BLOCK is 1/3.
_THIRD = ((3.0, 0.0, 9.0, 0.0, 9.0, 6.0, 3.0, 6.0),)
# 5×5, and the 5×4 inside it. IoU is above one half. Not the same polygon.
_TALL = ((0.0, 0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0),)
_SHORT = ((0.0, 0.0, 5.0, 0.0, 5.0, 4.0, 0.0, 4.0),)
_CANVAS = 12


class _UnreadCategory:
    """Stands in for an annotation. Reading ``category_id`` fails the test."""

    def __init__(self, segmentation: tuple[tuple[float, ...], ...]) -> None:
        self.segmentation = segmentation

    @property
    def category_id(self) -> int:
        raise AssertionError("category_id must stay unread")


def _annotation(
    annotation_id: int, category_id: int, segmentation: tuple[tuple[float, ...], ...]
) -> FilamentAnnotation:
    return FilamentAnnotation(
        id=annotation_id,
        image_id=1,
        category_id=category_id,
        spine=None,
        segmentation=segmentation,
    )


def _image(
    image_id: int,
    *,
    date_captured: str | None,
    file_name: str,
) -> TrainImage:
    return TrainImage(
        id=image_id,
        file_name=file_name,
        date_captured=date_captured,
        height=4,
        width=4,
        annotations=(),
    )


def test_resolution_is_an_untuned_quarter_of_the_disk() -> None:
    assert TRAIN_SIDE == 512
    assert FULL_DISK_SIDE == 2048
    assert FULL_DISK_SIDE % TRAIN_SIDE == 0
    assert train_data.__doc__ is not None
    assert "untuned" in train_data.__doc__


def test_source_calls_split_by_date_and_does_not_import_torch() -> None:
    source = Path(train_data.__file__).read_text(encoding="utf-8")
    assert "split_by_date" in source
    assert "date_split_key" not in source
    assert "torch" not in source
    assert "cv2" not in source


def test_empty_segmentations_are_an_all_zero_mask() -> None:
    mask = semantic_target([], 3, 5)
    assert mask.shape == (3, 5)
    assert mask.dtype == np.uint8
    assert int(mask.sum()) == 0
    again = semantic_target_from_annotations((), 3, 5)
    assert np.array_equal(again, mask)


def test_semantic_target_unions_every_category_without_reading_it() -> None:
    segmentations = (_LEFT, _RIGHT)
    expected = np.maximum(
        rasterize_polygons(_LEFT, 6, 8),
        rasterize_polygons(_RIGHT, 6, 8),
    )
    overlap = (rasterize_polygons(_LEFT, 6, 8) > 0) & (
        rasterize_polygons(_RIGHT, 6, 8) > 0
    )
    assert overlap.any()

    got = semantic_target(segmentations, 6, 8)
    assert got.dtype == np.uint8
    assert int(got.max()) == 1
    assert np.array_equal(got, expected)
    assert np.array_equal(got[overlap], np.ones(int(overlap.sum()), dtype=np.uint8))

    annotations = (
        _annotation(1, 1, _LEFT),
        _annotation(2, 4, _RIGHT),
    )
    assert np.array_equal(
        semantic_target_from_annotations(annotations, 6, 8),
        expected,
    )
    guarded = semantic_target_from_annotations(
        (_UnreadCategory(_LEFT), _UnreadCategory(_RIGHT)),
        6,
        8,
    )
    assert np.array_equal(guarded, expected)


def test_semantic_target_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="height"):
        semantic_target([], 0, 4)
    with pytest.raises(ValueError, match="width"):
        semantic_target([], 4, False)  # type: ignore[arg-type]


def test_area_and_nearest_downscale_a_divisible_pair() -> None:
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint8)
    # Corners of a stride-2 grid are 0 at the sampled centers except as set.
    mask[0, 0] = 1
    mask[0, 2] = 1
    mask[1, 1] = 1
    mask[1, 3] = 1
    mask[3, 1] = 1
    # mask[3, 3] stays 0. A mean of this block would not be binary.

    small_image, small_mask = downscale_pair(image, mask, side=2)

    assert small_image.dtype == np.float32
    assert small_image.shape == (2, 2)
    assert np.allclose(
        small_image,
        np.array([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32),
    )
    assert small_mask.dtype == np.uint8
    assert np.array_equal(small_mask, np.array([[1, 1], [1, 0]], dtype=np.uint8))
    assert set(np.unique(small_mask)).issubset({0, 1})


def test_channels_stay_separate_under_area_averaging() -> None:
    image = np.array(
        [
            [[0, 0, 0], [2, 0, 0]],
            [[0, 4, 0], [2, 0, 8]],
        ],
        dtype=np.uint8,
    )
    mask = np.zeros((2, 2), dtype=np.uint8)
    small_image, small_mask = downscale_pair(image, mask, side=1)
    assert small_image.shape == (1, 1, 3)
    assert np.allclose(small_image[0, 0], np.array([1.0, 1.0, 2.0]))
    assert small_mask.shape == (1, 1)


def test_bilinear_is_used_when_the_side_does_not_divide() -> None:
    # 3×3 to 2×2 is not an integer block. Pixel (0, 0) lands at source
    # (0.25, 0.25): 0.75*0.75*1 + 0.75*0.25*2 + 0.25*0.75*4 + 0.25*0.25*5 = 2.
    image = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 1
    mask[0, 2] = 1
    mask[1, 1] = 1
    mask[2, 0] = 1
    small_image, small_mask = downscale_pair(image, mask, side=2)
    assert np.allclose(
        small_image,
        np.array([[2.0, 3.5], [6.5, 8.0]], dtype=np.float32),
    )
    # Centers of the output pixels read input indexes 0 and 2, not 1.
    assert np.array_equal(small_mask, np.array([[1, 1], [1, 0]], dtype=np.uint8))


def test_downscale_rejects_a_mismatch_or_an_upscale() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="differ"):
        downscale_pair(image, mask, side=2)
    with pytest.raises(ValueError, match="downscale"):
        downscale_pair(image, np.zeros((4, 4), dtype=np.uint8), side=8)
    with pytest.raises(ValueError, match="side"):
        downscale_pair(image, np.zeros((4, 4), dtype=np.uint8), side=True)  # type: ignore[arg-type]


def test_training_split_matches_split_by_date() -> None:
    # Newest day first, two clocks on that day, and one file name with no
    # date_captured. The sides must be whatever split_by_date returns.
    images = [
        _image(1, date_captured="2022-01-10 22:00:00", file_name="late-b.jpeg"),
        _image(2, date_captured="2022-01-10 01:00:00", file_name="late-a.jpeg"),
        _image(3, date_captured=None, file_name="20220101120000Bh.jpeg"),
        _image(4, date_captured="2022-01-09 12:00:00", file_name="ninth.jpeg"),
        _image(5, date_captured="  ", file_name="20220102120000Bh.jpeg"),
    ]
    for day in range(3, 9):
        images.append(
            _image(
                day + 3,
                date_captured=f"2022-01-{day:02d} 12:00:00",
                file_name=f"202201{day:02d}120000Bh.jpeg",
            )
        )

    split = split_training_images(images)
    expected = split_by_date(
        [
            {"file_name": image.file_name, "date_captured": image.date_captured}
            for image in images
        ],
        val_fraction=0.2,
    )

    assert tuple(image.file_name for image in split.train) == expected.train
    assert tuple(image.file_name for image in split.holdout) == expected.val
    by_name = {image.file_name: image for image in images}
    assert all(image is by_name[image.file_name] for image in split.train)
    assert all(image is by_name[image.file_name] for image in split.holdout)
    # Both clocks on 10 January stay on the holdout side together.
    holdout_names = {image.file_name for image in split.holdout}
    assert {"late-a.jpeg", "late-b.jpeg"} <= holdout_names
    train_ids = {image.id for image in split.train}
    holdout_ids = {image.id for image in split.holdout}
    assert train_ids.isdisjoint(holdout_ids)
    assert train_ids | holdout_ids == {image.id for image in images}


def test_training_split_passes_only_file_name_and_date(monkeypatch) -> None:
    seen: dict[str, object] = {}
    real = split_by_date

    def spy(images, val_fraction=0.2):
        seen["images"] = list(images)
        seen["fraction"] = val_fraction
        return real(seen["images"], val_fraction)

    monkeypatch.setattr(train_data, "split_by_date", spy)
    images = (
        _image(1, date_captured="2022-06-21 18:53:32", file_name="newer.jpeg"),
        _image(2, date_captured=None, file_name="20150125172714Mh.jpeg"),
    )
    split_training_images(images)
    payload = seen["images"]
    assert isinstance(payload, list)
    assert seen["fraction"] == 0.2
    assert [set(record) for record in payload] == [
        {"file_name", "date_captured"},
        {"file_name", "date_captured"},
    ]
    assert payload[0]["date_captured"] == "2022-06-21 18:53:32"
    assert payload[1]["date_captured"] is None


def test_one_date_is_rejected() -> None:
    one = (_image(1, date_captured="2015-01-01 00:00:00", file_name="only.jpeg"),)
    with pytest.raises(ValueError, match="two distinct observation dates"):
        split_training_images(one)


class _GuardAnnotation:
    """Rasterizes a block. Reading ``category_id`` fails the test."""

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def rasterize(self, height: int, width: int) -> np.ndarray:
        if (height, width) != self._mask.shape:
            raise ValueError("rasterize size does not match the stored mask")
        return self._mask

    @property
    def category_id(self) -> int:
        raise AssertionError("category_id must stay unread")


class _GuardImage:
    def __init__(self, annotations: tuple[object, ...], side: int) -> None:
        self.annotations = annotations
        self.height = side
        self.width = side


def test_instance_pq_matches_downscaled_ground_truth() -> None:
    full = np.zeros((8, 8), dtype=np.uint8)
    full[2:6, 2:6] = 1
    image = _GuardImage((_GuardAnnotation(full),), side=8)
    _gray, small = downscale_pair(np.full((8, 8), 200, dtype=np.uint8), full, side=4)
    gray = np.full(small.shape, 200, dtype=np.uint8)
    assert int(small.sum()) > 0
    assert instance_pq(small, gray, image) == 1.0


def test_instance_pq_uses_the_module_area_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    # 256×256 scales the 400 px default to an area above one pixel.
    # A one-pixel prediction must not be kept by a cutoff this module invents.
    side = 256
    assert starting_min_area(side, side) > 1
    seen: dict[str, object] = {}
    real = labels_from_semantic

    def spy(semantic, gray, *, min_area=None, close_size=None):
        seen["min_area"] = min_area
        seen["close_size"] = close_size
        return real(semantic, gray, min_area=min_area, close_size=close_size)

    monkeypatch.setattr(train_data, "labels_from_semantic", spy)
    speck = np.zeros((side, side), dtype=np.uint8)
    speck[side // 2, side // 2] = 1
    gray = np.full((side, side), 200, dtype=np.uint8)
    image = _GuardImage((_GuardAnnotation(speck),), side=side)
    assert instance_pq(speck, gray, image) == 0.0
    assert seen["min_area"] is None
    assert seen["close_size"] is None


def test_empty_prediction_and_empty_ground_truth_are_nan() -> None:
    gray = np.full((16, 16), 200, dtype=np.uint8)
    semantic = np.zeros((16, 16), dtype=np.uint8)
    score = instance_pq(semantic, gray, _GuardImage((), side=16))
    assert math.isnan(score)


def _disk(
    image_id: int,
    annotations: tuple[FilamentAnnotation, ...],
    *,
    file_name: str = "20150101120000Bh.jpeg",
    date_captured: str | None = "2015-01-01 12:00:00",
    height: int = _CANVAS,
    width: int = _CANVAS,
) -> TrainImage:
    return TrainImage(
        id=image_id,
        file_name=file_name,
        date_captured=date_captured,
        height=height,
        width=width,
        annotations=annotations,
    )


def _other_day() -> TrainImage:
    return TrainImage(
        id=50,
        file_name="20150102120000Bh.jpeg",
        date_captured="2015-01-02 00:00:00",
        height=4,
        width=4,
        annotations=(),
    )


def _only(split: train_data.TrainingSplit, file_name: str) -> TrainImage:
    found = [
        image
        for image in (*split.train, *split.holdout)
        if image.file_name == file_name
    ]
    assert len(found) == 1
    return found[0]


def _logged_counts(caplog: pytest.LogCaptureFixture) -> dict[str, int]:
    messages = [
        record.getMessage()
        for record in caplog.records
        if "duplicated_file_names=" in record.getMessage()
    ]
    assert len(messages) == 1
    counts: dict[str, int] = {}
    for token in messages[0].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        counts[key] = int(value)
    return counts


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero((left != 0) & (right != 0)))
    union = int(np.count_nonzero((left != 0) | (right != 0)))
    if union == 0:
        return 0.0
    return intersection / union


def _enable_collapse_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="filament.train_data")


def test_exact_duplicate_inside_one_entry_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_collapse_logs(caplog)
    partner = _other_day()
    split = split_training_images(
        (
            _disk(1, (_annotation(1, 1, _BLOCK), _annotation(2, 1, _BLOCK))),
            partner,
        )
    )
    collapsed = _only(split, "20150101120000Bh.jpeg")
    assert len(collapsed.annotations) == 1
    assert collapsed.annotations[0].segmentation == _BLOCK
    assert len(collapsed.views) == 1
    assert len(collapsed.views[0]) == 2
    counts = _logged_counts(caplog)
    assert counts["exact_polygons_dropped"] == 1
    assert counts["near_duplicate_instances_merged"] == 0
    assert counts["duplicated_file_names"] == 0
    assert _only(split, partner.file_name) is partner


def test_exact_duplicate_polygon_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_collapse_logs(caplog)
    polygon = _BLOCK
    partner = _other_day()
    split = split_training_images(
        (
            _disk(1, (_annotation(1, 1, polygon),)),
            _disk(2, (_annotation(2, 1, polygon),)),
            partner,
        )
    )
    collapsed = _only(split, "20150101120000Bh.jpeg")
    assert collapsed.annotations[0].segmentation == polygon
    assert len(collapsed.annotations) == 1
    assert len(collapsed.views) == 2
    assert collapsed.views[0][0].segmentation == polygon
    assert collapsed.views[1][0].segmentation == polygon
    assert _logged_counts(caplog) == {
        "duplicated_file_names": 1,
        "extra_image_entries": 1,
        "exact_polygons_dropped": 1,
        "near_duplicate_instances_merged": 0,
    }
    assert _only(split, partner.file_name) is partner


def test_near_duplicate_entries_become_one_instance_and_the_union_covers_both(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_collapse_logs(caplog)
    left = rasterize_polygons(_BLOCK, _CANVAS, _CANVAS)
    right = rasterize_polygons(_BLOCK_SHIFT, _CANVAS, _CANVAS)
    assert _mask_iou(left, right) > 0.5
    assert np.any((left != 0) & (right == 0))
    assert np.any((right != 0) & (left == 0))

    split = split_training_images(
        (
            _disk(1, (_annotation(1, 1, _BLOCK),)),
            _disk(2, (_annotation(2, 2, _BLOCK_SHIFT),)),
            _other_day(),
        )
    )
    collapsed = _only(split, "20150101120000Bh.jpeg")
    assert len(collapsed.annotations) == 1
    kept = collapsed.annotations[0]
    assert kept.segmentation == _BLOCK
    assert len(collapsed.views) == 2
    assert collapsed.views[0][0].segmentation == _BLOCK
    assert collapsed.views[1][0].segmentation == _BLOCK_SHIFT
    assert kept.semantic_only == (_BLOCK_SHIFT,)
    union = np.maximum(left, right)
    got = semantic_target_from_annotations(collapsed.annotations, _CANVAS, _CANVAS)
    assert np.array_equal(got, union)
    kept_mask = kept.rasterize(_CANVAS, _CANVAS)
    assert np.array_equal(kept_mask, left)
    assert not np.array_equal(kept_mask, union)
    assert _logged_counts(caplog)["exact_polygons_dropped"] == 0
    assert _logged_counts(caplog)["near_duplicate_instances_merged"] == 1


@pytest.mark.parametrize("other", [_HALF, _THIRD])
def test_overlap_at_or_below_one_half_stays_two_instances(
    other: tuple[tuple[float, ...], ...],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_collapse_logs(caplog)
    left = rasterize_polygons(_BLOCK, _CANVAS, _CANVAS)
    right = rasterize_polygons(other, _CANVAS, _CANVAS)
    overlap = _mask_iou(left, right)
    assert 0.0 < overlap <= 0.5

    split = split_training_images(
        (
            _disk(1, (_annotation(1, 1, _BLOCK),)),
            _disk(2, (_annotation(2, 1, other),)),
            _other_day(),
        )
    )
    collapsed = _only(split, "20150101120000Bh.jpeg")
    assert [item.segmentation for item in collapsed.annotations] == [_BLOCK, other]
    assert _logged_counts(caplog)["near_duplicate_instances_merged"] == 0


def test_three_entries_count_one_duplicated_name_and_two_extra(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_collapse_logs(caplog)
    file_name = "20150101120000Bh.jpeg"
    split = split_training_images(
        (
            _disk(1, (), file_name=file_name, date_captured="2015-01-01 00:00:00"),
            _disk(2, (), file_name=file_name, date_captured="2015-01-01 00:00:00"),
            _disk(3, (), file_name=file_name, date_captured="2015-01-01 00:00:00"),
            _other_day(),
        )
    )
    counts = _logged_counts(caplog)
    assert counts["duplicated_file_names"] == 1
    assert counts["extra_image_entries"] == 2
    names = [image.file_name for image in (*split.train, *split.holdout)]
    assert names.count(file_name) == 1


def test_instance_pq_scores_views_not_the_kept_mask() -> None:
    # The stand-in list keeps _BLOCK and parks the shift on semantic_only.
    # Scoring that one mask would be PQ 1. Each entry is a view.
    split = split_training_images(
        (
            _disk(1, (_annotation(1, 1, _BLOCK),)),
            _disk(2, (_annotation(2, 2, _BLOCK_SHIFT),)),
            _other_day(),
        )
    )
    collapsed = _only(split, "20150101120000Bh.jpeg")
    assert len(collapsed.annotations) == 1
    left = rasterize_polygons(_BLOCK, _CANVAS, _CANVAS)
    right = rasterize_polygons(_BLOCK_SHIFT, _CANVAS, _CANVAS)
    overlap = _mask_iou(left, right)
    assert overlap > 0.5
    gray = np.full((_CANVAS, _CANVAS), 200, dtype=np.uint8)
    score = instance_pq(left, gray, collapsed)
    assert score == pytest.approx((1.0 + overlap) / 2.0)
    assert score != pytest.approx(1.0)
    union = semantic_target_from_annotations(collapsed.annotations, _CANVAS, _CANVAS)
    assert np.array_equal(union, np.maximum(left, right))


def test_instance_pq_counts_an_empty_annotator_view() -> None:
    split = split_training_images(
        (
            _disk(1, (_annotation(1, 1, _BLOCK),)),
            _disk(2, ()),
            _other_day(),
        )
    )
    collapsed = _only(split, "20150101120000Bh.jpeg")
    assert len(collapsed.views) == 2
    assert collapsed.views[1] == ()
    assert len(collapsed.annotations) == 1
    semantic = rasterize_polygons(_BLOCK, _CANVAS, _CANVAS)
    gray = np.full((_CANVAS, _CANVAS), 200, dtype=np.uint8)
    # One true positive on the drawn view, one false positive on the
    # empty view. The stand-in list alone would score 1.
    assert instance_pq(semantic, gray, collapsed) == pytest.approx(2 / 3)


def test_disjoint_entries_are_separate_views() -> None:
    split = split_training_images(
        (
            _disk(1, (_annotation(1, 1, _BLOCK),)),
            _disk(2, (_annotation(2, 1, _CORNER),)),
            _other_day(),
        )
    )
    collapsed = _only(split, "20150101120000Bh.jpeg")
    assert [view[0].segmentation for view in collapsed.views] == [_BLOCK, _CORNER]
    semantic = rasterize_polygons(_BLOCK, _CANVAS, _CANVAS)
    gray = np.full((_CANVAS, _CANVAS), 200, dtype=np.uint8)
    # View 1 matches. View 2 is a miss, so the prediction is also an FP
    # there: TP 1, FP 1, FN 1, PQ 0.5. One concatenated view would be
    # TP 1, FN 1, PQ 2/3.
    assert instance_pq(semantic, gray, collapsed) == pytest.approx(0.5)


def test_non_overlapping_filaments_from_different_entries_both_remain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_collapse_logs(caplog)
    left = rasterize_polygons(_BLOCK, _CANVAS, _CANVAS)
    right = rasterize_polygons(_CORNER, _CANVAS, _CANVAS)
    assert _mask_iou(left, right) == 0.0

    split = split_training_images(
        (
            _disk(1, (_annotation(1, 1, _BLOCK),)),
            _disk(2, (_annotation(2, 1, _CORNER),)),
            _other_day(),
        )
    )
    collapsed = _only(split, "20150101120000Bh.jpeg")
    assert [item.segmentation for item in collapsed.annotations] == [
        _BLOCK,
        _CORNER,
    ]
    assert np.array_equal(
        semantic_target_from_annotations(collapsed.annotations, _CANVAS, _CANVAS),
        np.maximum(left, right),
    )
    assert _logged_counts(caplog)["near_duplicate_instances_merged"] == 0
    names = [image.file_name for image in (*split.train, *split.holdout)]
    assert names.count("20150101120000Bh.jpeg") == 1
    assert len(names) == len(set(names))


def test_same_entry_overlap_is_not_merged_across_a_bridge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # _SHORT sits inside _TALL, and both sit inside _BLOCK. The second
    # entry may match both, but the two polygons from the first entry
    # stay two instances.
    _enable_collapse_logs(caplog)
    tall = rasterize_polygons(_TALL, _CANVAS, _CANVAS)
    short = rasterize_polygons(_SHORT, _CANVAS, _CANVAS)
    bridge = rasterize_polygons(_BLOCK, _CANVAS, _CANVAS)
    assert _mask_iou(tall, short) > 0.5
    assert not np.array_equal(tall, short)
    assert _mask_iou(tall, bridge) > 0.5
    assert _mask_iou(short, bridge) > 0.5

    split = split_training_images(
        (
            _disk(1, (_annotation(1, 1, _TALL), _annotation(2, 1, _SHORT))),
            _disk(2, (_annotation(3, 1, _BLOCK),)),
            _other_day(),
        )
    )
    collapsed = _only(split, "20150101120000Bh.jpeg")
    masks = [item.rasterize(_CANVAS, _CANVAS) for item in collapsed.annotations]
    assert len(masks) == 2
    assert any(np.array_equal(mask, short) for mask in masks)
    assert any(np.array_equal(mask, tall) for mask in masks)
    assert all(not np.array_equal(mask, bridge) for mask in masks)
    semantic = semantic_target_from_annotations(collapsed.annotations, _CANVAS, _CANVAS)
    assert np.array_equal(semantic, np.maximum(np.maximum(tall, short), bridge))
    assert _logged_counts(caplog)["near_duplicate_instances_merged"] == 1
    assert _logged_counts(caplog)["exact_polygons_dropped"] == 0


def test_disagreeing_dates_use_the_file_name_and_keep_a_whole_day(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_collapse_logs(caplog)
    file_name = "20150101120000Bh.jpeg"
    images = [
        _disk(1, (), file_name=file_name, date_captured="2015-01-09 01:00:00"),
        _disk(2, (), file_name=file_name, date_captured="2015-01-10 01:00:00"),
        _disk(
            11,
            (),
            file_name="20150101130000Mh.jpeg",
            date_captured="2015-01-01 13:00:00",
            height=4,
            width=4,
        ),
    ]
    for day in range(2, 11):
        images.append(
            _disk(
                100 + day,
                (),
                file_name=f"201501{day:02d}120000Bh.jpeg",
                date_captured=f"2015-01-{day:02d} 12:00:00",
                height=4,
                width=4,
            )
        )
    images.append(
        _disk(
            20,
            (),
            file_name="20150110180000Mh.jpeg",
            date_captured="2015-01-10 18:00:00",
            height=4,
            width=4,
        )
    )

    split = split_training_images(images)
    train = {image.file_name for image in split.train}
    holdout = {image.file_name for image in split.holdout}
    assert train.isdisjoint(holdout)
    # 1 January stays on the train side, including the file whose captured
    # timestamps said 9 and 10 January. 10 January stays on the holdout side.
    assert {file_name, "20150101130000Mh.jpeg"} <= train
    assert {
        "20150109120000Bh.jpeg",
        "20150110120000Bh.jpeg",
        "20150110180000Mh.jpeg",
    } <= holdout
    assert file_name not in holdout
    assert "20150110120000Bh.jpeg" not in train
    assert "20150110180000Mh.jpeg" not in train

    collapsed = _only(split, file_name)
    assert observation_date(
        {"file_name": "ignored.jpeg", "date_captured": collapsed.date_captured}
    ) == dt.date(2015, 1, 1)
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    ]
    assert len(warnings) == 1
    assert file_name in warnings[0]
    assert "2015-01-09 01:00:00" in warnings[0]
    assert "2015-01-10 01:00:00" in warnings[0]
    names = [image.file_name for image in (*split.train, *split.holdout)]
    assert names.count(file_name) == 1
    assert len(names) == len(set(names))


def test_agreeing_dates_keep_date_captured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    _enable_collapse_logs(caplog)
    file_name = "20150101120000Bh.jpeg"
    agreed = "2015-01-10 12:00:00"
    images = [
        _disk(1, (), file_name=file_name, date_captured=agreed),
        _disk(2, (), file_name=file_name, date_captured=agreed),
        _disk(
            11,
            (),
            file_name="20150101130000Mh.jpeg",
            date_captured="2015-01-01 13:00:00",
            height=4,
            width=4,
        ),
        _disk(
            20,
            (),
            file_name="20150110180000Mh.jpeg",
            date_captured="2015-01-10 18:00:00",
            height=4,
            width=4,
        ),
    ]
    for day in range(2, 11):
        images.append(
            _disk(
                100 + day,
                (),
                file_name=f"201501{day:02d}120000Bh.jpeg",
                date_captured=f"2015-01-{day:02d} 12:00:00",
                height=4,
                width=4,
            )
        )

    split = split_training_images(images)
    train = {image.file_name for image in split.train}
    holdout = {image.file_name for image in split.holdout}
    assert file_name in holdout
    assert file_name not in train
    assert {"20150110120000Bh.jpeg", "20150110180000Mh.jpeg"} <= holdout
    assert "20150101130000Mh.jpeg" in train
    assert _only(split, file_name).date_captured == agreed
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]


def test_repeated_file_name_with_two_shapes_is_rejected() -> None:
    images = (
        _disk(1, (), height=4, width=4),
        _disk(2, (), height=5, width=4),
    )
    with pytest.raises(ValueError, match="shapes"):
        split_training_images(images)
