from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.crop_data import (
    agreement_mask,
    check_mask_kind,
    decode_agreement_mask,
    load_train_cache,
    read_gray,
)
from filament.crops import CropSampler
from filament.train_data import (
    semantic_target_from_annotations,
    split_training_images,
)


def _record(name: str, size: int, polygon: list[float] | None):
    annotations = (
        (SimpleNamespace(segmentation=[polygon], category_id=1),) if polygon else ()
    )
    return SimpleNamespace(
        file_name=name, height=size, width=size, annotations=annotations
    )


def _write(path, size: int) -> None:
    image = np.full((size, size), 100, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_read_gray_checks_shape_and_existence(tmp_path) -> None:
    _write(tmp_path / "a.png", 16)
    assert read_gray(tmp_path / "a.png", (16, 16)).shape == (16, 16)
    with pytest.raises(ValueError, match="expected 8x8"):
        read_gray(tmp_path / "a.png", (8, 8))
    with pytest.raises(FileNotFoundError):
        read_gray(tmp_path / "missing.png")


def test_load_train_cache(tmp_path) -> None:
    _write(tmp_path / "a.png", 32)
    _write(tmp_path / "b.png", 32)
    records = [
        _record("a.png", 32, [4, 4, 20, 4, 20, 20, 4, 20]),
        _record("b.png", 32, None),
    ]
    lines: list[str] = []
    cache = load_train_cache(records, tmp_path, log=lines.append, progress_every=1)
    assert cache.file_names == ("a.png", "b.png")
    assert cache.images[0].shape == cache.masks[0].shape == (32, 32)
    assert cache.masks[0].dtype == np.uint8
    assert cache.masks[0].any()
    assert not cache.masks[1].any()
    assert cache.nbytes == 4 * 32 * 32
    assert 0 < cache.foreground_fraction() < 0.5
    assert lines[-1] == "prepared 2 / 2"


def test_load_train_cache_rejects_size_mismatch(tmp_path) -> None:
    _write(tmp_path / "a.png", 32)
    with pytest.raises(ValueError):
        load_train_cache([_record("a.png", 64, None)], tmp_path, log=lambda _: None)


# --- agreement weights --------------------------------------------------------

SQUARE = (4.0, 4.0, 20.0, 4.0, 20.0, 20.0, 4.0, 20.0)
LEFT = (4.0, 4.0, 12.0, 4.0, 12.0, 20.0, 4.0, 20.0)
FAR = (24.0, 24.0, 30.0, 24.0, 30.0, 30.0, 24.0, 30.0)


def _annotation(number: int, polygon: tuple[float, ...]) -> FilamentAnnotation:
    return FilamentAnnotation(
        id=number,
        image_id=0,
        category_id=1,
        spine=None,
        segmentation=(polygon,),
    )


def _view_image(*views: tuple[FilamentAnnotation, ...]) -> TrainImage:
    return TrainImage(
        id=0,
        file_name="a.png",
        date_captured="2020-01-01 12:00:00",
        height=32,
        width=32,
        annotations=views[0],
        views=tuple(views) if len(views) > 1 else (),
    )


def _weight(k: int, total: int) -> float:
    return 0.4 + 0.6 * k / total


def test_one_view_decodes_to_weight_one_everywhere() -> None:
    image = _view_image((_annotation(1, SQUARE),))
    stored = agreement_mask(image)
    assert stored.dtype == np.uint8
    assert set(np.unique(stored)) == {0, 255}
    target, weight = decode_agreement_mask(stored)
    assert target.dtype == np.uint8 and weight.dtype == np.float32
    assert np.array_equal(target, (stored != 0).astype(np.uint8))
    assert np.all(weight == 1.0)


def test_weights_follow_the_formula_within_the_uint8_rounding() -> None:
    # Three views: the left half of the square has k = 3, its right half k = 2,
    # and the far square k = 1.
    image = _view_image(
        (_annotation(1, SQUARE), _annotation(2, FAR)),
        (_annotation(3, SQUARE),),
        (_annotation(4, LEFT),),
    )
    target, weight = decode_agreement_mask(agreement_mask(image))
    left, right, far, empty = (10, 6), (10, 16), (27, 27), (0, 0)
    assert weight[left] == pytest.approx(_weight(3, 3), abs=1 / 510)
    assert weight[right] == pytest.approx(_weight(2, 3), abs=1 / 510)
    assert weight[far] == pytest.approx(_weight(1, 3), abs=1 / 510)
    assert target[left] == target[right] == target[far] == 1
    assert target[empty] == 0 and weight[empty] == 1.0


def test_one_annotator_with_overlapping_polygons_counts_once() -> None:
    image = _view_image(
        (_annotation(1, SQUARE), _annotation(2, LEFT)),
        (_annotation(3, FAR),),
    )
    _, weight = decode_agreement_mask(agreement_mask(image))
    assert weight[10, 6] == pytest.approx(_weight(1, 2), abs=1 / 510)


def test_a_view_that_drew_nothing_still_counts_in_r() -> None:
    image = _view_image((_annotation(1, SQUARE),), ())
    _, weight = decode_agreement_mask(agreement_mask(image))
    assert weight[10, 10] == pytest.approx(_weight(1, 2), abs=1 / 510)


def test_decode_rejects_anything_but_uint8() -> None:
    with pytest.raises(TypeError, match="uint8"):
        decode_agreement_mask(np.zeros((2, 2), dtype=np.int32))


def test_decode_works_on_a_batch_of_crops() -> None:
    batch = np.array([[[0, 255], [178, 0]]], dtype=np.uint8)
    target, weight = decode_agreement_mask(batch)
    assert target.tolist() == [[[0, 1], [1, 0]]]
    assert weight.shape == batch.shape
    assert weight[0, 1, 0] == pytest.approx(178 / 255)
    assert weight[0, 0, 0] == 1.0


def test_collapsed_views_add_up_to_the_union_of_the_split() -> None:
    # Repeated file name: an exact duplicate is dropped and a near duplicate
    # moves to semantic_only. The union has to be the same from either side.
    first = TrainImage(
        id=1,
        file_name="20200101120000Aa.png",
        date_captured="2020-01-01 12:00:00",
        height=32,
        width=32,
        annotations=(_annotation(1, SQUARE), _annotation(2, FAR)),
    )
    second = TrainImage(
        id=2,
        file_name="20200101120000Aa.png",
        date_captured="2020-01-01 12:00:00",
        height=32,
        width=32,
        annotations=(_annotation(3, SQUARE), _annotation(4, LEFT)),
    )
    later = TrainImage(
        id=3,
        file_name="20200901120000Aa.png",
        date_captured="2020-09-01 12:00:00",
        height=32,
        width=32,
        annotations=(_annotation(5, FAR),),
    )
    split = split_training_images([first, second, later])
    collapsed = next(i for i in split.train if i.file_name == first.file_name)
    assert len(collapsed.views) == 2
    union = semantic_target_from_annotations(collapsed.annotations, 32, 32)
    assert np.array_equal(agreement_mask(collapsed) != 0, union != 0)


def test_agreement_masks_draw_the_same_crops_as_the_union() -> None:
    image = _view_image(
        (_annotation(1, SQUARE), _annotation(2, FAR)),
        (_annotation(3, LEFT),),
    )
    stored = agreement_mask(image)
    union = semantic_target_from_annotations(image.views[0] + image.views[1], 32, 32)
    assert np.array_equal(stored != 0, union != 0)
    gray = np.arange(32 * 32, dtype=np.uint32).reshape(32, 32).astype(np.uint8)
    plain = CropSampler([gray], [union], 16, p_fg=0.67)
    weighted = CropSampler([gray], [stored], 16, p_fg=0.67)
    plain_batches = list(plain.epoch_batches(8, 4, np.random.default_rng(7)))
    weighted_batches = list(weighted.epoch_batches(8, 4, np.random.default_rng(7)))
    assert len(plain_batches) == len(weighted_batches) == 2
    for (images_a, masks_a), (images_b, masks_b) in zip(
        plain_batches, weighted_batches, strict=True
    ):
        assert np.array_equal(images_a, images_b)
        assert np.array_equal(masks_a, decode_agreement_mask(masks_b)[0])


def test_load_train_cache_stores_the_agreement_mask(tmp_path) -> None:
    _write(tmp_path / "a.png", 32)
    image = _view_image((_annotation(1, SQUARE),), (_annotation(2, LEFT),))
    plain = load_train_cache([image], tmp_path, log=lambda _: None)
    weighted = load_train_cache(
        [image], tmp_path, log=lambda _: None, agreement_weights=True
    )
    assert set(np.unique(plain.masks[0])) == {0, 1}
    assert np.array_equal(weighted.masks[0], agreement_mask(image))
    assert weighted.foreground_fraction() == plain.foreground_fraction()
    assert weighted.nbytes == plain.nbytes


def test_load_train_cache_stops_when_views_miss_part_of_the_union(tmp_path) -> None:
    _write(tmp_path / "a.png", 32)
    # A polygon on semantic_only that no annotator view has.
    extra = FilamentAnnotation(
        id=1,
        image_id=0,
        category_id=1,
        spine=None,
        segmentation=(SQUARE,),
        semantic_only=((FAR,),),
    )
    other = _annotation(2, SQUARE)
    image = dataclasses.replace(_view_image((other,), (other,)), annotations=(extra,))
    with pytest.raises(ValueError, match="do not add up"):
        load_train_cache([image], tmp_path, log=lambda _: None, agreement_weights=True)


def test_mask_kind_accepts_what_each_recipe_builds() -> None:
    image = _view_image((_annotation(1, SQUARE),), (_annotation(2, LEFT),))
    plain = semantic_target_from_annotations(image.views[0], 32, 32)
    stored = agreement_mask(image)
    empty = np.zeros((32, 32), dtype=np.uint8)
    check_mask_kind([plain, empty], agreement_weights=False)
    check_mask_kind([stored, empty], agreement_weights=True)


def test_mask_kind_stops_a_cache_built_for_the_other_recipe() -> None:
    image = _view_image((_annotation(1, SQUARE),), (_annotation(2, LEFT),))
    plain = semantic_target_from_annotations(image.views[0], 32, 32)
    stored = agreement_mask(image)
    with pytest.raises(ValueError, match="agreement_weights=True"):
        check_mask_kind([plain], agreement_weights=True)
    with pytest.raises(ValueError, match="agreement_weights=False"):
        check_mask_kind([stored], agreement_weights=False)


def test_every_stored_weight_is_far_from_a_plain_mask_value() -> None:
    # The check tells the kinds apart by value: the smallest weight, with one view
    # of many, still has to be well above 1.
    views = [(_annotation(1, SQUARE),)] + [()] * 9
    stored = agreement_mask(_view_image(*views))
    assert int(stored.max()) >= 102


def test_mask_kind_checks_every_mask_not_only_the_first() -> None:
    image = _view_image((_annotation(1, SQUARE),), (_annotation(2, LEFT),))
    plain = semantic_target_from_annotations(image.views[0], 32, 32)
    stored = agreement_mask(image)
    empty = np.zeros((32, 32), dtype=np.uint8)
    # A stale cache whose first frames hold no filament.
    with pytest.raises(ValueError, match="mask 2 is a plain"):
        check_mask_kind([empty, empty, plain], agreement_weights=True)
    with pytest.raises(ValueError, match="mask 2 holds agreement weights"):
        check_mask_kind([empty, empty, stored], agreement_weights=False)
    # The right kind after an empty one still passes.
    check_mask_kind([empty, plain, empty], agreement_weights=False)
    check_mask_kind([empty, stored, empty], agreement_weights=True)
    # A wrong mask late in the list is found even when the early ones are fine.
    with pytest.raises(ValueError, match="mask 1"):
        check_mask_kind([stored, plain], agreement_weights=True)
