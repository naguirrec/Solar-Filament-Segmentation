"""Detector windows, coordinate maps, splits and the COCO folders.

Everything is synthetic: 64 x 64 frames and hand-made polygons. Nothing here
reads a real image or a real annotation.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from filament.annotations import (
    FilamentAnnotation,
    TrainImage,
    rasterize_polygons,
)
from filament.detector_data import (
    ANNOTATIONS_NAME,
    DEFAULT_MIN_LABEL_AREA,
    FILAMENT_CATEGORY_ID,
    DetectorExample,
    DetectorSplits,
    Window,
    build_detector_splits,
    coco_payload,
    frame_to_window_points,
    frame_windows,
    image_examples,
    paste_window_mask,
    split_dates,
    window_instances,
    window_to_frame_points,
    write_detector_dataset,
)
from filament.split import split_by_date
from filament.train_data import split_training_images

FRAME = 64


def _rect(x0: float, y0: float, x1: float, y1: float) -> tuple[float, ...]:
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def _annotation(number: int, *polygons: tuple[float, ...]) -> FilamentAnnotation:
    return FilamentAnnotation(
        id=number,
        image_id=1,
        category_id=1,
        spine=None,
        segmentation=tuple(polygons),
    )


def _image(
    name: str,
    day: int = 1,
    views: tuple[tuple[FilamentAnnotation, ...], ...] = (),
    annotations: tuple[FilamentAnnotation, ...] = (),
    size: int = FRAME,
) -> TrainImage:
    return TrainImage(
        id=name,
        file_name=name,
        date_captured=f"2022-06-{day:02d} 10:00:00",
        height=size,
        width=size,
        annotations=annotations,
        views=views,
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return np.count_nonzero(left & right) / union


# ---------------------------------------------------------------- windows


def test_default_windows_are_four_quadrants_with_256_px_overlap() -> None:
    windows = frame_windows()
    assert windows == (
        Window(0, 0, 1152, 1152),
        Window(896, 0, 2048, 1152),
        Window(0, 896, 1152, 2048),
        Window(896, 896, 2048, 2048),
    )
    assert windows[0].x1 - windows[1].x0 == 256


@pytest.mark.parametrize(
    ("frame", "window", "stride"),
    [(100, 60, 50), (100, 60, 40), (100, 100, 30), (97, 20, 20), (2048, 768, 500)],
)
def test_windows_cover_the_frame_and_the_last_is_flush(frame, window, stride) -> None:
    windows = frame_windows(frame, window, stride)
    covered = np.zeros((frame, frame), dtype=int)
    for item in windows:
        assert item.width == item.height == window
        assert 0 <= item.x0 and item.x1 <= frame
        covered[item.y0 : item.y1, item.x0 : item.x1] += 1
    assert covered.min() >= 1
    assert max(w.x1 for w in windows) == frame
    assert max(w.y1 for w in windows) == frame


@pytest.mark.parametrize(
    ("frame", "window", "stride"),
    [(0, 10, 5), (100, 0, 5), (100, 10, 0), (100, 200, 50), (100, 50, 60)],
)
def test_frame_windows_validates_its_arguments(frame, window, stride) -> None:
    with pytest.raises(ValueError):
        frame_windows(frame, window, stride)


def test_frame_windows_rejects_non_ints() -> None:
    with pytest.raises(ValueError):
        frame_windows(100.0, 50, 25)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        frame_windows(True, 1, 1)  # type: ignore[arg-type]


def test_window_validates_and_is_frozen() -> None:
    with pytest.raises(ValueError):
        Window(5, 0, 5, 10)
    with pytest.raises(ValueError):
        Window(-1, 0, 5, 10)
    with pytest.raises(TypeError):
        Window(0.0, 0, 5, 10)  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        Window(0, 0, 5, 5).x0 = 1  # type: ignore[misc]


# ------------------------------------------------------ coordinate mapping


def test_points_map_to_the_resized_window_and_back() -> None:
    window = Window(896, 896, 2048, 2048)
    points = [[896.0, 896.0], [2048.0, 2048.0], [1472.0, 1000.0]]
    mapped = frame_to_window_points(points, window, 768)
    assert mapped[0].tolist() == [0.0, 0.0]
    assert mapped[1].tolist() == [768.0, 768.0]
    assert mapped[2, 0] == pytest.approx(384.0)
    back = window_to_frame_points(mapped, window, 768)
    np.testing.assert_allclose(back, points)


def test_points_need_shape_n_by_2() -> None:
    with pytest.raises(ValueError):
        frame_to_window_points([1.0, 2.0], Window(0, 0, 4, 4), 8)
    with pytest.raises(ValueError):
        frame_to_window_points([[1.0, 2.0]], Window(0, 0, 4, 4), 0)


def test_a_resized_window_mask_is_pasted_back_in_its_place() -> None:
    window = Window(24, 8, 64, 48)
    small = np.zeros((20, 20), dtype=np.uint8)
    small[0:10, 0:10] = 1  # the window's top-left quarter at target 20
    frame = paste_window_mask(small, window, (FRAME, FRAME))
    assert frame.shape == (FRAME, FRAME)
    assert frame.dtype == np.uint8
    assert frame[8:28, 24:44].all()
    assert frame.sum() == 20 * 20
    assert frame[:8].sum() == 0
    assert frame[:, :24].sum() == 0


def test_pasting_overlapping_windows_takes_the_maximum() -> None:
    out = np.zeros((FRAME, FRAME), dtype=np.uint8)
    paste_window_mask(
        np.ones((8, 8), np.uint8), Window(0, 0, 40, 40), (FRAME,) * 2, out
    )
    paste_window_mask(
        np.ones((8, 8), np.uint8), Window(24, 24, 64, 64), (FRAME,) * 2, out
    )
    assert out[30, 30] == 1
    assert out[0, 0] == 1
    assert out[63, 63] == 1
    assert out.sum() == 40 * 40 + 40 * 40 - 16 * 16


def test_pasting_accepts_bool_and_native_size_masks() -> None:
    window = Window(0, 0, 10, 10)
    native = np.ones((10, 10), dtype=bool)
    frame = paste_window_mask(native, window, (20, 20))
    assert frame.dtype == np.uint8
    assert frame.sum() == 100
    with pytest.raises(ValueError):
        paste_window_mask(native, Window(15, 15, 25, 25), (20, 20))
    with pytest.raises(ValueError):
        paste_window_mask(native, window, (20, 20), np.zeros((5, 5), np.uint8))


def test_a_float_mask_is_refused_by_an_integer_out() -> None:
    window = Window(0, 0, 10, 10)
    probability = np.full((10, 10), 0.9, dtype=np.float32)
    for dtype in (np.uint8, np.int32, np.bool_):
        with pytest.raises(TypeError, match="truncated"):
            paste_window_mask(probability, window, (20, 20), np.zeros((20, 20), dtype))
    # A float out keeps the values; so does the default out.
    kept = paste_window_mask(
        probability, window, (20, 20), np.zeros((20, 20), np.float32)
    )
    assert kept[:10, :10].min() == pytest.approx(0.9)
    assert paste_window_mask(probability, window, (20, 20)).max() == pytest.approx(0.9)
    # Integer into float is exact, so it is allowed.
    ones = paste_window_mask(
        np.ones((10, 10), np.uint8), window, (20, 20), np.zeros((20, 20), np.float32)
    )
    assert ones.sum() == 100


# --------------------------------------------------------------- instances


def test_a_window_instance_is_a_polygon_in_resized_coordinates() -> None:
    annotation = _annotation(1, _rect(4, 6, 36, 30))
    window = Window(0, 0, 40, 40)
    [instance] = window_instances(
        [annotation], window, 20, min_area=0, frame_shape=(FRAME, FRAME)
    )
    assert instance.annotation_id == 1
    assert instance.visible == 1.0
    assert len(instance.polygons) == 1
    # Back in the frame, the polygon covers the original mask.
    points = np.asarray(instance.polygons[0]).reshape(-1, 2)
    frame_points = window_to_frame_points(points, window, 20)
    rebuilt = rasterize_polygons([frame_points.reshape(-1).tolist()], FRAME, FRAME)
    original = annotation.rasterize(FRAME, FRAME)
    assert _iou(rebuilt > 0, original > 0) > 0.9
    x, y, w, h = instance.bbox
    assert (x, y) == pytest.approx((2.5, 3.5), abs=0.3)
    assert (w, h) == pytest.approx((15.0, 11.0), abs=1.0)
    assert instance.area == pytest.approx(32 * 24 * 0.25)


def test_a_filament_mostly_outside_is_dropped_by_min_visible() -> None:
    annotation = _annotation(1, _rect(30, 10, 50, 30))  # half in a 40 window
    window = Window(0, 0, 40, 40)
    kwargs = {"min_area": 0, "frame_shape": (FRAME, FRAME)}
    kept = window_instances([annotation], window, 20, min_visible=0.25, **kwargs)
    assert [round(i.visible, 2) for i in kept] == [0.5]
    assert window_instances([annotation], window, 20, min_visible=0.6, **kwargs) == ()
    # A filament wholly outside the window is never an instance.
    outside = _annotation(2, _rect(50, 50, 60, 60))
    assert window_instances([outside], window, 20, min_visible=0.0, **kwargs) == ()


def test_min_area_counts_pixels_inside_the_window_before_resizing() -> None:
    annotation = _annotation(1, _rect(4, 4, 14, 14))  # 100 px
    window = Window(0, 0, 40, 40)
    kwargs = {"frame_shape": (FRAME, FRAME)}
    assert window_instances([annotation], window, 20, min_area=100, **kwargs)
    assert not window_instances([annotation], window, 20, min_area=101, **kwargs)


def test_the_default_min_area_is_the_detectors_own_untuned_100() -> None:
    from filament import baseline

    assert DEFAULT_MIN_LABEL_AREA == 100
    assert DEFAULT_MIN_LABEL_AREA != baseline.DEFAULT_MIN_AREA  # the swept value
    window = Window(0, 0, 40, 40)
    kwargs = {"frame_shape": (FRAME, FRAME)}
    exact = _annotation(1, _rect(4, 4, 14, 14))  # 100 px
    short = _annotation(2, _rect(4, 4, 13, 15))  # 99 px
    assert window_instances([exact], window, 20, **kwargs)
    assert not window_instances([short], window, 20, **kwargs)
    image = _image("a.jpeg", annotations=(exact, short))
    [only] = image_examples(image, [window], 20)
    assert [i.annotation_id for i in only.instances] == [1]


def test_a_clipped_filament_in_two_pieces_is_one_instance() -> None:
    # A U whose bottom bar lies below the window: two bars remain.
    u_shape = (
        4.0, 4.0, 10.0, 4.0, 10.0, 34.0, 20.0, 34.0, 20.0, 4.0, 26.0, 4.0,
        26.0, 40.0, 4.0, 40.0,
    )  # fmt: skip
    annotation = _annotation(7, u_shape)
    window = Window(0, 0, 30, 30)
    result = window_instances(
        [annotation], window, 30, min_area=0, frame_shape=(FRAME, FRAME)
    )
    assert len(result) == 1
    assert len(result[0].polygons) == 2
    assert result[0].visible == pytest.approx(312 / 492, abs=0.01)


def test_an_annotation_with_two_polygons_keeps_both() -> None:
    annotation = _annotation(1, _rect(2, 2, 10, 10), _rect(20, 20, 30, 30))
    [instance] = window_instances(
        [annotation], Window(0, 0, 40, 40), 40, min_area=0, frame_shape=(FRAME,) * 2
    )
    assert len(instance.polygons) == 2


def test_a_piece_with_under_three_points_is_dropped() -> None:
    thin = _annotation(1, _rect(10, 5, 11, 30))  # one pixel wide
    result = window_instances(
        [thin], Window(0, 0, 40, 40), 40, min_area=0, min_visible=0.0,
        frame_shape=(FRAME, FRAME),
    )  # fmt: skip
    assert result == ()
    # The thin piece goes; the wide one stays, as a single polygon.
    both = _annotation(2, _rect(10, 5, 11, 30), _rect(20, 5, 30, 15))
    [instance] = window_instances(
        [both], Window(0, 0, 40, 40), 40, min_area=0, min_visible=0.0,
        frame_shape=(FRAME, FRAME),
    )  # fmt: skip
    assert len(instance.polygons) == 1


def test_window_instances_validates_its_arguments() -> None:
    annotation = _annotation(1, _rect(4, 4, 14, 14))
    window = Window(0, 0, 40, 40)
    kwargs = {"frame_shape": (FRAME, FRAME)}
    with pytest.raises(ValueError):
        window_instances([annotation], window, 0, **kwargs)
    with pytest.raises(ValueError):
        window_instances([annotation], window, 20, min_visible=1.5, **kwargs)
    with pytest.raises(ValueError):
        window_instances([annotation], window, 20, min_area=-1, **kwargs)
    with pytest.raises(ValueError):
        window_instances([annotation], Window(40, 40, 80, 80), 20, **kwargs)


# ---------------------------------------------------------------- examples


def test_each_view_is_one_example_per_window() -> None:
    first = (_annotation(1, _rect(4, 4, 20, 20)),)
    second = (
        _annotation(2, _rect(4, 4, 24, 24)),
        _annotation(3, _rect(40, 40, 60, 60)),
    )
    third: tuple[FilamentAnnotation, ...] = ()  # an annotator who drew nothing
    image = _image("20220601000000Bh.jpeg", views=(first, second, third))
    windows = frame_windows(FRAME, 40, 24)
    examples = image_examples(image, windows, 20, min_area=4)
    assert len(examples) == 3 * len(windows)
    assert {(e.window_index, e.view_index) for e in examples} == {
        (w, v) for w in range(len(windows)) for v in range(3)
    }
    top_left = {e.view_index: e for e in examples if e.window_index == 0}
    assert [len(top_left[v].instances) for v in range(3)] == [1, 1, 0]
    # The view that drew nothing is a negative example in every window.
    negatives = [e for e in examples if e.view_index == 2]
    assert len(negatives) == len(windows)
    assert all(e.instances == () for e in negatives)
    # Same pixels, different labels: both views share the window.
    assert top_left[0].window == top_left[1].window
    assert top_left[0].instances != top_left[1].instances


def test_an_image_without_views_uses_its_annotations_as_the_one_view() -> None:
    image = _image("a.jpeg", annotations=(_annotation(1, _rect(4, 4, 20, 20)),))
    examples = image_examples(image, frame_windows(FRAME, 40, 24), 20, min_area=4)
    assert {e.view_index for e in examples} == {0}
    assert len(examples) == 4


def test_image_examples_rejects_a_window_outside_the_image() -> None:
    image = _image("a.jpeg")
    with pytest.raises(ValueError):
        image_examples(image, [Window(0, 0, 80, 80)], 20)
    with pytest.raises(ValueError):
        image_examples(image, [], 20)


# ------------------------------------------------------------------ splits


def _split_images() -> list[TrainImage]:
    """Ten dates; day 3 has two files; ``f05`` has two annotator entries."""
    images = [_image(f"f{day:02d}.jpeg", day) for day in range(1, 11)]
    images.append(_image("f03b.jpeg", 3))
    images.append(
        _image("f05.jpeg", 5, annotations=(_annotation(9, _rect(1, 1, 9, 9)),))
    )
    return images


def test_splits_share_no_file_and_no_date() -> None:
    splits = build_detector_splits(_split_images())
    parts = {"train": splits.train, "val": splits.val, "holdout": splits.holdout}
    names = {k: [i.file_name for i in v] for k, v in parts.items()}
    dates = {k: split_dates(v) for k, v in parts.items()}
    for left in parts:
        assert len(names[left]) == len(set(names[left]))  # f05 counted once
        for right in parts:
            if left < right:
                assert not set(names[left]) & set(names[right]), (left, right)
                assert not dates[left] & dates[right], (left, right)
    every = set().union(*(set(v) for v in names.values()))
    assert every == {i.file_name for i in _split_images()}


def test_the_holdout_is_the_split_the_unet_uses() -> None:
    images = _split_images()
    splits = build_detector_splits(images)
    unet = split_training_images(images)
    assert [i.file_name for i in splits.holdout] == [i.file_name for i in unet.holdout]
    assert splits.holdout == unet.holdout
    payload = [
        {"file_name": i.file_name, "date_captured": i.date_captured} for i in unet.train
    ] + [
        {"file_name": i.file_name, "date_captured": i.date_captured}
        for i in unet.holdout
    ]
    assert set(split_by_date(payload, 0.2).val) == {i.file_name for i in splits.holdout}
    assert split_dates(splits.holdout) == {"2022-06-09", "2022-06-10"}


def test_detector_validation_is_the_newest_of_the_remaining_dates() -> None:
    splits = build_detector_splits(_split_images())
    # Ten dates: two held out, then one tenth of the remaining eight rounds to one.
    assert split_dates(splits.val) == {"2022-06-08"}
    assert max(split_dates(splits.train)) == "2022-06-07"
    assert max(split_dates(splits.val)) < min(split_dates(splits.holdout))
    assert {i.file_name for i in splits.train} >= {"f03.jpeg", "f03b.jpeg"}


def test_a_holdout_other_than_the_default_needs_an_explicit_override() -> None:
    images = _split_images()
    for fraction in (0.1, 0.4):
        with pytest.raises(ValueError, match="allow_nondefault_holdout"):
            build_detector_splits(images, val_fraction=fraction)
    # Spelling out the default is fine.
    assert build_detector_splits(images, val_fraction=0.2).holdout == (
        build_detector_splits(images).holdout
    )
    # With the override the holdout follows ``val_fraction``. It is then no
    # longer the U-Net's, which is the point of asking.
    wide = build_detector_splits(
        images, val_fraction=0.4, allow_nondefault_holdout=True
    )
    assert wide.holdout == split_training_images(images, val_fraction=0.4).holdout
    assert len(split_dates(wide.holdout)) == 4
    assert len(build_detector_splits(images).holdout) < len(wide.holdout)


def test_a_smaller_holdout_is_the_leak_the_default_guards_against() -> None:
    # 20 dates: the U-Net holds out the last four. A 0.1 holdout would put two
    # of those dates into detector train and val.
    images = [_image(f"f{day:02d}.jpeg", day) for day in range(1, 21)]
    unet_holdout = {i.file_name for i in split_training_images(images).holdout}
    small = build_detector_splits(
        images, val_fraction=0.1, allow_nondefault_holdout=True
    )
    leaked = unet_holdout & {i.file_name for i in small.train + small.val}
    assert leaked == {"f17.jpeg", "f18.jpeg"}


def test_the_annotations_object_is_accepted() -> None:
    from types import SimpleNamespace

    images = _split_images()
    wrapped = SimpleNamespace(images=tuple(images))
    assert build_detector_splits(wrapped) == build_detector_splits(images)


# -------------------------------------------------------------------- COCO


def _example(file_name: str, window_index: int, view_index: int, n: int):
    instances = window_instances(
        [_annotation(i, _rect(2 + 10 * i, 2, 10 + 10 * i, 12)) for i in range(n)],
        Window(0, 0, 40, 40),
        20,
        min_area=0,
        frame_shape=(FRAME, FRAME),
    )
    return DetectorExample(
        file_name, window_index, view_index, Window(0, 0, 40, 40), 20, instances
    )


def test_coco_payload_has_one_category_and_unique_names() -> None:
    examples = [
        _example("20220601000000Bh.jpeg", 0, 0, 2),
        _example("20220601000000Bh.jpeg", 0, 1, 0),
        _example("20220601000000Bh.jpeg", 1, 0, 1),
    ]
    payload = coco_payload(examples)
    assert [c["id"] for c in payload["categories"]] == [FILAMENT_CATEGORY_ID]
    assert [i["file_name"] for i in payload["images"]] == [
        "20220601000000Bh__w0__v0.jpg",
        "20220601000000Bh__w0__v1.jpg",
        "20220601000000Bh__w1__v0.jpg",
    ]
    assert [i["id"] for i in payload["images"]] == [1, 2, 3]
    assert all(i["width"] == i["height"] == 20 for i in payload["images"])
    annotations = payload["annotations"]
    assert [a["image_id"] for a in annotations] == [1, 1, 3]
    assert [a["id"] for a in annotations] == [1, 2, 3]
    assert {a["category_id"] for a in annotations} == {FILAMENT_CATEGORY_ID}
    assert all(a["iscrowd"] == 0 and len(a["bbox"]) == 4 for a in annotations)
    assert all(isinstance(p, list) for a in annotations for p in a["segmentation"])
    json.dumps(payload)  # serialisable


def test_coco_payload_refuses_a_repeated_image_name() -> None:
    twice = [_example("a.jpeg", 0, 0, 1), _example("a.jpeg", 0, 0, 1)]
    with pytest.raises(ValueError, match="duplicate"):
        coco_payload(twice)
    # Two files with the same stem would also collide.
    clash = [_example("a.jpeg", 0, 0, 1), _example("a.jpg", 0, 0, 1)]
    with pytest.raises(ValueError, match="duplicate"):
        coco_payload(clash)


# ----------------------------------------------------------- folder layout


def _dataset():
    """Eight dates. Views: f01 has three, the rest one (annotations)."""
    box = (_annotation(1, _rect(4, 4, 30, 30)),)
    other = (_annotation(2, _rect(8, 8, 28, 28)),)
    images = []
    for day in range(1, 9):
        name = f"2022060{day}000000Bh.jpeg"
        if day == 1:
            images.append(_image(name, day, views=(box, other, ())))
        else:
            images.append(_image(name, day, annotations=box))
    return images


def _loader(calls: list[str]):
    def load(image: TrainImage) -> np.ndarray:
        calls.append(image.file_name)
        gradient = np.arange(FRAME * FRAME, dtype=np.uint32) % 251
        return gradient.reshape(FRAME, FRAME).astype(np.uint8)

    return load


def test_the_dataset_has_the_rfdetr_layout(tmp_path) -> None:
    images = _dataset()
    splits = build_detector_splits(images, detector_val_fraction=0.25)
    calls: list[str] = []
    windows = frame_windows(FRAME, 40, 24)
    counts = write_detector_dataset(
        tmp_path,
        splits,
        _loader(calls),
        windows=windows,
        target=16,
        min_area=4,
    )
    holdout = {i.file_name for i in splits.holdout}
    assert holdout
    assert not holdout & set(calls)  # holdout frames are never loaded
    assert sorted(calls) == sorted(i.file_name for i in splits.train + splits.val)

    for folder in ("train", "valid", "test"):
        payload = json.loads((tmp_path / folder / ANNOTATIONS_NAME).read_text())
        jpgs = sorted(p.name for p in (tmp_path / folder).glob("*.jpg"))
        assert jpgs == sorted(i["file_name"] for i in payload["images"])
        assert counts[folder].files == len(jpgs)
        assert counts[folder].instances == len(payload["annotations"])
        assert not any(name.startswith(tuple(holdout_stems(holdout))) for name in jpgs)

    # One JPEG per (file, window, view): views count, not files.
    views = {i.file_name: len(_views(i)) for i in splits.train}
    expected = sum(v * len(windows) for v in views.values())
    assert counts["train"].files == expected
    assert counts["test"] == counts["valid"]


def holdout_stems(names) -> set[str]:
    return {n.split(".")[0] for n in names}


def _views(image: TrainImage):
    return image.views or (image.annotations,)


def test_views_share_pixels_and_the_frame_is_loaded_once(tmp_path) -> None:
    box = (_annotation(1, _rect(4, 4, 30, 30)),)
    image = _image("20220601000000Bh.jpeg", 1, views=(box, (), box))
    later = _image("20220602000000Bh.jpeg", 2)
    splits = build_detector_splits([image, later, _image("20220603000000Bh.jpeg", 3)])
    calls: list[str] = []
    write_detector_dataset(
        tmp_path, splits, _loader(calls), windows=frame_windows(FRAME, 40, 24),
        target=16, min_area=4,
    )  # fmt: skip
    assert calls.count(image.file_name) == 1  # three views, one load
    train = tmp_path / "train"
    stem = "20220601000000Bh"
    raw = [(train / f"{stem}__w0__v{v}.jpg").read_bytes() for v in range(3)]
    assert raw[0] == raw[1] == raw[2]
    decoded = cv2.imread(str(train / f"{stem}__w0__v0.jpg"), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (16, 16, 3)
    assert decoded.dtype == np.uint8
    spread = decoded.max(axis=2).astype(int) - decoded.min(axis=2).astype(int)
    assert spread.max() <= 2  # gray repeated in three channels
    payload = json.loads((train / ANNOTATIONS_NAME).read_text())
    by_name = {i["file_name"]: i["id"] for i in payload["images"]}
    counts = {name: 0 for name in by_name}
    for annotation in payload["annotations"]:
        name = next(n for n, i in by_name.items() if i == annotation["image_id"])
        counts[name] += 1
    assert counts[f"{stem}__w0__v0.jpg"] == 1
    assert counts[f"{stem}__w0__v1.jpg"] == 0  # the empty view is a negative
    assert counts[f"{stem}__w0__v2.jpg"] == 1


def test_a_folder_that_already_has_files_is_not_reused(tmp_path) -> None:
    splits = build_detector_splits(_dataset())
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "stale.jpg").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        write_detector_dataset(
            tmp_path, splits, _loader([]), windows=frame_windows(FRAME, 40, 24)
        )


def test_a_bad_frame_stops_the_writer(tmp_path) -> None:
    splits = build_detector_splits(_dataset())
    windows = frame_windows(FRAME, 40, 24)
    with pytest.raises(ValueError, match="uint8"):
        write_detector_dataset(
            tmp_path / "a", splits, lambda _: np.zeros((FRAME, FRAME)), windows=windows
        )
    with pytest.raises(ValueError, match="shape"):
        write_detector_dataset(
            tmp_path / "b",
            splits,
            lambda _: np.zeros((FRAME // 2, FRAME), np.uint8),
            windows=windows,
        )


def test_a_file_in_train_and_val_is_refused(tmp_path) -> None:
    splits = build_detector_splits(_dataset())
    leaky = type(splits)(splits.train, splits.train[:1], splits.holdout)
    with pytest.raises(ValueError, match="both"):
        write_detector_dataset(
            tmp_path, leaky, _loader([]), windows=frame_windows(FRAME, 40, 24)
        )


def test_the_writer_refuses_any_holdout_overlap_with_train_or_val(tmp_path) -> None:
    splits = build_detector_splits(_dataset())
    windows = frame_windows(FRAME, 40, 24)
    kinds = type(splits)
    held = splits.holdout[0]
    same_file = {
        "train": kinds(splits.train + (held,), splits.val, splits.holdout),
        "val": kinds(splits.train, splits.val + (held,), splits.holdout),
    }
    for where, leaky in same_file.items():
        with pytest.raises(ValueError, match=f"holdout files in {where}"):
            write_detector_dataset(
                tmp_path / where, leaky, _loader([]), windows=windows, target=16
            )
    # A different file on a holdout date is a leak too: the split is by date.
    sibling = _image("sibling.jpeg", day=int(held.date_captured[8:10]))
    by_date = kinds(splits.train + (sibling,), splits.val, splits.holdout)
    with pytest.raises(ValueError, match="holdout dates in train"):
        write_detector_dataset(
            tmp_path / "date", by_date, _loader([]), windows=windows, target=16
        )
    # Nothing is written on refusal, and the loader is never called.
    assert not any(tmp_path.rglob("*.jpg"))


def test_the_writer_refuses_windows_that_leave_pixels_uncovered(tmp_path) -> None:
    splits = build_detector_splits(_dataset())
    calls: list[str] = []
    one_window = [Window(0, 0, 40, 40)]
    with pytest.raises(ValueError, match="cover"):
        write_detector_dataset(
            tmp_path / "a", splits, _loader(calls), windows=one_window, target=16
        )
    # Coverage is judged against each image's own size, not FRAME_SIZE: windows
    # that cover 64 x 64 leave a 96 x 96 image partly uncovered.
    big = _image("20220601000000Bh.jpeg", 1, size=96)
    gap = [Window(0, 0, 64, 64), Window(32, 32, 96, 96)]  # misses the far corners
    sized = type(splits)((big,), (), ())
    with pytest.raises(ValueError, match=r"96 x 96.*x 64:96, y 0:32"):
        write_detector_dataset(
            tmp_path / "b", sized, _loader(calls), windows=gap, target=16
        )
    assert not calls
    assert not any(tmp_path.rglob("*.jpg"))


def test_full_cover_by_overlapping_windows_is_accepted(tmp_path) -> None:
    splits = build_detector_splits(_dataset())
    write_detector_dataset(
        tmp_path,
        splits,
        _loader([]),
        windows=frame_windows(FRAME, 40, 24),
        target=16,
        min_area=4,
    )
    assert (tmp_path / "train" / ANNOTATIONS_NAME).exists()


# ------------------------------------------------- pixels and labels line up


def _centroid(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return np.array([cols.mean(), rows.mean()])


def test_pixels_and_labels_line_up_in_windows_away_from_the_origin(tmp_path) -> None:
    """Write, decode the JPEG, rasterize the JSON polygon, compare them.

    Two off-diagonal bars (wide and short, tall and narrow), each in exactly one
    non-origin window of ``frame_windows(64, 40, 24)``: bar A in window 1
    (x 24:64, y 0:40) and bar B in window 2 (x 0:40, y 24:64). A crop with x and
    y swapped, a writer that always crops window 0, labels cropped without the
    window origin, labels swapped, or every example taking window 0's labels
    all give a JPEG and a polygon that disagree here.
    """
    target = 32
    windows = frame_windows(FRAME, 40, 24)
    bar_a = _rect(40, 4, 62, 16)
    bar_b = _rect(6, 42, 18, 62)
    frame = np.full((FRAME, FRAME), 20, dtype=np.uint8)
    frame[4:16, 40:62] = 235
    frame[42:62, 6:18] = 235
    image = _image(
        "20220601000000Bh.jpeg",
        1,
        annotations=(_annotation(1, bar_a), _annotation(2, bar_b)),
    )
    splits = DetectorSplits(
        train=(image,), val=(_image("20220602000000Bh.jpeg", 2),), holdout=()
    )
    write_detector_dataset(
        tmp_path, splits, lambda _: frame, windows=windows, target=target, min_area=4
    )

    train = tmp_path / "train"
    payload = json.loads((train / ANNOTATIONS_NAME).read_text())
    stem = "20220601000000Bh"
    with_bar = set()
    for index in range(len(windows)):
        name = f"{stem}__w{index}__v0.jpg"
        entry = next(i for i in payload["images"] if i["file_name"] == name)
        box = windows[index]
        assert entry["window"] == [box.x0, box.y0, box.x1, box.y1]
        decoded = cv2.imread(str(train / name), cv2.IMREAD_UNCHANGED)
        assert decoded.shape == (target, target, 3)
        pixels = decoded[:, :, 0] > 128
        polygons = [
            a["segmentation"]
            for a in payload["annotations"]
            if a["image_id"] == entry["id"]
        ]
        drawn = np.zeros((target, target), dtype=bool)
        for segmentation in polygons:
            drawn |= rasterize_polygons(segmentation, target, target) > 0
        if not pixels.any():
            assert not polygons, f"window {index} has labels over a dark picture"
            continue
        with_bar.add(index)
        assert drawn.any(), f"window {index} has a bar in the picture and no label"
        # A contour runs through boundary pixel centers, so on bars this small
        # the polygon is narrower than the picture (IoU about 0.8). A crop or
        # label from the wrong place shows a different bar: IoU 0.
        assert _iou(pixels, drawn) > 0.7, f"window {index}: label and pixels differ"
        assert np.abs(_centroid(pixels) - _centroid(drawn)).max() < 1.0
    assert with_bar == {1, 2}
    # Each bar sits where its window says: wide and short in window 1, tall and
    # narrow in window 2.
    shapes = {}
    for index in (1, 2):
        decoded = cv2.imread(str(train / f"{stem}__w{index}__v0.jpg"))
        rows, cols = np.nonzero(decoded[:, :, 0] > 128)
        shapes[index] = (np.ptp(cols) + 1, np.ptp(rows) + 1)
    assert shapes[1][0] > shapes[1][1]
    assert shapes[2][0] < shapes[2][1]
