"""Detections on disk, on synthetic windows."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from filament.detector_data import Window, frame_windows, resize_window
from filament.detector_export import (
    EXPORT_MANIFEST,
    DetectionWriter,
    detections_name,
    find_detection_export,
    frame_detections,
    read_detections,
    window_input,
    window_rects,
)
from filament.detector_fusion import Detection

SHAPE = (64, 64)


def test_masks_are_mapped_into_the_frame_by_the_window_geometry() -> None:
    window = Window(32, 16, 64, 48)  # 32 x 32 at x 32..64, y 16..48
    masks = np.zeros((3, 16, 16), dtype=bool)  # the detector sees it at half size
    masks[0, 2:6, 4:10] = True
    masks[1] = False  # empty: not a detection
    masks[2, 0, 0] = True
    found, empty = frame_detections(masks, [0.9, 0.8, 0.02], 1, window, SHAPE, 0.05)
    assert empty == 1
    assert [d.score for d in found] == [0.9]
    (only,) = found
    # Rows 2..6 and columns 4..10 at half size are rows 4..12 and columns 8..20
    # of the window, so rows 20..28 and columns 40..52 of the frame.
    assert only.bbox == (20, 40, 28, 52)
    assert only.window == 1 and only.shape == SHAPE
    assert only.decode().sum() == 8 * 12


def test_the_threshold_is_inclusive_and_inputs_are_checked() -> None:
    window = Window(0, 0, 32, 32)
    masks = np.ones((2, 32, 32), dtype=bool)
    found, _ = frame_detections(masks, [0.05, 0.0499], 0, window, SHAPE, 0.05)
    assert len(found) == 1
    with pytest.raises(ValueError, match="scores"):
        frame_detections(masks, [0.5], 0, window, SHAPE)
    with pytest.raises(ValueError, match="masks"):
        frame_detections(masks[0], [0.5], 0, window, SHAPE)


def _detections() -> list[Detection]:
    first = np.zeros(SHAPE, dtype=bool)
    first[5:15, 3:30] = True
    second = np.zeros(SHAPE, dtype=bool)
    second[40:44, 20:60] = True
    return [Detection.from_mask(first, 0.97, 0), Detection.from_mask(second, 0.013, 3)]


def test_the_file_round_trips_frames_scores_windows_and_masks(tmp_path: Path) -> None:
    path = tmp_path / "out" / detections_name("holdout")
    detections = _detections()
    with DetectionWriter(path) as writer:
        writer.write_frame("a.jpg", SHAPE, detections)
        writer.write_frame("b.jpg", SHAPE, [])
    assert (writer.frames, writer.detections) == (2, 2)
    frames = list(read_detections(path))
    assert [name for name, _, _ in frames] == ["a.jpg", "b.jpg"]
    _, shape, back = frames[0]
    assert shape == SHAPE and frames[1][2] == []
    for original, restored in zip(detections, back, strict=True):
        assert restored == original
        assert np.array_equal(restored.decode(), original.decode())


def test_a_frame_written_twice_or_with_another_shape_is_refused(tmp_path: Path) -> None:
    with DetectionWriter(tmp_path / "x.jsonl.gz") as writer:
        writer.write_frame("a.jpg", SHAPE, _detections())
        with pytest.raises(ValueError, match="twice"):
            writer.write_frame("a.jpg", SHAPE, [])
        with pytest.raises(ValueError, match="shape"):
            writer.write_frame("c.jpg", (32, 32), _detections())


def test_window_rects_are_top_left_bottom_right() -> None:
    windows = frame_windows(64, 40, 24)
    rects = window_rects(windows)
    assert rects[1] == (0, 24, 40, 64)
    assert len(rects) == len(windows)


def test_window_input_is_the_training_window_after_the_jpeg_round_trip(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    window = Window(0, 16, 48, 64)
    image = window_input(gray, window, target=24, quality=95)
    assert image.shape == (24, 24, 3) and image.dtype == np.uint8
    assert np.array_equal(image[:, :, 0], image[:, :, 2])
    # The training files are JPEGs of the same resize: reading one gives the input.
    small = resize_window(gray[16:64, 0:48], 24)
    path = tmp_path / "w.jpg"
    cv2.imwrite(
        str(path),
        np.repeat(small[:, :, None], 3, axis=2),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    assert np.array_equal(image, cv2.imread(str(path))[:, :, ::-1])


def test_window_input_checks_its_arguments() -> None:
    gray = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="fit"):
        window_input(gray, Window(0, 0, 80, 80), 24)
    with pytest.raises(ValueError, match="uint8"):
        window_input(gray.astype(np.float32), Window(0, 0, 32, 32), 24)


def test_split_names_are_safe() -> None:
    assert detections_name("test") == "detections_test.jsonl.gz"
    for bad in ("", "../x", ".hidden"):
        with pytest.raises(ValueError):
            detections_name(bad)


def _export(directory: Path, split: str = "holdout") -> Path:
    directory.mkdir(parents=True)
    (directory / detections_name(split)).write_bytes(b"")
    (directory / EXPORT_MANIFEST).write_text("{}")
    return directory


def test_find_export_walks_any_depth_and_refuses_none_or_two(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="export_detections"):
        find_detection_export(tmp_path, "holdout")
    nested = _export(tmp_path / "notebooks" / "me" / "export-detections")
    assert find_detection_export(tmp_path, "holdout") == nested
    with pytest.raises(FileNotFoundError):  # no file for that split
        find_detection_export(tmp_path, "test")
    _export(tmp_path / "other")
    with pytest.raises(RuntimeError, match="Several"):
        find_detection_export(tmp_path, "holdout")


def test_the_competition_tree_is_never_walked(tmp_path: Path) -> None:
    _export(tmp_path / "competitions" / "x")
    with pytest.raises(FileNotFoundError):
        find_detection_export(tmp_path, "holdout")
