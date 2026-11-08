"""The torch-free half of the reproduce notebook (``filament.reproduce``), on synthetic data."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.baseline import on_disk
from filament.components import (
    ADOPTED_OBJECT_FILTER,
    ObjectCut,
    filter_components,
)
from filament.gap_analysis import CURRENT_POINT
from filament.instances import labels_for_min_areas
from filament.reproduce import (
    DATA_FOLDER,
    EXPECTED_HOLDOUT_PQ,
    FINAL_CONFIG,
    REPO_ROOT,
    HoldoutPqError,
    PipelineConfig,
    Timing,
    check_accelerator,
    check_holdout_pq,
    iter_submission_rows,
    labels_for_frame,
    overlay,
    pick_overlay_names,
    resolve_data_root,
    score_holdout,
    write_timing,
)
from filament.submission import write_submission
from filament.validate import validate_submission

SIDE = 128


def _gray() -> np.ndarray:
    gray = np.full((SIDE, SIDE), 10, dtype=np.uint8)
    rows, cols = np.ogrid[:SIDE, :SIDE]
    gray[(rows - 64) ** 2 + (cols - 64) ** 2 <= 60**2] = 200
    return gray


BOX_A = (40, 30, 50, 90)  # top, left, bottom, right
BOX_B = (70, 30, 80, 90)


def _mask(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = True
    return mask


def _polygon(box: tuple[int, int, int, int]) -> tuple[float, ...]:
    top, left, bottom, right = box
    return (left, top, right, top, right, bottom, left, bottom)


def _image(day: int) -> TrainImage:
    annotations = tuple(
        FilamentAnnotation(
            id=number,
            image_id=day,
            category_id=1,
            spine=None,
            segmentation=(_polygon(b),),
        )
        for number, b in enumerate((BOX_A, BOX_B), start=1)
    )
    return TrainImage(
        id=day,
        file_name=f"frame{day:02d}.jpeg",
        date_captured=f"2022-06-{day:02d} 10:00:00",
        height=SIDE,
        width=SIDE,
        annotations=annotations,
        views=(annotations,),
    )


def _probability() -> np.ndarray:
    probability = np.full((SIDE, SIDE), 0.05, dtype=np.float32)
    probability[_mask(BOX_A, BOX_B)] = 0.95
    return probability


# ----------------------------------------------------------------- config


def test_the_final_config_is_the_documented_pipeline() -> None:
    assert FINAL_CONFIG.threshold == 0.45
    assert (FINAL_CONFIG.close_size, FINAL_CONFIG.min_area) == (11, 400)
    assert FINAL_CONFIG.object_filter == {"stat": "mean", "min": 0.8}
    assert FINAL_CONFIG.tta is True
    assert FINAL_CONFIG.grid_point == CURRENT_POINT
    assert FINAL_CONFIG.object_cut == ObjectCut("mean", 0.8)
    assert EXPECTED_HOLDOUT_PQ == 0.4255


# ----------------------------------------------------------- environment


def test_no_gpu_stops_unless_the_cpu_is_allowed() -> None:
    assert check_accelerator(True, False) == "cuda"
    assert check_accelerator(True, True) == "cuda"
    assert check_accelerator(False, True) == "cpu"
    with pytest.raises(RuntimeError, match="No GPU found.*ALLOW_CPU"):
        check_accelerator(False, False)


def test_the_data_root_comes_from_the_setting_then_the_environment(
    tmp_path: Path,
) -> None:
    def make(name: str) -> Path:
        root = tmp_path / name
        (root / "test" / "test_images").mkdir(parents=True)
        return root

    one, two, three = make("one"), make("two"), make("three")
    nowhere = tmp_path / "nowhere"
    kwargs = {"kaggle_root": nowhere, "local_roots": (three,)}
    assert resolve_data_root(one, environ={}, **kwargs) == one
    assert (
        resolve_data_root(None, environ={"FILAMENT_DATA_ROOT": str(two)}, **kwargs)
        == two
    )
    assert resolve_data_root(None, environ={}, **kwargs) == three
    assert resolve_data_root(nowhere, environ={}, **kwargs) == three  # skipped


def test_local_data_is_found_from_the_repo_not_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Jupyter starts the kernel in notebooks/. The data the docs say to put in
    # <repo>/data/MAGFiLO_1.0_Kaggle_2026 has to be found from there.
    repo = tmp_path / "repo"
    data = repo / "data" / DATA_FOLDER
    (data / "test" / "test_images").mkdir(parents=True)
    (repo / "notebooks").mkdir()
    monkeypatch.chdir(repo / "notebooks")
    nowhere = tmp_path / "nowhere"
    assert resolve_data_root(None, environ={}, kaggle_root=nowhere, repo_root=repo) == (
        data
    )
    # The plain data/ folder of the repo is the second choice.
    plain = tmp_path / "plain"
    (plain / "data" / "test" / "test_images").mkdir(parents=True)
    assert resolve_data_root(
        None, environ={}, kaggle_root=nowhere, repo_root=plain
    ) == (plain / "data")
    # A data/ folder under the working directory is not looked at.
    other = tmp_path / "other"
    (other / "data" / DATA_FOLDER / "test" / "test_images").mkdir(parents=True)
    monkeypatch.chdir(other)
    with pytest.raises(FileNotFoundError, match="Tried"):
        resolve_data_root(
            None, environ={}, kaggle_root=nowhere, repo_root=tmp_path / "empty"
        )


def test_the_default_repo_root_is_the_checkout_that_holds_this_module() -> None:
    assert (REPO_ROOT / "src" / "filament" / "reproduce.py").is_file()


def test_a_missing_data_root_lists_what_was_tried(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Tried") as caught:
        resolve_data_root(
            tmp_path / "a",
            environ={},
            kaggle_root=tmp_path / "b",
            local_roots=(tmp_path / "c",),
        )
    for name in "abc":
        assert str(tmp_path / name) in str(caught.value)


# ------------------------------------------------------------- one frame


def test_a_frame_keeps_the_strong_filaments_and_drops_the_rest() -> None:
    gray = _gray()
    probability = _probability()
    probability[_mask((100, 30, 110, 90))] = 0.6  # above the threshold, mean < 0.8
    probability[_mask((20, 30, 25, 40))] = 0.95  # strong but 50 px: under min_area
    labels = labels_for_frame(gray, probability, FINAL_CONFIG)
    assert labels.max() == 2
    assert (labels > 0).sum() == 2 * 10 * 60
    assert (
        labels[45, 60] != labels[75, 60] and labels[45, 60] > 0 and labels[75, 60] > 0
    )
    assert labels[105, 60] == 0 and labels[22, 35] == 0


def test_two_pieces_less_than_close_size_apart_are_joined_by_the_closing() -> None:
    gray = _gray()
    probability = np.full((SIDE, SIDE), 0.05, dtype=np.float32)
    # 8 px gap (rows 50 to 57): under close_size 11, so one filament.
    probability[_mask((40, 30, 50, 90), (58, 30, 68, 90))] = 0.95
    joined = labels_for_frame(gray, probability, FINAL_CONFIG)
    assert joined.max() == 1
    assert joined[45, 60] == joined[63, 60] > 0
    # Without the closing they are two, so this test depends on close_size.
    apart = labels_for_frame(gray, probability, replace(FINAL_CONFIG, close_size=0))
    assert apart.max() == 2
    assert apart[45, 60] != apart[63, 60]


def test_the_disk_gate_clears_what_is_off_the_disk() -> None:
    gray = _gray()
    probability = np.full((SIDE, SIDE), 0.05, dtype=np.float32)
    probability[2:12, 5:65] = 0.95  # corner of the frame: sky
    assert on_disk(gray)[7, 30] == False
    assert labels_for_frame(gray, probability, FINAL_CONFIG).max() == 0


def test_labels_for_frame_is_the_label_then_filter_code_of_the_sweeps() -> None:
    gray, probability = _gray(), _probability()
    probability[_mask((100, 30, 110, 90))] = 0.6
    semantic = (probability >= 0.45).astype(np.uint8)
    expected = filter_components(
        labels_for_min_areas(semantic, gray, [400], close_size=11, disk=on_disk(gray))[
            400
        ],
        probability,
        ObjectCut("mean", 0.8),
    )
    np.testing.assert_array_equal(labels_for_frame(gray, probability), expected)


def test_a_probability_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="does not match"):
        labels_for_frame(_gray(), np.zeros((4, 4), dtype=np.float32))


def test_without_an_object_filter_the_weak_component_stays() -> None:
    gray, probability = _gray(), _probability()
    probability[_mask((100, 30, 110, 90))] = 0.6
    config = PipelineConfig(0.45, 11, 400, None, True)
    assert labels_for_frame(gray, probability, config).max() == 3


# ------------------------------------------------------------ submission


def test_submission_rows_are_written_validated_and_timed(tmp_path: Path) -> None:
    import time

    paths = [tmp_path / "20220101000000Mh.jpeg", tmp_path / "20220102000000Mh.jpeg"]
    calls: list[tuple[int, ...]] = []

    def predict(gray: np.ndarray) -> np.ndarray:
        calls.append(gray.shape)
        time.sleep(0.01)
        return _probability()

    timing = Timing()
    seen: list[int] = []
    out = tmp_path / "submission.csv"
    write_submission(
        out,
        iter_submission_rows(
            paths,
            predict,
            FINAL_CONFIG,
            timing,
            reader=lambda path: _gray(),
            progress=lambda index, total, path, count: seen.append(count),
        ),
    )
    validate_submission(out, height=SIDE, width=SIDE)
    lines = out.read_text().splitlines()
    assert lines[0] == "filament_id,segmentation_rle"
    ids = [line.split(",")[0] for line in lines[1:]]
    assert ids == [
        "20220101000000Mh_1",
        "20220101000000Mh_2",
        "20220102000000Mh_1",
        "20220102000000Mh_2",
    ]
    assert seen == [2, 2] and calls == [(SIDE, SIDE)] * 2
    assert timing.images == 2
    assert 0.02 <= timing.inference_seconds <= timing.total_seconds
    summary = timing.summary("test")
    assert summary["images"] == 2
    assert summary["inference_seconds_per_image"] >= 0.01
    assert summary["total_seconds_per_image"] >= summary["inference_seconds_per_image"]


def test_timing_with_no_images_is_nan_and_writes_json(tmp_path: Path) -> None:
    timing = Timing()
    assert np.isnan(timing.inference_per_image) and np.isnan(timing.total_per_image)
    write_timing(tmp_path / "t.json", [timing.summary("empty")])
    assert '"label": "empty"' in (tmp_path / "t.json").read_text()


# --------------------------------------------------------------- holdout


def test_the_holdout_is_scored_with_the_existing_pooled_pq(tmp_path: Path) -> None:
    images = [_image(day) for day in range(1, 7)]
    for image in images:
        cv2.imwrite(str(tmp_path / image.file_name), _gray())
    timing = Timing()
    progress: list[int] = []
    kept = [images[1].file_name, images[4].file_name]
    result = score_holdout(
        images,
        tmp_path,
        lambda gray: _probability(),
        FINAL_CONFIG,
        timing,
        keep=kept,
        progress=lambda index, total, image: progress.append(index),
    )
    # The probability map is the ground truth, so the masks are found.
    assert result.pq > 0.8
    assert (result.files, result.views) == (6, 6)
    assert sorted(result.probabilities) == sorted(kept)
    assert progress == [1, 2, 3, 4, 5, 6]
    assert timing.images == 6 and timing.inference_seconds <= timing.total_seconds


def test_a_bad_prediction_scores_low(tmp_path: Path) -> None:
    images = [_image(day) for day in range(1, 4)]
    for image in images:
        cv2.imwrite(str(tmp_path / image.file_name), _gray())
    result = score_holdout(
        images,
        tmp_path,
        lambda gray: np.zeros((SIDE, SIDE), dtype=np.float32),
        FINAL_CONFIG,
        Timing(),
    )
    assert result.pq == 0.0


def test_the_pq_check_passes_inside_the_tolerance_and_stops_outside_it() -> None:
    assert "OK" in check_holdout_pq(0.4255)
    assert "OK" in check_holdout_pq(0.4255 + 0.0019)
    assert "OK" in check_holdout_pq(0.4255 - 0.0019)
    for measured in (0.4276, 0.4234, 0.38, 0.0):
        with pytest.raises(HoldoutPqError, match="Stopping"):
            check_holdout_pq(measured)
    with pytest.raises(HoldoutPqError):
        check_holdout_pq(float("nan"))


def test_a_pilot_never_stops_and_says_it_is_not_a_result() -> None:
    line = check_holdout_pq(0.1, pilot=True)
    assert "PILOT" in line and "not a result" in line


# -------------------------------------------------------------- overlays


def test_the_overlay_frames_are_fixed_by_the_seed() -> None:
    names = [f"frame{n:03d}.jpeg" for n in range(40)]
    first = pick_overlay_names(names, 6, 0)
    assert first == pick_overlay_names(list(reversed(names)), 6, 0)
    assert len(first) == len(set(first)) == 6 and first == sorted(first)
    assert first != pick_overlay_names(names, 6, 1)
    assert pick_overlay_names(names[:4], 6, 0) == sorted(names[:4])


def test_the_overlay_tints_instances_and_leaves_the_rest() -> None:
    gray = _gray()
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    labels[_mask(BOX_A)] = 1
    labels[_mask(BOX_B)] = 2
    picture = overlay(gray, labels)
    assert picture.shape == (SIDE, SIDE, 3) and picture.dtype == np.uint8
    assert not np.array_equal(picture[45, 60], picture[75, 60])
    assert (picture[45, 60] != gray[45, 60]).any()
    np.testing.assert_array_equal(picture[5, 5], [10, 10, 10])
    plain = overlay(gray, np.zeros_like(labels))
    np.testing.assert_array_equal(plain, np.stack([gray] * 3, axis=-1))
    with pytest.raises(ValueError, match="do not match"):
        overlay(gray, np.zeros((4, 4), dtype=np.int32))


def test_the_documented_object_filter_is_the_adopted_one() -> None:
    assert FINAL_CONFIG.object_filter is ADOPTED_OBJECT_FILTER
