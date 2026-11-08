"""Helpers of the detector training notebook, on synthetic data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.detector_data import Window, frame_windows
from filament.detector_train import (
    BEST_CHECKPOINTS,
    LAST_CHECKPOINT,
    check_coco_folder,
    check_resolution,
    copy_atomic,
    dataset_report,
    guard_seconds,
    pick_checkpoint,
    project_full_run,
    sha256_file,
    subsample_evenly,
    sync_best_checkpoint,
)

FRAME = 64


def _box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, ...]:
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def _image(*boxes: tuple[float, ...]) -> TrainImage:
    annotations = tuple(
        FilamentAnnotation(
            id=index, image_id=1, category_id=1, spine=None, segmentation=(box,)
        )
        for index, box in enumerate(boxes, start=1)
    )
    return TrainImage(
        id=1,
        file_name="20200101000000Mh.jpg",
        date_captured="2020-01-01 00:00:00",
        height=FRAME,
        width=FRAME,
        annotations=annotations,
        views=(annotations,),
    )


# --------------------------------------------------------------- resolution


@pytest.mark.parametrize("resolution", [504, 768, 24, 1536])
def test_resolutions_that_fit_patch_12_with_2_windows_pass(resolution) -> None:
    check_resolution(resolution, 12, 2)


@pytest.mark.parametrize("resolution", [500, 750, 770, 12])
def test_other_resolutions_fail_loudly(resolution) -> None:
    with pytest.raises(ValueError, match="multiple"):
        check_resolution(resolution, 12, 2)


def test_nano_has_one_window_so_312_passes() -> None:
    check_resolution(312, 12, 1)
    with pytest.raises(ValueError):
        check_resolution(0, 12, 1)
    with pytest.raises(ValueError):
        check_resolution(768, 12, True)  # type: ignore[arg-type]


# ---------------------------------------------------------------- subsample


def test_subsample_is_even_ordered_and_bounded() -> None:
    items = list(range(100))
    assert subsample_evenly(items, 10) == tuple(range(0, 100, 10))
    assert subsample_evenly(items, 100) == tuple(items)
    assert subsample_evenly(items, 500) == tuple(items)
    assert len(set(subsample_evenly(list(range(7)), 3))) == 3
    with pytest.raises(ValueError):
        subsample_evenly(items, 0)


# ------------------------------------------------------------ dataset report


def test_report_counts_negatives_and_the_instances_min_area_drops() -> None:
    # One big filament in the first window, one 3 x 3 speck (9 px) in the second,
    # and nothing at all in the third.
    image = _image(_box(2, 2, 14, 14), _box(40, 40, 43, 43))
    windows = (Window(0, 0, 32, 32), Window(32, 32, 64, 64), Window(32, 0, 64, 32))
    report = dataset_report([image], windows, target=32, min_visible=0.25, min_area=20)
    assert report.files == 1
    assert report.examples == 3
    assert report.instances == 1
    assert report.dropped_by_min_area == 1
    assert report.negative == 2
    assert report.negative_fraction == pytest.approx(2 / 3)
    kept = dataset_report([image], windows, target=32, min_area=0)
    assert kept.instances == 2 and kept.dropped_by_min_area == 0 and kept.negative == 1


def test_report_of_nothing_is_zero() -> None:
    report = dataset_report([], frame_windows(64, 32, 32), target=32)
    assert (report.files, report.examples, report.negative_fraction) == (0, 0, 0.0)


# -------------------------------------------------------------- coco folder


def _write_folder(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.jpg").write_bytes(b"x")
    (root / "b.jpg").write_bytes(b"x")
    payload = {
        "categories": [{"id": 1, "name": "filament"}],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 32, "height": 32},
            {"id": 2, "file_name": "b.jpg", "width": 32, "height": 32},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[1, 1, 9, 1, 9, 9]],
                "bbox": [1, 1, 8, 8],
                "area": 32,
                "iscrowd": 0,
            }
        ],
    }
    (root / "_annotations.coco.json").write_text(json.dumps(payload))
    return root


def test_a_good_folder_is_counted(tmp_path: Path) -> None:
    check = check_coco_folder(_write_folder(tmp_path / "train"))
    assert (check.images, check.annotations, check.negative_images) == (2, 1, 1)
    assert check.category_ids == (1,)


def test_a_bad_folder_lists_its_problems(tmp_path: Path) -> None:
    folder = _write_folder(tmp_path / "a")
    path = folder / "_annotations.coco.json"
    payload = json.loads(path.read_text())
    payload["images"].append(
        {"id": 3, "file_name": "gone.jpg", "width": 8, "height": 8}
    )
    payload["annotations"] += [
        {
            "id": 2,
            "image_id": 9,
            "category_id": 1,
            "segmentation": [[1, 1, 2, 2, 3, 3]],
            "bbox": [0, 0, 1, 5],
        },
        {
            "id": 3,
            "image_id": 3,
            "category_id": 7,
            "segmentation": [[1, 1, 2, 2]],
            "bbox": [0, 0, 20, 5],
        },
    ]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as caught:
        check_coco_folder(folder, max_problems=10)
    message = str(caught.value)
    for fragment in ("missing file gone.jpg", "unknown image 9", "unknown category"):
        assert fragment in message
    assert "under 3 points" in message and "box outside" in message


# -------------------------------------------------------------- checkpoints


def test_best_prefers_total_then_ema_then_regular() -> None:
    names = [
        "checkpoint.pth",
        "checkpoint_best_regular.pth",
        "checkpoint_best_ema.pth",
        "last_ema.pth",
    ]
    assert pick_checkpoint(names, "best") == "checkpoint_best_ema.pth"
    assert pick_checkpoint([*names, BEST_CHECKPOINTS[0]], "best") == BEST_CHECKPOINTS[0]
    assert pick_checkpoint(names, "last") == LAST_CHECKPOINT == "last_ema.pth"


def test_the_pilot_listing_of_rfdetr_1_12_0_resolves() -> None:
    # Files the 2-epoch Kaggle pilot actually wrote.
    pilot = ["checkpoint_best_ema.pth", "checkpoint_best_total.pth", "last_ema.pth"]
    assert pick_checkpoint(pilot, "best") == "checkpoint_best_total.pth"
    assert pick_checkpoint(pilot, "last") == "last_ema.pth"


def test_last_never_takes_the_every_tenth_epoch_checkpoint() -> None:
    # checkpoint.pth is written every checkpoint_interval (10) epochs: after 16 epochs
    # it holds epoch 10, so it must not be called the final epoch.
    with pytest.raises(FileNotFoundError, match=r"checkpoint\.pth"):
        pick_checkpoint(["checkpoint.pth", "checkpoint_best_total.pth"], "last")


def test_a_renamed_best_checkpoint_is_found_only_when_unambiguous() -> None:
    assert pick_checkpoint(["checkpoint.pth", "model_best.pth"]) == "model_best.pth"
    with pytest.raises(FileNotFoundError, match="model_best"):
        pick_checkpoint(["a_best.pth", "b_best.pth", "model_best_x.pth"])


def test_no_checkpoint_lists_what_was_there() -> None:
    with pytest.raises(FileNotFoundError, match="events.out"):
        pick_checkpoint(["events.out.tfevents"], "last")
    with pytest.raises(ValueError):
        pick_checkpoint([], "newest")


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "w.pth"
    data = b"abc" * 100_000
    path.write_bytes(data)
    assert sha256_file(path, chunk=1000) == hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- projection


def test_projection_scales_train_and_validation_files_separately() -> None:
    # The Kaggle pilot: 218 s for 2 epochs on 40 train + 10 validation files.
    projection = project_full_run(
        pilot_seconds=218,
        pilot_epochs=2,
        pilot_train_files=40,
        pilot_val_files=10,
        full_train_files=520,
        full_val_files=58,
        full_epochs=16,
    )
    assert projection.seconds_per_epoch == 109
    # Validation at no cost: 109 s x 520 / 40, not x 578 / 50 (which would be 1260 s).
    assert projection.projected_epoch_seconds == pytest.approx(109 * 520 / 40)
    assert projection.projected_hours == pytest.approx(109 * 13 * 16 / 3600)
    assert projection.projected_hours == pytest.approx(6.3, abs=0.05)


def test_projection_with_a_validation_cost() -> None:
    # Validation files cost half a train file: units 40 + 5 in the pilot, 520 + 29.
    projection = project_full_run(200, 2, 40, 10, 520, 58, 10, val_cost_ratio=0.5)
    assert projection.projected_epoch_seconds == pytest.approx(100 * 549 / 45)
    # With the same mix on both sides the split does not matter.
    even = project_full_run(100, 1, 40, 10, 400, 100, 1, val_cost_ratio=0.3)
    assert even.projected_epoch_seconds == pytest.approx(1000)


def test_projection_refuses_nonsense() -> None:
    with pytest.raises(ValueError):
        project_full_run(0, 2, 40, 10, 520, 58, 16)
    with pytest.raises(ValueError):
        project_full_run(10, 0, 40, 10, 520, 58, 16)
    with pytest.raises(ValueError):
        project_full_run(10, 2, 0, 10, 520, 58, 16)
    with pytest.raises(ValueError):
        project_full_run(10, 2, 40, -1, 520, 58, 16)
    with pytest.raises(ValueError):
        project_full_run(10, 2, 40, 10, 520, 58, 16, val_cost_ratio=1.5)


# ------------------------------------------------------------------ the guard


def test_guard_seconds_uses_the_stop_after_setting_when_it_is_given() -> None:
    kwargs = {"max_train_hours": 7.0, "pilot_max_minutes": 20.0}
    assert guard_seconds(pilot=False, **kwargs) == 7 * 3600
    assert guard_seconds(pilot=True, **kwargs) == 20 * 60
    assert guard_seconds(pilot=True, stop_after_minutes=3, **kwargs) == 180
    assert guard_seconds(pilot=False, stop_after_minutes=0.5, **kwargs) == 30
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            guard_seconds(pilot=True, stop_after_minutes=bad, **kwargs)


# ------------------------------------------------- copying to the saved output


def test_copy_atomic_leaves_no_temp_file_and_keeps_the_old_file_on_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run" / "best.pth"
    source.parent.mkdir()
    source.write_bytes(b"new weights")
    target = tmp_path / "out" / "detector.pth"
    target.parent.mkdir()
    target.write_bytes(b"old weights")
    copy_atomic(source, target)
    assert target.read_bytes() == b"new weights"
    assert [p.name for p in target.parent.iterdir()] == ["detector.pth"]
    with pytest.raises(FileNotFoundError):
        copy_atomic(tmp_path / "missing.pth", target)
    assert target.read_bytes() == b"new weights"
    assert [p.name for p in target.parent.iterdir()] == ["detector.pth"]


def test_the_watcher_copies_the_best_checkpoint_whenever_it_changes(
    tmp_path: Path,
) -> None:
    run, out = tmp_path / "run", tmp_path / "out" / "detector.pth"
    run.mkdir()
    seen: dict[str, tuple[int, int]] = {}
    # Nothing yet, and a last-epoch file alone is not a best checkpoint.
    assert sync_best_checkpoint(run, out, seen) is None
    (run / "last_ema.pth").write_bytes(b"last")
    assert sync_best_checkpoint(run, out, seen) is None
    assert not out.exists()
    # The first best checkpoint is copied; an unchanged one is not copied again.
    (run / "checkpoint_best_ema.pth").write_bytes(b"epoch 2")
    assert sync_best_checkpoint(run, out, seen) == "checkpoint_best_ema.pth"
    assert out.read_bytes() == b"epoch 2"
    out.write_bytes(b"touched")
    assert sync_best_checkpoint(run, out, seen) is None
    assert out.read_bytes() == b"touched"
    # A better epoch rewrites the file (new size): copied again.
    (run / "checkpoint_best_ema.pth").write_bytes(b"epoch 5, better")
    assert sync_best_checkpoint(run, out, seen) == "checkpoint_best_ema.pth"
    assert out.read_bytes() == b"epoch 5, better"
    # The preferred name (total) takes over once it exists.
    (run / "checkpoint_best_total.pth").write_bytes(b"total")
    assert sync_best_checkpoint(run, out, seen) == "checkpoint_best_total.pth"
    assert out.read_bytes() == b"total"
