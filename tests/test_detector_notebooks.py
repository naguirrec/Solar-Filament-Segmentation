"""The three detector notebooks: structure, guards, and their cells on synthetic data.

Nothing here needs torch, rfdetr or a GPU. The sweep and export notebooks have
their compute cells run with stand-ins for the Kaggle inputs, so a wrong call into
``src`` fails here and not on a Kaggle session.
"""

from __future__ import annotations

import ast
import json
import re
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.detector_data import frame_windows
from filament.detector_export import read_detections, window_rects
from filament.detector_fusion import Detection
from filament.gap_analysis import CURRENT_POINT
from filament.stats import weighted_pq

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
TRAIN = NOTEBOOK_DIR / "train_detector.ipynb"
EXPORT = NOTEBOOK_DIR / "export_detections.ipynb"
SWEEP = NOTEBOOK_DIR / "sweep_detector.ipynb"
MERGE = NOTEBOOK_DIR / "sweep_detector_merge.ipynb"
ALL = [TRAIN, EXPORT, SWEEP, MERGE]

SUBMIT_PATTERNS = re.compile(
    r"competition_submit|competitions\s+submit|KaggleApi|import\s+kaggle\b"
    r"|from\s+kaggle\b|kaggle\.api|\.submit\s*\(|kaggle\s+competitions",
    re.IGNORECASE,
)


def _cells(path: Path, kind: str) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == kind
    ]


def _code(path: Path) -> str:
    return "\n".join(_cells(path, "code"))


def _constants(path: Path) -> dict[str, object]:
    found: dict[str, object] = {}
    for source in _cells(path, "code"):
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        found[target.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        pass
    return found


def _module_names(module: str) -> set[str]:
    source = (ROOT / "src" / "filament" / f"{module}.py").read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


# ------------------------------------------------------------------ structure


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_everything_imported_from_filament_exists(path: Path) -> None:
    for source in _cells(path, "code"):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "filament."
            ):
                names = _module_names(node.module.split(".", 1)[1])
                for alias in node.names:
                    assert alias.name in names, (node.module, alias.name)


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_never_submits_and_names_its_inputs(path: Path) -> None:
    assert not SUBMIT_PATTERNS.search(_code(path))
    first = _cells(path, "markdown")[0]
    assert "filament-segmentation-2026" in first
    assert "ignacioaguirrecandia/solar-code" in first
    assert "Save & Run All" in first
    assert "PILOT" in first
    assert "does not submit" in first


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_the_settings_live_in_one_cell_and_pilot_defaults_to_false(path: Path) -> None:
    settings = [
        c for c in _cells(path, "code") if re.search(r"^PILOT = ", c, re.MULTILINE)
    ]
    assert len(settings) == 1
    assert _constants(path)["PILOT"] is False
    # No cell asks for input.
    assert not re.search(r"(?<![\w/.])input\s*\(|getpass", _code(path))


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.name)
def test_the_src_cell_looks_for_the_module_the_notebook_needs(path: Path) -> None:
    source = _cells(path, "code")[0]
    module = {
        TRAIN: "detector_train",
        EXPORT: "detector_export",
        SWEEP: "detector_sweep",
        MERGE: "detector_merge_sweep",
    }[path]
    assert f'"{module}.py"' in source or f"'{module}.py'" in source or module in source


def test_train_settings() -> None:
    constants = _constants(TRAIN)
    assert constants["MODEL_CLASS"] == "RFDETRSegLarge"
    assert constants["RESOLUTION"] == 768
    assert constants["RFDETR_VERSION"] == "1.12.0"
    assert constants["CHECKPOINT_CHOICE"] == "best"
    for name in ("EPOCHS", "BATCH_SIZE", "MAX_TRAIN_HOURS"):
        assert name in constants
    assert constants["PILOT_TRAIN_FILES"] == 40
    assert constants["PILOT_EPOCHS"] == 2
    assert constants["PILOT_MAX_MINUTES"] == 20.0


def test_train_uses_the_defaults_of_the_splits_and_never_builds_one_by_hand() -> None:
    code = _code(TRAIN)
    calls = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "build_detector_splits"
    ]
    assert len(calls) == 1
    assert [ast.unparse(a) for a in calls[0].args] == ["loaded"] and not calls[
        0
    ].keywords
    assert "DetectorSplits(" not in code
    assert "write_detector_dataset(" in code
    # The holdout is only counted, never loaded as frames.
    assert "full_splits.holdout" in code and "splits.holdout" not in code.replace(
        "full_splits.holdout", ""
    )


def test_train_checks_everything_before_training_and_installs_first() -> None:
    code = _code(TRAIN)
    order = [
        '"pip"',
        "check_resolution(RESOLUTION",
        "build_detector_splits(loaded)",
        "write_detector_dataset(",
        "check_coco_folder(path)",
        "model_class(resolution=RESOLUTION)",
        "model.train(",
        "pick_checkpoint(",
        "sha256_file(",
    ]
    positions = [code.find(item) for item in order]
    assert all(p >= 0 for p in positions), positions
    assert positions == sorted(positions)
    assert "Apache License" in code and "license_texts" in code


def test_train_guards_the_clock_and_the_pilot_prints_its_projection() -> None:
    code = _code(TRAIN)
    for fragment in (
        "threading.Timer",
        "interrupt_main",
        "guard_fired",
        "max_memory_allocated",
        "project_full_run(",
        "PILOT seconds per epoch",
        "PILOT peak GPU memory",
        "PILOT projected full run",
        "checkpoint_sha256",
    ):
        assert fragment in code, fragment
    assert "train_kwargs" in code and "SegmentationTrainConfig" in code


def test_train_does_not_read_the_test_images_or_the_holdout_frames() -> None:
    code = _code(TRAIN)
    assert "test_images" not in code
    # The holdout is only counted and reported, never turned into frames.
    for line in code.splitlines():
        if "holdout" in line:
            allowed = (
                "full_splits.holdout",
                "holdout_not_used",
                "never the holdout",
                "read no holdout",
            )
            assert any(item in line for item in allowed), line
    # The one place a frame is read is the loader of train and validation files.
    assert code.count("read_gray(") == 1 and "def load_frame(image)" in code


def test_export_settings() -> None:
    constants = _constants(EXPORT)
    assert constants["SPLITS"] == ("holdout", "test")
    assert constants["SCORE_THRESHOLD"] == 0.01
    assert constants["RESOLUTION"] == 768
    assert "CHECKPOINT" in constants


def test_export_checks_the_checkpoint_hash_and_writes_a_manifest() -> None:
    code = _code(EXPORT)
    for fragment in (
        "sha256_file(",
        "checkpoint_sha256",
        "EXPORT_MANIFEST",
        "inference_seconds_per_image",
        "total_seconds_per_image",
        "window_input(",
        "frame_detections(",
        "DetectionWriter(",
    ):
        assert fragment in code, fragment
    assert "fit(" not in code and "train(" not in code


def test_sweep_settings_and_base() -> None:
    constants = _constants(SWEEP)
    assert constants["MODEL"] == "r34"
    assert constants["EXPECTED_BASE_PQ"] == 0.4255
    assert constants["ALLOW_UNVERIFIED_WEIGHTS"] is False
    code = _code(SWEEP)
    assert "BASE = CURRENT_POINT" in code
    assert CURRENT_POINT.threshold == 0.45
    assert (CURRENT_POINT.close_size, CURRENT_POINT.min_area) == (11, 400)
    assert "ADOPTED_OBJECT_FILTER" in code and "on_disk" in code
    text = "\n".join(_cells(SWEEP, "markdown"))
    assert "+0.01" in text and "Nacho" in text
    assert "export-probs-r34" in text


def test_sweep_prints_the_verdict_line_last_and_reads_no_test_set() -> None:
    last = _cells(SWEEP, "code")[-1].strip().splitlines()[-1]
    assert last == "print(verdict_line(summary))"
    code = _code(SWEEP)
    assert "test_images" not in code and '"test"' not in code
    assert "crossfit_rows(" in code and "crossfit_summary(" in code


# ------------------------------------------------------ cells on synthetic data

SIDE = 128


def _namespace_from_imports(source: str) -> dict[str, object]:
    tree = ast.parse(source)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace: dict[str, object] = {"Path": Path}
    exec(compile(ast.Module(keep, []), "<imports>", "exec"), namespace)  # noqa: S102
    return namespace


def _scene_gray() -> np.ndarray:
    gray = np.full((SIDE, SIDE), 10, dtype=np.uint8)
    rows, cols = np.ogrid[:SIDE, :SIDE]
    gray[(rows - 64) ** 2 + (cols - 64) ** 2 <= 60**2] = 200
    return gray


BOX_A = (40, 30, 50, 90)  # top, left, bottom, right
BOX_B = (70, 30, 80, 90)


def _polygon(box: tuple[int, int, int, int]) -> tuple[float, ...]:
    top, left, bottom, right = box
    return (left, top, right, top, right, bottom, left, bottom)


def _mask(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=bool)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = True
    return mask


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


class _Cache:
    def __init__(self) -> None:
        probability = np.full((SIDE, SIDE), 0.05, dtype=np.float32)
        probability[_mask(BOX_A, BOX_B)] = 0.95
        self._probability = probability

    def mean(self, file_name: str) -> np.ndarray:
        return self._probability.copy()


def test_the_sweep_cells_run_end_to_end_on_a_synthetic_holdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cells = _cells(SWEEP, "code")
    namespace = _namespace_from_imports(next(c for c in cells if "EXPECTED =" in c))
    images = [_image(day) for day in range(1, 13)]
    for image in images:
        cv2.imwrite(str(tmp_path / image.file_name), _scene_gray())
    junk = Detection.from_mask(_mask((100, 10, 110, 30)), 0.05, 0)
    windows = frame_windows(SIDE, 80, 48)
    detections = {
        image.file_name: [
            Detection.from_mask(_mask(BOX_A), 0.9, 0),
            Detection.from_mask(_mask(BOX_B), 0.8, 1),
            junk,
        ]
        for image in images
    }
    namespace.update(
        PILOT=True,
        EXPECTED_BASE_PQ=0.0,
        BASE_PQ_TOLERANCE=1.0,
        BASE=CURRENT_POINT,
        OBJECT_CUT=namespace["object_cut_from_config"](
            namespace["ADOPTED_OBJECT_FILTER"]
        ),
        IMAGE_DIR=tmp_path,
        OUTPUT_PATH=tmp_path / "out" / "sweep.csv",
        holdout=images,
        cache=_Cache(),
        detections_by_file=detections,
        RECTS=window_rects(windows),
    )
    for source in cells:
        if "AXES = axis_settings()" in source or "rows = axis_rows(" in source:
            exec(compile(source, "<cell>", "exec"), namespace)  # noqa: S102
    output = capsys.readouterr().out
    last = output.strip().splitlines()[-1]
    assert re.fullmatch(
        r"VERDICT \S.* pq=\d\.\d{6} diff=[+-]\d\.\d{6} fraction_better=\d\.\d{3}"
        r" adopted=\d/5 base_fixed_in_sample=\d\.\d{6}",
        last,
    ), last
    # A pilot never submits without asking, and the line before says what diff is against.
    assert "no pilot: no\n" in output
    assert "measured against the base at its fixed setting, in sample" in output
    assert "dedup:" in output and "cross-window" in output
    assert "cross-fit adoption" in output and "ADOPT" in output
    csv_text = (tmp_path / "out" / "sweep.csv").read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    assert header.startswith("axis,setting,pooled_pq")
    for axis in (
        "base",
        "score_raw",
        "score_calibrated",
        "dilation",
        "sanity",
        "dedup",
    ):
        assert f"\n{axis}," in csv_text, axis
    assert "crossfit_verdict" in csv_text
    # The base mask finds both filaments, so the base PQ is high.
    assert namespace["base_pq"] > 0.8
    # The solar-disk gate runs inside every claim.
    assert "disk=gate" in (ROOT / "src/filament/detector_sweep.py").read_text()


def test_merge_sweep_settings_and_base() -> None:
    constants = _constants(MERGE)
    assert constants["MODEL"] == "r34"
    assert constants["EXPECTED_BASE_PQ"] == 0.4255
    assert constants["BASE_PQ_TOLERANCE"] == 0.002
    assert constants["ALLOW_UNVERIFIED_WEIGHTS"] is False
    code = _code(MERGE)
    assert "BASE = CURRENT_POINT" in code
    assert "ADOPTED_OBJECT_FILTER" in code and "on_disk" in code
    assert "holdout_detector_merge_sweep.csv" in code
    assert "MIN_PARTITIONS_ADOPTED" in code
    text = "\n".join(_cells(MERGE, "markdown"))
    assert "+0.01" in text and "Nacho" in text
    assert "export-probs-r34" in text and "no one-to-many" in text
    last = _cells(MERGE, "code")[-1].strip().splitlines()[-1]
    assert last == "print(verdict_line(summary))"
    assert "test_images" not in code and '"test"' not in code
    assert "crossfit_rows(" in code and "crossfit_summary(" in code


def test_the_merge_sweep_cells_run_end_to_end_on_a_synthetic_holdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cells = _cells(MERGE, "code")
    namespace = _namespace_from_imports(next(c for c in cells if "EXPECTED =" in c))
    images = [_image(day) for day in range(1, 13)]
    for image in images:
        cv2.imwrite(str(tmp_path / image.file_name), _scene_gray())
    junk = Detection.from_mask(_mask((100, 10, 110, 30)), 0.05, 0)
    windows = frame_windows(SIDE, 80, 48)
    # One detection spans both filaments (the U-Net had them apart), so it joins them.
    detections = {
        image.file_name: [Detection.from_mask(_mask(BOX_A, BOX_B), 0.85, 0), junk]
        for image in images
    }
    namespace.update(
        PILOT=True,
        EXPECTED_BASE_PQ=0.0,
        BASE_PQ_TOLERANCE=1.0,
        BASE=CURRENT_POINT,
        OBJECT_CUT=namespace["object_cut_from_config"](
            namespace["ADOPTED_OBJECT_FILTER"]
        ),
        IMAGE_DIR=tmp_path,
        OUTPUT_PATH=tmp_path / "out" / "merge.csv",
        holdout=images,
        cache=_Cache(),
        detections_by_file=detections,
        RECTS=window_rects(windows),
    )
    for source in cells:
        if "AXES = merge_axis_settings()" in source or "rows = axis_rows(" in source:
            exec(compile(source, "<cell>", "exec"), namespace)  # noqa: S102
    output = capsys.readouterr().out
    last = output.strip().splitlines()[-1]
    assert re.fullmatch(
        r"VERDICT \S.* pq=\d\.\d{6} diff=[+-]\d\.\d{6} fraction_better=\d\.\d{3}"
        r" adopted=\d/5 base_fixed_in_sample=\d\.\d{6}",
        last,
    ), last
    assert "7 settings on 3 axes" in output
    assert "no pilot: no\n" in output
    assert "measured against the base at its fixed setting, in sample" in output
    assert "not reported, src has no helper" in output
    # The diagnostics: every setting prints its parts and its merges.
    assert "tp=" in output and "sq=" in output and "rq=" in output
    assert "merges=12 frames_changed=12" in output  # default: one join in each frame
    csv_text = (tmp_path / "out" / "merge.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("axis,setting,pooled_pq")
    for axis in ("base", "merge_cut", "merge_coverage", "merge_dedup", "claim_fusion"):
        assert f"\n{axis}," in csv_text, axis
    assert "crossfit_verdict" in csv_text
    assert "merges=12 frames_changed=12" in csv_text and "tp=" in csv_text
    assert "merges=0 frames_changed=0" in csv_text  # cut 0.9 is above the score 0.85
    # Joining two filaments the ground truth draws apart loses both matches, so the
    # merge settings fall below the base, and the cut above the score is the base.
    pqs = {str(k): weighted_pq(v) for k, v in namespace["setting_counts"].items()}
    base_pq = namespace["base_pq"]
    assert base_pq > 0.8
    assert pqs["merge cut=0.5 coverage=0.5 dedup=0"] < base_pq
    assert pqs["merge cut=0.9 coverage=0.5 dedup=0"] == pytest.approx(base_pq)


class _Detections:
    def __init__(self, masks: np.ndarray, confidence: np.ndarray) -> None:
        self.mask = masks
        self.confidence = confidence

    def __len__(self) -> int:
        return len(self.confidence)


class _Predictor:
    """A stand-in for the library's predictor: one blob in the middle, one weak."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.calls = 0

    def predict(self, image: np.ndarray, threshold: float) -> _Detections:
        self.calls += 1
        assert image.shape == (self.size, self.size, 3) and image.dtype == np.uint8
        masks = np.zeros((3, self.size, self.size), dtype=bool)
        masks[0, 10:20, 10:60] = True
        masks[1, 30:40, 5:25] = True
        # masks[2] stays empty and must be skipped
        return _Detections(masks, np.array([0.9, 0.02, 0.5]))


class _Cuda:
    @staticmethod
    def synchronize() -> None:
        return None


def test_the_export_cells_write_readable_files_and_a_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cells = _cells(EXPORT, "code")
    data = next(c for c in cells if "def frames_of" in c)
    run = next(c for c in cells if "def predict_window" in c)
    manifest = next(c for c in cells if "manifest = {" in c)
    namespace = _namespace_from_imports(data)
    namespace.update(_namespace_from_imports(manifest))
    size = 32
    windows = frame_windows(SIDE, 80, 48)
    names = [f"frame{n}.jpeg" for n in range(3)]
    for name in names:
        cv2.imwrite(str(tmp_path / name), _scene_gray())
    namespace.update(
        np=np,
        time=__import__("time"),
        json=json,
        torch=types.SimpleNamespace(
            cuda=_Cuda(),
            __version__="stub",
        ),
        RESOLUTION=size,
        SCORE_THRESHOLD=0.05,
        SPLITS=("holdout",),
        WINDOWS=windows,
        FRAMES={"holdout": [(n, tmp_path / n, (SIDE, SIDE)) for n in names]},
        OUTPUT_DIR=tmp_path / "out",
        predictor=_Predictor(size),
        PILOT=False,
        CHECKPOINT_PATH=Path("detector_stub.pth"),
        CHECKPOINT_SHA256="0" * 64,
        MODEL_CLASS="RFDETRSegLarge",
        RFDETR_VERSION="1.12.0",
    )
    namespace["torch"].cuda.get_device_name = lambda index: "stub"
    for key in (
        "EXPORT_MANIFEST",
        "DetectionWriter",
        "detections_name",
        "window_input",
    ):
        assert key in namespace, key
    (tmp_path / "out").mkdir()
    exec(compile(run, "<run>", "exec"), namespace)  # noqa: S102
    exec(compile(manifest, "<manifest>", "exec"), namespace)  # noqa: S102

    stats = namespace["split_stats"]["holdout"]
    # 4 windows x 3 frames, one kept blob per window: the 0.02 one is under 0.05.
    assert stats["frames"] == 3 and stats["detections"] == 12
    assert stats["empty_masks_skipped"] == 12
    frames = list(read_detections(tmp_path / "out" / "detections_holdout.jsonl.gz"))
    assert [name for name, _, _ in frames] == names
    for _, shape, found in frames:
        assert shape == (SIDE, SIDE) and len(found) == 4
        assert sorted(d.window for d in found) == [0, 1, 2, 3]
        assert all(d.shape == shape and d.score == 0.9 for d in found)
    saved = json.loads((tmp_path / "out" / "detector_export_manifest.json").read_text())
    assert saved["checkpoint_sha256"] == "0" * 64
    assert saved["windows"] == [[w.x0, w.y0, w.x1, w.y1] for w in windows]
    assert saved["splits"]["holdout"]["frames"] == 3
    output = capsys.readouterr().out
    assert "first window" in output
    assert "read back detections_holdout.jsonl.gz: 3 frames, 12 detections" in output


def test_the_export_refuses_a_file_that_does_not_read_back(tmp_path: Path) -> None:
    cells = _cells(EXPORT, "code")
    data = next(c for c in cells if "def frames_of" in c)
    run = next(c for c in cells if "def predict_window" in c)
    namespace = _namespace_from_imports(data)
    size = 32
    names = ["frame0.jpeg", "frame1.jpeg"]
    for name in names:
        cv2.imwrite(str(tmp_path / name), _scene_gray())

    class _Truncating(namespace["DetectionWriter"]):  # type: ignore[misc]
        def write_frame(self, file_name, shape, detections):  # type: ignore[no-untyped-def]
            super().write_frame(file_name, shape, detections)
            self.frames += 1  # the counter says more than the file holds

    namespace.update(
        np=np,
        time=__import__("time"),
        json=json,
        torch=types.SimpleNamespace(cuda=_Cuda(), __version__="stub"),
        RESOLUTION=size,
        SCORE_THRESHOLD=0.05,
        SPLITS=("holdout",),
        WINDOWS=frame_windows(SIDE, 80, 48),
        FRAMES={"holdout": [(n, tmp_path / n, (SIDE, SIDE)) for n in names]},
        OUTPUT_DIR=tmp_path / "out",
        predictor=_Predictor(size),
        DetectionWriter=_Truncating,
    )
    with pytest.raises(RuntimeError, match="read back"):
        exec(compile(run, "<run>", "exec"), namespace)  # noqa: S102


def test_the_export_settings_are_a_pilot_of_6_frames_and_a_checkpoint_path() -> None:
    constants = _constants(EXPORT)
    assert constants["PILOT"] is False
    assert constants["PILOT_FRAMES"] == 6
    assert constants["CHECKPOINT"] is None
    code = _code(EXPORT)
    assert 'f"rfdetr[train,loggers]=={RFDETR_VERSION}"' in code
    assert "_pilot" in code and "read_detections(path)" in code
    assert constants["RFDETR_VERSION"] == _constants(TRAIN)["RFDETR_VERSION"]


def test_the_sweep_pilot_reads_the_frames_of_the_export_pilot() -> None:
    assert _constants(SWEEP)["PILOT_FILES"] == _constants(EXPORT)["PILOT_FRAMES"] == 6
    assert "submit_without_asking(summary, pilot=PILOT)" in _code(SWEEP)


def test_no_stale_text_about_epochs_or_the_checkpoint() -> None:
    constants = _constants(TRAIN)
    assert constants["EPOCHS"] == 16
    assert constants["PILOT_STOP_AFTER_MINUTES"] is None
    text = "\n".join(
        [
            *("\n".join(_cells(TRAIN, kind)) for kind in ("markdown", "code")),
            (ROOT / "docs" / "kaggle-gpu.md").read_text(encoding="utf-8"),
        ]
    )
    for stale in (
        "EPOCHS = 20",
        "`checkpoint.pth` every epoch",
        "none has run on Kaggle yet",
        "the final epoch",
        "placeholders until",
    ):
        assert stale not in text, stale


# ------------------------------------------------ the train cells, with stand-ins

_TRAIN_KEYS = (
    "dataset_dir",
    "output_dir",
    "epochs",
    "batch_size",
    "grad_accum_steps",
    "early_stopping",
    "early_stopping_patience",
    "seed",
    "num_workers",
    "tensorboard",
    "run_test",
)


def _train_namespace(tmp_path: Path, train, **settings) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """The train notebook's cells 6 and 7 with a model whose ``train`` is ``train``."""
    import _thread
    import dataclasses
    import shutil
    import threading
    import time
    import traceback

    cells = _cells(TRAIN, "code")
    namespace = _namespace_from_imports(
        next(c for c in cells if "from filament.detector_train import" in c)
    )
    namespace.update(_constants(TRAIN))
    run_dir, out = tmp_path / "run", tmp_path / "out"
    run_dir.mkdir()
    out.mkdir()

    class _Model:
        model_config = types.SimpleNamespace(resolution=768)

        def train(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            assert set(kwargs) == set(_TRAIN_KEYS)
            train(run_dir, namespace)

    cuda = types.SimpleNamespace(
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 2**30,
        max_memory_reserved=lambda: 2**30,
    )
    namespace.update(
        _thread=_thread,
        dataclasses=dataclasses,
        json=json,
        shutil=shutil,
        threading=threading,
        time=time,
        traceback=traceback,
        model_class=lambda resolution: _Model(),
        rfdetr_config=types.SimpleNamespace(
            SegmentationTrainConfig=types.SimpleNamespace(
                model_fields=dict.fromkeys(_TRAIN_KEYS)
            )
        ),
        torch=types.SimpleNamespace(cuda=cuda, __version__="stub"),
        GPU=types.SimpleNamespace(name="stub"),
        PATCH_SIZE=12,
        NUM_WINDOWS=2,
        DATA_ROOT=tmp_path / "data",
        RUN_DIR=run_dir,
        SCRATCH=tmp_path,
        OUTPUT_DIR=out,
        CHECKPOINT_PATH=out / "detector.pth",
        WATCH_PATH=out / "detector.pth",
        MANIFEST_PATH=out / "manifest.json",
        splits=types.SimpleNamespace(train=(), val=()),
        full_splits=types.SimpleNamespace(train=(), val=(), holdout=()),
        REPORTS={},
    )
    namespace.update(settings)
    return namespace


def _run_train_cell(namespace: dict[str, object]) -> None:
    source = next(c for c in _cells(TRAIN, "code") if "model.train(" in c)
    exec(compile(source, "<train>", "exec"), namespace)  # noqa: S102


def _run_checkpoint_cell(namespace: dict[str, object]) -> None:
    source = next(c for c in _cells(TRAIN, "code") if "names = [path.name" in c)
    exec(compile(source, "<checkpoint>", "exec"), namespace)  # noqa: S102


def _run_watcher_once(namespace: dict[str, object]) -> None:
    """One poll of the notebook's watcher, without waiting a minute."""
    ticks = iter([False, True])
    original = namespace["stop_watching"]
    namespace["stop_watching"] = types.SimpleNamespace(wait=lambda _s: next(ticks))
    try:
        namespace["_watch"]()  # type: ignore[operator]
    finally:
        namespace["stop_watching"] = original


@pytest.mark.parametrize(
    "failure",
    [
        SystemExit(1),
        RuntimeError("CUDA out of memory"),
        KeyboardInterrupt(),
    ],
    ids=["SystemExit", "Exception", "KeyboardInterrupt"],
)
def test_a_stopped_training_still_leaves_the_best_checkpoint_in_the_output(
    tmp_path: Path, failure: BaseException, capsys: pytest.CaptureFixture[str]
) -> None:
    def train(run_dir: Path, namespace: dict[str, object]) -> None:
        (run_dir / "checkpoint_best_total.pth").write_bytes(b"epoch 3 weights")
        (run_dir / "last_ema.pth").write_bytes(b"epoch 3 last")
        _run_watcher_once(namespace)
        raise failure

    namespace = _train_namespace(tmp_path, train)
    _run_train_cell(namespace)  # does not raise
    out = tmp_path / "out"
    # The watcher already put the best checkpoint in the output, whatever happened next.
    assert (out / "detector.pth").read_bytes() == b"epoch 3 weights"
    assert str(namespace["train_error"]).startswith(type(failure).__name__)
    assert "TRAINING STOPPED EARLY" in capsys.readouterr().out
    _run_checkpoint_cell(namespace)
    assert (out / "detector.pth").read_bytes() == b"epoch 3 weights"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_error"] == namespace["train_error"]
    assert manifest["checkpoint_source"] == "checkpoint_best_total.pth"
    assert [p.name for p in out.iterdir() if p.name.startswith(".")] == []


def test_a_failure_with_no_checkpoint_at_all_fails_loudly(tmp_path: Path) -> None:
    def train(run_dir: Path, namespace: dict[str, object]) -> None:
        raise RuntimeError("CUDA out of memory at epoch 0")

    namespace = _train_namespace(tmp_path, train)
    _run_train_cell(namespace)
    with pytest.raises(RuntimeError, match=r"No checkpoint.*out of memory"):
        _run_checkpoint_cell(namespace)
    assert not (tmp_path / "out" / "detector.pth").exists()


def test_the_wall_clock_guard_fires_and_the_checkpoint_survives_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import time

    def train(run_dir: Path, namespace: dict[str, object]) -> None:
        (run_dir / "checkpoint_best_ema.pth").write_bytes(b"weights before the guard")
        _run_watcher_once(namespace)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:  # the guard interrupts this sleep
            time.sleep(0.01)
        raise AssertionError("the guard did not fire")

    namespace = _train_namespace(tmp_path, train, PILOT_STOP_AFTER_MINUTES=0.001)
    _run_train_cell(namespace)
    assert namespace["guard_fired"].is_set()  # type: ignore[attr-defined]
    assert namespace["train_error"] is None
    assert namespace["limit_seconds"] == pytest.approx(0.06)
    assert "stopped by the wall-clock guard" in capsys.readouterr().out
    _run_checkpoint_cell(namespace)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["stopped_by_guard"] is True and manifest["train_error"] is None
    assert (
        tmp_path / "out" / "detector.pth"
    ).read_bytes() == b"weights before the guard"


def test_the_last_choice_hands_on_last_ema_and_keeps_the_best_apart(
    tmp_path: Path,
) -> None:
    def train(run_dir: Path, namespace: dict[str, object]) -> None:
        (run_dir / "checkpoint.pth").write_bytes(b"epoch 10")
        (run_dir / "checkpoint_best_total.pth").write_bytes(b"best")
        (run_dir / "last_ema.pth").write_bytes(b"end of training")

    out = tmp_path / "out"
    namespace = _train_namespace(
        tmp_path,
        train,
        CHECKPOINT_CHOICE="last",
        WATCH_PATH=out / "detector_best_so_far.pth",
    )
    _run_train_cell(namespace)
    _run_checkpoint_cell(namespace)
    assert (out / "detector.pth").read_bytes() == b"end of training"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_source"] == "last_ema.pth"
