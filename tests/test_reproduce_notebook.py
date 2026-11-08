"""notebooks/reproduce.ipynb: structure, guards, and its cells on synthetic data.

Nothing here needs torch or a GPU. The cells that load the weights are only
checked as text; the cells after them run with a stand-in ``predict``.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.components import ADOPTED_OBJECT_FILTER
from filament.gap_analysis import CURRENT_POINT
from filament.reproduce import (
    EXPECTED_HOLDOUT_PQ,
    FINAL_CONFIG,
    PQ_TOLERANCE,
    HoldoutPqError,
    PipelineConfig,
)
from filament.weights import R34_FP16_FILE

ROOT = Path(__file__).resolve().parents[1]
REPRODUCE = ROOT / "notebooks" / "reproduce.ipynb"
INFER = ROOT / "notebooks" / "infer_unet_crops.ipynb"

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


def _code(path: Path = REPRODUCE) -> str:
    return "\n".join(_cells(path, "code"))


def _constants(path: Path = REPRODUCE) -> dict[str, object]:
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


def _cell_with(fragment: str) -> str:
    matches = [c for c in _cells(REPRODUCE, "code") if fragment in c]
    assert len(matches) == 1, (fragment, len(matches))
    return matches[0]


# ------------------------------------------------------------- structure


def test_everything_imported_from_filament_exists() -> None:
    for source in _cells(REPRODUCE, "code"):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "filament."
            ):
                names = _module_names(node.module.split(".", 1)[1])
                for alias in node.names:
                    assert alias.name in names, (node.module, alias.name)


def test_never_submits_never_asks_for_input_and_names_its_inputs() -> None:
    assert not SUBMIT_PATTERNS.search(_code())
    assert not re.search(r"(?<![\w/.])input\s*\(|getpass", _code())
    first = _cells(REPRODUCE, "markdown")[0]
    assert "filament-segmentation-2026" in first
    assert "ignacioaguirrecandia/solar-code" in first
    assert "Save & Run All" in first
    assert "PILOT" in first and "does not submit" in first
    assert "train-unet-r34-crops-80ep" in first


def test_the_jury_route_needs_nothing_private() -> None:
    first = _cells(REPRODUCE, "markdown")[0]
    assert "jury route, nothing private" in first
    assert "uploaded by you as your own Kaggle dataset" in first
    assert "filament/reproduce.py" in first
    # solar-code is named only as the author's own route.
    assert "*Author's own route:*" in first
    assert first.count("solar-code") == 1
    assert first.index("solar-code") > first.index("*Author's own route:*")
    code = _code()
    assert "Do not make the GitHub" not in code
    assert "make the GitHub repository public" not in code
    assert "ignacioaguirrecandia" not in code
    assert "upload this repository as a dataset of" in code
    # The marker is the module this notebook needs, so an older dataset is caught.
    assert code.count('"reproduce.py"') == 2
    assert "crop_training.py" not in code


def test_the_predictor_cell_uses_the_tta_and_overlap_settings() -> None:
    cell = _cell_with("predict = ensemble_predictor(")
    assert "tta=USE_TTA" in cell
    assert "overlap=TILE_OVERLAP" in cell
    assert "batch_size=INFER_BATCH_SIZE" in cell
    assert "use_amp=use_amp" in cell and "tile_size=TILE_SIZE" in cell


def test_local_data_is_looked_up_from_the_repo_root() -> None:
    assert "resolve_data_root(DATA_ROOT, repo_root=REPO_ROOT)" in _code()


def test_the_checkpoint_mismatch_message_names_no_override_this_notebook_lacks() -> (
    None
):
    cell = _cell_with("verify_and_report(")
    assert "remedy=(" in cell
    assert "ALLOW_UNVERIFIED_WEIGHTS" not in cell


def test_the_settings_are_in_the_first_code_cell_and_pilot_is_false() -> None:
    cells = _cells(REPRODUCE, "code")
    assert re.search(r"^PILOT = ", cells[0], re.MULTILINE)
    assert sum(bool(re.search(r"^PILOT = ", c, re.MULTILINE)) for c in cells) == 1
    constants = _constants()
    assert constants["PILOT"] is False
    assert constants["ALLOW_CPU"] is False
    assert "filament" not in cells[0]  # the settings cell imports nothing


def test_the_settings_are_the_documented_final_pipeline() -> None:
    constants = _constants()
    assert constants["THRESHOLD"] == CURRENT_POINT.threshold == 0.45
    assert constants["CLOSE_SIZE"] == CURRENT_POINT.close_size == 11
    assert constants["MIN_AREA"] == CURRENT_POINT.min_area == 400
    assert constants["OBJECT_FILTER"] == ADOPTED_OBJECT_FILTER
    assert constants["USE_TTA"] is True
    assert constants["EXPECTED_HOLDOUT_PQ"] == EXPECTED_HOLDOUT_PQ == 0.4255
    assert constants["PQ_TOLERANCE"] == PQ_TOLERANCE == 0.002
    assert constants["WEIGHTS_FILE"] == R34_FP16_FILE
    assert constants["TILE_OVERLAP"] == 128 and constants["INFER_BATCH_SIZE"] == 8
    rebuilt = PipelineConfig(
        threshold=constants["THRESHOLD"],
        close_size=constants["CLOSE_SIZE"],
        min_area=constants["MIN_AREA"],
        object_filter=constants["OBJECT_FILTER"],
        tta=constants["USE_TTA"],
    )
    assert rebuilt == FINAL_CONFIG
    assert constants["OVERLAY_COUNT"] == 6


def test_the_weights_are_checked_and_a_wrong_hash_has_no_way_around() -> None:
    code = _code()
    assert "ALLOW_UNVERIFIED_WEIGHTS" not in code
    assert "allow_unverified" not in code
    assert (
        "verify_and_report(\n        fingerprint,\n        expected=R34_CROPS," in code
    )
    assert "allow_unverified" not in code
    assert "verify_repo_weights(checkpoint, weights_file, entry)" in code
    assert "locate_weights(REPO_ROOT, WEIGHTS_FILE, KAGGLE_WEIGHTS_FILE)" in code
    assert 'print(f"weights from the {source.kind}: {source.path}")' in code


def test_the_order_is_gpu_then_weights_then_holdout_check_then_the_csv() -> None:
    code = _code()
    order = [
        "check_accelerator(torch.cuda.is_available(), ALLOW_CPU)",
        "locate_weights(REPO_ROOT",
        "model = ct.model_from_checkpoint(",
        "score_holdout(",
        "check_holdout_pq(",
        "labels_for_frame(gray",
        "write_submission(",
        "validate_submission(SUBMISSION_PATH)",
        "write_timing(",
    ]
    positions = [code.find(item) for item in order]
    assert all(p >= 0 for p in positions), positions
    assert positions == sorted(positions)


def test_the_logic_is_in_src_and_not_copied_into_cells() -> None:
    code = _code()
    for helper in (
        "def predict_probability",
        "def labels_from_semantic",
        "def filter_components",
        "def weighted_pq",
        "def tensor_sha256",
        "def encode_mask",
    ):
        assert helper not in code
    for call in (
        "ensemble_predictor(",
        "iter_submission_rows(",
        "score_holdout(",
        "overlay(gray, labels)",
    ):
        assert call in code


def test_the_notebook_reports_the_seconds_per_image() -> None:
    code = _code()
    assert "inference_seconds_per_image" in code
    assert "total_seconds_per_image" in code
    assert "timing.json" in code


# ------------------------------------------------------ the infer notebook


def test_infer_unet_crops_stops_without_a_gpu_unless_allowed() -> None:
    constants = _constants(INFER)
    assert constants["ALLOW_CPU"] is False
    cell = next(c for c in _cells(INFER, "code") if "device = torch.device(" in c)
    assert 'if device.type != "cuda" and not ALLOW_CPU:' in cell
    assert "raise RuntimeError(" in cell
    assert "No GPU found" in cell
    text = "\n".join(_cells(INFER, "markdown"))
    assert "the notebook **stops**" in text
    assert "On CPU the notebook still runs" not in text


# ------------------------------------------------ cells on synthetic data

SIDE = 128
BOX_A = (40, 30, 50, 90)
BOX_B = (70, 30, 80, 90)
FULL = 2048


def _mask(side: int, *boxes: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = True
    return mask


def _disk_gray(side: int, radius: int) -> np.ndarray:
    gray = np.full((side, side), 10, dtype=np.uint8)
    rows, cols = np.ogrid[:side, :side]
    centre = side // 2
    gray[(rows - centre) ** 2 + (cols - centre) ** 2 <= radius**2] = 200
    return gray


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


def _predict(gray: np.ndarray) -> np.ndarray:
    """A stand-in for the network: the filaments are where the boxes are."""
    side = gray.shape[0]
    probability = np.full(gray.shape, 0.05, dtype=np.float32)
    if side == SIDE:
        probability[_mask(side, BOX_A, BOX_B)] = 0.95
    else:  # a 2048 px test frame: one filament and one weak blob
        probability[_mask(side, (900, 800, 960, 1300))] = 0.95
        probability[_mask(side, (1100, 800, 1160, 1300))] = 0.6
    return probability


@pytest.fixture
def scene(tmp_path: Path) -> dict[str, object]:
    images = [_image(day) for day in range(1, 9)]
    for image in images:
        cv2.imwrite(str(tmp_path / image.file_name), _disk_gray(SIDE, 60))
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    test_paths = []
    for stem in ("20220301000000Mh", "20220302000000Mh"):
        path = test_dir / f"{stem}.jpeg"
        cv2.imwrite(str(path), _disk_gray(FULL, 900))
        test_paths.append(path)
    out = tmp_path / "out"
    out.mkdir()
    return {
        "holdout": images,
        "IMAGE_DIR": tmp_path,
        "test_paths": test_paths,
        "OUTPUT": out,
        "predict": _predict,
        "PIPELINE": FINAL_CONFIG,
        "DEVICE": "cpu",
        "Path": Path,
        **_constants(),
    }


def _run(fragment: str, namespace: dict[str, object]) -> None:
    exec(compile(_cell_with(fragment), "<cell>", "exec"), namespace)  # noqa: S102


def test_the_holdout_cell_stops_when_the_pq_is_not_the_documented_one(
    scene: dict[str, object],
) -> None:
    with pytest.raises(HoldoutPqError, match="Stopping"):
        _run("holdout_result = score_holdout(", scene)
    assert scene["holdout_result"].pq > 0.8  # type: ignore[attr-defined]


def test_the_cells_run_end_to_end_on_a_synthetic_pilot(
    scene: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    scene["PILOT"] = True
    _run("holdout_result = score_holdout(", scene)
    _run("figure, axes = plt.subplots(", scene)
    _run("test_timing = Timing()", scene)
    _run("summaries = [", scene)
    output = capsys.readouterr().out
    out = scene["OUTPUT"]
    assert isinstance(out, Path)

    assert "PILOT: holdout PQ" in output and "not a result" in output
    assert (out / "overlays_holdout.png").stat().st_size > 0
    lines = (out / "submission.csv").read_text().splitlines()
    assert lines[0] == "filament_id,segmentation_rle"
    # One strong filament per test image; the weak blob is dropped by the filter.
    assert [line.split(",")[0] for line in lines[1:]] == [
        "20220301000000Mh_1",
        "20220302000000Mh_1",
    ]
    assert "wrote 2 rows for 2 images" in output
    assert "validated with filament.validate" in output
    timing = json.loads((out / "timing.json").read_text())
    assert (
        timing[0]["images"] == 2 and "PILOT (not a measurement)" in timing[0]["label"]
    )
    assert "CPU (not a GPU measurement)" in timing[0]["label"]
    assert "inference" in output and "whole pipeline" in output
    assert output.strip().splitlines()[-1] == (
        "This notebook wrote the CSV. It did not submit it."
    )
    assert len(scene["overlay_names"]) == 6  # type: ignore[arg-type]


def test_a_documented_pq_that_matches_lets_the_run_go_on(
    scene: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    # The stand-in finds the boxes, so its PQ is its own number; use that as the
    # "documented" one to see the success line.
    scene["EXPECTED_HOLDOUT_PQ"] = 1.0
    scene["PQ_TOLERANCE"] = 1.0
    _run("holdout_result = score_holdout(", scene)
    assert ": OK" in capsys.readouterr().out
