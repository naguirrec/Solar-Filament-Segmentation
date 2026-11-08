"""``notebooks/evaluate_final.ipynb``: structure, guards, and its cells on synthetic data.

Nothing here needs torch or a GPU. The compute cells are run with stand-ins for
the Kaggle inputs, so a wrong call into ``src`` fails here and not on a Kaggle
session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ast
import json
import re

import cv2
import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.final_eval import EXAMPLE_KINDS

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "evaluate_final.ipynb"
MODULES = ("final_metrics", "final_eval", "final_figures")

SUBMIT_PATTERNS = re.compile(
    r"competition_submit|competitions\s+submit|KaggleApi|import\s+kaggle\b"
    r"|from\s+kaggle\b|kaggle\.api|\.submit\s*\(|kaggle\s+competitions",
    re.IGNORECASE,
)


def _cells(kind: str) -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == kind
    ]


def _code() -> str:
    return "\n".join(_cells("code"))


def _constants() -> dict[str, object]:
    found: dict[str, object] = {}
    for source in _cells("code"):
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


def test_everything_imported_from_filament_exists() -> None:
    for source in _cells("code"):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "filament."
            ):
                names = _module_names(node.module.split(".", 1)[1])
                for alias in node.names:
                    assert alias.name in names, (node.module, alias.name)


def test_never_submits_and_names_its_inputs() -> None:
    assert not SUBMIT_PATTERNS.search(_code())
    first = _cells("markdown")[0]
    assert "filament-segmentation-2026" in first
    assert "ignacioaguirrecandia/solar-code" in first
    assert "ignacioaguirrecandia/export-probs-r34" in first
    assert "Save & Run All" in first
    assert "PILOT" in first
    assert "does not submit" in first
    assert "self-evaluation-notebook" in first


def test_the_settings_live_in_one_cell_and_pilot_defaults_to_false() -> None:
    settings = [c for c in _cells("code") if re.search(r"^PILOT = ", c, re.MULTILINE)]
    assert len(settings) == 1
    constants = _constants()
    assert constants["PILOT"] is False
    assert constants["MODEL"] == "r34"
    assert constants["ALLOW_UNVERIFIED_WEIGHTS"] is False
    assert constants["EXPECTED_BASE_PQ"] == 0.4255
    assert constants["BASE_PQ_TOLERANCE"] == 0.002
    assert not re.search(r"(?<![\w/.])input\s*\(|getpass", _code())


def test_the_src_cell_looks_for_the_modules_the_notebook_needs() -> None:
    source = _cells("code")[0]
    assert '"final_figures.py"' in source
    # final_figures imports the other two, so one marker covers the dataset version.
    figures = (ROOT / "src/filament/final_figures.py").read_text(encoding="utf-8")
    assert "from filament.final_eval import" in figures
    assert "from filament.final_metrics import" in figures
    for module in MODULES:
        assert (ROOT / "src/filament" / f"{module}.py").is_file()


def test_reads_no_test_image_and_the_pipeline_is_not_a_setting() -> None:
    code = _code()
    assert "test_images" not in code and '"test"' not in code
    assert "write_submission" not in code and "submission.csv" not in code.replace(
        "no submission", ""
    ).replace("submission.csv`", "")
    # The pipeline comes from filament.final_eval, not from numbers typed here.
    for name in ("THRESHOLD", "CLOSE_SIZE", "MIN_AREA"):
        assert name not in _constants()


def test_the_pooled_pq_guard_precedes_every_write() -> None:
    code = _code()
    assert code.index("BASE_PQ_TOLERANCE}") < code.index("write_summary(OUTPUT_PATH")
    assert code.index("write_summary(OUTPUT_PATH") < code.index("save_figure(")


def test_captions_carry_the_attribution_for_every_figure() -> None:
    captions = next(c for c in _cells("markdown") if "Figure captions" in c)
    for name in (
        "iou_distribution.png",
        "dice_distribution.png",
        "relations.png",
        *(f"overlay_{kind}.png" for kind in EXAMPLE_KINDS),
    ):
        assert name in captions, name
    assert captions.count("GONG data, NSO/AURA/NSF") >= 3
    assert "ATTRIBUTION" in _code()


# ------------------------------------------------------ cells on synthetic data

SIDE = 128
BOX_A = (40, 30, 50, 90)  # top, left, bottom, right
BOX_B = (70, 30, 80, 90)
BOX_C = (10, 40, 22, 90)  # annotated, but the probability map has nothing there
BOX_FP = (100, 50, 120, 90)  # predicted, not annotated


def _scene_gray() -> np.ndarray:
    gray = np.full((SIDE, SIDE), 10, dtype=np.uint8)
    rows, cols = np.ogrid[:SIDE, :SIDE]
    gray[(rows - 64) ** 2 + (cols - 64) ** 2 <= 60**2] = 200
    return gray


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
        for number, b in enumerate((BOX_A, BOX_B, BOX_C), start=1)
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
        probability[_mask(BOX_A, BOX_B, BOX_FP)] = 0.95
        self._probability = probability

    def mean(self, file_name: str) -> np.ndarray:
        return self._probability.copy()


def _namespace(tmp_path: Path, *, pilot: bool, expected: float) -> dict[str, object]:
    cells = _cells("code")
    imports = next(c for c in cells if "EXPECTED =" in c)
    tree = ast.parse(imports)
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace: dict[str, object] = {"Path": Path}
    exec(compile(ast.Module(keep, []), "<imports>", "exec"), namespace)  # noqa: S102
    images = [_image(day) for day in range(1, 13)]
    for image in images:
        cv2.imwrite(str(tmp_path / image.file_name), _scene_gray())
    namespace.update(
        PILOT=pilot,
        EXPECTED_BASE_PQ=expected,
        BASE_PQ_TOLERANCE=0.002,
        EXAMPLES_PER_KIND=2,
        SEED=0,
        IMAGE_DIR=tmp_path,
        OUTPUT_DIR=tmp_path / "out",
        OUTPUT_PATH=tmp_path / "out" / "holdout_final_metrics.csv",
        holdout=images,
        cache=_Cache(),
    )
    return namespace


def _run(namespace: dict[str, object], *markers: str) -> None:
    cells = _cells("code")
    for marker in markers:
        source = next(c for c in cells if marker in c)
        exec(compile(source, "<cell>", "exec"), namespace)  # noqa: S102


def test_the_compute_cells_run_end_to_end_on_a_synthetic_holdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    namespace = _namespace(tmp_path, pilot=True, expected=0.0)
    _run(
        namespace,
        "results = []",
        "rows = summary_rows(results)",
        "iou_scores = positive_scores",
        "attribution for every figure",
    )
    output = capsys.readouterr().out
    last = output.strip().splitlines()[-1]
    assert re.fullmatch(
        r"FINAL pq=\d\.\d{6} sq=\d\.\d{6} rq=\d\.\d{6} mean_dice=\d\.\d{6}"
        r" one_to_many=\d+ many_to_one=\d+",
        last,
    ), last
    assert "PILOT: a plumbing check" in output
    assert "GONG data, NSO/AURA/NSF" in output
    assert "did not submit" in output

    # Per frame: A and B matched, C missed, the FP blob predicted. 12 frames, 1 view each.
    totals = namespace["totals"]
    assert (totals.tp, totals.fp, totals.fn) == (24, 12, 12)  # type: ignore[attr-defined]
    assert 0.5 < namespace["base_pq"] < 0.8  # type: ignore[operator]

    csv_lines = (
        (tmp_path / "out" / "holdout_final_metrics.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert csv_lines[0] == "metric,value,source"
    metrics = {line.split(",")[0]: line.split(",")[1] for line in csv_lines[1:]}
    for name in (
        "pq", "sq", "rq", "tp", "fp", "fn", "mean_dice", "iou_q05", "iou_q95",
        "dice_q05", "dice_q95", "one_to_many", "many_to_one",
    ):  # fmt: skip
        assert name in metrics, name
    assert metrics["tp"] == "24" and metrics["gt_miss_1to0"] == "12"
    assert metrics["pred_miss_0to1"] == "12"

    for name in (
        "iou_distribution.png",
        "dice_distribution.png",
        "relations.png",
        "overlay_good_match.png",
        "overlay_miss.png",
        "overlay_false_positive.png",
    ):
        path = tmp_path / "out" / name
        assert path.is_file() and path.stat().st_size < 400_000, name
    # The kinds the synthetic scene has no example of are skipped, not drawn empty.
    assert "no fragmentation example" in output and "no over_merge example" in output
    assert not (tmp_path / "out" / "overlay_fragmentation.png").exists()
    # Nothing but numbers and PNGs is written.
    suffixes = {p.suffix for p in (tmp_path / "out").iterdir()}
    assert suffixes == {".csv", ".png"}


def test_a_real_run_stops_when_the_pooled_pq_does_not_reproduce(
    tmp_path: Path,
) -> None:
    namespace = _namespace(tmp_path, pilot=False, expected=0.4255)
    with pytest.raises(RuntimeError, match="differs from 0.4255"):
        _run(namespace, "results = []")
    # Stopped before anything was written.
    assert not (tmp_path / "out").exists()


def test_a_pilot_does_not_stop_on_the_pq_guard(tmp_path: Path) -> None:
    namespace = _namespace(tmp_path, pilot=True, expected=0.4255)
    _run(namespace, "results = []")
    assert namespace["base_pq"] != pytest.approx(0.4255, abs=0.002)
