"""Private Kaggle dataset ``src`` lookup from the notebooks.

The helper has to run before ``filament`` can be imported, so it lives in
the notebook. These tests load that function against a temporary tree.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
REAL_PARTS = (
    "datasets",
    "ignacioaguirrecandia",
    "week1-solar-filament-segmentation",
    "Solar-Filament-Segmentation-main",
    "src",
)


def _load(notebook: str, name: str, directory: Path) -> Callable[..., Path | None]:
    path = NOTEBOOKS / notebook
    notebook_json = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook_json["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        if f"def {name}" not in source:
            continue
        module = ast.parse(source)
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                segment = ast.get_source_segment(source, node)
                assert segment is not None
                file = directory / f"{path.stem}_{name}.py"
                file.write_text(
                    "import os\nfrom pathlib import Path\n\n" + segment + "\n",
                    encoding="utf-8",
                )
                spec = importlib.util.spec_from_file_location(
                    f"{path.stem}_{name}",
                    file,
                )
                assert spec is not None
                assert spec.loader is not None
                loaded = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(loaded)
                return getattr(loaded, name)
    raise AssertionError(name)


def _marker(src: Path, filename: str) -> None:
    package = src / "filament"
    package.mkdir(parents=True, exist_ok=True)
    (package / filename).write_text("# marker\n", encoding="utf-8")


def _real_layout(root: Path) -> Path:
    return root.joinpath(*REAL_PARTS)


@pytest.mark.parametrize(
    ("notebook", "marker"),
    [
        ("train_unet.ipynb", "train_data.py"),
        ("week1_submission.ipynb", "runtime.py"),
    ],
)
def test_finds_real_layout_and_skips_competitions(
    tmp_path: Path,
    notebook: str,
    marker: str,
) -> None:
    find = _load(notebook, "_private_dataset_src", tmp_path)
    src = _real_layout(tmp_path)
    _marker(src, marker)
    # Shallower than the private dataset. Must not win, and must not be walked.
    _marker(
        tmp_path / "competitions" / "filament-segmentation-2026" / "src",
        marker,
    )
    assert find(tmp_path) == src


def test_shallow_match_wins(tmp_path: Path) -> None:
    find = _load("train_unet.ipynb", "_private_dataset_src", tmp_path)
    shallow = tmp_path / "attached" / "src"
    _marker(shallow, "train_data.py")
    _marker(_real_layout(tmp_path), "train_data.py")
    assert find(tmp_path) == shallow


def test_same_depth_uses_path_order(tmp_path: Path) -> None:
    find = _load("train_unet.ipynb", "_private_dataset_src", tmp_path)
    earlier = tmp_path / "a-dataset" / "src"
    later = tmp_path / "b-dataset" / "src"
    _marker(later, "train_data.py")
    _marker(earlier, "train_data.py")
    assert find(tmp_path) == earlier


def test_competitions_tree_alone_is_not_a_match(tmp_path: Path) -> None:
    find = _load("train_unet.ipynb", "_private_dataset_src", tmp_path)
    _marker(tmp_path / "competitions" / "deep" / "src", "train_data.py")
    assert find(tmp_path) is None
    assert find(tmp_path / "absent") is None


def test_week1_does_not_require_train_data(tmp_path: Path) -> None:
    train = _load("train_unet.ipynb", "_private_dataset_src", tmp_path)
    week1 = _load("week1_submission.ipynb", "_private_dataset_src", tmp_path)
    src = _real_layout(tmp_path)
    _marker(src, "runtime.py")
    assert week1(tmp_path) == src
    assert train(tmp_path) is None


def test_error_names_the_real_kaggle_layout() -> None:
    for name in ("train_unet.ipynb", "week1_submission.ipynb"):
        notebook_json = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            if isinstance(cell["source"], list)
            else cell["source"]
            for cell in notebook_json["cells"]
            if cell["cell_type"] == "code"
        )
        assert "/kaggle/input/datasets/ignacioaguirrecandia/" in code
        assert "week1-solar-filament-segmentation/" in code
        assert "Solar-Filament-Segmentation-main/src" in code


# --- the crop notebooks -------------------------------------------------------

CROP_NOTEBOOKS = ("train_unet_crops.ipynb", "infer_unet_crops.ipynb")
SOLAR_CODE_PARTS = (
    "datasets",
    "ignacioaguirrecandia",
    "solar-code",
    "Solar-Filament-Segmentation-main",
    "src",
)


@pytest.mark.parametrize("notebook", CROP_NOTEBOOKS)
def test_crop_notebooks_find_the_solar_code_layout(
    tmp_path: Path, notebook: str
) -> None:
    find = _load(notebook, "_private_dataset_src", tmp_path)
    src = tmp_path.joinpath(*SOLAR_CODE_PARTS)
    _marker(src, "crop_training.py")
    # Shallower, but under competitions: must not win and must not be walked.
    _marker(
        tmp_path / "competitions" / "filament-segmentation-2026" / "src",
        "crop_training.py",
    )
    assert find(tmp_path) == src


@pytest.mark.parametrize("notebook", CROP_NOTEBOOKS)
def test_crop_notebooks_reject_a_dataset_without_the_crop_modules(
    tmp_path: Path, notebook: str
) -> None:
    # An older version of the dataset has train_data.py but not crop_training.py.
    find = _load(notebook, "_private_dataset_src", tmp_path)
    _marker(tmp_path.joinpath(*SOLAR_CODE_PARTS), "train_data.py")
    assert find(tmp_path) is None
    assert find(tmp_path / "absent") is None


@pytest.mark.parametrize("notebook", CROP_NOTEBOOKS)
def test_crop_notebooks_shallow_match_and_path_order(
    tmp_path: Path, notebook: str
) -> None:
    find = _load(notebook, "_private_dataset_src", tmp_path)
    _marker(tmp_path.joinpath(*SOLAR_CODE_PARTS), "crop_training.py")
    shallow = tmp_path / "attached" / "src"
    _marker(shallow, "crop_training.py")
    assert find(tmp_path) == shallow
    same_depth = tmp_path / "a-dataset" / "src"
    _marker(same_depth, "crop_training.py")
    assert find(tmp_path) == same_depth  # equal depth: path order decides


@pytest.mark.parametrize("notebook", CROP_NOTEBOOKS)
def test_crop_notebooks_name_the_real_kaggle_layout_in_the_error(notebook: str) -> None:
    notebook_json = json.loads((NOTEBOOKS / notebook).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook_json["cells"]
        if cell["cell_type"] == "code"
    )
    assert "/kaggle/input/datasets/ignacioaguirrecandia/" in code
    assert "solar-code/" in code
    assert "Solar-Filament-Segmentation-main/src" in code
    assert "copytree" not in code  # src is used in place, not copied to working
