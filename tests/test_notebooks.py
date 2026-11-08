"""Structure and contents of the notebooks. Pure Python: no torch, no execution."""

from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from filament.checkpoint import CROP_CHECKPOINTS, R18_12EP, R34_CROPS
from filament.crop_config import (
    CROP_MODELS,
    DEFAULT_CROP_MODEL,
    DEFAULT_INFER_BATCH_SIZE,
    DEFAULT_THRESHOLD,
    DEFAULT_TILE_OVERLAP,
    CropTrainConfig,
)
from filament.sweep import (
    DEFAULT_CLOSE_SIZES,
    DEFAULT_GRID_MIN_AREAS,
    DEFAULT_MIN_AREAS,
    DEFAULT_THRESHOLDS,
    TTA_CLOSE_SIZES,
    TTA_MIN_AREAS,
    TTA_THRESHOLDS,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
ALL_NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
TRAIN = NOTEBOOK_DIR / "train_unet_crops.ipynb"
INFER = NOTEBOOK_DIR / "infer_unet_crops.ipynb"
POSTPROC = NOTEBOOK_DIR / "sweep_postprocess_r34.ipynb"
TTA_ENSEMBLE = NOTEBOOK_DIR / "sweep_tta_ensemble.ipynb"
EXPORT = NOTEBOOK_DIR / "export_probs.ipynb"
GAP = NOTEBOOK_DIR / "analyze_holdout_gap.ipynb"
OBJECT_FILTER = NOTEBOOK_DIR / "sweep_object_filter.ipynb"
SELECTOR = NOTEBOOK_DIR / "sweep_selector.ipynb"
CROP_ENSEMBLE = NOTEBOOK_DIR / "sweep_crop_ensemble.ipynb"
MODEL_NOTEBOOKS = [
    TRAIN,
    INFER,
    POSTPROC,
    TTA_ENSEMBLE,
    EXPORT,
    GAP,
    OBJECT_FILTER,
    SELECTOR,
]

# Anything that would submit to Kaggle from a notebook.
SUBMIT_PATTERNS = re.compile(
    r"competition_submit|competitions\s+submit|KaggleApi|import\s+kaggle\b"
    r"|from\s+kaggle\b|kaggle\.api|\.submit\s*\(|kaggle\s+competitions",
    re.IGNORECASE,
)


def _text(source: object) -> str:
    return "".join(source) if isinstance(source, list) else str(source)


def _cells(path: Path, kind: str) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [_text(c["source"]) for c in notebook["cells"] if c["cell_type"] == kind]


def _code(path: Path) -> str:
    return "\n".join(_cells(path, "code"))


def _constants(path: Path) -> dict[str, object]:
    """Top-level ``NAME = <literal>`` assignments over all code cells."""
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


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_clean_v4_json(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 0
    assert isinstance(notebook["metadata"], dict)
    ids = []
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}, index
        assert isinstance(cell["source"], (str, list)), index
        assert isinstance(cell["metadata"], dict), index
        if notebook["nbformat_minor"] >= 5:
            ids.append(cell["id"])
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], f"{path.name} cell {index} has outputs"
            assert cell["execution_count"] is None, f"{path.name} cell {index}"
        else:
            assert "outputs" not in cell and "execution_count" not in cell
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_validates_with_nbformat_when_installed(path: Path) -> None:
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse(path: Path) -> None:
    for source in _cells(path, "code"):
        ast.parse(source)


def test_the_two_new_notebooks_exist() -> None:
    assert TRAIN.is_file() and INFER.is_file()


# --- training notebook --------------------------------------------------------


def test_train_constants() -> None:
    constants = _constants(TRAIN)
    expected = {
        "MODEL": "r34",
        "CROPS_PER_EPOCH": 4520,
        "CROP_SIZE": 512,
        "USE_AMP": True,
        "SEED": 0,
        "P_FG": 0.67,
        "TILE_OVERLAP": 128,
        "INFER_BATCH_SIZE": 8,
        "THRESHOLD": 0.5,
        "CLOSE_SIZE": 0,
        "MIN_AREAS": (80, 140, 240, 400, 700, 1200, 2000, 3000),
    }
    for name, value in expected.items():
        assert constants[name] == value, name
        assert type(constants[name]) is type(value), name


def test_model_is_the_first_setting_of_the_config_cell() -> None:
    for source in _cells(TRAIN, "code"):
        if "EPOCHS = " in source:
            first = next(
                node for node in ast.parse(source).body if isinstance(node, ast.Assign)
            )
            assert first.targets[0].id == "MODEL"
            return
    raise AssertionError("no cell sets EPOCHS")


def test_train_settings_that_differ_by_model_come_from_the_preset() -> None:
    code = _code(TRAIN)
    assert "PRESET = crop_model_config(MODEL)" in code
    assert "EPOCHS = PRESET.epochs" in code
    assert "BATCH_SIZE = PRESET.batch_size" in code
    assert "MAX_TRAIN_HOURS = PRESET.max_train_hours" in code
    assert "LEARNING_RATE = PRESET.learning_rate" in code
    assert "encoder_name=PRESET.encoder_name" in code
    assert "encoder_weights=PRESET.encoder_weights" in code
    assert 'encoder_name="resnet34"' not in code


def _train_config_keywords() -> dict[str, str]:
    for source in _cells(TRAIN, "code"):
        for node in ast.parse(source).body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "CONFIG"
            ):
                return {kw.arg: ast.unparse(kw.value) for kw in node.value.keywords}
    raise AssertionError("no cell builds CONFIG")


def test_train_config_passes_every_field_so_no_recipe_setting_is_lost() -> None:
    keywords = _train_config_keywords()
    assert set(keywords) == {f.name for f in dataclasses.fields(CropTrainConfig)}
    for name in (
        "optimizer",
        "weight_decay",
        "lr_schedule",
        "warmup_epochs",
        "grad_clip_norm",
        "ema_decay",
        "agreement_weights",
    ):
        assert keywords[name] == f"PRESET.{name}"


def test_train_notebook_builds_the_cache_the_recipe_needs() -> None:
    assert "agreement_weights=CONFIG.agreement_weights" in _code(TRAIN)


def test_train_constants_agree_with_the_library_defaults() -> None:
    constants = _constants(TRAIN)
    defaults = CropTrainConfig()
    assert defaults.epochs == R34_CROPS.epochs
    assert constants["CROPS_PER_EPOCH"] == defaults.crops_per_epoch
    assert constants["CROP_SIZE"] == defaults.crop_size
    assert constants["P_FG"] == defaults.p_fg
    assert constants["TILE_OVERLAP"] == DEFAULT_TILE_OVERLAP
    assert constants["INFER_BATCH_SIZE"] == DEFAULT_INFER_BATCH_SIZE
    assert constants["THRESHOLD"] == DEFAULT_THRESHOLD
    assert constants["MIN_AREAS"] == DEFAULT_MIN_AREAS


def test_train_notebook_never_submits_or_reads_the_test_set() -> None:
    code = _code(TRAIN)
    assert not SUBMIT_PATTERNS.search(code)
    assert "submission.csv" not in code
    assert "write_submission" not in code
    assert "test_images" not in code.replace(
        'print("test/test_images is not used")', ""
    )
    assert "input(" not in code


def test_train_notebook_outputs_and_pins() -> None:
    code = _code(TRAIN)
    assert "OUTPUT_DIR / Path(EXPECTED.weights_path).name" in code
    assert 'OUTPUT_DIR / f"holdout_sweep_{MODEL}.csv"' in code
    assert "predictor_name=Path(EXPECTED.weights_path).stem" in code
    assert 'OUTPUT_DIR = Path("/kaggle/working")' in code
    assert 'SMP_VERSION = "0.6.0"' in code
    assert "smp.__version__ != SMP_VERSION" in code
    assert "ct.require_cuda(6.0)" in code
    assert "checkpoint_fingerprint" in code
    assert "train-unet-r34-crops" not in code  # one source for the path


# --- inference notebook -------------------------------------------------------


def test_infer_constants() -> None:
    constants = _constants(INFER)
    expected = {
        "MIN_AREA": 400,
        "CLOSE_SIZE": 0,
        "TILE_OVERLAP": 128,
        "INFER_BATCH_SIZE": 8,
        "THRESHOLD": 0.5,
        "USE_AMP": True,
        "ALLOW_UNVERIFIED_WEIGHTS": False,
        "SEED": 0,
    }
    for name, value in expected.items():
        assert constants[name] == value, name
        assert type(constants[name]) is type(value), name
    assert constants["TILE_OVERLAP"] == DEFAULT_TILE_OVERLAP
    assert constants["INFER_BATCH_SIZE"] == DEFAULT_INFER_BATCH_SIZE


def test_infer_default_weights_come_from_the_config() -> None:
    expected = (
        "/kaggle/input/notebooks/ignacioaguirrecandia/"
        "train-unet-r34-crops-80ep/unet_r34_crops.pt"
    )
    assert R34_CROPS.weights_path == expected
    code = _code(INFER)
    assert "WEIGHTS_PATH = EXPECTED.weights_path" in code
    assert "EXPECTED_TENSOR_SHA256 = EXPECTED.tensor_sha256" in code
    assert "PREFERRED_NAME = Path(EXPECTED.weights_path).name" in code
    assert "ignacioaguirrecandia" not in code.replace(
        "ignacioaguirrecandia/solar-code", ""
    ).replace("/kaggle/input/datasets/ignacioaguirrecandia/", "")
    assert expected in _cells(INFER, "markdown")[0]
    assert "train-unet-r34-crops" in _cells(INFER, "markdown")[0]


def test_infer_writes_the_csv_and_never_submits() -> None:
    code = _code(INFER)
    assert not SUBMIT_PATTERNS.search(code)
    assert 'Path("/kaggle/working/submission.csv")' in code
    assert "write_submission(" in code
    assert "validate_submission(" in code
    assert "verify_and_report(" in code
    assert "expected=EXPECTED" in code
    assert "train_images" not in code
    assert "input(" not in code
    assert 'SMP_VERSION = "0.6.0"' in code


def test_infer_unet_r18_takes_its_path_from_the_config() -> None:
    code = _code(NOTEBOOK_DIR / "infer_unet.ipynb")
    assert "WEIGHTS_PATH = R18_12EP.weights_path" in code
    assert "train-unet-r18-12ep-2026-10-31" not in code
    assert "from filament.checkpoint import tensor_sha256 as _tensor_sha256" in code
    assert "def _tensor_sha256" not in code
    assert R18_12EP.weights_path.endswith("train-unet-r18-12ep-2026-10-31/unet.pt")


# --- both new notebooks -------------------------------------------------------


@pytest.mark.parametrize("path", [TRAIN, INFER], ids=lambda p: p.name)
def test_first_cell_lists_the_inputs_and_settings(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    first = notebook["cells"][0]
    assert first["cell_type"] == "markdown"
    text = _text(first["source"])
    assert "filament-segmentation-2026" in text
    assert "ignacioaguirrecandia/solar-code" in text
    assert "GPU T4" in text
    assert "single T4" in text
    assert "Internet" in text and "**On**" in text
    assert "Save & Run All" in text
    assert "train-unet-r34-crops" in text
    assert (
        "does not submit" in text.lower()
        or "did not submit" in text.lower()
        or ("does not submit" in text)
    )


def _module_names(module: str) -> set[str]:
    source = (ROOT / "src" / "filament" / f"{module}.py").read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


@pytest.mark.parametrize(
    "path", [*MODEL_NOTEBOOKS, CROP_ENSEMBLE], ids=lambda p: p.name
)
def test_everything_imported_from_filament_exists(path: Path) -> None:
    """Catch a renamed function before a Kaggle session does."""
    for source in _cells(path, "code"):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "filament":
                    for alias in node.names:
                        assert (
                            ROOT / "src" / "filament" / f"{alias.name}.py"
                        ).is_file()
                elif node.module.startswith("filament."):
                    names = _module_names(node.module.split(".", 1)[1])
                    for alias in node.names:
                        assert alias.name in names, (node.module, alias.name)
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "ct"
            ):
                assert node.attr in _module_names("crop_training"), node.attr


# --- selecting the crop model by name ------------------------------------------


@pytest.mark.parametrize("path", MODEL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_selects_the_model_by_name(path: Path) -> None:
    assert _constants(path)["MODEL"] == DEFAULT_CROP_MODEL == "r34"
    code = _code(path)
    assert "EXPECTED = crop_checkpoint(MODEL)" in code
    # One source for what the file has to be: nothing is written for one model.
    assert "R34_CROPS" not in code
    assert "unet_r34_crops" not in code
    assert '"r34"' not in code.replace('MODEL = "r34"', "")


@pytest.mark.parametrize("path", MODEL_NOTEBOOKS, ids=lambda p: p.name)
def test_first_cell_lists_every_model(path: Path) -> None:
    text = _cells(path, "markdown")[0]
    assert set(CROP_MODELS) == set(CROP_CHECKPOINTS)
    for name, expected in CROP_CHECKPOINTS.items():
        weights = Path(expected.weights_path)
        assert f"`{name}`" in text
        assert f"`{expected.encoder}`" in text
        assert f"`{weights.parent.name}`" in text
        assert f"`{weights.name}`" in text


# --- post-processing grid notebook ---------------------------------------------


def test_postproc_notebook_grid_matches_the_library_defaults() -> None:
    constants = _constants(POSTPROC)
    assert constants["THRESHOLDS"] == DEFAULT_THRESHOLDS
    assert constants["CLOSE_SIZES"] == DEFAULT_CLOSE_SIZES
    assert constants["MIN_AREAS"] == DEFAULT_GRID_MIN_AREAS
    assert constants["CURRENT"] == (0.5, 0, 400)
    assert constants["ALLOW_UNVERIFIED_WEIGHTS"] is False
    assert constants["TILE_OVERLAP"] == DEFAULT_TILE_OVERLAP
    assert constants["INFER_BATCH_SIZE"] == DEFAULT_INFER_BATCH_SIZE


def test_postproc_notebook_writes_the_csv_and_nothing_else() -> None:
    code = _code(POSTPROC)
    assert not SUBMIT_PATTERNS.search(code)
    assert 'Path(f"/kaggle/working/holdout_postproc_{MODEL}.csv")' in code
    assert "run_postproc_grid(" in code
    assert "predictor_name=Path(EXPECTED.weights_path).stem" in code
    assert "expected=EXPECTED" in code
    assert "WEIGHTS_PATH = EXPECTED.weights_path" in code
    assert "EXPECTED_TENSOR_SHA256 = EXPECTED.tensor_sha256" in code
    assert "EXPECTED.holdout_pq" in code  # the cross-check follows the model
    assert 'SMP_VERSION = "0.6.0"' in code
    # Never reads the test set, never trains, never writes a submission.
    for forbidden in ("test_images", "train_model(", "write_submission(", "input("):
        assert forbidden not in code.replace("test/test_images is not used", "")
    assert R34_CROPS.weights_path not in code  # one source for the path


# --- TTA / ensemble notebook and the inference flags ---------------------------


def test_tta_ensemble_notebook_grid() -> None:
    constants = _constants(TTA_ENSEMBLE)
    assert constants["THRESHOLDS"] == TTA_THRESHOLDS == (0.40, 0.45, 0.50)
    # 0 is the cross-check against the training notebook; the rest is the grid.
    assert constants["CLOSE_SIZES"] == (0, *TTA_CLOSE_SIZES)
    assert constants["MIN_AREAS"] == TTA_MIN_AREAS == (400,)
    assert constants["REFERENCE"] == (0.45, 15, 400)
    assert constants["ENSEMBLE_WEIGHTS"] == ((0.5, 0.5), (0.3, 0.7))
    assert constants["ALLOW_UNVERIFIED_WEIGHTS"] is False
    assert constants["TILE_OVERLAP"] == DEFAULT_TILE_OVERLAP
    assert constants["INFER_BATCH_SIZE"] == DEFAULT_INFER_BATCH_SIZE
    assert constants["R18_SIDE"] == 512
    # Only odd kernels reach label_foreground (10 and 20 were asked for).
    assert all(size == 0 or size % 2 == 1 for size in constants["CLOSE_SIZES"])


def test_tta_ensemble_notebook_guards_both_checkpoints_and_never_submits() -> None:
    code = _code(TTA_ENSEMBLE)
    assert not SUBMIT_PATTERNS.search(code)
    assert 'Path("/kaggle/working/holdout_tta_ensemble.csv")' in code
    assert "expected=expected" in code  # the shared guard helper
    assert "R18_12EP" in code
    assert "WEIGHTS_PATH = EXPECTED.weights_path" in code
    assert "TRAINING_NOTEBOOK_PQ = EXPECTED.holdout_pq" in code
    assert "R18_WEIGHTS_PATH = R18_12EP.weights_path" in code
    assert "sweep_configs(" in code and "write_config_rows(" in code
    for forbidden in ("train_model(", "write_submission(", "input("):
        assert forbidden not in code
    assert "test/test_images" not in code.replace("test/test_images is not used", "")
    assert 'SMP_VERSION = "0.6.0"' in code
    assert "EvalConfig(MODEL, {MODEL: 1.0})" in code
    assert 'EvalConfig(f"{MODEL}_tta", {MODEL: 1.0}, tta=True)' in code
    names = _cells(TTA_ENSEMBLE, "markdown")[0]
    for name in ("ens_50_50", "ens_30_70", "ens_50_50_tta", "ens_30_70_tta"):
        assert name in names


# --- probability export notebook -------------------------------------------------


def test_export_notebook_settings() -> None:
    constants = _constants(EXPORT)
    assert constants["SPLITS"] == ("holdout", "test")
    assert constants["FULL_SIDE"] == 2048
    assert constants["ALLOW_UNVERIFIED_WEIGHTS"] is False
    assert constants["TILE_OVERLAP"] == DEFAULT_TILE_OVERLAP
    assert constants["INFER_BATCH_SIZE"] == DEFAULT_INFER_BATCH_SIZE
    assert constants["USE_AMP"] is True
    assert 'OUTPUT_ROOT = Path("/kaggle/working/probs")' in _code(EXPORT)


def test_export_notebook_guards_the_checkpoint_and_records_it() -> None:
    code = _code(EXPORT)
    assert "WEIGHTS_PATH = EXPECTED.weights_path" in code
    assert "expected=EXPECTED" in code
    assert "verify_and_report(" in code
    assert 'SMP_VERSION = "0.6.0"' in code
    assert "write_manifest(" in code
    assert '"tensor_sha256": fingerprint["tensor_sha256"]' in code


def test_export_notebook_accepts_only_the_expected_weights_name() -> None:
    code = _code(EXPORT)
    assert "using the only .pt found" not in code
    assert "name == PREFERRED_NAME" in code
    assert '"weights_file": str(weights_file)' in code


def test_export_notebook_writes_the_manifest_before_the_cross_check() -> None:
    code = _code(EXPORT)
    assert code.index("write_manifest(OUTPUT_ROOT, MODEL, MANIFEST)") < code.index(
        "sweep_postproc_grid(\n"
    )
    assert code.count("write_manifest(") == 2  # before, and again with the result
    assert '"cross_check": "not run"' in code
    assert 'MANIFEST["cross_check"] = "FAILED"' in code
    assert "raise" not in code[code.index("CROSS_CHECK_FLOOR_MARGIN") :]


def test_export_notebook_never_trains_submits_or_scores_the_test_set() -> None:
    code = _code(EXPORT)
    assert not SUBMIT_PATTERNS.search(code)
    for forbidden in ("train_model(", "write_submission(", "submission.csv", "input("):
        assert forbidden not in code
    # The scoring cell only ever reads the holdout side of the cache.
    assert 'ProbCache(split_dir(OUTPUT_ROOT, MODEL, "holdout"))' in code
    assert 'ProbCache(split_dir(OUTPUT_ROOT, MODEL, "test"))' not in code
    assert code.count("ProbCache(") == 1
    assert code.count("sweep_postproc_grid(") == 1
    assert "split.holdout, _read, _cached" in code


# --- holdout gap analysis notebook -------------------------------------------------


def test_gap_notebook_reads_only_the_holdout_of_the_cache() -> None:
    code = _code(GAP)
    assert not SUBMIT_PATTERNS.search(code)
    assert 'split_dir(CACHE_ROOT, MODEL, "holdout")' in code
    assert '"test"' not in code
    assert "test_images" not in code
    for forbidden in ("torch", "train_model(", "write_submission(", "input("):
        assert forbidden not in code


def test_gap_notebook_guards_the_cache_with_the_checkpoint_entry() -> None:
    code = _code(GAP)
    assert "manifest_fingerprint(" in code
    assert "expected=EXPECTED" in code
    assert "ALLOW_UNVERIFIED_WEIGHTS = False" in code
    assert "find_cache_root(MODEL" in code


# --- object filter notebook --------------------------------------------------------


def test_object_filter_notebook_reads_only_the_holdout_of_the_cache() -> None:
    code = _code(OBJECT_FILTER)
    assert not SUBMIT_PATTERNS.search(code)
    assert 'split_dir(CACHE_ROOT, MODEL, "holdout")' in code
    assert '"test"' not in code
    assert "test_images" not in code
    for forbidden in ("torch", "train_model(", "write_submission(", "input("):
        assert forbidden not in code
    assert "manifest_fingerprint(" in code
    assert "expected=EXPECTED" in code
    assert "ALLOW_UNVERIFIED_WEIGHTS = False" in code


def test_object_filter_notebook_decides_by_the_crossfit_not_the_sweep_peak() -> None:
    code = _code(OBJECT_FILTER)
    assert "crossfit_rows(" in code
    assert "plateau_center(" in code
    assert "NO_OBJECT_FILTER" in code


# --- selector notebook -------------------------------------------------------------


def test_selector_notebook_reads_only_the_holdout_of_the_cache() -> None:
    code = _code(SELECTOR)
    assert not SUBMIT_PATTERNS.search(code)
    assert 'split_dir(CACHE_ROOT, MODEL, "holdout")' in code
    assert '"test"' not in code
    assert "test_images" not in code
    for forbidden in ("torch", "train_model(", "write_submission(", "input("):
        assert forbidden not in code
    assert "manifest_fingerprint(" in code
    assert "expected=EXPECTED" in code
    assert "ALLOW_UNVERIFIED_WEIGHTS = False" in code


def test_selector_notebook_decides_by_the_out_of_fold_verdict() -> None:
    code = _code(SELECTOR)
    assert "evaluate_selector(" in code
    assert "SELECTOR_MODELS" in code and "CANDIDATE_POINTS" in code
    assert ".fit(" not in code  # models are only fitted out of fold, in the library


def test_infer_object_filter_comes_from_the_library_and_runs_before_encoding() -> None:
    code = _code(INFER)
    assert "OBJECT_FILTER = ADOPTED_OBJECT_FILTER" in code
    assert "OBJECT_CUT = object_cut_from_config(OBJECT_FILTER)" in code
    assert "object_filter_mismatches(" in code
    # Applied to the labels of each frame, after post-processing, before encoding.
    filtered = code.index("labels = filter_components(labels, probability, OBJECT_CUT)")
    assert code.index("labels = labels_from_semantic(") < filtered
    assert filtered < code.index("iter_filament_counts(labels)")
    assert code.count("filter_components(") == 1
    # No copy of the filter's logic in the notebook.
    assert "component_scores(" not in code and "keep_components(" not in code


def test_infer_keeps_its_guards_with_the_object_filter() -> None:
    code = _code(INFER)
    assert "verify_and_report(" in code
    assert "EXPECTED_TENSOR_SHA256 = EXPECTED.tensor_sha256" in code
    assert "validate_submission(" in code


def test_infer_crop_ensemble_is_off_by_default_and_next_to_the_weights() -> None:
    constants = _constants(INFER)
    assert constants["CROP_ENSEMBLE"] is None
    assert constants["ENSEMBLE_WEIGHTS_PATHS"] == {}
    settings = next(cell for cell in _cells(INFER, "code") if "WEIGHTS_PATH = " in cell)
    assert settings.index("WEIGHTS_PATH = EXPECTED.weights_path") < settings.index(
        "CROP_ENSEMBLE = None"
    )


def _calls(path: Path, function: str) -> list[ast.Call]:
    found = []
    for source in _cells(path, "code"):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                target = node.func
                name = (
                    target.id
                    if isinstance(target, ast.Name)
                    else getattr(target, "attr", None)
                )
                if name == function:
                    found.append(node)
    return found


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def test_infer_model_weights_come_from_the_library_once_before_any_loading() -> None:
    assigned = []
    for source in _cells(INFER, "code"):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MODEL_WEIGHTS"
                for t in node.targets
            ):
                assigned.append(node)
    assert len(assigned) == 1
    call = assigned[0].value
    assert isinstance(call, ast.Call) and call.func.id == "submission_model_weights"
    assert [ast.unparse(a) for a in call.args] == ["MODEL", "CROP_ENSEMBLE"]
    assert {kw.arg: ast.unparse(kw.value) for kw in call.keywords} == {
        "use_r18_ensemble": "USE_ENSEMBLE",
        "r18_weights": "ENSEMBLE_WEIGHTS",
        "allow_unverified": "ALLOW_UNVERIFIED_WEIGHTS",
    }
    code = _code(INFER)
    assert code.index("MODEL_WEIGHTS = submission_model_weights(") < code.index(
        "WEIGHTS_FILE = _resolve_weights(WEIGHTS_PATH)"
    )
    assert 'CONFIG = EvalConfig("submission", MODEL_WEIGHTS, tta=USE_TTA)' in code
    assert "for name in members_to_load(MODEL_WEIGHTS, MODEL):" in code


GUARDED_NOTEBOOKS = [
    path for path in ALL_NOTEBOOKS if "verify_and_report(" in _code(path)
]


@pytest.mark.parametrize("path", GUARDED_NOTEBOOKS, ids=lambda p: p.name)
def test_no_checkpoint_guard_is_forced_open(path: Path) -> None:
    """Every guard call obeys the notebook's own setting, which defaults to False."""
    constants = _constants(path)
    if "ALLOW_UNVERIFIED_WEIGHTS" in constants:
        assert constants["ALLOW_UNVERIFIED_WEIGHTS"] is False
    for call in _calls(path, "verify_and_report"):
        allow = _keyword(call, "allow_unverified")
        if allow is not None:
            assert isinstance(allow, ast.Name), ast.unparse(call)
            assert allow.id == "ALLOW_UNVERIFIED_WEIGHTS", ast.unparse(call)
        expected = _keyword(call, "expected")
        if expected is not None:
            assert not isinstance(expected, ast.Constant), ast.unparse(call)


@pytest.mark.parametrize("path", [INFER, CROP_ENSEMBLE], ids=lambda p: p.name)
def test_every_ensemble_member_gets_its_own_guard(path: Path) -> None:
    guards = _calls(path, "verify_and_report")
    assert all(_keyword(call, "allow_unverified") is not None for call in guards)
    expected = {ast.unparse(_keyword(call, "expected")) for call in guards}
    member = "member" if path == INFER else "expected"
    assert member in expected  # the loop over members checks each one


def _member_loop(path: Path, iterates: str) -> str:
    """The source of the notebook's top-level ``for name in <iterates>:`` loop."""
    for source in _cells(path, "code"):
        for node in ast.parse(source).body:
            if (
                isinstance(node, ast.For)
                and ast.unparse(node.target) == "name"
                and ast.unparse(node.iter) == iterates
            ):
                return ast.unparse(node)
    raise AssertionError(f"no loop over {iterates} in {path.name}")


# What each stand-in checkpoint stores. The tile sizes differ, so a test can see which
# checkpoint a member's tile size came from, and that tile_size wins over crop_size.
_STORED_FIELDS = {
    "r34": {"crop_size": 512, "tile_size": 384},
    "r34_v2": {"crop_size": 512, "tile_size": 256},
    "r34_v2_agree": {"crop_size": 640},
}


def _notebook_assignment(path: Path, name: str) -> str:
    """The source of the notebook's top-level ``name = ...`` statement."""
    for source in _cells(path, "code"):
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.unparse(node)
    raise AssertionError(f"no assignment to {name} in {path.name}")


def _run_member_loop(
    path: Path,
    iterates: str,
    members: list[str],
    tmp_path: Path,
    *,
    settings: dict[str, object] | None = None,
) -> tuple[list[tuple[str, object, object]], dict[str, object], dict[str, object]]:
    """Run the notebook's member loop for real, with test doubles for torch.

    Each entry of ``filament.checkpoint`` keeps its real fields but points at a file
    in ``tmp_path`` named after its member, so a loop that follows ``weights_path``
    runs as on Kaggle. The doubles carry the file's name through the checkpoint, the
    fingerprint and the model, so the test can see which checkpoint each guard checked
    and each predictor was built from. Returns the guard calls, the predictors, and
    the entries the loop was given.
    """
    from filament.checkpoint import crop_checkpoint
    from filament.crop_ensemble import members_to_load

    entries = {}
    for name in {"r34", *members}:
        file = tmp_path / f"{name}.pt"
        file.write_text(name)
        entries[name] = dataclasses.replace(
            crop_checkpoint(name), weights_path=str(file)
        )
    guarded: list[tuple[str, object, object]] = []

    def verify(fingerprint, *, allow_unverified, expected):
        guarded.append((fingerprint["source"], expected, allow_unverified))

    fake_ct = SimpleNamespace(
        load_checkpoint=lambda file: {
            "source": Path(file).stem,
            **_STORED_FIELDS.get(Path(file).stem, {"crop_size": 768}),
        },
        model_from_checkpoint=lambda checkpoint, _device: (
            f"model:{checkpoint['source']}"
        ),
        tiled_predictor=lambda model, _device, **kwargs: (model, kwargs["tile_size"]),
    )
    predictors: dict[str, object] = {"r34": "the MODEL predictor, loaded before"}
    namespace = {
        "Path": Path,
        "ct": fake_ct,
        "crop_checkpoint": entries.__getitem__,
        "members_to_load": members_to_load,
        "checkpoint_fingerprint": lambda c: {
            "source": c["source"],
            "tensor_sha256": "x",
        },
        "verify_and_report": verify,
        "_materialize_checkpoint": lambda file: file,
        "print": lambda *_args, **_kwargs: None,
        "device": "cpu",
        "use_amp": True,
        "USE_AMP": True,
        "TILE_SIZE": 999,  # MODEL's tile size, which no member should read
        "TILE_OVERLAP": 128,
        "INFER_BATCH_SIZE": 8,
        "ALLOW_UNVERIFIED_WEIGHTS": False,
        "MODEL": "r34",
        "EXPECTED": entries["r34"],
        "WEIGHTS_PATH": entries["r34"].weights_path,
        "model": "the MODEL network",
        "MODEL_WEIGHTS": {name: 1.0 / len(members) for name in members},
        "MEMBERS": tuple(members),
        "PREDICTORS": predictors,
        **(settings or {}),
    }
    if path == CROP_ENSEMBLE:
        # The notebook's own default paths, as a Kaggle run has them.
        exec(_notebook_assignment(path, "WEIGHTS_PATHS"), namespace)  # noqa: S102
    exec(_member_loop(path, iterates), namespace)  # noqa: S102
    return guarded, predictors, entries


_INFER_LOOP = "members_to_load(MODEL_WEIGHTS, MODEL)"


def test_infer_member_loop_checks_and_builds_each_member_from_its_own_file(
    tmp_path: Path,
) -> None:
    # The notebook's default {}: each member loads from its own entry's weights_path.
    assert _constants(INFER)["ENSEMBLE_WEIGHTS_PATHS"] == {}
    guarded, predictors, entries = _run_member_loop(
        INFER,
        _INFER_LOOP,
        ["r34", "r34_v2_agree", "r34_v2"],
        tmp_path,
        settings={"ENSEMBLE_WEIGHTS_PATHS": {}},
    )
    # MODEL was loaded and guarded before the loop; the loop only adds the others.
    assert guarded == [
        ("r34_v2_agree", entries["r34_v2_agree"], False),
        ("r34_v2", entries["r34_v2"], False),
    ]
    assert predictors["r34_v2_agree"] == ("model:r34_v2_agree", 640)  # its crop_size
    assert predictors["r34_v2"] == ("model:r34_v2", 256)  # tile_size wins
    assert predictors["r34"] == "the MODEL predictor, loaded before"


def test_infer_member_path_override_applies_to_that_member_only(tmp_path: Path) -> None:
    other = tmp_path / "other.pt"
    other.write_text("other")
    guarded, predictors, entries = _run_member_loop(
        INFER,
        _INFER_LOOP,
        ["r34", "r34_v2_agree", "r34_v2"],
        tmp_path,
        settings={"ENSEMBLE_WEIGHTS_PATHS": {"r34_v2_agree": str(other)}},
    )
    # The file changes, and the guard still checks it against the member's own entry.
    assert guarded == [
        ("other", entries["r34_v2_agree"], False),
        ("r34_v2", entries["r34_v2"], False),
    ]
    assert predictors["r34_v2_agree"] == ("model:other", 768)
    assert predictors["r34_v2"] == ("model:r34_v2", 256)


def test_infer_member_with_a_missing_file_stops_and_names_it(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="r34_v2_agree"):
        _run_member_loop(
            INFER,
            _INFER_LOOP,
            ["r34", "r34_v2_agree"],
            tmp_path,
            settings={"ENSEMBLE_WEIGHTS_PATHS": {"r34_v2_agree": str(missing)}},
        )


def test_holdout_member_loop_checks_and_builds_each_member_from_its_own_file(
    tmp_path: Path,
) -> None:
    guarded, predictors, entries = _run_member_loop(
        CROP_ENSEMBLE, "MEMBERS", ["r34", "r34_v2_agree"], tmp_path
    )
    assert guarded == [
        ("r34", entries["r34"], False),
        ("r34_v2_agree", entries["r34_v2_agree"], False),
    ]
    assert predictors["r34"] == ("model:r34", 384)  # tile_size wins
    assert predictors["r34_v2_agree"] == ("model:r34_v2_agree", 640)  # its crop_size


def test_infer_object_filter_warning_knows_the_models() -> None:
    (call,) = _calls(INFER, "object_filter_mismatches")
    assert ast.unparse(_keyword(call, "models")) == "MODEL_WEIGHTS"
    code = _code(INFER)
    assert code.index("MODEL_WEIGHTS = submission_model_weights(") < code.index(
        "OBJECT_CUT = object_cut_from_config(OBJECT_FILTER)"
    )


def test_infer_header_lists_the_inputs_of_an_ensemble() -> None:
    header = _cells(INFER, "markdown")[0]
    assert "CROP_ENSEMBLE" in header and "every other member" in header


def test_train_notebook_says_d2_is_pinned() -> None:
    text = "\n".join(_cells(TRAIN, "markdown"))
    assert "`R34_V2_AGREE` (D2) is pinned" in text
    assert "and `R34_V2_AGREE` are not pinned yet" not in text


def test_crop_ensemble_notebook_scores_the_holdout_only_and_guards_each_member() -> (
    None
):
    code = _code(CROP_ENSEMBLE)
    assert not SUBMIT_PATTERNS.search(code)
    for forbidden in ("train_model(", "write_submission(", "submission.csv", "input("):
        assert forbidden not in code
    assert "test/test_images" not in code.replace("test/test_images is not used", "")
    assert "CONFIGS = HOLDOUT_ENSEMBLE_CONFIGS" in code
    assert "POINT = ENSEMBLE_POINT" in code
    assert "object_cut_from_config(ADOPTED_OBJECT_FILTER)" in code
    assert "expected=expected" in code and "verify_and_report(" in code
    assert "require_pinned(MEMBERS, allow_unverified=ALLOW_UNVERIFIED_WEIGHTS)" in code
    assert "ALLOW_UNVERIFIED_WEIGHTS = False" in code
    assert 'Path("/kaggle/working/holdout_crop_ensemble.csv")' in code
    assert 'SMP_VERSION = "0.6.0"' in code
    assert "ct.require_cuda(6.0)" in code


def test_infer_flags_are_off_by_default() -> None:
    constants = _constants(INFER)
    assert constants["USE_TTA"] is False
    assert constants["USE_ENSEMBLE"] is False
    assert constants["ENSEMBLE_WEIGHTS"] == (0.5, 0.5)
    assert constants["R18_SIDE"] == 512
    code = _code(INFER)
    assert "R18_WEIGHTS_PATH = R18_12EP.weights_path" in code
    assert "expected=R18_12EP" in code  # the r18 checkpoint is guarded too
    assert "predict = ensemble_predictor(PREDICTORS, CONFIG)" in code
    assert "probability = predict(gray)" in code
    assert "ct.predict_probability(" not in code
    assert not SUBMIT_PATTERNS.search(code)
