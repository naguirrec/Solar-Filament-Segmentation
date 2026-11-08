"""requirements.txt is the pinned export of uv.lock with the ``gpu`` extra."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _requirements() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match:
            pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def _locked() -> dict[str, str]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {p["name"]: p["version"] for p in lock["package"] if "version" in p}


def test_every_pin_is_the_version_in_the_lock() -> None:
    locked = _locked()
    pins = _requirements()
    assert pins
    wrong = {n: (v, locked.get(n)) for n, v in pins.items() if locked.get(n) != v}
    assert wrong == {}


def test_the_export_command_in_the_header_includes_the_gpu_extra() -> None:
    header = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()[:2]
    assert "--extra gpu" in header[1]
    assert "--frozen" in header[1] and "--no-dev" in header[1]


def test_the_runtime_packages_the_notebook_needs_are_pinned() -> None:
    pins = _requirements()
    assert pins["torch"] == "2.11.0"
    assert pins["segmentation-models-pytorch"] == "0.6.0"
    for name in ("torchvision", "timm", "numpy", "opencv-python-headless"):
        assert name in pins, name
    for name in ("pycocotools", "scikit-image", "scikit-learn", "matplotlib"):
        assert name in pins, name


def test_the_declared_dependencies_are_all_pinned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = [
        *project["project"]["dependencies"],
        *project["project"]["optional-dependencies"]["gpu"],
    ]
    pins = _requirements()
    for requirement in declared:
        name = re.split(r"[<>=!~\s\[]", requirement, maxsplit=1)[0].lower()
        assert name.replace("_", "-") in pins, requirement


def test_torch_is_an_optional_extra_so_ci_does_not_install_it() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert not any("torch" in d for d in project["project"]["dependencies"])
    gpu = project["project"]["optional-dependencies"]["gpu"]
    assert "torch==2.11.0" in gpu
    assert "segmentation-models-pytorch==0.6.0" in gpu
