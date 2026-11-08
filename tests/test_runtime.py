"""Runtime profile tests.

Monkeypatch environment variables only. No dataset files and no network.
Autodetect of ``/kaggle/working`` is not fixture-tested; these tests assume
that path is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# src/ is not an installed package in this pull request.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from filament.runtime import DataRootError, resolve_runtime

_ENV_KEYS = (
    "FILAMENT_PROFILE",
    "FILAMENT_DATA_ROOT",
    "FILAMENT_OUTPUT_DIR",
    "KAGGLE_KERNEL_RUN_TYPE",
    "COLAB_RELEASE_TAG",
)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    ("env", "profile", "data_root", "output_dir"),
    [
        ({}, "local", Path("data"), Path("outputs")),
        (
            {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"},
            "kaggle",
            None,
            Path("/kaggle/working"),
        ),
        (
            {"COLAB_RELEASE_TAG": "release"},
            "colab",
            None,
            Path("/content/outputs"),
        ),
    ],
)
def test_profile_defaults(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    profile: str,
    data_root: Path | None,
    output_dir: Path,
) -> None:
    _clear_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    runtime = resolve_runtime()

    assert runtime.profile == profile
    assert runtime.data_root == data_root
    assert runtime.output_dir == output_dir
    assert runtime.device == "cpu"
    if data_root is None:
        with pytest.raises(DataRootError, match="FILAMENT_DATA_ROOT"):
            runtime.require_data_root()
    else:
        assert runtime.require_data_root() == data_root


def test_env_overrides_beat_autodetect(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release")
    monkeypatch.setenv("FILAMENT_PROFILE", " Colab ")
    monkeypatch.setenv("FILAMENT_DATA_ROOT", "/content/competition-data")
    monkeypatch.setenv("FILAMENT_OUTPUT_DIR", "/content/custom-out")

    runtime = resolve_runtime()

    assert runtime.profile == "colab"
    assert runtime.data_root == Path("/content/competition-data")
    assert runtime.output_dir == Path("/content/custom-out")
    assert runtime.device == "cpu"
    assert runtime.require_data_root() == Path("/content/competition-data")


def test_kaggle_wins_when_colab_is_also_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release")

    runtime = resolve_runtime()

    assert runtime.profile == "kaggle"
    assert runtime.output_dir == Path("/kaggle/working")
    assert runtime.data_root is None


def test_empty_kaggle_variable_still_selects_kaggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "")

    runtime = resolve_runtime()

    assert runtime.profile == "kaggle"
    assert runtime.device == "cpu"


def test_blank_profile_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FILAMENT_PROFILE", "   ")

    runtime = resolve_runtime()

    assert runtime.profile == "local"
    assert runtime.data_root == Path("data")
    assert runtime.output_dir == Path("outputs")


def test_unknown_profile_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FILAMENT_PROFILE", "gpu-box")

    with pytest.raises(ValueError, match="FILAMENT_PROFILE"):
        resolve_runtime()


def test_local_output_and_data_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("FILAMENT_PROFILE", "local")
    monkeypatch.setenv("FILAMENT_DATA_ROOT", "data/raw")
    monkeypatch.setenv("FILAMENT_OUTPUT_DIR", "outputs/smoke")

    runtime = resolve_runtime()

    assert runtime.profile == "local"
    assert runtime.data_root == Path("data/raw")
    assert runtime.output_dir == Path("outputs/smoke")
    assert runtime.device == "cpu"
