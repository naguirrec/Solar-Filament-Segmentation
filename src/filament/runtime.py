"""Resolve data and output paths for local, Kaggle, and Colab runs.

Profiles are ``local``, ``kaggle``, and ``colab``. ``FILAMENT_PROFILE``
overrides autodetection. ``FILAMENT_DATA_ROOT`` and ``FILAMENT_OUTPUT_DIR``
override the data and output directories. The device is always ``cpu``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProfileName = Literal["local", "kaggle", "colab"]

PROFILES: tuple[ProfileName, ...] = ("local", "kaggle", "colab")

_KAGGLE_WORKDIR = Path("/kaggle/working")
_LOCAL_DATA = Path("data")
_DEFAULT_OUTPUT: dict[ProfileName, Path] = {
    "local": Path("outputs"),
    "kaggle": _KAGGLE_WORKDIR,
    "colab": Path("/content/outputs"),
}


class DataRootError(RuntimeError):
    """Raised when a reader asks for a data root that was never configured."""


@dataclass(frozen=True, slots=True)
class Runtime:
    """Resolved profile, paths, and device for one run."""

    profile: ProfileName
    data_root: Path | None
    output_dir: Path
    device: Literal["cpu"]

    def require_data_root(self) -> Path:
        """Return the data root.

        Local runs default to ``data``. Kaggle and Colab leave it unset until
        ``FILAMENT_DATA_ROOT`` names the real folder. On Kaggle that folder is
        under ``/kaggle/input`` and stays ``[VERIFICAR]`` until the Data tab
        is read.
        """
        if self.data_root is None:
            raise DataRootError(
                f"Data root is unset for profile {self.profile!r}. Set "
                "FILAMENT_DATA_ROOT to the competition data folder. On Kaggle, "
                "read the real directory under /kaggle/input (the folder name "
                "is [VERIFICAR] until the Data tab is read)."
            )
        return self.data_root


def detect_profile() -> ProfileName:
    """Pick ``kaggle``, ``colab``, or ``local`` from the process environment.

    Kaggle wins when ``KAGGLE_KERNEL_RUN_TYPE`` is set or ``/kaggle/working``
    exists. Colab is next, when ``COLAB_RELEASE_TAG`` is set.
    """
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or _KAGGLE_WORKDIR.exists():
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    return "local"


def resolve_runtime() -> Runtime:
    """Build a runtime from the environment and the profile defaults."""
    profile = _selected_profile()
    data_root = _optional_path("FILAMENT_DATA_ROOT")
    output_dir = _optional_path("FILAMENT_OUTPUT_DIR")
    if data_root is None and profile == "local":
        data_root = _LOCAL_DATA
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT[profile]
    return Runtime(
        profile=profile,
        data_root=data_root,
        output_dir=output_dir,
        # CPU only in this change. Week-2 training can choose a device later.
        device="cpu",
    )


def _selected_profile() -> ProfileName:
    chosen = _nonblank("FILAMENT_PROFILE")
    if chosen is None:
        return detect_profile()
    normalized = chosen.lower()
    for profile in PROFILES:
        if normalized == profile:
            return profile
    known = ", ".join(PROFILES)
    raise ValueError(f"Unknown FILAMENT_PROFILE {chosen!r}. Expected one of: {known}.")


def _nonblank(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = raw.strip()
    if text == "":
        return None
    return text


def _optional_path(name: str) -> Path | None:
    text = _nonblank(name)
    if text is None:
        return None
    return Path(text)
