"""Export a training checkpoint to a small fp16 file, and find and verify it later.

The final delivery needs the model weights in the repository, under
``weights/``, so that ``notebooks/reproduce.ipynb`` can predict without
retraining. A training checkpoint holds fp32 tensors (about 98 MB for the
ResNet-34 U-Net); the exported file holds the same ``state_dict`` in fp16
(about 49 MB) plus the few fields needed to rebuild the network.

``weights/MANIFEST.json`` records, for each exported file:

* ``file_sha256``: the SHA-256 of the file bytes,
* ``tensor_sha256``: :func:`filament.checkpoint.tensor_sha256` of the fp16 state,
* ``tensor_sha256_original``: the same hash of the fp32 checkpoint it came from
  (for the final model, the pin ``3ac77e98…`` of :data:`filament.checkpoint.R34_CROPS`),
* ``bytes``, ``encoder``, ``epochs``.

fp16 changes the values slightly. The hashes prove which weights these are, not
that the conversion is harmless: the holdout PQ check of the reproduce notebook
does that.

This module does not import torch at import time. Only :func:`half_state_dict`
and :func:`export_weights` need it, and they import it when called, so the
lookup and verification half runs in CI without torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filament.checkpoint import (
    R34_CROPS,
    ExpectedCheckpoint,
    checkpoint_fingerprint,
    checkpoint_problems,
    tensor_sha256,
)

MANIFEST_NAME = "MANIFEST.json"
MANIFEST_FORMAT = 1
# GitHub refuses a pushed file over 100 MiB. 100 MB (decimal) leaves some margin.
MAX_FILE_BYTES = 100 * 1000 * 1000
WEIGHTS_DIR_NAME = "weights"
# The file the reproduce notebook looks for by default.
R34_FP16_FILE = "unet_r34_crops_fp16.pt"

# What an exported file keeps from the training checkpoint, when it is there.
KEPT_FIELDS = (
    "encoder_name",
    "encoder_weights",
    "smp_version",
    "in_channels",
    "classes",
    "epochs",
    "epochs_requested",
    "status",
    "crop_size",
    "training_resolution",
    "native_shape",
    "tile_size",
    "tile_overlap",
)


class WeightsError(RuntimeError):
    """A weights file is missing, too big, or is not the one the manifest names."""


# ----------------------------------------------------------------- hashing


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of the bytes of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------- manifest


def manifest_path(weights_dir: str | Path) -> Path:
    return Path(weights_dir) / MANIFEST_NAME


def read_weights_manifest(weights_dir: str | Path) -> dict[str, Any]:
    """The manifest of ``weights_dir``, or an empty one if there is none yet."""
    path = manifest_path(weights_dir)
    if not path.is_file():
        return {"format": MANIFEST_FORMAT, "weights": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("format") != MANIFEST_FORMAT
        or not isinstance(data.get("weights"), dict)
    ):
        raise WeightsError(
            f"{path} is not a weights manifest of format {MANIFEST_FORMAT}"
        )
    return data


def write_weights_manifest(
    weights_dir: str | Path, manifest: Mapping[str, Any]
) -> Path:
    """Write the manifest as sorted, indented JSON (so a diff shows what changed)."""
    path = manifest_path(weights_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def manifest_entry(weights_dir: str | Path, file_name: str) -> dict[str, Any]:
    """The manifest entry of ``file_name``, or :class:`WeightsError`."""
    manifest = read_weights_manifest(weights_dir)
    entry = manifest["weights"].get(file_name)
    if entry is None:
        known = sorted(manifest["weights"]) or "none"
        raise WeightsError(
            f"{file_name} is not in {manifest_path(weights_dir)} (entries: {known})"
        )
    return entry


# ------------------------------------------------------------------- export


def half_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """The ``state_dict`` with every floating tensor in fp16.

    Integer tensors (``num_batches_tracked`` of the batch norms) are kept as
    they are. Needs torch; it is imported here.
    """
    import torch

    converted: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            converted[key] = value.detach().cpu().to(torch.float16).contiguous()
        elif isinstance(value, torch.Tensor):
            converted[key] = value.detach().cpu().contiguous()
        else:
            converted[key] = value
    return converted


def export_weights(
    checkpoint_path: str | Path,
    weights_dir: str | Path,
    *,
    file_name: str | None = None,
) -> dict[str, Any]:
    """Write ``<weights_dir>/<file_name>`` and its manifest entry, and return the entry.

    ``checkpoint_path`` is a training checkpoint (``state_dict`` plus fields,
    :func:`filament.crop_training.save_checkpoint`). The exported file holds the
    fp16 ``state_dict`` and :data:`KEPT_FIELDS`. It is refused, and removed, when it
    is over :data:`MAX_FILE_BYTES`. Needs torch.
    """
    import torch

    source = Path(checkpoint_path)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        raise WeightsError(f"{source} is not a checkpoint with a state_dict")
    original_hash = tensor_sha256(checkpoint["state_dict"])
    half = half_state_dict(checkpoint["state_dict"])
    exported: dict[str, Any] = {"state_dict": half, "dtype": "float16"}
    exported["source_tensor_sha256"] = original_hash
    for key in KEPT_FIELDS:
        if key in checkpoint:
            exported[key] = checkpoint[key]

    name = file_name or f"{source.stem}_fp16.pt"
    if Path(name).name != name:
        raise WeightsError(f"file name {name!r} must not contain a directory")
    destination = Path(weights_dir) / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(exported, temporary)
    size = temporary.stat().st_size
    if size > MAX_FILE_BYTES:
        temporary.unlink()
        raise WeightsError(
            f"{name} would be {size} bytes, over the limit of {MAX_FILE_BYTES} "
            "(100 MB). Nothing was written."
        )
    os.replace(temporary, destination)

    entry = {
        "file_sha256": sha256_file(destination),
        "tensor_sha256": tensor_sha256(half),
        "tensor_sha256_original": original_hash,
        "dtype": "float16",
        "bytes": size,
        "encoder": checkpoint.get("encoder_name"),
        "epochs": checkpoint.get("epochs"),
    }
    manifest = read_weights_manifest(weights_dir)
    manifest["weights"][name] = entry
    write_weights_manifest(weights_dir, manifest)
    return entry


# ------------------------------------------------------------- verification


def repo_weights_problems(
    entry: Mapping[str, Any],
    *,
    file_sha256: str,
    fingerprint: Mapping[str, object],
    expected: ExpectedCheckpoint = R34_CROPS,
) -> list[str]:
    """Why a ``weights/`` file is not the pinned model, or ``[]`` if it is.

    ``entry`` is the manifest entry, ``file_sha256`` the hash of the file on disk,
    ``fingerprint`` is :func:`filament.checkpoint.checkpoint_fingerprint` of the
    loaded file. The entry has to descend from ``expected.tensor_sha256`` (the
    fp32 pin), the file has to match the entry's file and tensor hashes, and the
    stored epochs and encoder have to be ``expected``'s.
    """
    if expected.tensor_sha256 is None:
        return [f"'{expected.label}' has no pinned tensor_sha256 to descend from"]
    problems: list[str] = []
    if str(entry.get("tensor_sha256_original", "")).lower() != expected.tensor_sha256:
        problems.append(
            "the manifest's tensor_sha256_original is "
            f"{entry.get('tensor_sha256_original')}, not the pinned "
            f"{expected.tensor_sha256}"
        )
    if file_sha256.lower() != str(entry.get("file_sha256", "")).lower():
        problems.append(
            f"file_sha256={file_sha256}, the manifest says {entry.get('file_sha256')}"
        )
    pinned_fp16 = entry.get("tensor_sha256")
    if pinned_fp16 is None:
        problems.append("the manifest has no tensor_sha256 for this file")
        return problems
    # The fp16 hash replaces the fp32 pin; epochs and encoder are still checked.
    problems += checkpoint_problems(
        fingerprint, expected=expected, expected_tensor_sha256=str(pinned_fp16)
    )
    return problems


def verify_repo_weights(
    checkpoint: Mapping[str, object],
    path: str | Path,
    entry: Mapping[str, Any],
    *,
    expected: ExpectedCheckpoint = R34_CROPS,
) -> dict[str, object]:
    """Raise :class:`WeightsError` unless the loaded file is the manifest's one.

    Returns the fingerprint. ``checkpoint`` is the loaded file (``state_dict``
    plus fields), ``path`` where it was read from.
    """
    fingerprint = checkpoint_fingerprint(checkpoint)
    problems = repo_weights_problems(
        entry,
        file_sha256=sha256_file(path),
        fingerprint=fingerprint,
        expected=expected,
    )
    if problems:
        raise WeightsError(
            f"{path} is not the weights the manifest names: " + "; ".join(problems)
        )
    return fingerprint


# ------------------------------------------------------------------ lookup


@dataclass(frozen=True, slots=True)
class WeightsSource:
    """Where the weights were found. ``kind`` is ``repo`` or ``kaggle``."""

    kind: str
    path: Path


def kaggle_candidates(input_root: str | Path, file_name: str) -> list[Path]:
    """Files or folders named ``file_name`` under ``input_root``.

    A folder counts: Kaggle extracts an uploaded ``.pt`` (a zip) into a folder of
    that name. ``competitions`` is not walked.
    """
    root = Path(input_root)
    found: list[Path] = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        kept = []
        for name in sorted(dirnames):
            if name == "competitions":
                continue
            if name == file_name:
                found.append(directory / name)
                continue
            kept.append(name)
        dirnames[:] = kept
        if file_name in filenames:
            found.append(directory / file_name)
    return found


def locate_weights(
    repo_root: str | Path,
    repo_file: str,
    kaggle_file: str,
    *,
    input_root: str | Path = "/kaggle/input",
) -> WeightsSource:
    """``weights/<repo_file>`` of the repo, else the Kaggle training output.

    The Kaggle file is a stopgap until the weights are committed. Several
    candidates are an error (the notebook does not guess), none is an error that
    says where it looked.
    """
    in_repo = Path(repo_root) / WEIGHTS_DIR_NAME / repo_file
    if in_repo.is_file():
        return WeightsSource("repo", in_repo)
    found = kaggle_candidates(input_root, kaggle_file)
    if len(found) == 1:
        return WeightsSource("kaggle", found[0])
    if found:
        listed = "\n".join(f"  {path}" for path in found)
        raise WeightsError(
            f"{in_repo} does not exist and {len(found)} files named {kaggle_file} "
            f"were found under {input_root}:\n{listed}"
        )
    raise WeightsError(
        f"No weights. {in_repo} does not exist and no file named {kaggle_file} was "
        f"found under {input_root}. Add the weights to the repo, or on Kaggle attach "
        "the output of the notebook train-unet-r34-crops-80ep. If the repo file is in "
        "an attached dataset, Kaggle may have extracted the .pt into a folder."
    )


def materialize_checkpoint(path: str | Path, work_dir: str | Path) -> Path:
    """A real checkpoint file at ``path``; a folder Kaggle extracted is re-zipped.

    ``torch.save`` writes a zip. Kaggle may unpack an uploaded ``.pt`` into a
    folder ``unet.pt/unet/data.pkl``. That folder is stored again, uncompressed,
    as ``<work_dir>/<name>-from-extracted.pt``, which ``torch.load`` can read.
    """
    source = Path(path)
    if source.is_file():
        return source
    if not source.is_dir():
        raise WeightsError(f"{source} does not exist")
    pickles = sorted(
        p for p in source.rglob("data.pkl") if p.is_file() and not p.is_symlink()
    )
    if len(pickles) != 1:
        raise WeightsError(
            f"{source} is a folder, not a checkpoint, and it holds {len(pickles)} "
            "data.pkl files (expected one)"
        )
    archive = pickles[0].parent
    members = sorted(p for p in archive.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in archive.rglob("*")):
        raise WeightsError(f"{source} contains a symlink, not re-zipping it")
    stem = source.stem if source.name.endswith(".pt") else source.name
    destination = Path(work_dir) / f"{stem}-from-extracted.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as handle:
        for member in members:
            handle.write(member, arcname=member.relative_to(archive.parent).as_posix())
    return destination
