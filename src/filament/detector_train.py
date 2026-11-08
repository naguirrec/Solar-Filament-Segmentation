"""Small checks and reports for ``notebooks/train_detector.ipynb``.

Everything here is plain Python and numpy: no torch and no ``rfdetr`` import, so
the notebook's decisions can be tested without a GPU. The notebook does the
training; this module decides what to train on, what to print about it, and
which checkpoint it hands on.

Nothing in this module ran against a real RF-DETR training run. The checkpoint
names in :data:`BEST_CHECKPOINTS` and :data:`LAST_CHECKPOINT` are what
``rfdetr`` 1.x writes as far as the author knows [VERIFY on the pilot: the
notebook lists the output directory and fails with that listing if none
matches]. The 2-epoch pilot on Kaggle (``rfdetr`` 1.12.0) wrote
``checkpoint_best_ema.pth``, ``checkpoint_best_total.pth`` and ``last_ema.pth``;
no per-epoch ``checkpoint.pth`` appeared, because the library's default
``checkpoint_interval`` is 10. The holdout is never an input here: :func:`dataset_report` and
:func:`check_coco_folder` read only the train and validation sides.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from filament.annotations import TrainImage
from filament.detector_data import (
    ANNOTATIONS_NAME,
    DEFAULT_MIN_LABEL_AREA,
    MIN_VISIBLE,
    TARGET,
    Window,
    image_examples,
)

# What ``rfdetr`` 1.12.0 wrote into ``output_dir`` in the 2-epoch Kaggle pilot.
BEST_CHECKPOINTS = (
    "checkpoint_best_total.pth",
    "checkpoint_best_ema.pth",
    "checkpoint_best_regular.pth",
)
# The EMA weights after the last epoch that ran. ``checkpoint.pth`` is not used: the
# library writes it every ``checkpoint_interval`` (default 10) epochs, so after 16
# epochs it holds epoch 10, not the end of training.
LAST_CHECKPOINT = "last_ema.pth"
CHECKPOINT_CHOICES = ("best", "last")


def check_resolution(resolution: int, patch_size: int, num_windows: int) -> None:
    """Raise ``ValueError`` unless the detector can take this input size.

    RF-DETR's DINOv2 backbone cuts the input into ``patch_size`` squares and
    splits that grid into ``num_windows`` x ``num_windows`` windows, so the side
    has to be a multiple of ``patch_size * num_windows``. For ``RFDETRSegLarge``
    the config says 12 and 2, so 24: 504 (its default) and 768 pass, 750 does not.
    """
    for name, value in (
        ("resolution", resolution),
        ("patch_size", patch_size),
        ("num_windows", num_windows),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    unit = patch_size * num_windows
    if resolution % unit:
        raise ValueError(
            f"resolution {resolution} is not a multiple of patch_size "
            f"({patch_size}) x num_windows ({num_windows}) = {unit}"
        )


def subsample_evenly[T](items: Sequence[T], limit: int) -> tuple[T, ...]:
    """At most ``limit`` items, evenly spaced across ``items``, order kept.

    A pilot that took the first ``limit`` files of a date-sorted split would see
    only the oldest dates. ``limit >= len(items)`` returns everything.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError(f"limit must be a positive int, got {limit!r}")
    count = len(items)
    if limit >= count:
        return tuple(items)
    return tuple(items[(index * count) // limit] for index in range(limit))


@dataclass(frozen=True, slots=True)
class DatasetReport:
    """What the labels of a split look like before the min_area cut and after it.

    ``examples`` counts (file, window, annotator view) triples. ``negative`` is
    how many of them have no instance left. ``instances`` are the ones kept at
    ``min_area``. ``dropped_by_min_area`` are the ones that passed the visibility
    rule and were under ``min_area`` pixels inside the window.
    """

    files: int
    examples: int
    negative: int
    instances: int
    dropped_by_min_area: int

    @property
    def negative_fraction(self) -> float:
        return self.negative / self.examples if self.examples else 0.0


def dataset_report(
    images: Iterable[TrainImage],
    windows: Sequence[Window],
    target: int = TARGET,
    min_visible: float = MIN_VISIBLE,
    min_area: int = DEFAULT_MIN_LABEL_AREA,
) -> DatasetReport:
    """Count examples, negatives and the instances ``min_area`` drops.

    Runs :func:`filament.detector_data.image_examples` twice per file, once with
    ``min_area`` and once with 0, so the difference is exactly what the cut
    removes. It reads annotations only, no pixels.
    """
    files = examples = negative = kept = dropped = 0
    for image in images:
        files += 1
        with_cut = image_examples(image, windows, target, min_visible, min_area)
        without_cut = image_examples(image, windows, target, min_visible, 0)
        examples += len(with_cut)
        negative += sum(1 for example in with_cut if not example.instances)
        kept += sum(len(example.instances) for example in with_cut)
        dropped += sum(len(example.instances) for example in without_cut) - sum(
            len(example.instances) for example in with_cut
        )
    return DatasetReport(
        files=files,
        examples=examples,
        negative=negative,
        instances=kept,
        dropped_by_min_area=dropped,
    )


@dataclass(frozen=True, slots=True)
class FolderCheck:
    """Counts from :func:`check_coco_folder`."""

    images: int
    annotations: int
    negative_images: int
    category_ids: tuple[int, ...]


def check_coco_folder(folder: str | Path, max_problems: int = 5) -> FolderCheck:
    """Check one RF-DETR folder (``_annotations.coco.json`` plus its JPEGs).

    A cheap guard that does not need the library: every image file exists, ids
    are unique, every annotation points at a known image and category, every
    polygon has at least 3 points, every box lies inside its image and has a
    positive size. It is not a replacement for the library's own loader, which
    the training notebook also runs. Raises ``ValueError`` listing up to
    ``max_problems`` problems.
    """
    root = Path(folder)
    payload = json.loads((root / ANNOTATIONS_NAME).read_text(encoding="utf-8"))
    problems: list[str] = []

    def note(text: str) -> None:
        problems.append(text)

    images = {}
    for image in payload["images"]:
        if image["id"] in images:
            note(f"duplicate image id {image['id']}")
        images[image["id"]] = image
        if not (root / image["file_name"]).is_file():
            note(f"missing file {image['file_name']}")
    categories = {category["id"] for category in payload["categories"]}
    if not categories:
        note("no category")
    seen_annotations: set[int] = set()
    with_labels: set[int] = set()
    for annotation in payload["annotations"]:
        identifier = annotation["id"]
        if identifier in seen_annotations:
            note(f"duplicate annotation id {identifier}")
        seen_annotations.add(identifier)
        image = images.get(annotation["image_id"])
        if image is None:
            note(f"annotation {identifier}: unknown image {annotation['image_id']}")
            continue
        with_labels.add(annotation["image_id"])
        if annotation["category_id"] not in categories:
            note(f"annotation {identifier}: unknown category")
        polygons = annotation["segmentation"]
        if not polygons or any(len(polygon) < 6 for polygon in polygons):
            note(f"annotation {identifier}: a polygon has under 3 points")
        x, y, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            note(f"annotation {identifier}: empty box")
        elif (
            x < 0
            or y < 0
            or x + width > image["width"] + 1e-6
            or y + height > image["height"] + 1e-6
        ):
            note(f"annotation {identifier}: box outside its image")
    if problems:
        shown = "; ".join(problems[:max_problems])
        raise ValueError(
            f"{root}: {len(problems)} problem(s) in the COCO folder: {shown}"
        )
    return FolderCheck(
        images=len(images),
        annotations=len(payload["annotations"]),
        negative_images=len(images) - len(with_labels),
        category_ids=tuple(sorted(categories)),
    )


def pick_checkpoint(names: Iterable[str], choice: str = "best") -> str:
    """Name the checkpoint file to hand on, out of the files in ``output_dir``.

    ``"best"`` is the library's best on the validation folder of the dataset,
    which is the detector validation dates and never the holdout (the holdout is
    not written to disk). The first of :data:`BEST_CHECKPOINTS` that is present
    wins; with none of those, the one file with ``best`` in its name, if there is
    exactly one. ``"last"`` is :data:`LAST_CHECKPOINT`, the EMA weights after the last epoch that
    ran. A ``checkpoint.pth`` is never taken for it. Anything
    else is a ``FileNotFoundError`` that lists what was there, so a changed
    library naming fails loudly and not on the wrong file.
    """
    if choice not in CHECKPOINT_CHOICES:
        raise ValueError(f"choice must be one of {CHECKPOINT_CHOICES}, got {choice!r}")
    present = sorted(set(names))
    if choice == "last":
        if LAST_CHECKPOINT in present:
            return LAST_CHECKPOINT
    else:
        for name in BEST_CHECKPOINTS:
            if name in present:
                return name
        loose = [name for name in present if "best" in name and name.endswith(".pth")]
        if len(loose) == 1:
            return loose[0]
    raise FileNotFoundError(
        f"no checkpoint for choice {choice!r}; the output directory holds: {present}"
    )


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RunProjection:
    """A pilot's speed and what it implies for the full run."""

    seconds_per_epoch: float
    projected_epoch_seconds: float
    projected_hours: float


def project_full_run(
    pilot_seconds: float,
    pilot_epochs: int,
    pilot_train_files: int,
    pilot_val_files: int,
    full_train_files: int,
    full_val_files: int,
    full_epochs: int,
    val_cost_ratio: float = 0.0,
) -> RunProjection:
    """Scale a pilot to the full run, train and validation files separately.

    ``pilot_seconds`` is the wall time of the pilot's ``train`` call, including the
    library's start-up, so the projection leans long. An epoch is modelled as
    ``train_files + val_cost_ratio * val_files`` units of work, where
    ``val_cost_ratio`` is what a validation file costs relative to a train file.
    The pilot mixes train and validation 4:1 and the full run about 9:1, so
    scaling them together would under-project. The default ratio of 0 puts all of
    the pilot's time on the train files, the longest (safest) projection; a
    forward-only validation pass is cheaper than a train step, so the truth lies
    between 0 and 1 and nearer 0. Untested model: the second pilot epoch is the
    better guide when the first one paid for downloads and warm-up.
    """
    if (
        pilot_epochs < 1
        or full_epochs < 1
        or pilot_train_files < 1
        or full_train_files < 1
    ):
        raise ValueError("epochs and train file counts must be positive")
    if pilot_val_files < 0 or full_val_files < 0:
        raise ValueError("validation file counts cannot be negative")
    if not 0.0 <= val_cost_ratio <= 1.0:
        raise ValueError(f"val_cost_ratio must be in [0, 1], got {val_cost_ratio!r}")
    if pilot_seconds <= 0:
        raise ValueError(f"pilot_seconds must be positive, got {pilot_seconds!r}")
    per_epoch = pilot_seconds / pilot_epochs
    pilot_units = pilot_train_files + val_cost_ratio * pilot_val_files
    full_units = full_train_files + val_cost_ratio * full_val_files
    full_epoch = per_epoch * full_units / pilot_units
    return RunProjection(
        seconds_per_epoch=per_epoch,
        projected_epoch_seconds=full_epoch,
        projected_hours=full_epoch * full_epochs / 3600.0,
    )


def guard_seconds(
    *,
    pilot: bool,
    max_train_hours: float,
    pilot_max_minutes: float,
    stop_after_minutes: float | None = None,
) -> float:
    """Seconds after which the wall-clock guard stops the training.

    ``stop_after_minutes`` (``PILOT_STOP_AFTER_MINUTES``) wins when it is set, so a
    pilot can make the guard fire and show that the checkpoint survives it.
    """
    if stop_after_minutes is not None:
        if isinstance(stop_after_minutes, bool) or stop_after_minutes <= 0:
            raise ValueError(
                f"stop_after_minutes must be positive or None, got {stop_after_minutes!r}"
            )
        return float(stop_after_minutes) * 60.0
    return pilot_max_minutes * 60.0 if pilot else max_train_hours * 3600.0


def copy_atomic(source: str | Path, destination: str | Path) -> Path:
    """Copy ``source`` to ``destination`` through a temp file and ``os.replace``.

    A reader (or a kill at any instant) sees the old file or the new one, never half
    of one. The temp file sits next to the destination so the replace stays on one
    filesystem, and is removed if the copy fails.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def sync_best_checkpoint(
    run_dir: str | Path, destination: str | Path, seen: dict[str, tuple[int, int]]
) -> str | None:
    """Copy the current best checkpoint of ``run_dir`` to ``destination`` if it changed.

    Returns the name copied, or ``None`` when there is no best checkpoint yet or it
    has not changed since the last call. ``seen`` is the caller's state: file name
    to ``(mtime_ns, size)`` of the last copy. The stat is taken *before* the copy,
    so a file the library rewrites during the copy differs on the next call and is
    copied again. The destination always holds a complete file
    (:func:`copy_atomic`).
    """
    root = Path(run_dir)
    try:
        name = pick_checkpoint([p.name for p in root.glob("*.pth")], "best")
    except FileNotFoundError:
        return None
    stat = (root / name).stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    if seen.get(name) == signature:
        return None
    copy_atomic(root / name, destination)
    seen[name] = signature
    return name
