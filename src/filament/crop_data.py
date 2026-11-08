"""Decode train frames once and keep them in RAM at native resolution.

Each :class:`filament.annotations.TrainImage` becomes a uint8 gray frame and
a uint8 union mask of the same size (no resizing). The mask is
:func:`filament.train_data.semantic_target_from_annotations`: the union of
every polygon on the file, all four categories as one class.

At 2048 x 2048 that is 4 MiB per frame and 4 MiB per mask, about 4.5 GiB for
565 train files. The arrays are plain numpy buffers, so a forked process
shares them without copying them.

With agreement weights the mask stays one uint8 array, nonzero exactly on the
union, but each positive pixel holds its BCE weight instead of 1
(:func:`agreement_mask`). The crop sampler only looks at which pixels are
nonzero, so it draws the same crops as for the plain union mask, and the two
runs compare on the same data. The weight is ``0.4 + 0.6 * k / R``, where ``k``
of the file's ``R`` annotator views cover the pixel. It is stored as
``round(255 * weight)``, so it is off by at most 1/510. Background pixels weigh
1. The formula is the proposal of ``docs/ROADMAP_V2.md``, not a tuned value.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from filament.annotations import TrainImage, annotator_views
from filament.train_data import semantic_target_from_annotations

AGREEMENT_BASE = 0.4
AGREEMENT_SLOPE = 0.6

# BCE weight of each stored uint8 value: 1 on background (0), value / 255 else.
_STORED_WEIGHT = np.arange(256, dtype=np.float32) / 255.0
_STORED_WEIGHT[0] = 1.0


@dataclass(frozen=True, slots=True)
class TrainCache:
    """Frames and union masks for a list of train files, in the same order.

    Nonzero mask pixels are the union. The values are 1, or the stored weights
    of :func:`agreement_mask` when the cache was built with agreement weights.
    """

    images: list[np.ndarray]
    masks: list[np.ndarray]
    file_names: tuple[str, ...]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.images) + sum(a.nbytes for a in self.masks)

    def foreground_fraction(self) -> float:
        """Share of foreground pixels over all masks."""
        pixels = sum(mask.size for mask in self.masks)
        return sum(int(np.count_nonzero(m)) for m in self.masks) / max(pixels, 1)


def read_gray(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read one JPEG as a C-contiguous uint8 gray array.

    Raises when the file is missing or unreadable, or when ``expected_shape``
    is given and does not match.
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(f"{path.name} is not a uint8 grayscale image")
    if expected_shape is not None and gray.shape != expected_shape:
        raise ValueError(
            f"{path.name} is {gray.shape[0]}x{gray.shape[1]}, "
            f"expected {expected_shape[0]}x{expected_shape[1]}"
        )
    return np.ascontiguousarray(gray)


def agreement_mask(record: TrainImage) -> np.ndarray:
    """Return the record's uint8 mask with the BCE weight stored on each positive.

    ``R`` counts every annotator view of the file, also one that drew nothing,
    as the pooled metric does. A file with one view stores 255 on every positive
    pixel, which decodes to weight 1, so it trains as with the plain union.
    """
    views = annotator_views(record)
    count = np.zeros((record.height, record.width), dtype=np.uint16)
    for view in views:
        mask = semantic_target_from_annotations(view, record.height, record.width)
        count += mask != 0
    weight = AGREEMENT_BASE + AGREEMENT_SLOPE * (count / len(views))
    return np.where(count > 0, np.rint(255.0 * weight), 0).astype(np.uint8)


def decode_agreement_mask(stored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split :func:`agreement_mask` values into ``(target, weight)``.

    ``target`` is uint8, 1 on the union. ``weight`` is float32, 1 on background.
    Works on a whole mask or on a batch of crops.
    """
    array = np.asarray(stored)
    if array.dtype != np.uint8:
        raise TypeError(f"expected a uint8 mask, got dtype {array.dtype}")
    return (array != 0).astype(np.uint8), _STORED_WEIGHT[array]


def check_mask_kind(masks: Sequence[np.ndarray], agreement_weights: bool) -> None:
    """Stop when the masks in RAM are not the kind this recipe expects.

    A plain mask holds 0 and 1. A stored weight is at least ``255 * 0.4``. Reading
    one kind as the other gives no error, only wrong targets or weights: a stored
    weight read as a target even makes the loss negative. An all-zero mask fits both.
    """
    for index, mask in enumerate(masks):
        top = int(np.max(mask)) if np.size(mask) else 0
        if top == 0:
            continue
        if agreement_weights and top <= 1:
            raise ValueError(
                f"mask {index} is a plain union mask, but the recipe uses agreement "
                "weights. Build the cache with agreement_weights=True."
            )
        if not agreement_weights and top > 1:
            raise ValueError(
                f"mask {index} holds agreement weights, but the recipe does not use "
                "them. Build the cache with agreement_weights=False."
            )


def load_train_cache(
    records: Sequence[TrainImage],
    image_dir: Path,
    *,
    log: Callable[[str], None] = print,
    progress_every: int = 50,
    agreement_weights: bool = False,
) -> TrainCache:
    """Decode every record's JPEG and rasterize its mask, once.

    The JPEG size has to equal the ``height`` and ``width`` in the record.
    Prints a progress line every ``progress_every`` files. With
    ``agreement_weights`` the masks are :func:`agreement_mask`, and a file whose
    nonzero pixels are not its union mask stops the run: it would change the
    crops the comparison with the plain union relies on.
    """
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    total = len(records)
    for index, record in enumerate(records):
        if index % progress_every == 0:
            log(f"prepared {index} / {total}")
        gray = read_gray(
            Path(image_dir) / record.file_name, (record.height, record.width)
        )
        mask = semantic_target_from_annotations(
            record.annotations, record.height, record.width
        )
        if agreement_weights:
            stored = agreement_mask(record)
            if not np.array_equal(stored != 0, mask != 0):
                raise ValueError(
                    f"{record.file_name}: the annotator views do not add up to "
                    "the union mask"
                )
            mask = stored
        images.append(gray)
        masks.append(mask)
    log(f"prepared {total} / {total}")
    return TrainCache(
        images=images,
        masks=masks,
        file_names=tuple(record.file_name for record in records),
    )
