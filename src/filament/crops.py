"""Random training crops at native resolution, pure numpy.

Frames are 2048 x 2048 and filaments cover a small share of them, so a
uniform crop is mostly empty. With probability ``p_fg`` a crop is centered
on a random foreground pixel (plus jitter, clamped inside the image); with
probability ``1 - p_fg`` its position is uniform. An image with no
foreground always falls back to uniform.

Everything is deterministic given a ``numpy.random.Generator``. Nothing here
imports torch, cv2 or the annotation code.

Augmentation is the dihedral group: horizontal flip, vertical flip and a
90 degree rotation. It changes the apparent orientation relative to the
solar disk. The target is one filament class (chirality is not a label), and
the effect on the score is untuned.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

# Cap on stored foreground pixels per image (4 bytes each). A mask with more
# foreground is subsampled at even spacing, which keeps the memory of 565
# images under about 150 MiB.
MAX_FOREGROUND_POINTS = 65_536


def foreground_index(mask: np.ndarray, max_points: int = MAX_FOREGROUND_POINTS):
    """Return flat (row-major) indices of foreground pixels as int32.

    At most ``max_points`` are kept. When there are more, they are taken at
    even spacing, so the choice is deterministic. An all-zero mask returns an
    empty array.
    """
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {array.shape}")
    if max_points < 1:
        raise ValueError("max_points must be >= 1")
    flat = np.flatnonzero(array)
    if flat.size > max_points:
        pick = np.linspace(0, flat.size - 1, num=max_points).astype(np.int64)
        flat = flat[pick]
    return flat.astype(np.int32)


def sample_crop_origin(
    shape: tuple[int, int],
    crop_size: int,
    rng: np.random.Generator,
    *,
    foreground: np.ndarray | None = None,
    p_fg: float = 0.67,
    jitter: int | None = None,
) -> tuple[int, int]:
    """Return the ``(top, left)`` corner of one square crop.

    ``foreground`` is the output of :func:`foreground_index` for this image,
    or ``None``. With probability ``p_fg`` and a non-empty ``foreground``,
    the crop is centered on one random foreground pixel, shifted by a
    uniform integer in ``[-jitter, jitter]`` on each axis (default
    ``crop_size // 4``), then clamped so the crop lies inside the image.
    Otherwise the corner is uniform over every position that fits.

    Raises ``ValueError`` when the crop does not fit inside the image.
    """
    height, width = int(shape[0]), int(shape[1])
    if crop_size < 1:
        raise ValueError("crop_size must be >= 1")
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"crop_size {crop_size} is larger than the image {height}x{width}"
        )
    if not 0.0 <= p_fg <= 1.0:
        raise ValueError("p_fg must be in [0, 1]")
    spread = crop_size // 4 if jitter is None else int(jitter)
    if spread < 0:
        raise ValueError("jitter must be >= 0")

    max_top = height - crop_size
    max_left = width - crop_size
    use_foreground = (
        foreground is not None and len(foreground) > 0 and rng.random() < p_fg
    )
    if use_foreground:
        flat = int(foreground[int(rng.integers(0, len(foreground)))])
        center_row, center_col = divmod(flat, width)
        shift_row = int(rng.integers(-spread, spread + 1))
        shift_col = int(rng.integers(-spread, spread + 1))
        top = center_row - crop_size // 2 + shift_row
        left = center_col - crop_size // 2 + shift_col
        return (
            int(min(max(top, 0), max_top)),
            int(min(max(left, 0), max_left)),
        )
    return int(rng.integers(0, max_top + 1)), int(rng.integers(0, max_left + 1))


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one random flip / 90 degree rotation to image and mask together.

    Returns C-contiguous copies. Square inputs keep their shape.
    """
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        image, mask = image[::-1, :], mask[::-1, :]
    turns = int(rng.integers(0, 4))
    if turns:
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class CropSampler:
    """Draw crops from images held in RAM.

    ``images`` and ``masks`` are lists of 2D uint8 arrays; image ``i`` and
    mask ``i`` share a shape. Crops are slices of those arrays, so nothing
    is decoded or resized while sampling.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        crop_size: int,
        *,
        p_fg: float = 0.67,
        jitter: int | None = None,
        augment: bool = True,
    ) -> None:
        if len(images) != len(masks):
            raise ValueError(f"{len(images)} images but {len(masks)} masks")
        if len(images) == 0:
            raise ValueError("no images to sample from")
        for index, (image, mask) in enumerate(zip(images, masks, strict=True)):
            if image.ndim != 2 or image.shape != mask.shape:
                raise ValueError(
                    f"image {index} shape {image.shape} and mask shape "
                    f"{mask.shape} must be the same 2D shape"
                )
            if crop_size > min(image.shape):
                raise ValueError(
                    f"crop_size {crop_size} is larger than image {index} {image.shape}"
                )
        self.images = images
        self.masks = masks
        self.crop_size = int(crop_size)
        self.p_fg = float(p_fg)
        self.jitter = jitter
        self.augment = bool(augment)
        self._foreground = [foreground_index(mask) for mask in masks]

    def __len__(self) -> int:
        return len(self.images)

    def sample(
        self, index: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one ``(image_crop, mask_crop)`` pair from image ``index``."""
        image, mask = self.images[index], self.masks[index]
        top, left = sample_crop_origin(
            image.shape,
            self.crop_size,
            rng,
            foreground=self._foreground[index],
            p_fg=self.p_fg,
            jitter=self.jitter,
        )
        rows = slice(top, top + self.crop_size)
        cols = slice(left, left + self.crop_size)
        crop_image, crop_mask = image[rows, cols], mask[rows, cols]
        if self.augment:
            return augment_pair(crop_image, crop_mask, rng)
        return np.ascontiguousarray(crop_image), np.ascontiguousarray(crop_mask)

    def image_order(self, n_crops: int, rng: np.random.Generator) -> np.ndarray:
        """Return which image each of ``n_crops`` crops comes from.

        Images are repeated as evenly as possible (``n_crops`` divisible by
        the image count gives every image the same number of crops) and then
        shuffled.
        """
        if n_crops < 1:
            raise ValueError("n_crops must be >= 1")
        repeats = -(-n_crops // len(self))
        order = np.tile(np.arange(len(self)), repeats)
        return rng.permutation(order)[:n_crops]

    def epoch_batches(
        self,
        n_crops: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(images, masks)`` uint8 batches of shape ``(B, S, S)``.

        ``n_crops`` crops in total. The last batch is smaller when
        ``n_crops`` is not a multiple of ``batch_size``.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        order = self.image_order(n_crops, rng)
        for start in range(0, len(order), batch_size):
            pairs = [
                self.sample(int(i), rng) for i in order[start : start + batch_size]
            ]
            yield (
                np.stack([pair[0] for pair in pairs]),
                np.stack([pair[1] for pair in pairs]),
            )
