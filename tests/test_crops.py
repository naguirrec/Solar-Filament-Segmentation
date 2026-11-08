"""Crop sampling. Pure numpy: no torch, no real frames."""

from __future__ import annotations

import numpy as np
import pytest

from filament.crops import (
    CropSampler,
    augment_pair,
    foreground_index,
    sample_crop_origin,
)


def _blob_mask(side: int = 1024, row: int = 700, col: int = 300, size: int = 16):
    mask = np.zeros((side, side), dtype=np.uint8)
    mask[row : row + size, col : col + size] = 1
    return mask


def test_foreground_index_is_row_major_int32() -> None:
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1, 2] = 1
    mask[3, 4] = 1
    index = foreground_index(mask)
    assert index.dtype == np.int32
    assert index.tolist() == [1 * 5 + 2, 3 * 5 + 4]
    assert foreground_index(np.zeros((4, 5), dtype=np.uint8)).size == 0


def test_foreground_index_caps_points_evenly() -> None:
    mask = np.ones((100, 100), dtype=np.uint8)
    index = foreground_index(mask, max_points=50)
    assert index.size == 50
    assert index[0] == 0
    assert index[-1] == 100 * 100 - 1
    assert np.all(np.diff(index) > 0)


def test_origin_is_always_inside_the_image() -> None:
    rng = np.random.default_rng(0)
    for height, width, crop in [(512, 512, 512), (600, 777, 512), (2048, 2048, 512)]:
        foreground = np.array(
            [0, width - 1, (height - 1) * width, height * width - 1], dtype=np.int32
        )
        for p_fg in (0.0, 0.5, 1.0):
            for _ in range(500):
                top, left = sample_crop_origin(
                    (height, width), crop, rng, foreground=foreground, p_fg=p_fg
                )
                assert 0 <= top <= height - crop
                assert 0 <= left <= width - crop


def test_single_foreground_pixel_is_always_inside_the_crop() -> None:
    # Jitter plus clamping must never push the chosen pixel out of the crop.
    rng = np.random.default_rng(1)
    for row, col in [(3, 1000), (1000, 3), (0, 0), (1023, 1023), (512, 512)]:
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[row, col] = 1
        sampler = CropSampler([mask], [mask], 128, p_fg=1.0, augment=False)
        for _ in range(200):
            _image, crop = sampler.sample(0, rng)
            assert crop.shape == (128, 128)
            assert crop.sum() == 1


def test_crop_shape_and_dtype() -> None:
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(300, 400), dtype=np.uint8)
    mask = (image > 200).astype(np.uint8)
    sampler = CropSampler([image], [mask], 64)
    crop_image, crop_mask = sampler.sample(0, rng)
    assert crop_image.shape == crop_mask.shape == (64, 64)
    assert crop_image.dtype == crop_mask.dtype == np.uint8
    assert crop_image.flags["C_CONTIGUOUS"] and crop_mask.flags["C_CONTIGUOUS"]


def test_foreground_bias_raises_foreground_fraction() -> None:
    mask = _blob_mask()
    image = np.zeros_like(mask)

    def mean_fraction(p_fg: float) -> float:
        rng = np.random.default_rng(3)
        sampler = CropSampler([image], [mask], 128, p_fg=p_fg, augment=False)
        return float(np.mean([sampler.sample(0, rng)[1].mean() for _ in range(400)]))

    uniform = mean_fraction(0.0)
    biased = mean_fraction(0.67)
    assert uniform > 0  # the blob is sometimes hit by chance
    assert biased > 5 * uniform


def test_empty_mask_falls_back_to_uniform() -> None:
    rng = np.random.default_rng(4)
    empty = np.zeros((600, 600), dtype=np.uint8)
    sampler = CropSampler([empty], [empty], 128, p_fg=1.0)
    origins = {
        sample_crop_origin(
            empty.shape, 128, rng, foreground=sampler._foreground[0], p_fg=1.0
        )
        for _ in range(200)
    }
    assert len(origins) > 50  # spread over the image, not a fixed place
    assert sampler.sample(0, rng)[0].shape == (128, 128)


def test_crop_larger_than_image_is_an_error() -> None:
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="larger than the image"):
        sample_crop_origin((100, 200), 128, rng)
    small = np.zeros((100, 200), dtype=np.uint8)
    with pytest.raises(ValueError, match="larger than image"):
        CropSampler([small], [small], 128)


def test_bad_arguments_are_errors() -> None:
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        sample_crop_origin((100, 100), 0, rng)
    with pytest.raises(ValueError):
        sample_crop_origin((100, 100), 10, rng, p_fg=1.5)
    with pytest.raises(ValueError):
        sample_crop_origin((100, 100), 10, rng, jitter=-1)
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        CropSampler([image], [], 4)
    with pytest.raises(ValueError):
        CropSampler([], [], 4)
    with pytest.raises(ValueError):
        CropSampler([image], [np.zeros((9, 10), dtype=np.uint8)], 4)


def test_same_seed_same_batches_different_seed_differs() -> None:
    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, size=(256, 256), dtype=np.uint8) for _ in range(3)]
    masks = [(image > 240).astype(np.uint8) for image in images]
    sampler = CropSampler(images, masks, 64)

    def run(seed: int):
        return list(sampler.epoch_batches(10, 4, np.random.default_rng(seed)))

    first, again, other = run(11), run(11), run(12)
    for (a_img, a_mask), (b_img, b_mask) in zip(first, again, strict=True):
        assert np.array_equal(a_img, b_img)
        assert np.array_equal(a_mask, b_mask)
    assert not all(
        np.array_equal(a[0], b[0]) for a, b in zip(first, other, strict=True)
    )


def test_epoch_batches_sizes_and_even_image_use() -> None:
    images = [np.full((64, 64), value, dtype=np.uint8) for value in (10, 20, 30, 40)]
    masks = [np.zeros((64, 64), dtype=np.uint8) for _ in images]
    sampler = CropSampler(images, masks, 32, augment=False)
    batches = list(sampler.epoch_batches(20, 8, np.random.default_rng(0)))
    assert [batch[0].shape[0] for batch in batches] == [8, 8, 4]
    assert batches[0][0].shape == (8, 32, 32)
    # 20 crops over 4 images: 5 each. The constant fill identifies the source.
    seen = np.concatenate([batch[0][:, 0, 0] for batch in batches])
    assert {int(v): int((seen == v).sum()) for v in (10, 20, 30, 40)} == {
        10: 5,
        20: 5,
        30: 5,
        40: 5,
    }


def test_default_epoch_gives_every_file_eight_crops() -> None:
    images = [np.zeros((8, 8), dtype=np.uint8)] * 565
    sampler = CropSampler(images, images, 4)
    order = sampler.image_order(4520, np.random.default_rng(0))
    assert np.bincount(order, minlength=565).tolist() == [8] * 565


def test_augment_keeps_image_and_mask_aligned() -> None:
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    mask = (image > 128).astype(np.uint8)
    moved = set()
    for _ in range(100):
        new_image, new_mask = augment_pair(image, mask, rng)
        assert np.array_equal(new_mask, (new_image > 128).astype(np.uint8))
        assert sorted(new_image.ravel()) == sorted(image.ravel())
        moved.add(new_image.tobytes())
    assert len(moved) == 8  # the 8 elements of the dihedral group


def test_no_augment_returns_the_plain_slice() -> None:
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, size=(100, 100), dtype=np.uint8)
    mask = np.zeros_like(image)
    sampler = CropSampler([image], [mask], 32, augment=False)
    crop, _ = sampler.sample(0, np.random.default_rng(10))
    top, left = sample_crop_origin((100, 100), 32, np.random.default_rng(10))
    assert np.array_equal(crop, image[top : top + 32, left : left + 32])
