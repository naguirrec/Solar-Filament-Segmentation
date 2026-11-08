"""Tile positions and stitching. Pure numpy."""

from __future__ import annotations

import numpy as np
import pytest

from filament.tiling import predict_tiled, taper_window, tile_boxes, tile_starts


@pytest.mark.parametrize(
    ("length", "tile", "overlap"),
    [
        (2048, 512, 128),
        (2048, 512, 0),
        (1000, 512, 128),  # (1000 - 512) = 488 is not a multiple of 384
        (513, 512, 128),
        (777, 100, 30),
        (512, 512, 128),
        (600, 512, 511),
        (5, 1, 0),
    ],
)
def test_starts_cover_every_pixel_and_end_flush(
    length: int, tile: int, overlap: int
) -> None:
    starts = tile_starts(length, tile, overlap)
    covered = np.zeros(length, dtype=bool)
    for start in starts:
        assert 0 <= start <= max(length - tile, 0)
        covered[start : start + tile] = True
    assert covered.all()
    assert starts[0] == 0
    assert starts[-1] == max(length - tile, 0)  # last tile flush with the edge
    assert starts == sorted(set(starts))


def test_default_grid_is_five_by_five() -> None:
    assert tile_starts(2048, 512, 128) == [0, 384, 768, 1152, 1536]
    assert len(tile_boxes(2048, 2048, 512, 128)) == 25


def test_flush_tile_when_size_is_not_a_multiple_of_stride() -> None:
    assert tile_starts(1000, 512, 128) == [0, 384, 488]


def test_image_not_longer_than_tile_has_one_start() -> None:
    assert tile_starts(300, 512, 128) == [0]
    assert tile_starts(512, 512, 128) == [0]


def test_bad_tiling_arguments() -> None:
    with pytest.raises(ValueError):
        tile_starts(1000, 512, 512)
    with pytest.raises(ValueError):
        tile_starts(1000, 512, -1)
    with pytest.raises(ValueError):
        tile_starts(1000, 0, 0)
    with pytest.raises(ValueError):
        tile_starts(0, 512, 0)


def _identity(tiles: np.ndarray) -> np.ndarray:
    return tiles.astype(np.float32)


@pytest.mark.parametrize(
    ("shape", "tile", "overlap"),
    [((64, 64), 32, 8), ((50, 70), 32, 8), ((45, 45), 16, 5), ((32, 32), 32, 8)],
)
def test_identity_predict_reproduces_the_input_exactly(
    shape: tuple[int, int], tile: int, overlap: int
) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)
    out = predict_tiled(image, _identity, tile_size=tile, overlap=overlap, batch_size=3)
    assert out.dtype == np.float32
    assert out.shape == shape
    assert np.array_equal(out, image.astype(np.float32))


def test_overlap_pixels_are_the_mean_of_the_covering_tiles() -> None:
    # 20 x 10 image, 10 x 10 tiles, overlap 4: starts [0, 6, 10] on rows.
    # Tile k predicts the constant 10 * (k + 1), so the stitched value is the
    # mean over the tiles covering each row.
    calls: list[int] = []

    def constant_per_tile(tiles: np.ndarray) -> np.ndarray:
        out = np.empty(tiles.shape, dtype=np.float32)
        for slot in range(tiles.shape[0]):
            calls.append(len(calls))
            out[slot] = 10.0 * len(calls)
        return out

    image = np.zeros((20, 10), dtype=np.uint8)
    out = predict_tiled(image, constant_per_tile, tile_size=10, overlap=4, batch_size=2)
    starts = tile_starts(20, 10, 4)
    assert starts == [0, 6, 10]
    expected = np.zeros(20, dtype=np.float32)
    count = np.zeros(20, dtype=np.float32)
    for k, start in enumerate(starts):
        expected[start : start + 10] += 10.0 * (k + 1)
        count[start : start + 10] += 1
    expected /= count
    assert np.allclose(out[:, 0], expected)
    assert np.allclose(out[:, 9], expected)
    assert count.max() == 2
    assert out[8, 0] == pytest.approx((10 + 20) / 2)  # tiles 0 and 1
    assert out[12, 0] == pytest.approx((20 + 30) / 2)  # tiles 1 and 2
    assert out[0, 0] == pytest.approx(10)  # covered by the first tile only


def test_counts_are_never_zero_for_awkward_sizes() -> None:
    ones = lambda tiles: np.ones(tiles.shape, dtype=np.float32)
    for height, width in [(33, 47), (31, 31), (100, 17), (1, 1)]:
        image = np.zeros((height, width), dtype=np.uint8)
        out = predict_tiled(image, ones, tile_size=16, overlap=5, batch_size=4)
        assert out.shape == (height, width)
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 1.0)


def test_image_smaller_than_tile_is_padded_and_cropped_back() -> None:
    seen: list[tuple[int, ...]] = []

    def spy(tiles: np.ndarray) -> np.ndarray:
        seen.append(tiles.shape)
        return tiles.astype(np.float32)

    image = np.arange(30 * 20, dtype=np.uint8).reshape(30, 20)
    out = predict_tiled(image, spy, tile_size=64, overlap=16, batch_size=8)
    assert seen == [(1, 64, 64)]
    assert out.shape == (30, 20)
    assert np.array_equal(out, image.astype(np.float32))


def test_batches_never_exceed_batch_size() -> None:
    sizes: list[int] = []

    def spy(tiles: np.ndarray) -> np.ndarray:
        sizes.append(tiles.shape[0])
        return tiles.astype(np.float32)

    image = np.zeros((64, 64), dtype=np.uint8)
    predict_tiled(image, spy, tile_size=16, overlap=4, batch_size=5)
    assert sum(sizes) == len(tile_boxes(64, 64, 16, 4))
    assert max(sizes) <= 5


def test_wrong_predict_shape_is_an_error() -> None:
    def bad(tiles: np.ndarray) -> np.ndarray:
        return np.zeros((tiles.shape[0], 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="predict_batch returned shape"):
        predict_tiled(np.zeros((32, 32), dtype=np.uint8), bad, tile_size=16, overlap=4)


def test_window_weights_the_overlap() -> None:
    # Two 8-px tiles on a 12-px row; tile 0 predicts 0, tile 1 predicts 1.
    # With a window the overlap is a weighted mean, not 0.5.
    window = np.ones((8, 8), dtype=np.float32)
    window[:, :2] = 0.5  # the left columns of every tile count half

    image = np.zeros((8, 12), dtype=np.uint8)
    state = {"n": 0}

    def alternating(tiles: np.ndarray) -> np.ndarray:
        out = np.full(tiles.shape, state["n"], dtype=np.float32)
        state["n"] += 1
        return out

    uniform = predict_tiled(image, alternating, tile_size=8, overlap=4, batch_size=1)
    state["n"] = 0
    weighted = predict_tiled(
        image, alternating, tile_size=8, overlap=4, batch_size=1, window=window
    )
    assert np.allclose(uniform[:, 4:8], 0.5)  # columns 4..7 are covered twice
    # Column 4 is column 4 of tile 0 (weight 1) and column 0 of tile 1 (0.5).
    assert weighted[0, 4] == pytest.approx((0 * 1.0 + 1 * 0.5) / 1.5)
    assert np.allclose(weighted[:, 0], 0.0)


def test_window_must_be_positive_and_tile_shaped() -> None:
    image = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(ValueError, match="window shape"):
        predict_tiled(image, _identity, tile_size=8, overlap=2, window=np.ones((4, 4)))
    bad = np.ones((8, 8))
    bad[0, 0] = 0
    with pytest.raises(ValueError, match="> 0"):
        predict_tiled(image, _identity, tile_size=8, overlap=2, window=bad)


def test_taper_window_is_positive_symmetric_and_stitches_constants() -> None:
    window = taper_window(32, 8)
    assert window.shape == (32, 32)
    assert window.min() > 0
    assert window.max() == pytest.approx(1.0)
    assert np.allclose(window, window.T)
    assert np.allclose(window, window[::-1, ::-1])
    ones = lambda tiles: np.full(tiles.shape, 0.25, dtype=np.float32)
    out = predict_tiled(
        np.zeros((70, 90), dtype=np.uint8),
        ones,
        tile_size=32,
        overlap=8,
        window=window,
    )
    assert np.allclose(out, 0.25)
