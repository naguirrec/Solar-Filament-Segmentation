"""Full-resolution holdout scores on tiny synthetic images.

No MAGFiLO frames, no network, and no torch. A 512 semantic mask is
upsampled onto a larger synthetic frame and scored there.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from filament.annotations import rasterize_polygons
from filament.baseline import segment_filaments
from filament.baseline_holdout import score_holdout
from filament.instances import labels_from_semantic
from filament.metrics import fast_view_counts

# Rows 10:60, columns 10:56 on an 80×80 frame. Area 2300, above the
# baseline default min_area of 400. pycocotools treats the far edges
# as exclusive, so this polygon is that rectangle.
_RECTANGLE = ((10.0, 10.0, 56.0, 10.0, 56.0, 60.0, 10.0, 60.0),)
_RECT_ROWS = slice(10, 60)
_RECT_COLS = slice(10, 56)


class _UnreadCategory:
    """Rasterizes stored polygons. Reading ``category_id`` fails the test."""

    def __init__(self, segmentation: tuple[tuple[float, ...], ...]) -> None:
        self._segmentation = segmentation

    def rasterize(self, height: int, width: int) -> np.ndarray:
        return rasterize_polygons(self._segmentation, height, width)

    @property
    def category_id(self) -> int:
        raise AssertionError("category_id must stay unread")


class _StoredMask:
    """Returns one prepared mask. Reading ``category_id`` fails the test."""

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def rasterize(self, height: int, width: int) -> np.ndarray:
        if (height, width) != self._mask.shape:
            raise ValueError("rasterize size does not match the stored mask")
        return self._mask

    @property
    def category_id(self) -> int:
        raise AssertionError("category_id must stay unread")


class _Image:
    def __init__(
        self,
        file_name: str,
        height: int,
        width: int,
        annotations: tuple[object, ...],
        views: tuple[tuple[object, ...], ...] = (),
    ) -> None:
        self.file_name = file_name
        self.height = height
        self.width = width
        self.annotations = annotations
        self.views = views


def _bright_rectangle() -> np.ndarray:
    image = np.full((80, 80), 220, dtype=np.uint8)
    image[_RECT_ROWS, _RECT_COLS] = 30
    return image


def _rectangle_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[_RECT_ROWS, _RECT_COLS] = 1
    return mask


def _nearest_upsample(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Same center rule the scorer documents. Independent of its private helper."""

    def index(length: int, out_length: int) -> np.ndarray:
        scale = length / out_length
        picked = np.floor((np.arange(out_length, dtype=np.float64) + 0.5) * scale)
        return np.clip(picked.astype(np.int64), 0, length - 1)

    rows = index(mask.shape[0], out_h)
    cols = index(mask.shape[1], out_w)
    return np.ascontiguousarray(mask[np.ix_(rows, cols)])


def test_perfect_dark_filament_scores_one(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real = segment_filaments

    def spy(image: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        calls.append((args, kwargs))
        return real(image)

    monkeypatch.setattr("filament.baseline_holdout.segment_filaments", spy)
    gray = _bright_rectangle()
    image = _Image("filament.jpeg", 80, 80, (_UnreadCategory(_RECTANGLE),))
    scores = score_holdout(
        (image,),
        {"filament.jpeg": gray},
        {"filament.jpeg": _rectangle_mask(gray.shape)},
    )

    assert scores.baseline_pq == 1.0
    assert scores.unet_pq == 1.0
    assert scores.image_count == 1
    assert calls == [((), {})]


def test_512_mask_is_scored_at_the_large_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = np.zeros((512, 512), dtype=np.uint8)
    semantic[200:232, 180:220] = 1
    large = (520, 520)
    assert semantic.shape != large
    expected = _nearest_upsample(semantic, large[0], large[1])
    assert expected.shape == large
    assert int(expected.sum()) > 0

    seen: dict[str, object] = {}
    real_labels = labels_from_semantic

    def spy_labels(
        mask: np.ndarray,
        gray: np.ndarray,
        *,
        min_area: int | None = None,
        close_size: int | None = None,
    ) -> np.ndarray:
        seen["shape"] = mask.shape
        seen["gray_shape"] = gray.shape
        seen["min_area"] = min_area
        seen["close_size"] = close_size
        return real_labels(mask, gray, min_area=min_area, close_size=close_size)

    recorded: list[tuple[np.ndarray, list[np.ndarray]]] = []
    real_counts = fast_view_counts

    def spy_counts(pred_labels: np.ndarray, gt_masks: object) -> object:
        actual = [np.asarray(mask) for mask in gt_masks]  # type: ignore[union-attr]
        recorded.append((np.asarray(pred_labels), actual))
        return real_counts(pred_labels, actual)

    monkeypatch.setattr("filament.baseline_holdout.labels_from_semantic", spy_labels)
    monkeypatch.setattr("filament.baseline_holdout.fast_view_counts", spy_counts)

    gray = np.full(large, 210, dtype=np.uint8)
    image = _Image("wide.jpeg", large[0], large[1], (_StoredMask(expected),))
    scores = score_holdout(
        (image,),
        lambda file_name: gray,
        lambda file_name: semantic,
    )

    assert seen["shape"] == large
    assert seen["gray_shape"] == large
    assert seen["shape"] != (512, 512)
    assert seen["min_area"] is None
    assert seen["close_size"] is None
    assert scores.baseline_pq == 0.0
    assert scores.unet_pq == 1.0
    assert scores.image_count == 1
    assert len(recorded) == 2
    baseline_labels, baseline_truth = recorded[0]
    unet_labels, unet_truth = recorded[1]
    assert int(baseline_labels.max()) == 0
    assert unet_labels.shape == large
    assert unet_labels.shape != (512, 512)
    assert np.array_equal(unet_labels == 1, expected)
    assert len(baseline_truth) == 1
    assert len(unet_truth) == 1
    assert baseline_truth[0].shape == large
    assert np.array_equal(baseline_truth[0], unet_truth[0])
    assert np.array_equal(unet_truth[0], expected)


def test_both_paths_see_the_same_ground_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = ((2.0, 2.0, 12.0, 2.0, 12.0, 14.0, 2.0, 14.0),)
    right = ((20.0, 18.0, 36.0, 18.0, 36.0, 30.0, 20.0, 30.0),)
    recorded: list[list[np.ndarray]] = []
    real = fast_view_counts

    def spy(pred_labels: np.ndarray, ground_truth: object) -> object:
        recorded.append([np.asarray(mask) for mask in ground_truth])  # type: ignore[union-attr]
        return real(pred_labels, recorded[-1])

    monkeypatch.setattr("filament.baseline_holdout.fast_view_counts", spy)
    gray = _bright_rectangle()
    image = _Image(
        "two.jpeg",
        80,
        80,
        (_UnreadCategory(left), _UnreadCategory(right)),
    )
    score_holdout(
        (image,),
        {"two.jpeg": gray},
        {"two.jpeg": np.zeros((8, 8), dtype=np.uint8)},
    )

    assert len(recorded) == 2
    assert len(recorded[0]) == 2
    assert len(recorded[1]) == 2
    for baseline_mask, unet_mask in zip(recorded[0], recorded[1], strict=True):
        assert baseline_mask.shape == (80, 80)
        assert np.array_equal(baseline_mask, unet_mask)
    assert np.array_equal(recorded[0][0], rasterize_polygons(left, 80, 80))
    assert np.array_equal(recorded[0][1], rasterize_polygons(right, 80, 80))


def test_empty_prediction_against_empty_image_is_nan() -> None:
    gray = np.full((24, 24), 180, dtype=np.uint8)

    def load_gray(file_name: str) -> np.ndarray:
        assert file_name == "empty.jpeg"
        return gray

    image = _Image("empty.jpeg", 24, 24, ())
    scores = score_holdout(
        (image,),
        load_gray,
        lambda file_name: np.zeros((6, 6), dtype=np.uint8),
    )
    assert math.isnan(scores.baseline_pq)
    assert math.isnan(scores.unet_pq)
    assert scores.image_count == 1


def test_counts_are_pooled_across_images() -> None:
    filament = _Image("filament.jpeg", 80, 80, (_UnreadCategory(_RECTANGLE),))
    empty = _Image("empty.jpeg", 24, 24, ())
    grays = {
        "filament.jpeg": _bright_rectangle(),
        "empty.jpeg": np.full((24, 24), 180, dtype=np.uint8),
    }
    # The filament image gets an empty semantic mask, so the U-Net misses it.
    semantics = {
        "filament.jpeg": np.zeros((8, 8), dtype=np.uint8),
        "empty.jpeg": np.zeros((4, 4), dtype=np.uint8),
    }
    scores = score_holdout((filament, empty), grays, semantics)
    # The empty image adds nothing. The filament is one TP for the
    # baseline and one FN for the U-Net. A mean of 1 and NaN is not this.
    assert scores.baseline_pq == 1.0
    assert scores.unet_pq == 0.0
    assert scores.image_count == 2


def test_blank_annotation_is_not_an_instance() -> None:
    gray = _bright_rectangle()
    blank = np.zeros(gray.shape, dtype=np.uint8)
    image = _Image(
        "filament.jpeg",
        80,
        80,
        (_StoredMask(blank), _UnreadCategory(_RECTANGLE)),
    )
    scores = score_holdout(
        (image,),
        {"filament.jpeg": gray},
        {"filament.jpeg": _rectangle_mask(gray.shape)},
    )
    assert scores.baseline_pq == 1.0
    assert scores.unet_pq == 1.0
    assert scores.image_count == 1


def test_duplicate_file_names_are_pooled_not_averaged() -> None:
    gray = _bright_rectangle()
    matched = _Image("same.jpeg", 80, 80, (_UnreadCategory(_RECTANGLE),))
    unmatched = _Image("same.jpeg", 80, 80, ())
    scores = score_holdout(
        (matched, unmatched),
        {"same.jpeg": gray},
        {"same.jpeg": _rectangle_mask(gray.shape)},
    )
    # One exact match (TP, IoU 1) and one empty view with that same
    # prediction (FP). Pooled PQ is 1 / 1.5. The mean of 1 and 0 is 0.5.
    assert scores.image_count == 2
    assert scores.baseline_pq == pytest.approx(2 / 3)
    assert scores.unet_pq == pytest.approx(2 / 3)


def test_empty_annotator_view_is_a_false_positive() -> None:
    # The stand-in annotation list has the filament. A second annotator
    # drew nothing. Keeping only the first mask would score 1.
    gray = _bright_rectangle()
    kept = _UnreadCategory(_RECTANGLE)
    image = _Image(
        "filament.jpeg",
        80,
        80,
        (kept,),
        views=((kept,), ()),
    )
    scores = score_holdout(
        (image,),
        {"filament.jpeg": gray},
        {"filament.jpeg": _rectangle_mask(gray.shape)},
    )
    assert scores.baseline_pq == pytest.approx(2 / 3)
    assert scores.unet_pq == pytest.approx(2 / 3)
    assert scores.image_count == 1


def test_empty_holdout_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        score_holdout((), {}, {})


def test_source_does_not_store_a_leaderboard_figure() -> None:
    source = Path(score_holdout.__code__.co_filename).read_text(encoding="utf-8")
    assert "torch" not in source
    assert "split_training_images" in source
    assert "0.06" not in source
    assert "2048" in source
    assert "category_id" in source
