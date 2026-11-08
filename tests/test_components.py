"""Per-component probability scores and the object filter. Pure numpy."""

from __future__ import annotations

import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.baseline import on_disk
from filament.components import (
    ADOPTED_OBJECT_FILTER,
    FEATURE_NAMES,
    NO_OBJECT_FILTER,
    OBJECT_FILTER_VALIDATED_ON,
    ObjectCut,
    component_features,
    component_scores,
    component_view_hits,
    expected_matched_iou,
    filter_components,
    keep_components,
    object_cut_from_config,
    object_filter_mismatches,
    per_file_object_counts,
    selection_counts,
)
from filament.instances import labels_for_min_areas
from filament.metrics import PqCounts, fast_view_counts
from filament.sweep import GridPoint, per_file_counts

SIDE = 128


def test_scores_of_known_components() -> None:
    labels = np.array([[1, 1, 0, 2], [1, 1, 0, 0]], dtype=np.int32)
    probability = np.array([[0.2, 0.4, 0.99, 0.9], [0.6, 0.8, 0.99, 0.99]], np.float32)
    scores = component_scores(labels, probability)
    assert len(scores) == 2
    assert scores.area.tolist() == [4, 1]
    assert scores.mean.tolist() == pytest.approx([0.5, 0.9])
    assert scores.max.tolist() == pytest.approx([0.8, 0.9])
    # Background pixels (0.99) count for nothing.
    assert scores.p90.tolist() == pytest.approx([0.8, 0.9])


def test_p90_matches_a_sorted_lookup_on_many_pixels() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 6, (200, 200), dtype=np.int32)
    probability = rng.random((200, 200), dtype=np.float32)
    scores = component_scores(labels, probability)
    for label in range(1, 6):
        pixels = np.sort(probability[labels == label])
        rank = int(np.ceil(0.9 * pixels.size)) - 1
        assert scores.p90[label - 1] == pixels[rank]
        assert scores.max[label - 1] == pixels[-1]
        assert scores.mean[label - 1] == pytest.approx(pixels.mean(), abs=1e-5)
        assert scores.area[label - 1] == pixels.size


def test_an_id_without_pixels_scores_zero_and_no_components_is_fine() -> None:
    labels = np.array([[0, 3]], dtype=np.int32)
    scores = component_scores(labels, np.array([[0.5, 0.7]], np.float32))
    assert scores.area.tolist() == [0, 0, 1]
    assert scores.mean.tolist() == pytest.approx([0.0, 0.0, 0.7])
    empty = component_scores(np.zeros((3, 3), np.int32), np.zeros((3, 3), np.float32))
    assert len(empty) == 0


def test_scores_refuse_another_shape_and_statistic() -> None:
    with pytest.raises(ValueError, match="differ"):
        component_scores(np.zeros((2, 2), np.int32), np.zeros((3, 3)))
    scores = component_scores(np.ones((2, 2), np.int32), np.ones((2, 2)))
    with pytest.raises(ValueError, match="unknown statistic"):
        scores.statistic("median")


def test_keep_components_renumbers_and_leaves_pixels_alone() -> None:
    labels = np.array([[1, 0, 2], [3, 3, 2]], dtype=np.int32)
    kept = keep_components(labels, np.array([False, True, True]))
    assert kept.tolist() == [[0, 0, 1], [2, 2, 1]]
    assert keep_components(labels, np.array([True] * 3)).tolist() == labels.tolist()
    assert not keep_components(labels, np.array([False] * 3)).any()
    with pytest.raises(ValueError, match="flags"):
        keep_components(labels, np.array([True]))


@pytest.mark.parametrize(
    "bad", [("median", 0.5), ("mean", -0.1), ("mean", 1.0), ("max", 2.0)]
)
def test_object_cut_refuses_bad_settings(bad: tuple[str, float]) -> None:
    with pytest.raises(ValueError):
        ObjectCut(*bad)


def test_an_object_cut_prints_as_a_short_rule() -> None:
    assert str(ObjectCut("mean", 0.7)) == "mean>=0.7"


def _gray() -> np.ndarray:
    rows, cols = np.ogrid[:SIDE, :SIDE]
    disk = (rows - 64) ** 2 + (cols - 64) ** 2 <= 58**2
    return np.where(disk, 200, 5).astype(np.uint8)


def _box(x0: int, y0: int, x1: int, y1: int) -> tuple[float, ...]:
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def _annotation(number: int, box: tuple[float, ...]) -> FilamentAnnotation:
    return FilamentAnnotation(
        id=number, image_id=1, category_id=1, spine=None, segmentation=(box,)
    )


def _frame() -> tuple[TrainImage, np.ndarray]:
    """A real filament at 0.9, a false blob at 0.55. One view."""
    real = _annotation(1, _box(40, 40, 80, 60))
    image = TrainImage(
        id=1,
        file_name="a.jpeg",
        date_captured=None,
        height=SIDE,
        width=SIDE,
        annotations=(real,),
    )
    probability = np.zeros((SIDE, SIDE), dtype=np.float32)
    probability[np.asarray(real.rasterize(SIDE, SIDE)) != 0] = 0.9
    probability[80:95, 50:75] = 0.55
    return image, probability


BASE = GridPoint(0.5, 0, 20)


def _counts(cuts: list[ObjectCut]):
    image, probability = _frame()
    return per_file_object_counts(
        [image], lambda _i: _gray(), lambda _i: probability, BASE, cuts
    )


def test_no_filter_reproduces_the_plain_sweep() -> None:
    image, probability = _frame()
    plain = per_file_counts([image], lambda _i: _gray(), lambda _i: probability, [BASE])
    filtered = _counts([NO_OBJECT_FILTER])
    assert filtered.counts[NO_OBJECT_FILTER] == plain.counts[BASE]
    assert filtered.file_names == plain.file_names
    assert filtered.views == plain.views


def test_the_filter_drops_the_weak_blob_and_raises_pq() -> None:
    cut = ObjectCut("mean", 0.7)
    result = _counts([NO_OBJECT_FILTER, cut])
    before = result.counts[NO_OBJECT_FILTER][0]
    after = result.counts[cut][0]
    assert (before.tp, before.fp, before.fn) == (1, 1, 0)
    assert (after.tp, after.fp, after.fn) == (1, 0, 0)
    assert after.pq > before.pq
    assert after.iou_sum == pytest.approx(before.iou_sum)


def test_a_cut_above_everything_drops_the_real_filament_too() -> None:
    cut = ObjectCut("max", 0.95)
    counts = _counts([cut]).counts[cut][0]
    assert (counts.tp, counts.fp, counts.fn) == (0, 0, 1)


@pytest.mark.parametrize("statistic", ["mean", "max", "p90"])
def test_every_statistic_separates_these_two_components(statistic: str) -> None:
    cut = ObjectCut(statistic, 0.7)
    counts = _counts([cut]).counts[cut][0]
    assert (counts.tp, counts.fp) == (1, 0)


def test_per_file_object_counts_refuse_bad_input() -> None:
    image, probability = _frame()
    read, load = (lambda _i: _gray()), (lambda _i: probability)
    with pytest.raises(ValueError, match="holdout is empty"):
        per_file_object_counts([], read, load, BASE, [NO_OBJECT_FILTER])
    with pytest.raises(ValueError, match="distinct"):
        per_file_object_counts(
            [image], read, load, BASE, [NO_OBJECT_FILTER, NO_OBJECT_FILTER]
        )
    with pytest.raises(ValueError, match="does not match"):
        per_file_object_counts(
            [image], read, lambda _i: np.zeros((2, 2)), BASE, [NO_OBJECT_FILTER]
        )


# --- features ----------------------------------------------------------------------


def _feature(row: np.ndarray, name: str) -> float:
    return float(row[FEATURE_NAMES.index(name)])


def _feature_frame():
    """A long thin dark filament and a compact pale blob on a disk."""
    gray = _gray().copy()
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    labels[60:64, 30:100] = 1
    labels[80:92, 50:62] = 2
    gray[labels == 1] = 60
    gray[labels == 2] = 180
    probability = np.zeros((SIDE, SIDE), dtype=np.float32)
    probability[labels == 1] = 0.9
    probability[labels == 2] = 0.6
    spread = np.zeros((SIDE, SIDE), dtype=np.float32)
    spread[labels == 2] = 0.2
    return labels, probability, spread, gray


def test_features_describe_shape_certainty_and_contrast() -> None:
    labels, probability, spread, gray = _feature_frame()
    features = component_features(labels, probability, spread, gray, on_disk(gray))
    assert features.shape == (2, len(FEATURE_NAMES))
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))
    thin, blob = features
    assert _feature(thin, "log_area") == pytest.approx(np.log(4 * 70))
    assert _feature(thin, "elongation") > _feature(blob, "elongation") + 2
    assert _feature(thin, "skeleton_length") > 50
    assert _feature(thin, "mean_width") < _feature(blob, "mean_width") + 1
    assert _feature(thin, "probability_mean") == pytest.approx(0.9)
    assert _feature(blob, "probability_max") == pytest.approx(0.6)
    assert _feature(thin, "flip_std_mean") == pytest.approx(0.0)
    assert _feature(blob, "flip_std_mean") == pytest.approx(0.2)
    assert _feature(thin, "solidity") == pytest.approx(1.0, abs=0.05)
    # The dark filament stands out from the disk, the pale blob much less.
    assert _feature(thin, "contrast") == pytest.approx((200 - 60) / 200, abs=0.01)
    assert _feature(blob, "contrast") == pytest.approx((200 - 180) / 200, abs=0.01)
    assert _feature(thin, "contrast") > _feature(blob, "contrast")


def test_position_features_know_the_limb() -> None:
    gray = _gray()
    disk = on_disk(gray)
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    labels[62:66, 60:68] = 1  # the center of the disk
    rows, cols = np.nonzero(disk)
    edge_col = int(cols.max())
    edge_row = int(rows[cols == edge_col].mean())
    labels[edge_row - 1 : edge_row + 2, edge_col - 6 : edge_col + 1] = 2
    zeros = np.zeros((SIDE, SIDE), dtype=np.float32)
    features = component_features(labels, zeros, zeros, gray, disk)
    center, edge = features
    assert _feature(center, "radial_distance") < 0.1
    assert _feature(center, "touches_limb") == 0.0
    assert _feature(edge, "radial_distance") > 0.9
    assert _feature(edge, "touches_limb") == 1.0


def test_features_survive_degenerate_shapes() -> None:
    gray = _gray()
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    labels[64, 30:100] = 1  # one pixel wide
    labels[90, 64] = 2  # one pixel
    labels[40:43, 40:43] = 3  # fragments under one id, as a closing can give
    labels[40:43, 60:63] = 3
    zeros = np.zeros((SIDE, SIDE), dtype=np.float32)
    features = component_features(labels, zeros, zeros, gray, on_disk(gray))
    assert features.shape == (3, len(FEATURE_NAMES))
    assert np.all(np.isfinite(features))


def test_no_components_give_an_empty_table() -> None:
    gray = _gray()
    zeros = np.zeros((SIDE, SIDE), dtype=np.float32)
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    features = component_features(labels, zeros, zeros, gray, on_disk(gray))
    assert features.shape == (0, len(FEATURE_NAMES))


def test_features_refuse_mismatched_shapes() -> None:
    gray = _gray()
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    with pytest.raises(ValueError, match="share a shape"):
        component_features(labels, np.zeros((2, 2)), labels, gray, on_disk(gray))


# --- hits, and the metric without the pixels ----------------------------------------


def _random_scene(seed: int) -> tuple[np.ndarray, list[list[np.ndarray]]]:
    """Disjoint components, and views whose masks overlap them to varying degrees."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    boxes = []
    for index in range(8):
        top = 4 + (index % 4) * 30
        left = 4 + (index // 4) * 60
        height, width = int(rng.integers(6, 24)), int(rng.integers(10, 50))
        labels[top : top + height, left : left + width] = index + 1
        boxes.append((top, left, height, width))
    views = []
    for _ in range(3):
        masks = []
        for top, left, height, width in boxes:
            if rng.random() < 0.3:
                continue
            mask = np.zeros((SIDE, SIDE), dtype=np.uint8)
            shift = int(rng.integers(0, max(height, 2)))
            mask[top + shift : top + height + shift, left : left + width] = 1
            masks.append(mask)
        extra = np.zeros((SIDE, SIDE), dtype=np.uint8)
        extra[110:120, 100:120] = 1  # a filament nobody predicted
        masks.append(extra)
        views.append(masks)
    return labels, views


@pytest.mark.parametrize("seed", range(5))
def test_selection_counts_match_the_metric_on_the_pixels(seed: int) -> None:
    labels, views = _random_scene(seed)
    hits = [component_view_hits(labels, masks) for masks in views]
    rng = np.random.default_rng(100 + seed)
    subsets = [np.ones(8, bool), np.zeros(8, bool), rng.random(8) < 0.5]
    assert sum(view.candidate.size for view in hits) > 0  # not vacuous
    for emit in subsets:
        kept = keep_components(labels, emit)
        expected = sum((fast_view_counts(kept, masks) for masks in views), PqCounts())
        found = selection_counts(hits, emit)
        assert (found.tp, found.fp, found.fn) == (expected.tp, expected.fp, expected.fn)
        assert found.iou_sum == pytest.approx(expected.iou_sum)


def test_expected_matched_iou_is_per_view_and_adds_up_to_the_metric() -> None:
    labels, views = _random_scene(1)
    hits = [component_view_hits(labels, masks) for masks in views]
    target = expected_matched_iou(hits, 8)
    total = sum((fast_view_counts(labels, masks) for masks in views), PqCounts())
    assert target.sum() * len(views) == pytest.approx(total.iou_sum)
    assert total.tp > 0


def test_expected_matched_iou_counts_a_view_that_missed_it() -> None:
    labels = np.zeros((SIDE, SIDE), dtype=np.int32)
    labels[10:20, 10:40] = 1
    mask = (labels == 1).astype(np.uint8)
    hits = [component_view_hits(labels, [mask]), component_view_hits(labels, [])]
    assert expected_matched_iou(hits, 1)[0] == pytest.approx(0.5)
    counts = selection_counts(hits, np.array([True]))
    assert (counts.tp, counts.fp, counts.fn) == (1, 1, 0)


# --- the filter on the inference path -------------------------------------------------


def _filter_scene() -> tuple[np.ndarray, np.ndarray]:
    """Two components from a thresholded map: a confident one and a weak one."""
    probability = np.zeros((SIDE, SIDE), dtype=np.float32)
    probability[40:60, 40:80] = 0.9
    probability[80:95, 50:75] = 0.55
    semantic = (probability >= 0.5).astype(np.uint8)
    labels = labels_for_min_areas(semantic, _gray(), [20], close_size=0)[20]
    return labels, probability


def test_no_filter_returns_the_labels_untouched() -> None:
    labels, probability = _filter_scene()
    assert filter_components(labels, probability, None) is labels


def test_the_filter_drops_the_weak_component_and_keeps_the_other_pixels() -> None:
    labels, probability = _filter_scene()
    assert int(labels.max()) == 2
    kept = filter_components(labels, probability, ObjectCut("mean", 0.7))
    assert int(kept.max()) == 1
    assert np.array_equal(kept != 0, probability >= 0.7)
    # A cut below every component keeps them all, renumbered as before.
    assert np.array_equal(
        filter_components(labels, probability, ObjectCut("mean", 0.5)), labels
    )


def test_the_filter_is_the_one_the_sweep_scored() -> None:
    labels, probability = _filter_scene()
    for statistic in ("mean", "max", "p90"):
        cut = ObjectCut(statistic, 0.7)
        scores = component_scores(labels, probability).statistic(statistic)
        assert np.array_equal(
            filter_components(labels, probability, cut),
            keep_components(labels, scores >= 0.7),
        )


def test_filtered_components_encode_to_disjoint_submission_masks() -> None:
    labels, probability = _filter_scene()
    kept = filter_components(labels, probability, ObjectCut("max", 0.6))
    ids = [i for i in np.unique(kept) if i != 0]
    assert ids == list(range(1, len(ids) + 1))


def test_config_reader() -> None:
    assert object_cut_from_config(None) is None
    assert object_cut_from_config({"stat": "p90", "min": 0.8}) == ObjectCut("p90", 0.8)
    with pytest.raises(ValueError, match="exactly the keys"):
        object_cut_from_config({"stat": "mean", "minimum": 0.8})
    with pytest.raises(ValueError, match="exactly the keys"):
        object_cut_from_config({"stat": "mean"})
    with pytest.raises(ValueError, match="unknown statistic"):
        object_cut_from_config({"stat": "median", "min": 0.8})


def test_the_adopted_filter_is_valid_and_its_base_is_recorded() -> None:
    # The cut the B1 cross-fit chose in 7 of 10 folds (docs/RESULTS.md).
    assert ADOPTED_OBJECT_FILTER == {"stat": "mean", "min": 0.8}
    assert object_cut_from_config(ADOPTED_OBJECT_FILTER) == ObjectCut("mean", 0.8)
    assert OBJECT_FILTER_VALIDATED_ON == {
        "threshold": 0.45,
        "close_size": 11,
        "min_area": 400,
        "tta": True,
    }


def test_mismatches_name_every_setting_that_differs() -> None:
    assert (
        object_filter_mismatches(threshold=0.45, close_size=11, min_area=400, tta=True)
        == []
    )
    found = object_filter_mismatches(
        threshold=0.5, close_size=0, min_area=400, tta=False
    )
    assert [line.split("=")[0] for line in found] == ["threshold", "close_size", "tta"]


def test_mismatches_know_the_models_the_filter_was_validated_on() -> None:
    base = {"threshold": 0.45, "close_size": 11, "min_area": 400, "tta": True}
    assert object_filter_mismatches(**base, models={"r34": 1.0}) == []
    assert object_filter_mismatches(**base, models={"r34": 3.0}) == []  # normalized
    assert object_filter_mismatches(**base) == []  # models not given: not checked
    alone = object_filter_mismatches(**base, models={"r34_v2_agree": 1.0})
    assert len(alone) == 1 and alone[0].startswith("models=")
    assert "validated with {'r34': 1.0}" in alone[0]
    ensemble = object_filter_mismatches(
        **base, models={"r34": 0.7, "r34_v2_agree": 0.3}
    )
    assert len(ensemble) == 1 and "r34_v2_agree" in ensemble[0]
    r18 = object_filter_mismatches(**base, models={"r18": 0.5, "r34": 0.5})
    assert len(r18) == 1


@pytest.mark.parametrize(
    "models", [{"r18": 0.0, "r34": 0.0}, {}, {"r34": float("nan")}]
)
def test_mismatches_refuse_weights_without_a_positive_sum(models) -> None:
    base = {"threshold": 0.45, "close_size": 11, "min_area": 400, "tta": True}
    with pytest.raises(ValueError, match="positive sum"):
        object_filter_mismatches(**base, models=models)
