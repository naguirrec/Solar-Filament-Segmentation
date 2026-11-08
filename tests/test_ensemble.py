"""Flip / unflip, averaging and the TTA / ensemble sweep. Pure numpy, no torch."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.ensemble import (
    FLIP_NAMES,
    EvalConfig,
    combine,
    ensemble_predictor,
    flip_image,
    flipped_predictions,
    mean_maps,
    needed_flips,
    resize_probability,
    unflip,
    weighted_mean,
)
from filament.metrics import PqCounts
from filament.sweep import (
    TTA_CLOSE_SIZES,
    TTA_COLUMNS,
    TTA_MIN_AREAS,
    TTA_THRESHOLDS,
    GridPoint,
    config_rows,
    format_config_table,
    sweep_configs,
    sweep_postproc_grid,
    write_config_rows,
)

SIDE = 96


def _asymmetric(rows: int = 6, cols: int = 8) -> np.ndarray:
    """No flip of this array equals the array or another flip of it."""
    return np.arange(rows * cols, dtype=np.float32).reshape(rows, cols) ** 1.5


# --- flips -------------------------------------------------------------------


@pytest.mark.parametrize("name", FLIP_NAMES)
def test_unflip_undoes_flip(name: str) -> None:
    array = _asymmetric()
    assert np.array_equal(unflip(flip_image(array, name), name), array)


def test_each_flip_reverses_the_expected_axes() -> None:
    array = _asymmetric()
    assert np.array_equal(flip_image(array, "none"), array)
    assert np.array_equal(flip_image(array, "h"), array[:, ::-1])
    assert np.array_equal(flip_image(array, "v"), array[::-1, :])
    assert np.array_equal(flip_image(array, "hv"), array[::-1, ::-1])
    # hv is h then v, in either order.
    assert np.array_equal(
        flip_image(flip_image(array, "h"), "v"), flip_image(array, "hv")
    )


def test_flips_are_distinct_and_return_new_contiguous_arrays() -> None:
    array = _asymmetric()
    flipped = [flip_image(array, name) for name in FLIP_NAMES]
    for i in range(len(flipped)):
        assert flipped[i].flags["C_CONTIGUOUS"]
        for j in range(i + 1, len(flipped)):
            assert not np.array_equal(flipped[i], flipped[j])
    copy = flip_image(array, "none")
    copy[0, 0] = -1
    assert array[0, 0] == 0  # the input is never aliased


def test_flip_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="unknown flip"):
        flip_image(_asymmetric(), "d")
    with pytest.raises(ValueError, match="2D"):
        flip_image(np.zeros((2, 3, 4)), "h")


def test_an_equivariant_predictor_gives_identical_flipped_predictions() -> None:
    # A pointwise model commutes with flips, so every un-flipped map is the
    # same as the plain prediction. This is the correctness check for TTA.
    gray = (np.arange(48, dtype=np.uint8).reshape(6, 8) * 5) % 251

    def predict(frame: np.ndarray) -> np.ndarray:
        return frame.astype(np.float32) / 255.0

    maps = flipped_predictions(predict, gray)
    assert list(maps) == list(FLIP_NAMES)
    for name in FLIP_NAMES:
        assert np.array_equal(maps[name], predict(gray))


def test_a_position_dependent_predictor_is_put_back_in_the_frame() -> None:
    # This model marks the top-left pixel of the image it is given. After a
    # flip that pixel is a different corner of the original frame, and the
    # un-flipped map must put the mark back there.
    gray = np.zeros((4, 5), dtype=np.uint8)

    def predict(frame: np.ndarray) -> np.ndarray:
        out = np.zeros(frame.shape, dtype=np.float32)
        out[0, 0] = 1.0
        return out

    maps = flipped_predictions(predict, gray)
    corner = {"none": (0, 0), "h": (0, 4), "v": (3, 0), "hv": (3, 4)}
    for name, (row, col) in corner.items():
        assert maps[name][row, col] == 1.0 and maps[name].sum() == 1.0, name


def test_flipped_predictions_checks_its_inputs() -> None:
    gray = np.zeros((4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="non-empty and distinct"):
        flipped_predictions(lambda f: f, gray, ())
    with pytest.raises(ValueError, match="non-empty and distinct"):
        flipped_predictions(lambda f: f, gray, ("h", "h"))
    with pytest.raises(ValueError, match="expected"):
        flipped_predictions(lambda f: f[:2], gray)


# --- averaging ---------------------------------------------------------------


def test_weighted_mean_normalizes_weights_and_keeps_inputs() -> None:
    a = np.full((3, 3), 0.2, dtype=np.float32)
    b = np.full((3, 3), 0.8, dtype=np.float32)
    a0, b0 = a.copy(), b.copy()
    for weights in ((0.3, 0.7), (3, 7), (30.0, 70.0)):
        out = weighted_mean([a, b], weights)
        assert out.dtype == np.float32
        assert np.allclose(out, 0.3 * 0.2 + 0.7 * 0.8)
    assert np.allclose(weighted_mean([a, b], (0.5, 0.5)), 0.5)
    assert np.array_equal(a, a0) and np.array_equal(b, b0)


def test_mean_of_one_map_is_that_map_and_zero_weight_drops_a_model() -> None:
    a = _asymmetric() / 100
    assert np.allclose(mean_maps([a]), a)
    assert np.allclose(weighted_mean([a, a * 0 + 9.0], (1, 0)), a)


def test_averaging_stays_a_probability() -> None:
    rng = np.random.default_rng(0)
    maps = [rng.random((5, 5)).astype(np.float32) for _ in range(4)]
    out = mean_maps(maps)
    assert out.min() >= min(m.min() for m in maps)
    assert out.max() <= max(m.max() for m in maps)
    assert np.allclose(out, np.mean(maps, axis=0), atol=1e-6)


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ((0.5,), "weights"),
        ((0.0, 0.0), "sum to > 0"),
        ((-1.0, 2.0), "non-negative"),
        ((float("nan"), 1.0), "finite"),
    ],
)
def test_weighted_mean_rejects_bad_weights(weights, match) -> None:
    maps = [np.zeros((2, 2)), np.ones((2, 2))]
    with pytest.raises(ValueError, match=match):
        weighted_mean(maps, weights)


def test_weighted_mean_rejects_empty_and_mismatched_maps() -> None:
    with pytest.raises(ValueError, match="no maps"):
        weighted_mean([], [])
    with pytest.raises(ValueError, match="shapes differ"):
        weighted_mean([np.zeros((2, 2)), np.zeros((2, 3))], (1, 1))


def test_resize_probability_keeps_range_and_shape() -> None:
    rng = np.random.default_rng(0)
    small = rng.random((8, 8)).astype(np.float32)
    big = resize_probability(small, (32, 32))
    assert big.shape == (32, 32) and big.dtype == np.float32
    assert big.min() >= small.min() - 1e-6 and big.max() <= small.max() + 1e-6
    assert np.allclose(resize_probability(np.full((8, 8), 0.3), (32, 32)), 0.3)
    same = resize_probability(small, (8, 8))
    assert np.array_equal(same, small) and same is not small
    with pytest.raises(ValueError, match="2D"):
        resize_probability(np.zeros((2, 2, 2)), (4, 4))


# --- configs -----------------------------------------------------------------


def test_needed_flips_asks_only_for_what_a_config_uses() -> None:
    r34 = EvalConfig("r34", {"r34": 1.0})
    r34_tta = EvalConfig("r34_tta", {"r34": 1.0}, tta=True)
    ens = EvalConfig("ens", {"r18": 0.5, "r34": 0.5})
    assert needed_flips([r34]) == {"r34": ("none",)}
    assert needed_flips([r34, ens]) == {"r34": ("none",), "r18": ("none",)}
    assert needed_flips([r34, r34_tta, ens]) == {
        "r34": FLIP_NAMES,
        "r18": ("none",),
    }


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        EvalConfig("", {"r34": 1.0})
    with pytest.raises(ValueError, match="no models"):
        EvalConfig("x", {})


def test_combine_matches_a_manual_average() -> None:
    rng = np.random.default_rng(2)
    cache = {
        model: {flip: rng.random((4, 4)).astype(np.float32) for flip in FLIP_NAMES}
        for model in ("r18", "r34")
    }
    plain = EvalConfig("r34", {"r34": 1.0})
    assert np.array_equal(combine(cache, plain), cache["r34"]["none"])

    tta = EvalConfig("r34_tta", {"r34": 1.0}, tta=True)
    manual = np.mean([cache["r34"][f] for f in FLIP_NAMES], axis=0)
    assert np.allclose(combine(cache, tta), manual, atol=1e-6)

    ens = EvalConfig("ens", {"r18": 0.3, "r34": 0.7}, tta=True)
    manual = 0.3 * np.mean([cache["r18"][f] for f in FLIP_NAMES], axis=0)
    manual += 0.7 * np.mean([cache["r34"][f] for f in FLIP_NAMES], axis=0)
    assert np.allclose(combine(cache, ens), manual, atol=1e-6)


def test_combine_reports_missing_models_and_flips() -> None:
    cache = {"r34": {"none": np.zeros((2, 2), dtype=np.float32)}}
    with pytest.raises(KeyError, match="needs model 'r18'"):
        combine(cache, EvalConfig("x", {"r18": 1.0}))
    with pytest.raises(KeyError, match="no cached flips"):
        combine(cache, EvalConfig("x", {"r34": 1.0}, tta=True))


def test_ensemble_predictor_runs_each_needed_flip_once() -> None:
    gray = (np.arange(30, dtype=np.uint8).reshape(5, 6) * 7) % 200
    calls = {"a": 0, "b": 0}

    def make(name: str, scale: float):
        def predict(frame: np.ndarray) -> np.ndarray:
            calls[name] += 1
            return frame.astype(np.float32) / 255.0 * scale

        return predict

    predictors = {"a": make("a", 1.0), "b": make("b", 0.5), "unused": make("c", 1.0)}
    config = EvalConfig("ens", {"a": 0.4, "b": 0.6}, tta=True)
    out = ensemble_predictor(predictors, config)(gray)
    assert calls == {"a": 4, "b": 4}
    expected = gray.astype(np.float32) / 255.0 * (0.4 * 1.0 + 0.6 * 0.5)
    assert np.allclose(out, expected, atol=1e-6)  # pointwise models: flips cancel
    with pytest.raises(KeyError, match="needs model"):
        ensemble_predictor({"a": predictors["a"]}, config)


# --- the sweep ---------------------------------------------------------------


def _annotation(number: int, x0: int, y0: int, x1: int, y1: int) -> FilamentAnnotation:
    box = (x0, y0, x1, y0, x1, y1, x0, y1)
    return FilamentAnnotation(
        id=number, image_id=1, category_id=1, spine=None, segmentation=(box,)
    )


def _holdout() -> list[TrainImage]:
    """Two frames with one big and one small filament, each a single view."""
    views = [
        ((_annotation(1, 20, 20, 50, 40), _annotation(2, 60, 60, 66, 66)),),
        ((_annotation(3, 20, 50, 70, 70),),),
    ]
    return [
        TrainImage(
            id=f"f{n}",
            file_name=f"2020010{n}000000Aa.jpeg",
            date_captured=None,
            height=SIDE,
            width=SIDE,
            annotations=view[0],
            views=view,
        )
        for n, view in enumerate(views, start=1)
    ]


def _gray() -> np.ndarray:
    rows, cols = np.ogrid[:SIDE, :SIDE]
    disk = (rows - 48) ** 2 + (cols - 48) ** 2 <= 44**2
    return np.where(disk, 200, 10).astype(np.uint8)


def _truth(image: TrainImage) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=np.float32)
    for annotation in image.annotations:
        mask = np.maximum(mask, annotation.rasterize(SIDE, SIDE).astype(np.float32))
    return mask


def _models(holdout: list[TrainImage]):
    """Two stand-in models that read the true mask of the current frame.

    They ignore the pixels they are given (the disk is symmetric anyway), so
    this tests the bookkeeping of configs, not the flips. Flip correctness
    is tested above with models that do look at the frame.
    """
    rng = np.random.default_rng(1)
    maps = {}
    for number, image in enumerate(holdout):
        noise = rng.random((SIDE, SIDE)).astype(np.float32)
        base = np.roll(_truth(image), 1 + number, axis=1)
        maps[image.file_name] = np.clip(0.8 * base + 0.45 * noise, 0.0, 1.0)
    current: dict[str, str] = {}

    def read(image: TrainImage) -> np.ndarray:
        current["name"] = image.file_name
        return _gray()

    def make(shift: int, scale: float):
        def predict(frame: np.ndarray) -> np.ndarray:
            return np.clip(np.roll(maps[current["name"]], shift, axis=0) * scale, 0, 1)

        return predict

    return read, {"r18": make(3, 0.9), "r34": make(0, 1.0)}


def test_sweep_configs_matches_scoring_each_combined_map_alone() -> None:
    holdout = _holdout()
    read, predictors = _models(holdout)
    configs = [
        EvalConfig("r34", {"r34": 1.0}),
        EvalConfig("ens", {"r18": 0.3, "r34": 0.7}),
    ]
    thresholds, closes, areas = (0.4, 0.5), (0, 5), (60, 400)
    result = sweep_configs(
        holdout, read, predictors, configs, thresholds, closes, areas
    )
    assert set(result.totals) == {"r34", "ens"}
    for config in configs:
        alone = sweep_postproc_grid(
            holdout,
            read,
            lambda gray, c=config: combine(
                {m: {"none": predictors[m](gray)} for m in c.weights}, c
            ),
            thresholds,
            closes,
            areas,
        )
        assert result.totals[config.name] == alone
    # Different probability maps give different scores somewhere on the grid.
    assert result.totals["r34"] != result.totals["ens"]


def test_sweep_configs_predicts_each_model_flip_once_per_image() -> None:
    holdout = _holdout()
    read, predictors = _models(holdout)
    calls = {"r18": 0, "r34": 0}

    def counted(name: str):
        def predict(frame: np.ndarray) -> np.ndarray:
            calls[name] += 1
            return predictors[name](frame)

        return predict

    configs = [
        EvalConfig("r34", {"r34": 1.0}),
        EvalConfig("r34_tta", {"r34": 1.0}, tta=True),
        EvalConfig("ens", {"r18": 0.5, "r34": 0.5}),
        EvalConfig("ens_tta", {"r18": 0.5, "r34": 0.5}, tta=True),
    ]
    sweep_configs(
        holdout,
        read,
        {name: counted(name) for name in predictors},
        configs,
        (0.5,),
        (0,),
        (60,),
    )
    # 2 images: every model runs its 4 flips once, not once per config.
    assert calls == {"r18": 2 * 4, "r34": 2 * 4}


def test_without_tta_each_model_runs_one_pass_per_image() -> None:
    holdout = _holdout()
    read, predictors = _models(holdout)
    calls = {"n": 0}

    def counted(frame: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return predictors["r34"](frame)

    sweep_configs(
        holdout,
        read,
        {"r34": counted},
        [EvalConfig("r34", {"r34": 1.0})],
        (0.5,),
        (0,),
        (60,),
    )
    assert calls["n"] == 2


def test_tta_on_a_flip_equivariant_model_changes_no_score() -> None:
    # A pointwise model gives the same map for every flip once it is put back,
    # so TTA must score exactly like no TTA. A wrong un-flip would not.
    holdout = _holdout()
    read, _ = _models(holdout)

    def equivariant(frame: np.ndarray) -> np.ndarray:
        return frame.astype(np.float32) / 255.0

    configs = [
        EvalConfig("plain", {"m": 1.0}),
        EvalConfig("tta", {"m": 1.0}, tta=True),
    ]
    result = sweep_configs(
        holdout, read, {"m": equivariant}, configs, (0.5,), (0, 5), (60,)
    )
    assert result.totals["plain"] == result.totals["tta"]


def test_sweep_configs_rejects_bad_setups() -> None:
    holdout = _holdout()
    read, predictors = _models(holdout)
    cfg = EvalConfig("a", {"r34": 1.0})
    with pytest.raises(ValueError, match="distinct"):
        sweep_configs(holdout, read, predictors, [cfg, cfg], (0.5,), (0,), (60,))
    with pytest.raises(ValueError, match="distinct"):
        sweep_configs(holdout, read, predictors, [], (0.5,), (0,), (60,))
    with pytest.raises(KeyError, match="no predictor"):
        sweep_configs(holdout, read, {}, [cfg], (0.5,), (0,), (60,))
    with pytest.raises(ValueError, match="holdout is empty"):
        sweep_configs([], read, predictors, [cfg], (0.5,), (0,), (60,))


def test_tta_grid_defaults_use_odd_close_sizes_around_the_requested_point() -> None:
    assert TTA_THRESHOLDS == (0.40, 0.45, 0.50)
    assert TTA_MIN_AREAS == (400,)
    # 10 and 20 were asked for; label_foreground only takes 0 or odd sizes.
    assert TTA_CLOSE_SIZES == (11, 15, 21, 25)


def test_config_rows_are_sorted_and_the_csv_has_the_header(tmp_path: Path) -> None:
    configs = [
        EvalConfig("r34", {"r34": 1.0}),
        EvalConfig("ens_tta", {"r18": 3.0, "r34": 7.0}, tta=True),
    ]
    totals = {
        "r34": {
            GridPoint(0.45, 15, 400): PqCounts(tp=1, fp=1, fn=0, iou_sum=1.0),
            GridPoint(0.5, 15, 400): PqCounts(),
        },
        "ens_tta": {
            GridPoint(0.45, 15, 400): PqCounts(tp=3, fp=0, fn=1, iou_sum=3.0),
        },
    }
    rows = config_rows(configs, totals, holdout_files=142, views=231)
    assert [(r["config"], r["pooled_pq"]) for r in rows] == [
        ("ens_tta", "0.857143"),
        ("r34", "0.666667"),
        ("r34", "nan"),
    ]
    assert rows[0] == {
        "config": "ens_tta",
        "models": "r18:0.3+r34:0.7",
        "tta": "yes",
        "threshold": "0.45",
        "close_size": "15",
        "min_area": "400",
        "pooled_pq": "0.857143",
        "holdout_files": "142",
        "annotator_views": "231",
    }
    assert rows[1]["models"] == "r34:1" and rows[1]["tta"] == "no"

    out = tmp_path / "holdout_tta_ensemble.csv"
    write_config_rows(out, rows)
    with out.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == TTA_COLUMNS
        assert list(reader) == rows
    assert len(format_config_table(rows, limit=2).splitlines()) == 3
