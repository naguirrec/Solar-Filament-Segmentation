"""Probability averaging across crop models, and the holdout ensemble scoring."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage
from filament.checkpoint import CROP_CHECKPOINTS
from filament.components import ObjectCut
from filament.crop_ensemble import (
    ENSEMBLE_COLUMNS,
    ENSEMBLE_POINT,
    HOLDOUT_ENSEMBLE_CONFIGS,
    describe_models,
    ensemble_rows,
    ensemble_weights,
    members_to_load,
    per_file_ensemble_counts,
    require_pinned,
    submission_model_weights,
    unpinned_members,
    write_ensemble_rows,
)
from filament.ensemble import (
    FLIP_NAMES,
    EvalConfig,
    combine,
    ensemble_predictor,
    flipped_predictions,
    mean_maps,
)
from filament.sweep import GridPoint

SIDE = 128


# --- weights -----------------------------------------------------------------------


def test_no_ensemble_is_the_model_alone() -> None:
    assert ensemble_weights(None, "r34") == {"r34": 1.0}


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        ((("r34", 0.7), ("r34_v2_agree", 0.3)), {"r34": 0.7, "r34_v2_agree": 0.3}),
        ((("r34", 1), ("r34_v2_agree", 1)), {"r34": 0.5, "r34_v2_agree": 0.5}),
        ((("r34_v2_agree", 3.0), ("r34", 1.0)), {"r34_v2_agree": 0.75, "r34": 0.25}),
        ((("r34", 2.0),), {"r34": 1.0}),
    ],
)
def test_weights_are_normalized_to_sum_to_one(members, expected) -> None:
    weights = ensemble_weights(members, "r34")
    assert weights == pytest.approx(expected)
    assert list(weights) == [name for name, _ in members]  # order kept
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "members",
    [
        (),
        (("r34", 0.5), ("r34", 0.5)),
        (("r34", 0.5), ("resnet34", 0.5)),
        (("r34", 0.0), ("r34_v2_agree", 1.0)),
        (("r34", -1.0), ("r34_v2_agree", 1.0)),
        (("r34", float("nan")), ("r34_v2_agree", 1.0)),
        (("r34", float("inf")),),
    ],
)
def test_bad_ensembles_are_refused(members) -> None:
    with pytest.raises(ValueError):
        ensemble_weights(members, "r34")


def test_the_two_ensemble_members_are_pinned_and_others_are_reported() -> None:
    assert unpinned_members({"r34": 0.5, "r34_v2_agree": 0.5}) == []
    unpinned = [n for n, e in CROP_CHECKPOINTS.items() if e.tensor_sha256 is None]
    assert unpinned  # r50, effb3 and r34_v2 are still unpinned
    assert unpinned_members({"r34": 0.5, unpinned[0]: 0.5}) == [unpinned[0]]


# --- the inference notebook's model weights -----------------------------------------


def _unpinned_name() -> str:
    return next(n for n, e in CROP_CHECKPOINTS.items() if e.tensor_sha256 is None)


def test_without_an_ensemble_the_weights_are_what_the_notebook_always_used() -> None:
    assert submission_model_weights("r34", None) == {"r34": 1.0}
    assert submission_model_weights("r34", None, allow_unverified=True) == {"r34": 1.0}
    # The ResNet-18 ensemble keeps its weights as given, as before: not normalized
    # here (filament.ensemble.combine normalizes them when it averages).
    assert submission_model_weights(
        "r34", None, use_r18_ensemble=True, r18_weights=(3, 7)
    ) == {"r18": 3, "r34": 7}
    assert submission_model_weights(
        "r34", None, use_r18_ensemble=True, r18_weights=(0.0, 1.0)
    ) == {"r18": 0.0, "r34": 1.0}


@pytest.mark.parametrize(
    "bad",
    [
        (0.0, 0.0),
        (-1.0, 2.0),
        (2.0, -1.0),
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
    ],
)
def test_bad_r18_weights_stop_before_any_loading(bad) -> None:
    with pytest.raises(ValueError, match="ENSEMBLE_WEIGHTS"):
        submission_model_weights("r34", None, use_r18_ensemble=True, r18_weights=bad)


def test_an_unpinned_model_is_refused_inside_an_ensemble() -> None:
    # The pin check covers MODEL too, not only the members loaded besides it.
    unpinned = _unpinned_name()
    with pytest.raises(RuntimeError, match=unpinned):
        submission_model_weights(unpinned, ((unpinned, 1), ("r34", 1)))
    weights = submission_model_weights(
        unpinned, ((unpinned, 1), ("r34", 1)), allow_unverified=True
    )
    assert members_to_load(weights, unpinned) == ["r34"]


def test_a_crop_ensemble_is_normalized_and_checked() -> None:
    weights = submission_model_weights("r34", (("r34", 7), ("r34_v2_agree", 3)))
    assert weights == pytest.approx({"r34": 0.7, "r34_v2_agree": 0.3})
    assert members_to_load(weights, "r34") == ["r34_v2_agree"]
    assert members_to_load({"r18": 0.5, "r34": 0.5}, "r34") == []


def test_a_bad_crop_ensemble_stops_before_any_loading() -> None:
    with pytest.raises(ValueError, match="not both"):
        submission_model_weights(
            "r34", (("r34", 1), ("r34_v2_agree", 1)), use_r18_ensemble=True
        )
    with pytest.raises(ValueError, match="has to be a member"):
        submission_model_weights("r34", (("r34_v2_agree", 1.0),))
    unpinned = _unpinned_name()
    with pytest.raises(RuntimeError, match="no pinned tensor_sha256"):
        submission_model_weights("r34", (("r34", 1), (unpinned, 1)))
    allowed = submission_model_weights(
        "r34", (("r34", 1), (unpinned, 1)), allow_unverified=True
    )
    assert allowed == pytest.approx({"r34": 0.5, unpinned: 0.5})


def test_require_pinned() -> None:
    require_pinned(["r34", "r34_v2_agree"])
    with pytest.raises(RuntimeError, match=_unpinned_name()):
        require_pinned(["r34", _unpinned_name()])
    require_pinned([_unpinned_name()], allow_unverified=True)


# --- averaging -----------------------------------------------------------------------


def _flip_dependent(scale: float, shift: float):
    """A predictor whose output depends on position, so the flips disagree."""

    def predict(gray: np.ndarray) -> np.ndarray:
        rows = np.linspace(0.0, 1.0, gray.shape[0], dtype=np.float32)[:, None]
        return np.clip(scale * (rows + gray / 255.0) / 2.0 + shift, 0.0, 1.0)

    return predict


def _gray(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (32, 48), dtype=np.uint8)


@pytest.mark.parametrize("tta", [False, True])
def test_single_model_path_is_bit_for_bit_the_model_alone(tta: bool) -> None:
    gray = _gray()
    predict = _flip_dependent(1.0, 0.0)
    flips = FLIP_NAMES if tta else ("none",)
    alone = mean_maps(list(flipped_predictions(predict, gray, flips).values()))
    config = EvalConfig("submission", ensemble_weights(None, "r34"), tta=tta)
    combined = ensemble_predictor({"r34": predict}, config)(gray)
    assert combined.dtype == np.float32
    assert np.array_equal(combined, alone)
    if not tta:
        assert np.array_equal(combined, predict(gray).astype(np.float32))


def test_two_models_average_after_each_ones_own_flips() -> None:
    gray = _gray(1)
    a, b = _flip_dependent(1.0, 0.0), _flip_dependent(0.5, 0.2)
    weights = ensemble_weights((("r34", 0.7), ("r34_v2_agree", 0.3)), "r34")
    config = EvalConfig("ens", weights, tta=True)
    cache = {
        "r34": flipped_predictions(a, gray),
        "r34_v2_agree": flipped_predictions(b, gray),
    }
    expected = 0.7 * mean_maps(list(cache["r34"].values())) + 0.3 * mean_maps(
        list(cache["r34_v2_agree"].values())
    )
    combined = combine(cache, config)
    assert combined.shape == gray.shape and combined.dtype == np.float32
    assert np.allclose(combined, expected, atol=1e-6)
    assert np.allclose(
        ensemble_predictor({"r34": a, "r34_v2_agree": b}, config)(gray), expected
    )
    assert 0.0 <= combined.min() and combined.max() <= 1.0


def test_unnormalized_weights_give_the_same_map() -> None:
    gray = _gray(2)
    cache = {
        "r34": flipped_predictions(_flip_dependent(1.0, 0.0), gray),
        "r34_v2_agree": flipped_predictions(_flip_dependent(0.5, 0.2), gray),
    }
    raw = combine(cache, EvalConfig("x", {"r34": 7.0, "r34_v2_agree": 3.0}, tta=True))
    normalized = combine(
        cache,
        EvalConfig(
            "y",
            ensemble_weights((("r34", 7.0), ("r34_v2_agree", 3.0)), "r34"),
            tta=True,
        ),
    )
    assert np.allclose(raw, normalized, atol=1e-6)


# --- the holdout configurations -----------------------------------------------------


def test_holdout_configs_are_the_four_asked_for() -> None:
    names = [config.name for config in HOLDOUT_ENSEMBLE_CONFIGS]
    assert names == [
        "r34_tta",
        "r34_v2_agree_tta",
        "r34+agree_50_50_tta",
        "r34+agree_70_30_tta",
    ]
    assert all(config.tta for config in HOLDOUT_ENSEMBLE_CONFIGS)
    assert [describe_models(c) for c in HOLDOUT_ENSEMBLE_CONFIGS] == [
        "r34:1",
        "r34_v2_agree:1",
        "r34:0.5+r34_v2_agree:0.5",
        "r34:0.7+r34_v2_agree:0.3",
    ]
    assert ENSEMBLE_POINT == GridPoint(0.45, 11, 400)
    for config in HOLDOUT_ENSEMBLE_CONFIGS:
        assert set(config.weights) <= set(CROP_CHECKPOINTS)
        assert unpinned_members(config.weights) == []


def _disk_gray() -> np.ndarray:
    rows, cols = np.ogrid[:SIDE, :SIDE]
    disk = (rows - 64) ** 2 + (cols - 64) ** 2 <= 58**2
    return np.where(disk, 200, 5).astype(np.uint8)


def _box(x0: int, y0: int, x1: int, y1: int) -> tuple[float, ...]:
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def _scene():
    """A frame with a dark filament (gray 60) and a paler blob (gray 120).

    Both predictors are pixelwise functions of the gray value, so every flip gives the
    same map once flipped back, and the flip-TTA mean is that map.
    """
    real = FilamentAnnotation(
        id=1,
        image_id=1,
        category_id=1,
        spine=None,
        segmentation=(_box(40, 40, 80, 60),),
    )
    image = TrainImage(
        id=1,
        file_name="a.jpeg",
        date_captured=None,
        height=SIDE,
        width=SIDE,
        annotations=(real,),
    )
    gray = _disk_gray()
    gray[np.asarray(real.rasterize(SIDE, SIDE)) != 0] = 60
    gray[80:95, 50:75] = 120
    calls = {"good": 0, "noisy": 0}

    def good(frame: np.ndarray) -> np.ndarray:
        calls["good"] += 1
        return np.where(frame == 60, 0.9, 0.0).astype(np.float32)

    def noisy(frame: np.ndarray) -> np.ndarray:
        calls["noisy"] += 1
        values = np.where(frame == 60, 0.9, 0.0)
        return np.where(frame == 120, 0.6, values).astype(np.float32)

    return image, gray, {"r34": good, "r34_v2_agree": noisy}, calls


def test_holdout_scoring_averages_then_filters_and_predicts_once_per_flip() -> None:
    image, gray, predictors, calls = _scene()
    configs = [
        EvalConfig("good", {"r34": 1.0}, tta=True),
        EvalConfig("noisy", {"r34_v2_agree": 1.0}, tta=True),
        # The noisy model first: scoring the first member's map instead of the
        # average would keep its blob.
        EvalConfig("mix", {"r34_v2_agree": 0.5, "r34": 0.5}, tta=True),
    ]
    point = GridPoint(0.45, 0, 20)
    per_file, seconds = per_file_ensemble_counts(
        [image], lambda _i: gray, predictors, configs, point=point
    )
    assert calls == {"good": 4, "noisy": 4}  # 4 flips each, not once per config
    assert seconds >= 0
    good, noisy, mix = (per_file.counts[c.name][0] for c in configs)
    assert (good.tp, good.fp, good.fn) == (1, 0, 0)
    assert (noisy.tp, noisy.fp) == (1, 1)
    # Averaged, the blob is at 0.3, below the threshold: it is gone.
    assert (mix.tp, mix.fp) == (1, 0)
    # The object filter drops the blob of the noisy model alone.
    filtered, _ = per_file_ensemble_counts(
        [image],
        lambda _i: gray,
        predictors,
        configs[1:2],
        point=point,
        cut=ObjectCut("mean", 0.8),
    )
    kept = filtered.counts["noisy"][0]
    assert (kept.tp, kept.fp) == (1, 0)


def test_the_object_filter_sees_the_averaged_map() -> None:
    image, gray, predictors, _ = _scene()
    bright = {**predictors}

    def confident_blob(frame: np.ndarray) -> np.ndarray:
        values = np.where(frame == 60, 0.9, 0.0)
        return np.where(frame == 120, 0.95, values).astype(np.float32)

    bright["r34_v2_agree"] = confident_blob
    # Averaged, the blob is 0.475: above the threshold, below the cut. Filtering on
    # the first member's map (0.95) would keep it.
    mix = EvalConfig("mix", {"r34_v2_agree": 0.5, "r34": 0.5}, tta=True)
    point = GridPoint(0.45, 0, 20)
    plain, _ = per_file_ensemble_counts(
        [image], lambda _i: gray, bright, [mix], point=point
    )
    filtered, _ = per_file_ensemble_counts(
        [image], lambda _i: gray, bright, [mix], point=point, cut=ObjectCut("mean", 0.8)
    )
    assert (plain.counts["mix"][0].tp, plain.counts["mix"][0].fp) == (1, 1)
    assert (filtered.counts["mix"][0].tp, filtered.counts["mix"][0].fp) == (1, 0)


def test_holdout_scoring_refuses_missing_models_and_bad_configs() -> None:
    image, gray, predictors, _ = _scene()
    with pytest.raises(KeyError, match="r50"):
        per_file_ensemble_counts(
            [image],
            lambda _i: gray,
            predictors,
            [EvalConfig("x", {"r50": 1.0})],
        )
    same = [EvalConfig("x", {"r34": 1.0}), EvalConfig("x", {"r34": 1.0})]
    with pytest.raises(ValueError, match="distinct"):
        per_file_ensemble_counts([image], lambda _i: gray, predictors, same)
    with pytest.raises(ValueError, match="empty"):
        per_file_ensemble_counts([], lambda _i: gray, predictors, same[:1])


def test_rows_compare_every_config_with_the_first(tmp_path: Path) -> None:
    image, gray, predictors, _ = _scene()
    configs = [
        EvalConfig("good", {"r34": 1.0}, tta=True),
        EvalConfig("noisy", {"r34_v2_agree": 1.0}, tta=True),
    ]
    per_file, _ = per_file_ensemble_counts(
        [image],
        lambda _i: gray,
        predictors,
        configs,
        point=GridPoint(0.45, 0, 20),
    )
    cut = ObjectCut("mean", 0.8)
    rows = ensemble_rows(
        per_file, configs, point=GridPoint(0.45, 0, 20), cut=cut, resamples=50
    )
    assert [row["config"] for row in rows] == ["good", "noisy"]
    assert float(rows[0]["diff_vs_first"]) == 0.0
    assert float(rows[1]["diff_vs_first"]) < 0
    assert rows[0]["object_filter"] == "mean>=0.8" and rows[0]["tta"] == "yes"
    assert rows[0]["holdout_files"] == "1" and rows[0]["annotator_views"] == "1"
    path = tmp_path / "ens.csv"
    write_ensemble_rows(path, rows)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == ENSEMBLE_COLUMNS
        assert list(reader) == rows
