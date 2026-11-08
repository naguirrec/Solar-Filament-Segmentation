"""Holdout sweep scoring on synthetic frames. Pure numpy and cv2."""

from __future__ import annotations

import ast
import csv
import math
from pathlib import Path

import numpy as np
import pytest

from filament.annotations import FilamentAnnotation, TrainImage, annotator_views
from filament.instances import labels_from_semantic
from filament.metrics import PqCounts, fast_view_counts
from filament.sweep import (
    COLUMNS,
    DEFAULT_CLOSE_SIZES,
    DEFAULT_GRID_MIN_AREAS,
    DEFAULT_MIN_AREAS,
    DEFAULT_THRESHOLDS,
    POSTPROC_COLUMNS,
    GridPoint,
    PerFileCounts,
    best_row,
    count_views,
    format_postproc_table,
    format_pq,
    format_table,
    per_file_counts,
    postproc_grid,
    postproc_rows,
    sweep_min_area,
    sweep_postproc_grid,
    sweep_rows,
    view_masks,
    write_postproc_rows,
    write_rows,
)

ROOT = Path(__file__).resolve().parents[1]
SIDE = 128


def _box(x0: int, y0: int, x1: int, y1: int) -> tuple[float, ...]:
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def _annotation(number: int, *boxes: tuple[float, ...]) -> FilamentAnnotation:
    return FilamentAnnotation(
        id=number,
        image_id=1,
        category_id=1,
        spine=None,
        segmentation=tuple(boxes),
    )


def _gray() -> np.ndarray:
    """Bright disk on dark sky, so the on-disk gate keeps the middle."""
    rows, cols = np.ogrid[:SIDE, :SIDE]
    disk = (rows - 64) ** 2 + (cols - 64) ** 2 <= 58**2
    return np.where(disk, 200, 10).astype(np.uint8)


def _image(name: str, views: tuple[tuple[FilamentAnnotation, ...], ...]) -> TrainImage:
    return TrainImage(
        id=name,
        file_name=f"{name}.jpeg",
        date_captured=None,
        height=SIDE,
        width=SIDE,
        annotations=views[0],
        views=views,
    )


def _holdout() -> list[TrainImage]:
    big = _annotation(1, _box(30, 30, 60, 50))  # 600 px
    small = _annotation(2, _box(80, 80, 86, 86))  # 36 px
    other = _annotation(3, _box(30, 70, 90, 90))  # 1200 px
    return [
        _image("20200101000000Aa", ((big, small), (big,))),
        _image("20200102000000Aa", ((other,),)),
    ]


def _union(image: TrainImage) -> np.ndarray:
    mask = np.zeros((SIDE, SIDE), dtype=np.uint8)
    for view in annotator_views(image):
        for masks in view_masks(image, view):
            mask |= masks
    return mask


def test_perfect_prediction_and_min_area_effect() -> None:
    holdout = _holdout()
    grays = {image.file_name: _gray() for image in holdout}
    unions = {image.file_name: _union(image) for image in holdout}
    current = {}

    def read(image: TrainImage) -> np.ndarray:
        current["name"] = image.file_name
        return grays[image.file_name]

    totals = sweep_min_area(
        holdout, read, lambda gray: unions[current["name"]], (10, 100, 800)
    )
    assert list(totals) == [10, 100, 800]
    # min_area 10 keeps the 36 px filament: both views of image 1 score it.
    # View 1 has {600, 36}, view 2 has {600}; image 2 has {1200}.
    assert totals[10].fp == 1 and totals[10].tp == 4 and totals[10].fn == 0
    # min_area 100 drops it: view 1 now misses it (fn), nothing else changes.
    assert totals[100].tp == 3 and totals[100].fn == 1 and totals[100].fp == 0
    # min_area 800 also drops the 600 px filament in both views of image 1.
    assert totals[800].tp == 1 and totals[800].fn == 3
    assert totals[100].pq == pytest.approx(3 / (3 + 0.5))


def test_predict_is_called_once_per_image_for_all_min_areas() -> None:
    holdout = _holdout()
    calls = {"predict": 0, "read": 0}

    def read(image: TrainImage) -> np.ndarray:
        calls["read"] += 1
        return _gray()

    def predict(gray: np.ndarray) -> np.ndarray:
        calls["predict"] += 1
        return np.zeros_like(gray)

    sweep_min_area(holdout, read, predict, DEFAULT_MIN_AREAS)
    assert calls == {"predict": 2, "read": 2}


def test_matches_the_per_min_area_loop_of_the_sweep_notebook() -> None:
    # The notebook runs the whole holdout once per min_area; the inverted
    # loop must add up the same counts, so the PQ is identical.
    # Three images; the semantic mask is the annotation union shifted by a few
    # pixels plus speckle, so IoU is partial (a few matches above 0.5, some
    # misses and false positives) and iou_sum is a real float sum.
    holdout = [
        *_holdout(),
        _image("20200103000000Aa", ((_annotation(4, _box(20, 20, 70, 40)),),)),
    ]
    rng = np.random.default_rng(0)
    semantics = {}
    for number, image in enumerate(holdout):
        shifted = np.roll(_union(image), 2 + number, axis=1)
        speckle = (rng.random((SIDE, SIDE)) < 0.01).astype(np.uint8)
        semantics[image.file_name] = shifted | speckle
    gray = _gray()
    current = {}

    def read(image: TrainImage) -> np.ndarray:
        current["name"] = image.file_name
        return gray

    areas = (5, 60, 400)
    got = sweep_min_area(
        holdout, read, lambda g: semantics[current["name"]], areas, close_size=0
    )
    for area in areas:
        total = PqCounts()
        for image in holdout:
            labels = labels_from_semantic(
                semantics[image.file_name], gray, min_area=area, close_size=0
            )
            for view in annotator_views(image):
                total = total + fast_view_counts(labels, view_masks(image, view))
        assert got[area].tp > 0 and got[area].iou_sum > 0.0
        assert got[area] == total
        assert got[area].pq == total.pq


def test_shape_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="does not match the frame"):
        sweep_min_area(_holdout(), lambda image: _gray(), lambda gray: gray[:10], (10,))
    with pytest.raises(ValueError, match="holdout is empty"):
        sweep_min_area([], lambda image: _gray(), lambda gray: gray, (10,))


def test_rows_csv_and_best(tmp_path: Path) -> None:
    totals = {
        10: PqCounts(tp=1, fp=1, fn=0, iou_sum=1.0),
        20: PqCounts(tp=3, fp=0, fn=1, iou_sum=3.0),
        30: PqCounts(),
    }
    rows = sweep_rows("unet_x", totals, close_size=0, holdout_files=142, views=231)
    assert rows[0] == {
        "predictor": "unet_x",
        "min_area": "10",
        "close_size": "0",
        "pooled_pq": "0.666667",
        "holdout_files": "142",
        "annotator_views": "231",
    }
    assert rows[2]["pooled_pq"] == "nan"
    best = best_row(rows)
    assert best is not None and best["min_area"] == "20"
    assert best_row(rows[2:]) is None

    out = tmp_path / "nested" / "sweep.csv"
    write_rows(out, rows)
    with out.open(newline="") as handle:
        read_back = list(csv.DictReader(handle))
    assert read_back == rows
    assert format_table(rows).splitlines()[0] == ",".join(COLUMNS)
    assert len(format_table(rows).splitlines()) == 4


def test_best_row_prefers_the_first_on_ties() -> None:
    rows = [
        {"pooled_pq": "0.3", "min_area": "80"},
        {"pooled_pq": "0.3", "min_area": "140"},
    ]
    assert best_row(rows)["min_area"] == "80"


def test_columns_match_the_existing_sweep_csv() -> None:
    header = (ROOT / "results" / "postprocess_sweep_2026-11-01.csv").read_text()
    assert header.splitlines()[0] == ",".join(COLUMNS)


def test_count_views_counts_annotator_entries() -> None:
    assert count_views(_holdout()) == 3


def _notebook_function(name: str):
    import json

    notebook = json.loads(
        (ROOT / "notebooks" / "sweep_postprocess.ipynb").read_text(encoding="utf-8")
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                namespace = {"np": np, "math": math, "TrainImage": TrainImage}
                exec(ast.get_source_segment(source, node), namespace)  # noqa: S102
                return namespace[name]
    raise AssertionError(name)


def test_view_masks_and_format_pq_match_the_sweep_notebook() -> None:
    notebook_masks = _notebook_function("_view_masks")
    notebook_format = _notebook_function("_format_pq")
    for image in _holdout():
        for view in annotator_views(image):
            ours = view_masks(image, view)
            theirs = notebook_masks(image, view)
            assert len(ours) == len(theirs)
            assert all(np.array_equal(a, b) for a, b in zip(ours, theirs, strict=True))
    for value in (0.0, 0.2886, 1 / 3, float("nan")):
        assert format_pq(value) == notebook_format(value)


@pytest.mark.parametrize("areas", [(), (5, 5)])
def test_min_areas_must_be_non_empty_and_distinct(areas: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="non-empty and distinct"):
        sweep_min_area(_holdout(), lambda i: _gray(), lambda g: g, areas)


# --- post-processing grid (threshold x close_size x min_area) -----------------


def _probabilities(holdout: list[TrainImage]) -> dict[str, np.ndarray]:
    """Soft maps: the annotation union at 0.8 over noise, shifted a little.

    The noise puts pixels on both sides of every threshold in 0.3-0.6, so
    different thresholds really give different masks.
    """
    rng = np.random.default_rng(1)
    maps = {}
    for number, image in enumerate(holdout):
        base = np.roll(_union(image), 1 + number, axis=1).astype(np.float32)
        noise = rng.random((SIDE, SIDE)).astype(np.float32)
        maps[image.file_name] = np.clip(0.8 * base + 0.45 * noise, 0.0, 1.0)
    return maps


def _grid_run(thresholds, close_sizes, min_areas):
    holdout = _holdout()
    maps = _probabilities(holdout)
    gray = _gray()
    current = {}
    calls = {"predict": 0}

    def read(image: TrainImage) -> np.ndarray:
        current["name"] = image.file_name
        return gray

    def predict(frame: np.ndarray) -> np.ndarray:
        calls["predict"] += 1
        return maps[current["name"]]

    totals = sweep_postproc_grid(
        holdout, read, predict, thresholds, close_sizes, min_areas
    )
    return holdout, maps, gray, totals, calls


def test_grid_has_every_combination_in_order() -> None:
    points = postproc_grid((0.4, 0.5), (0, 3), (100, 200, 300))
    assert len(points) == 12
    assert points[0] == GridPoint(0.4, 0, 100)
    assert points[2] == GridPoint(0.4, 0, 300)
    assert points[3] == GridPoint(0.4, 3, 100)
    assert points[-1] == GridPoint(0.5, 3, 300)
    assert len(set(points)) == 12


def test_default_grid_covers_the_requested_ranges() -> None:
    assert min(DEFAULT_THRESHOLDS) == 0.30 and max(DEFAULT_THRESHOLDS) == 0.60
    assert 0.5 in DEFAULT_THRESHOLDS  # the current setting is in the grid
    assert 0 in DEFAULT_CLOSE_SIZES
    assert DEFAULT_GRID_MIN_AREAS == (240, 320, 400, 500, 600, 700)
    points = postproc_grid(
        DEFAULT_THRESHOLDS, DEFAULT_CLOSE_SIZES, DEFAULT_GRID_MIN_AREAS
    )
    assert GridPoint(0.5, 0, 400) in points
    assert len(points) == 7 * 4 * 6


@pytest.mark.parametrize(
    ("thresholds", "close_sizes", "min_areas", "error"),
    [
        ((), (0,), (400,), ValueError),
        ((0.5, 0.5), (0,), (400,), ValueError),
        ((0.5,), (0, 0), (400,), ValueError),
        ((0.5,), (0,), (400, 400), ValueError),
        ((0.0,), (0,), (400,), ValueError),
        ((1.0,), (0,), (400,), ValueError),
        ((0.5,), (2,), (400,), ValueError),  # even kernel
        ((0.5,), (-3,), (400,), ValueError),
        ((0.5,), (0,), (0,), ValueError),
        ((0.5,), (0.0,), (400,), TypeError),
        ((0.5,), (0,), (True,), TypeError),
        (("0.5",), (0,), (400,), TypeError),
    ],
)
def test_grid_rejects_bad_axes(thresholds, close_sizes, min_areas, error) -> None:
    with pytest.raises(error):
        postproc_grid(thresholds, close_sizes, min_areas)


def test_network_runs_once_per_image_for_the_whole_grid() -> None:
    _, _, _, totals, calls = _grid_run(DEFAULT_THRESHOLDS, (0, 3), (240, 400))
    assert calls["predict"] == 2  # two holdout images, however many grid points
    assert len(totals) == len(DEFAULT_THRESHOLDS) * 2 * 2


def test_grid_reproduces_the_min_area_sweep_at_the_current_setting() -> None:
    # (threshold 0.5, close_size 0) must give the numbers of sweep_min_area.
    areas = (5, 60, 400)
    holdout, maps, gray, totals, _ = _grid_run((0.3, 0.5), (0, 5), areas)
    current = {}

    def read(image: TrainImage) -> np.ndarray:
        current["name"] = image.file_name
        return gray

    reference = sweep_min_area(
        holdout,
        read,
        lambda frame: (maps[current["name"]] >= 0.5).astype(np.uint8),
        areas,
        close_size=0,
    )
    for area in areas:
        assert totals[GridPoint(0.5, 0, area)] == reference[area]
    assert reference[5].tp > 0  # not a vacuous comparison


@pytest.mark.parametrize(("threshold", "close_size"), [(0.3, 7), (0.45, 3), (0.6, 0)])
def test_every_grid_point_matches_labeling_one_mask_at_a_time(
    threshold: float, close_size: int
) -> None:
    areas = (5, 60, 400)
    holdout, maps, gray, totals, _ = _grid_run((0.3, 0.45, 0.6), (0, 3, 7), areas)
    for area in areas:
        expected = PqCounts()
        for image in holdout:
            semantic = (maps[image.file_name] >= threshold).astype(np.uint8)
            labels = labels_from_semantic(
                semantic, gray, min_area=area, close_size=close_size
            )
            for view in annotator_views(image):
                expected = expected + fast_view_counts(labels, view_masks(image, view))
        assert totals[GridPoint(threshold, close_size, area)] == expected


def test_threshold_changes_the_score() -> None:
    _, _, _, totals, _ = _grid_run((0.3, 0.6), (0,), (60,))
    low, high = totals[GridPoint(0.3, 0, 60)], totals[GridPoint(0.6, 0, 60)]
    assert low != high


def test_probability_shape_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="does not match the frame"):
        sweep_postproc_grid(
            _holdout(),
            lambda i: _gray(),
            lambda g: g[:10].astype(float),
            (0.5,),
            (0,),
            (5,),
        )
    with pytest.raises(ValueError, match="holdout is empty"):
        sweep_postproc_grid([], lambda i: _gray(), lambda g: g, (0.5,), (0,), (5,))


def test_postproc_rows_are_sorted_best_first_with_nan_last(tmp_path: Path) -> None:
    totals = {
        GridPoint(0.4, 0, 100): PqCounts(tp=1, fp=1, fn=0, iou_sum=1.0),  # 0.6667
        GridPoint(0.5, 0, 100): PqCounts(),  # nan
        GridPoint(0.5, 3, 100): PqCounts(tp=3, fp=0, fn=1, iou_sum=3.0),  # 0.857
        GridPoint(0.6, 0, 100): PqCounts(tp=1, fp=1, fn=0, iou_sum=1.0),  # tie
    }
    rows = postproc_rows("unet_x", totals, holdout_files=142, views=231)
    assert [(r["threshold"], r["close_size"]) for r in rows] == [
        ("0.5", "3"),
        ("0.4", "0"),  # ties keep grid order
        ("0.6", "0"),
        ("0.5", "0"),
    ]
    assert rows[0] == {
        "predictor": "unet_x",
        "threshold": "0.5",
        "close_size": "3",
        "min_area": "100",
        "pooled_pq": "0.857143",
        "holdout_files": "142",
        "annotator_views": "231",
    }
    assert rows[-1]["pooled_pq"] == "nan"

    out = tmp_path / "nested" / "holdout_postproc_r34.csv"
    write_postproc_rows(out, rows)
    with out.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == POSTPROC_COLUMNS
        assert list(reader) == rows
    assert len(format_postproc_table(rows).splitlines()) == 5
    assert len(format_postproc_table(rows, limit=2).splitlines()) == 3


# --- per-file counts --------------------------------------------------------------


def _per_file_holdout() -> tuple[list[TrainImage], dict[str, np.ndarray]]:
    """Two files: one with two views that partly disagree, one with one view."""
    box_a, box_b = _box(40, 40, 70, 60), _box(40, 70, 60, 90)
    first = TrainImage(
        id=1,
        file_name="first.jpeg",
        date_captured=None,
        height=SIDE,
        width=SIDE,
        annotations=(_annotation(1, box_a), _annotation(2, box_b)),
        views=(
            (_annotation(1, box_a), _annotation(2, box_b)),
            (_annotation(3, box_a),),
        ),
    )
    second = TrainImage(
        id=2,
        file_name="second.jpeg",
        date_captured=None,
        height=SIDE,
        width=SIDE,
        annotations=(_annotation(4, box_b),),
    )
    maps = {}
    for image in (first, second):
        probability = np.zeros((SIDE, SIDE), dtype=np.float32)
        for annotation in image.annotations:
            mask = np.asarray(annotation.rasterize(SIDE, SIDE)) != 0
            probability[mask] = 0.9
        maps[image.file_name] = probability
    return [first, second], maps


def test_per_file_counts_add_up_to_the_pooled_sweep() -> None:
    holdout, maps = _per_file_holdout()
    frames = iter(holdout)
    pooled = sweep_postproc_grid(
        holdout,
        lambda _image: _gray(),
        lambda _gray_frame: maps[next(frames).file_name],
        (0.45, 0.5),
        (0, 3),
        (20, 400),
    )
    per_file = per_file_counts(
        holdout,
        lambda _image: _gray(),
        lambda image: maps[image.file_name],
        list(pooled),
    )
    assert isinstance(per_file, PerFileCounts)
    assert per_file.file_names == ("first.jpeg", "second.jpeg")
    assert per_file.views == (2, 1)
    for point, totals in pooled.items():
        assert len(per_file.counts[point]) == 2
        added = per_file.pooled(point)
        assert (added.tp, added.fp, added.fn) == (totals.tp, totals.fp, totals.fn)
        assert added.iou_sum == pytest.approx(totals.iou_sum)
    # The test is not vacuous: something matches, and the two files differ.
    point = GridPoint(0.5, 0, 20)
    assert per_file.pooled(point).tp > 0
    assert per_file.counts[point][0] != per_file.counts[point][1]


def test_per_file_counts_score_each_file_on_its_own_views() -> None:
    holdout, maps = _per_file_holdout()
    point = GridPoint(0.5, 0, 20)
    per_file = per_file_counts(
        holdout,
        lambda _image: _gray(),
        lambda image: maps[image.file_name],
        [point],
    )
    first, second = per_file.counts[point]
    # First file: two predictions. View 1 matches both; view 2 matches box a and
    # leaves box b as a false positive.
    assert (first.tp, first.fp, first.fn) == (3, 1, 0)
    assert (second.tp, second.fp, second.fn) == (1, 0, 0)


def test_per_file_counts_accept_points_that_are_not_a_full_grid() -> None:
    holdout, maps = _per_file_holdout()
    points = [GridPoint(0.5, 0, 20), GridPoint(0.45, 3, 400)]
    per_file = per_file_counts(
        holdout,
        lambda _image: _gray(),
        lambda image: maps[image.file_name],
        points,
    )
    assert set(per_file.counts) == set(points)


def test_per_file_counts_refuse_bad_input() -> None:
    holdout, maps = _per_file_holdout()
    point = GridPoint(0.5, 0, 20)
    read, load = (lambda _image: _gray()), (lambda image: maps[image.file_name])
    with pytest.raises(ValueError, match="holdout is empty"):
        per_file_counts([], read, load, [point])
    with pytest.raises(ValueError, match="distinct"):
        per_file_counts(holdout, read, load, [point, point])
    with pytest.raises(ValueError, match="does not match"):
        per_file_counts(
            holdout, read, lambda _image: np.zeros((4, 4), np.float32), [point]
        )
