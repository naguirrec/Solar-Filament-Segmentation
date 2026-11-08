"""The probability cache on disk. Pure numpy, no torch and no competition data."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from filament.checkpoint import R34_CROPS, checkpoint_problems
from filament.ensemble import (
    FLIP_NAMES,
    EvalConfig,
    combine,
    flipped_predictions,
)
from filament.probcache import (
    INDEX_FILE,
    MANIFEST_FILE,
    MEAN_FILE,
    STD_FILE,
    ExportTiming,
    ProbCache,
    ProbCacheWriter,
    export_prob_maps,
    find_cache_root,
    flip_statistics,
    manifest_fingerprint,
    model_dir,
    read_manifest,
    split_dir,
    write_manifest,
)

SHAPE = (24, 32)


def _map(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(SHAPE, dtype=np.float32)


def _write_cache(directory: Path, names: list[str]) -> dict[str, np.ndarray]:
    written = {}
    with ProbCacheWriter(directory, names, SHAPE) as writer:
        for seed, name in enumerate(names):
            written[name] = _map(seed)
            writer.write(name, written[name], written[name] / 4)
    return written


def test_layout_is_one_directory_per_model_and_split(tmp_path: Path) -> None:
    assert model_dir(tmp_path, "r34") == tmp_path / "r34"
    assert split_dir(tmp_path, "r34", "holdout") == tmp_path / "r34" / "holdout"


def test_a_split_leaves_three_files_whatever_the_frame_count(tmp_path: Path) -> None:
    _write_cache(tmp_path, [f"{i}.jpeg" for i in range(40)])
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [MEAN_FILE, STD_FILE, INDEX_FILE]
    )


def test_round_trip_is_close_and_returns_float32(tmp_path: Path) -> None:
    names = ["b.jpeg", "a.jpeg", "c.jpeg"]
    written = _write_cache(tmp_path, names)
    cache = ProbCache(tmp_path)
    assert cache.file_names == names  # the order of the arrays, not sorted
    assert len(cache) == 3
    assert "a.jpeg" in cache and "z.jpeg" not in cache
    for name in names:
        mean = cache.mean(name)
        assert mean.dtype == np.float32
        upper = written[name] >= 0.25
        assert np.abs(mean - written[name])[upper].max() <= 2.0**-12
        assert np.abs(cache.std(name) - written[name] / 4).max() <= 2.0**-12


def test_a_threshold_decision_only_changes_right_next_to_the_threshold(
    tmp_path: Path,
) -> None:
    written = _write_cache(tmp_path, ["a.jpeg"])["a.jpeg"]
    loaded = ProbCache(tmp_path).mean("a.jpeg")
    for cut in (0.3, 0.45, 0.5, 0.6):
        changed = (written >= cut) != (loaded >= cut)
        assert np.all(np.abs(written[changed] - cut) <= 2.0**-11)


def test_reading_keeps_a_threshold_at_full_precision(tmp_path: Path) -> None:
    # float16(0.45) lies just below 0.45. A float16 array would round the
    # threshold the same way and call this pixel foreground at 0.45.
    stored = np.float16(0.45)
    assert float(stored) < 0.45
    assert np.all(np.full(SHAPE, stored, dtype=np.float16) >= 0.45)
    level = np.full(SHAPE, float(stored), dtype=np.float32)
    with ProbCacheWriter(tmp_path, ["a.jpeg"], SHAPE) as writer:
        writer.write("a.jpeg", level, np.zeros(SHAPE))
    loaded = ProbCache(tmp_path).mean("a.jpeg")
    assert loaded.dtype == np.float32
    assert not np.any(loaded >= 0.45)


def test_returned_maps_are_writable_copies(tmp_path: Path) -> None:
    _write_cache(tmp_path, ["a.jpeg"])
    cache = ProbCache(tmp_path)
    first = cache.mean("a.jpeg")
    before = first.copy()
    first[:] = 0.0
    assert np.array_equal(cache.mean("a.jpeg"), before)


def test_unknown_frame_is_refused(tmp_path: Path) -> None:
    _write_cache(tmp_path, ["a.jpeg"])
    with pytest.raises(KeyError, match="z.jpeg"):
        ProbCache(tmp_path).mean("z.jpeg")


def test_writer_refuses_bad_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        ProbCacheWriter(tmp_path, ["a", "a"], SHAPE)
    with pytest.raises(ValueError, match="distinct"):
        ProbCacheWriter(tmp_path, [], SHAPE)
    writer = ProbCacheWriter(tmp_path, ["a", "b"], SHAPE)
    good = np.zeros(SHAPE)
    with pytest.raises(KeyError):
        writer.write("z", good, good)
    with pytest.raises(ValueError, match="shape"):
        writer.write("a", np.zeros((2, 2)), good)
    nan = good.copy()
    nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        writer.write("a", nan, good)
    logits = good.copy()
    logits[0, 0] = 3.0
    with pytest.raises(ValueError, match="not a probability"):
        writer.write("a", logits, good)
    negative = good.copy()
    negative[0, 0] = -0.5
    with pytest.raises(ValueError, match="not a probability"):
        writer.write("a", good, negative)
    writer.write("a", good, good)
    with pytest.raises(ValueError, match="already written"):
        writer.write("a", good, good)


def test_an_incomplete_export_cannot_be_read(tmp_path: Path) -> None:
    writer = ProbCacheWriter(tmp_path, ["a", "b"], SHAPE)
    writer.write("a", _map(0), _map(1))
    with pytest.raises(ValueError, match="1 of 2 frames"):
        writer.close()
    with pytest.raises(FileNotFoundError, match="not a finished export"):
        ProbCache(tmp_path)


def test_an_exception_during_the_export_leaves_no_index(tmp_path: Path) -> None:
    with (
        pytest.raises(RuntimeError, match="boom"),
        ProbCacheWriter(tmp_path, ["a"], SHAPE) as writer,
    ):
        writer.write("a", _map(0), _map(1))
        raise RuntimeError("boom")
    with pytest.raises(FileNotFoundError):
        ProbCache(tmp_path)


def test_a_restart_invalidates_the_previous_export(tmp_path: Path) -> None:
    _write_cache(tmp_path, ["a"])
    ProbCache(tmp_path)
    ProbCacheWriter(tmp_path, ["a"], SHAPE)
    with pytest.raises(FileNotFoundError):
        ProbCache(tmp_path)


def test_reader_refuses_arrays_that_do_not_match_the_index(tmp_path: Path) -> None:
    _write_cache(tmp_path, ["a", "b"])
    (tmp_path / INDEX_FILE).write_text('["a"]', encoding="utf-8")
    with pytest.raises(ValueError, match="2 frames"):
        ProbCache(tmp_path)


def test_reader_refuses_another_dtype(tmp_path: Path) -> None:
    _write_cache(tmp_path, ["a"])
    np.save(tmp_path / MEAN_FILE, np.zeros((1, *SHAPE), dtype=np.float32))
    with pytest.raises(ValueError, match="float16"):
        ProbCache(tmp_path)


def _asymmetric(gray: np.ndarray) -> np.ndarray:
    """A predictor that is not flip-equivariant: it depends on the position."""
    rows = np.linspace(0.0, 1.0, gray.shape[0], dtype=np.float32)[:, None]
    return (rows + gray / 255.0) / 2.0


def _gray(seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, SHAPE, dtype=np.uint8)


def test_stored_mean_is_the_map_flip_tta_scores() -> None:
    maps = flipped_predictions(_asymmetric, _gray())
    mean, std = flip_statistics(maps)
    config = EvalConfig("tta", {"m": 1.0}, tta=True)
    assert np.array_equal(mean, combine({"m": maps}, config))
    assert np.allclose(std, np.std(np.stack(list(maps.values())), axis=0), atol=1e-6)
    assert mean.dtype == np.float32 and std.dtype == np.float32
    assert std.max() > 0.01  # the flips really disagree here


def test_a_flip_equivariant_predictor_has_no_spread() -> None:
    maps = flipped_predictions(lambda gray: gray / 255.0, _gray())
    _mean, std = flip_statistics(maps)
    assert std.max() < 1e-6


def test_flip_statistics_refuses_nothing() -> None:
    with pytest.raises(ValueError):
        flip_statistics({})


def _write_frames(directory: Path, count: int) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"frame{index}.png"
        assert cv2.imwrite(str(path), _gray(index))
        paths.append(path)
    return paths


def test_export_writes_a_finished_cache_with_one_pass_per_flip(tmp_path: Path) -> None:
    paths = _write_frames(tmp_path / "in", 3)
    calls = 0

    def predict(gray: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return _asymmetric(gray)

    seen: list[int] = []
    ticks = iter(float(t) for t in range(1000))
    timing = export_prob_maps(
        paths,
        predict,
        tmp_path / "out",
        shape=SHAPE,
        progress=lambda index, _total, _path: seen.append(index),
        clock=lambda: next(ticks),
    )
    assert calls == 3 * len(FLIP_NAMES)
    assert seen == [1, 2, 3]
    assert isinstance(timing, ExportTiming)
    assert timing.files == 3
    assert timing.inference_seconds == 3.0  # each frame's two clock reads are 1 apart
    assert timing.seconds_per_image == 1.0
    assert timing.total_seconds > timing.inference_seconds
    cache = ProbCache(tmp_path / "out")
    assert cache.file_names == [p.name for p in paths]
    gray = cv2.imread(str(paths[1]), cv2.IMREAD_GRAYSCALE)
    expected, spread = flip_statistics(flipped_predictions(_asymmetric, gray))
    assert np.abs(cache.mean(paths[1].name) - expected).max() <= 2.0**-11
    assert np.abs(cache.std(paths[1].name) - spread).max() <= 2.0**-11


def test_export_stops_on_a_frame_of_another_size(tmp_path: Path) -> None:
    paths = _write_frames(tmp_path / "in", 1)
    with pytest.raises(ValueError, match="expected 8x8"):
        export_prob_maps(paths, _asymmetric, tmp_path / "out", shape=(8, 8))
    with pytest.raises(FileNotFoundError):
        ProbCache(tmp_path / "out")


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = {"model": "r34", "flips": list(FLIP_NAMES), "splits": {"holdout": 142}}
    path = write_manifest(tmp_path, "r34", manifest)
    assert path == tmp_path / "r34" / MANIFEST_FILE
    assert read_manifest(tmp_path, "r34") == manifest


def test_manifest_fingerprint_passes_the_checkpoint_guard() -> None:
    expected = R34_CROPS
    manifest = {
        "epochs": expected.epochs,
        "encoder": expected.encoder,
        "tensor_sha256": expected.tensor_sha256,
        "tile_size": 512,
    }
    fingerprint = manifest_fingerprint(manifest)
    assert checkpoint_problems(fingerprint, expected=expected) == []
    other = manifest_fingerprint({**manifest, "tensor_sha256": "ab"})
    assert checkpoint_problems(other, expected=expected)
    assert checkpoint_problems(manifest_fingerprint({}), expected=expected)


def test_find_cache_root_at_any_depth(tmp_path: Path) -> None:
    root = tmp_path / "datasets" / "someone" / "solar-probs-r34" / "probs"
    write_manifest(root, "r34", {"model": "r34"})
    assert find_cache_root("r34", tmp_path) == root
    with pytest.raises(FileNotFoundError, match="r50"):
        find_cache_root("r50", tmp_path)


def test_find_cache_root_skips_competitions(tmp_path: Path) -> None:
    write_manifest(tmp_path / "competitions" / "x", "r34", {})
    with pytest.raises(FileNotFoundError):
        find_cache_root("r34", tmp_path)


def test_find_cache_root_refuses_two_caches_of_one_model(tmp_path: Path) -> None:
    write_manifest(tmp_path / "a", "r34", {})
    write_manifest(tmp_path / "b", "r34", {})
    with pytest.raises(RuntimeError, match="Several caches"):
        find_cache_root("r34", tmp_path)
