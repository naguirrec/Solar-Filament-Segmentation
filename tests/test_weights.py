"""Weights export, manifest, lookup and verification (``filament.weights``).

The torch half (the fp16 export itself) is skipped when torch is not installed, as
the other torch tests are. Everything else runs on plain values.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from filament import weights
from filament.checkpoint import (
    R34_CROPS,
    checkpoint_fingerprint,
    tensor_sha256,
)
from filament.weights import (
    WeightsError,
    kaggle_candidates,
    locate_weights,
    manifest_entry,
    materialize_checkpoint,
    read_weights_manifest,
    repo_weights_problems,
    sha256_file,
    verify_repo_weights,
    write_weights_manifest,
)

# tensor_sha256 hashes anything that is not a tensor by its repr, so plain values
# stand in for the tensors here.
STATE = {"encoder.w": [1, 2, 3], "decoder.b": [4.5]}
ORIGINAL = {"encoder.w": [9, 9, 9]}
FP16_HASH = tensor_sha256(STATE)


def _checkpoint(**fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "state_dict": STATE,
        "epochs": 80,
        "encoder_name": "resnet34",
    }
    base.update(fields)
    return base


def _entry(file_hash: str, **overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "file_sha256": file_hash,
        "tensor_sha256": FP16_HASH,
        "tensor_sha256_original": R34_CROPS.tensor_sha256,
        "dtype": "float16",
        "bytes": 10,
        "encoder": "resnet34",
        "epochs": 80,
    }
    entry.update(overrides)
    return entry


# --------------------------------------------------------------- hashing


def test_sha256_file_is_the_hash_of_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1000)
    import hashlib

    assert sha256_file(path, chunk=7) == hashlib.sha256(b"abc" * 1000).hexdigest()


# -------------------------------------------------------------- manifest


def test_manifest_round_trip_and_empty_default(tmp_path: Path) -> None:
    assert read_weights_manifest(tmp_path) == {"format": 1, "weights": {}}
    manifest = {"format": 1, "weights": {"a.pt": _entry("0" * 64)}}
    path = write_weights_manifest(tmp_path / "weights", manifest)
    assert path.name == "MANIFEST.json"
    assert read_weights_manifest(tmp_path / "weights") == manifest
    assert manifest_entry(tmp_path / "weights", "a.pt")["epochs"] == 80
    with pytest.raises(WeightsError, match=r"b\.pt is not in"):
        manifest_entry(tmp_path / "weights", "b.pt")


def test_a_manifest_of_another_format_is_refused(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"format": 99, "weights": {}}))
    with pytest.raises(WeightsError, match="format 1"):
        read_weights_manifest(tmp_path)


# ---------------------------------------------------------- verification


def test_matching_weights_have_no_problems() -> None:
    fingerprint = checkpoint_fingerprint(_checkpoint())
    assert (
        repo_weights_problems(
            _entry("a" * 64), file_sha256="a" * 64, fingerprint=fingerprint
        )
        == []
    )


@pytest.mark.parametrize(
    ("entry_changes", "file_hash", "checkpoint_changes", "fragment"),
    [
        ({}, "b" * 64, {}, "file_sha256"),
        ({"tensor_sha256_original": "c" * 64}, "a" * 64, {}, "tensor_sha256_original"),
        ({"tensor_sha256": "d" * 64}, "a" * 64, {}, "tensor_sha256="),
        ({}, "a" * 64, {"epochs": 40}, "epochs=40"),
        ({}, "a" * 64, {"encoder_name": "resnet50"}, "encoder=resnet50"),
        ({}, "a" * 64, {"state_dict": ORIGINAL}, "tensor_sha256="),
    ],
    ids=["file", "original", "manifest-fp16", "epochs", "encoder", "tensors"],
)
def test_each_way_the_weights_can_differ_is_reported(
    entry_changes: dict[str, object],
    file_hash: str,
    checkpoint_changes: dict[str, object],
    fragment: str,
) -> None:
    fingerprint = checkpoint_fingerprint(_checkpoint(**checkpoint_changes))
    problems = repo_weights_problems(
        _entry("a" * 64, **entry_changes),
        file_sha256=file_hash,
        fingerprint=fingerprint,
    )
    assert any(fragment in problem for problem in problems), problems


def test_an_entry_without_an_fp16_hash_is_a_problem() -> None:
    entry = _entry("a" * 64)
    del entry["tensor_sha256"]
    problems = repo_weights_problems(
        entry,
        file_sha256="a" * 64,
        fingerprint=checkpoint_fingerprint(_checkpoint()),
    )
    assert problems == ["the manifest has no tensor_sha256 for this file"]


def test_verify_repo_weights_hashes_the_file_and_raises_on_a_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "w.pt"
    path.write_bytes(b"weights")
    entry = _entry(sha256_file(path))
    fingerprint = verify_repo_weights(_checkpoint(), path, entry)
    assert fingerprint["tensor_sha256"] == FP16_HASH
    path.write_bytes(b"weights, edited")
    with pytest.raises(WeightsError, match="file_sha256"):
        verify_repo_weights(_checkpoint(), path, entry)


# ---------------------------------------------------------------- lookup


def test_the_repo_file_wins_over_a_kaggle_file(tmp_path: Path) -> None:
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "m_fp16.pt").write_bytes(b"x")
    kaggle = tmp_path / "input" / "nb" / "m.pt"
    kaggle.parent.mkdir(parents=True)
    kaggle.write_bytes(b"y")
    found = locate_weights(tmp_path, "m_fp16.pt", "m.pt", input_root=tmp_path / "input")
    assert (found.kind, found.path) == ("repo", tmp_path / "weights" / "m_fp16.pt")


def test_the_kaggle_file_is_the_stopgap(tmp_path: Path) -> None:
    kaggle = tmp_path / "input" / "nb" / "m.pt"
    kaggle.parent.mkdir(parents=True)
    kaggle.write_bytes(b"y")
    found = locate_weights(tmp_path, "m_fp16.pt", "m.pt", input_root=tmp_path / "input")
    assert (found.kind, found.path) == ("kaggle", kaggle)


def test_a_folder_that_kaggle_extracted_counts_as_a_candidate(tmp_path: Path) -> None:
    folder = tmp_path / "input" / "nb" / "m.pt"
    (folder / "m").mkdir(parents=True)
    assert kaggle_candidates(tmp_path / "input", "m.pt") == [folder]


def test_competitions_are_not_searched(tmp_path: Path) -> None:
    hidden = tmp_path / "input" / "competitions" / "x" / "m.pt"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(b"z")
    with pytest.raises(WeightsError, match="No weights"):
        locate_weights(tmp_path, "m_fp16.pt", "m.pt", input_root=tmp_path / "input")


def test_several_candidates_are_listed_and_not_guessed(tmp_path: Path) -> None:
    for name in ("a", "b"):
        path = tmp_path / "input" / name / "m.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"y")
    with pytest.raises(WeightsError, match="2 files named m.pt") as caught:
        locate_weights(tmp_path, "m_fp16.pt", "m.pt", input_root=tmp_path / "input")
    assert "/a/m.pt" in str(caught.value) and "/b/m.pt" in str(caught.value)


def test_no_weights_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(WeightsError) as caught:
        locate_weights(tmp_path, "m_fp16.pt", "m.pt", input_root=tmp_path / "input")
    assert str(tmp_path / "weights" / "m_fp16.pt") in str(caught.value)
    assert "train-unet-r34-crops-80ep" in str(caught.value)


# ------------------------------------------------- extracted torch archives


def _extracted(root: Path) -> Path:
    folder = root / "unet_r34_crops.pt"
    (folder / "unet_r34_crops" / "data").mkdir(parents=True)
    (folder / "unet_r34_crops" / "data.pkl").write_bytes(b"pickle")
    (folder / "unet_r34_crops" / "version").write_bytes(b"3\n")
    (folder / "unet_r34_crops" / "data" / "0").write_bytes(b"tensor")
    return folder


def test_a_file_is_returned_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "a.pt"
    path.write_bytes(b"x")
    assert materialize_checkpoint(path, tmp_path / "work") == path


def test_an_extracted_folder_is_rezipped_with_the_archive_folder_name(
    tmp_path: Path,
) -> None:
    folder = _extracted(tmp_path / "input")
    result = materialize_checkpoint(folder, tmp_path / "work")
    assert result == tmp_path / "work" / "unet_r34_crops-from-extracted.pt"
    with zipfile.ZipFile(result) as archive:
        assert sorted(archive.namelist()) == [
            "unet_r34_crops/data.pkl",
            "unet_r34_crops/data/0",
            "unet_r34_crops/version",
        ]
        assert all(i.compress_type == zipfile.ZIP_STORED for i in archive.infolist())


def test_a_folder_without_exactly_one_pickle_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "x.pt"
    folder.mkdir()
    with pytest.raises(WeightsError, match="0 data.pkl"):
        materialize_checkpoint(folder, tmp_path / "work")
    with pytest.raises(WeightsError, match="does not exist"):
        materialize_checkpoint(tmp_path / "missing.pt", tmp_path / "work")


# -------------------------------------------------- the export (needs torch)


def _tiny_training_checkpoint(path: Path, torch) -> dict[str, object]:  # type: ignore[no-untyped-def]
    generator = torch.Generator().manual_seed(0)
    state = {
        "encoder.conv.weight": torch.randn(8, 3, 3, 3, generator=generator),
        "decoder.bias": torch.randn(8, generator=generator),
        "encoder.bn.num_batches_tracked": torch.tensor(7, dtype=torch.int64),
    }
    checkpoint = {
        "state_dict": state,
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "smp_version": "0.6.0",
        "epochs": 80,
        "crop_size": 512,
        "status": "complete",
        "epoch_losses": [0.5, 0.4],  # not kept by the export
    }
    torch.save(checkpoint, path)
    return checkpoint


def test_half_state_dict_converts_floats_and_keeps_integers() -> None:
    torch = pytest.importorskip("torch")
    state = {
        "w": torch.randn(4, dtype=torch.float32),
        "n": torch.tensor(3, dtype=torch.int64),
    }
    half = weights.half_state_dict(state)
    assert half["w"].dtype == torch.float16
    assert half["n"].dtype == torch.int64 and int(half["n"]) == 3


def test_export_writes_fp16_weights_and_a_manifest(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "unet_tiny.pt"
    original = _tiny_training_checkpoint(source, torch)
    out = tmp_path / "weights"

    entry = weights.export_weights(source, out, file_name="tiny_fp16.pt")

    exported = torch.load(out / "tiny_fp16.pt", map_location="cpu", weights_only=True)
    assert exported["state_dict"]["encoder.conv.weight"].dtype == torch.float16
    assert exported["state_dict"]["encoder.bn.num_batches_tracked"].dtype == torch.int64
    assert exported["epochs"] == 80 and exported["encoder_name"] == "resnet34"
    assert exported["crop_size"] == 512 and exported["smp_version"] == "0.6.0"
    assert "epoch_losses" not in exported
    # fp16 stays close to the original values.
    difference = (
        exported["state_dict"]["encoder.conv.weight"].float()
        - original["state_dict"]["encoder.conv.weight"]
    ).abs()
    assert float(difference.max()) < 1e-2

    assert entry["tensor_sha256"] == tensor_sha256(exported["state_dict"])
    assert entry["tensor_sha256_original"] == tensor_sha256(original["state_dict"])
    assert entry["tensor_sha256"] != entry["tensor_sha256_original"]
    assert entry["file_sha256"] == sha256_file(out / "tiny_fp16.pt")
    assert entry["bytes"] == (out / "tiny_fp16.pt").stat().st_size
    assert (entry["encoder"], entry["epochs"], entry["dtype"]) == (
        "resnet34",
        80,
        "float16",
    )
    assert manifest_entry(out, "tiny_fp16.pt") == entry
    assert not list(out.glob("*.tmp"))


def test_an_exported_file_passes_its_own_verification(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "unet_tiny.pt"
    _tiny_training_checkpoint(source, torch)
    out = tmp_path / "weights"
    entry = weights.export_weights(source, out, file_name="tiny_fp16.pt")
    loaded = torch.load(out / "tiny_fp16.pt", map_location="cpu", weights_only=True)
    # The pin of the real run is not this tiny state's hash: only that check fails.
    problems = repo_weights_problems(
        entry,
        file_sha256=sha256_file(out / "tiny_fp16.pt"),
        fingerprint=checkpoint_fingerprint(loaded),
    )
    assert len(problems) == 1 and "tensor_sha256_original" in problems[0]


def test_a_second_export_keeps_the_first_manifest_entry(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "unet_tiny.pt"
    _tiny_training_checkpoint(source, torch)
    out = tmp_path / "weights"
    weights.export_weights(source, out, file_name="one_fp16.pt")
    weights.export_weights(source, out, file_name="two_fp16.pt")
    assert sorted(read_weights_manifest(out)["weights"]) == [
        "one_fp16.pt",
        "two_fp16.pt",
    ]


def test_a_file_over_100_mb_fails_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    assert weights.MAX_FILE_BYTES == 100_000_000
    source = tmp_path / "unet_tiny.pt"
    _tiny_training_checkpoint(source, torch)
    out = tmp_path / "weights"
    monkeypatch.setattr(weights, "MAX_FILE_BYTES", 100)
    with pytest.raises(WeightsError, match="over the limit"):
        weights.export_weights(source, out, file_name="big_fp16.pt")
    assert list(out.iterdir()) == []


def test_export_refuses_a_file_that_is_not_a_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    source = tmp_path / "nothing.pt"
    torch.save({"model": 1}, source)
    with pytest.raises(WeightsError, match="state_dict"):
        weights.export_weights(source, tmp_path / "weights")
    with pytest.raises(WeightsError, match="directory"):
        torch.save({"state_dict": {}}, source)
        weights.export_weights(source, tmp_path / "weights", file_name="a/b.pt")


def test_an_fp16_file_loads_into_the_fp32_network_and_predicts_alike(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    import numpy as np

    from filament import crop_training as ct

    torch.manual_seed(0)
    network = ct.build_model("resnet34", None).eval()
    source = tmp_path / "unet_resnet34.pt"
    torch.save(
        {
            "state_dict": network.state_dict(),
            "encoder_name": "resnet34",
            "smp_version": ct.smp.__version__,
            "epochs": 80,
            "crop_size": 64,
        },
        source,
    )
    weights.export_weights(source, tmp_path / "weights", file_name="r34_fp16.pt")
    loaded = ct.load_checkpoint(tmp_path / "weights" / "r34_fp16.pt")
    half = ct.model_from_checkpoint(loaded, torch.device("cpu"))
    assert next(half.parameters()).dtype == torch.float32

    gray = np.random.default_rng(0).integers(0, 255, (128, 128), dtype=np.uint8)
    cpu = torch.device("cpu")
    kwargs = {"tile_size": 64, "overlap": 16, "batch_size": 4, "use_amp": False}
    reference = ct.tiled_predictor(network, cpu, **kwargs)(gray)
    converted = ct.tiled_predictor(half, cpu, **kwargs)(gray)
    assert reference.shape == converted.shape == (128, 128)
    assert float(np.abs(reference - converted).max()) < 0.05
