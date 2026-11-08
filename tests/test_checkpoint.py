"""Checkpoint fingerprint check. Pure Python: no torch."""

from __future__ import annotations

import dataclasses
from pathlib import PurePosixPath

import pytest

from filament.checkpoint import (
    EXPECTED_EPOCHS,
    EXPECTED_TENSOR_SHA256,
    THREE_EPOCH_TENSOR_SHA256,
    CheckpointMismatchError,
    ExpectedCheckpoint,
    checkpoint_problems,
    verify_checkpoint,
)

GOOD = {"epochs": 12, "tensor_sha256": EXPECTED_TENSOR_SHA256}


def test_expected_constants() -> None:
    assert EXPECTED_EPOCHS == 12
    assert EXPECTED_TENSOR_SHA256 == (
        "70c341c95596a62160983e335d176bad14e29e0755d9bda3e66787d7b9079a04"
    )
    assert THREE_EPOCH_TENSOR_SHA256 == (
        "5ed568146b35ee9380f417f8ae05ec234aaab16d8364bd1f9532348cfc288c0c"
    )


@pytest.mark.parametrize("epochs", [12.0, "12.0", " 12 "])
def test_integer_valued_epochs_pass(epochs: object) -> None:
    good = {"epochs": epochs, "tensor_sha256": EXPECTED_TENSOR_SHA256}
    assert checkpoint_problems(good) == []


@pytest.mark.parametrize("epochs", [12.5, True, "True", "nan", "inf", float("nan")])
def test_non_integer_epochs_fail(epochs: object) -> None:
    problems = checkpoint_problems(
        {"epochs": epochs, "tensor_sha256": EXPECTED_TENSOR_SHA256}
    )
    assert problems
    assert "epochs=" in problems[0]


def test_hash_is_normalized_for_match_and_note() -> None:
    padded = f"  {EXPECTED_TENSOR_SHA256.upper()}\n"
    assert checkpoint_problems({"epochs": 12, "tensor_sha256": padded}) == []
    three = f" {THREE_EPOCH_TENSOR_SHA256.upper()} "
    problems = checkpoint_problems({"epochs": 12, "tensor_sha256": three})
    assert len(problems) == 1
    assert "(the 3-epoch weights)" in problems[0]


def test_matching_fingerprint_passes() -> None:
    assert verify_checkpoint(GOOD) == []
    # The notebook prints values as text.
    assert (
        checkpoint_problems(
            {"epochs": "12", "tensor_sha256": EXPECTED_TENSOR_SHA256.upper()}
        )
        == []
    )


def test_three_epoch_checkpoint_fails() -> None:
    bad = {"epochs": 3, "tensor_sha256": THREE_EPOCH_TENSOR_SHA256}
    with pytest.raises(CheckpointMismatchError, match="epochs=3") as info:
        verify_checkpoint(bad)
    assert "3-epoch" in str(info.value)
    assert len(checkpoint_problems(bad)) == 2


def test_wrong_hash_with_right_epochs_fails() -> None:
    with pytest.raises(CheckpointMismatchError, match="tensor_sha256"):
        verify_checkpoint({"epochs": 12, "tensor_sha256": THREE_EPOCH_TENSOR_SHA256})
    with pytest.raises(CheckpointMismatchError, match="tensor_sha256"):
        verify_checkpoint({"epochs": 12, "tensor_sha256": "0" * 64})


def test_wrong_epochs_with_right_hash_fails() -> None:
    with pytest.raises(CheckpointMismatchError, match="epochs=11"):
        verify_checkpoint({"epochs": 11, "tensor_sha256": EXPECTED_TENSOR_SHA256})


def test_the_last_sentence_of_the_error_can_be_replaced_but_it_still_raises() -> None:
    bad = {"epochs": 3, "tensor_sha256": THREE_EPOCH_TENSOR_SHA256}
    with pytest.raises(CheckpointMismatchError, match="ALLOW_UNVERIFIED_WEIGHTS"):
        verify_checkpoint(bad)  # the default text is unchanged
    with pytest.raises(CheckpointMismatchError, match="no override here") as info:
        verify_checkpoint(bad, remedy="no override here")
    assert "ALLOW_UNVERIFIED_WEIGHTS" not in str(info.value)
    with pytest.raises(CheckpointMismatchError, match="no override here"):
        verify_and_report(bad, remedy="no override here", emit=lambda _line: None)


def test_override_flag_lets_a_mismatch_through() -> None:
    bad = {"epochs": 3, "tensor_sha256": THREE_EPOCH_TENSOR_SHA256}
    problems = verify_checkpoint(bad, allow_unverified=True)
    assert len(problems) == 2


@pytest.mark.parametrize(
    "fingerprint",
    [
        {},
        {"epochs": 12},
        {"tensor_sha256": EXPECTED_TENSOR_SHA256},
        {"epochs": "not stored", "tensor_sha256": EXPECTED_TENSOR_SHA256},
        {"epochs": 12, "tensor_sha256": "not stored"},
        {"epochs": None, "tensor_sha256": None},
        {"epochs": "twelve", "tensor_sha256": EXPECTED_TENSOR_SHA256},
    ],
)
def test_missing_or_unreadable_fields_fail(fingerprint: dict) -> None:
    assert checkpoint_problems(fingerprint)
    with pytest.raises(CheckpointMismatchError):
        verify_checkpoint(fingerprint)
    assert verify_checkpoint(fingerprint, allow_unverified=True)


# --- configurable expectation (R18 default, R34 crops) -----------------------

import hashlib

from filament.checkpoint import (
    CROP_CHECKPOINTS,
    R18_12EP,
    R34_CROPS,
    R34_CROPS_40EP,
    checkpoint_fingerprint,
    crop_checkpoint,
    tensor_sha256,
    unpinned_warnings,
    verify_and_report,
)
from filament.crop_config import CROP_MODELS, crop_model_config


class _FakeTensor:
    """Tensor-like: detach/cpu/contiguous/numpy, shape and dtype."""

    def __init__(self, array) -> None:
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self._array


def test_default_expectation_is_r18() -> None:
    assert R18_12EP.encoder == "resnet18"
    assert R18_12EP.tensor_sha256 == EXPECTED_TENSOR_SHA256
    assert checkpoint_problems(GOOD) == []
    # Old callers pass no encoder key: it is not checked.
    assert checkpoint_problems({**GOOD, "encoder": "resnet18"}) == []


def test_r18_guard_rejects_r34_encoder() -> None:
    problems = checkpoint_problems({**GOOD, "encoder": "resnet34"})
    assert any("encoder=resnet34, expected resnet18" in p for p in problems)


def test_r34_expectation_matches_encoder_and_epochs() -> None:
    fingerprint = {
        "epochs": 80,
        "encoder": "resnet34",
        "tensor_sha256": R34_CROPS.tensor_sha256,
    }
    assert checkpoint_problems(fingerprint, expected=R34_CROPS) == []
    # The R18 weights fail the R34 guard on epochs, encoder and hash.
    problems = checkpoint_problems({**GOOD, "encoder": "resnet18"}, expected=R34_CROPS)
    assert len(problems) == 3


def test_r34_hash_is_pinned_and_enforced() -> None:
    assert R34_CROPS.tensor_sha256 == (
        "3ac77e982e589dabcd56cbfaf1fa54c4adddb52bbeab28757639ef164c67ba5a"
    )
    assert R34_CROPS.epochs == 80
    assert R34_CROPS.weights_path.endswith(
        "/train-unet-r34-crops-80ep/unet_r34_crops.pt"
    )
    good = {
        "epochs": 80,
        "encoder": "resnet34",
        "tensor_sha256": R34_CROPS.tensor_sha256,
    }
    assert verify_checkpoint(good, expected=R34_CROPS) == []
    assert unpinned_warnings(good, expected=R34_CROPS) == []
    other = {**good, "tensor_sha256": "cd" * 32}
    with pytest.raises(CheckpointMismatchError):
        verify_checkpoint(other, expected=R34_CROPS)


def test_r34_40_epoch_weights_are_still_documented_but_not_the_default() -> None:
    assert R34_CROPS_40EP.tensor_sha256 == (
        "c8b3052443ebcac956e0e650d93010790ee7109670f17be6899d065c3257fc0c"
    )
    assert R34_CROPS_40EP.epochs == 40
    assert R34_CROPS_40EP.weights_path.endswith(
        "/train-unet-r34-crops/unet_r34_crops.pt"
    )
    old = {
        "epochs": 40,
        "encoder": "resnet34",
        "tensor_sha256": R34_CROPS_40EP.tensor_sha256,
    }
    assert verify_checkpoint(old, expected=R34_CROPS_40EP) == []
    # The default guard refuses the 40-epoch weights on epochs and hash.
    problems = checkpoint_problems(old, expected=R34_CROPS)
    assert len(problems) == 2
    with pytest.raises(CheckpointMismatchError):
        verify_checkpoint(old, expected=R34_CROPS)


def test_unpinned_hash_warns_but_verifies() -> None:
    unpinned = dataclasses.replace(R34_CROPS, tensor_sha256=None)
    fingerprint = {"epochs": 80, "encoder": "resnet34", "tensor_sha256": "cd" * 32}
    assert verify_checkpoint(fingerprint, expected=unpinned) == []
    warnings = unpinned_warnings(fingerprint, expected=unpinned)
    assert len(warnings) == 1 and "cd" * 32 in warnings[0]
    assert unpinned_warnings(GOOD, expected=R18_12EP) == []


def test_crop_checkpoints_match_the_crop_models_by_name() -> None:
    assert set(CROP_CHECKPOINTS) == set(CROP_MODELS)
    assert crop_checkpoint("r34") is R34_CROPS
    for name, expected in CROP_CHECKPOINTS.items():
        assert expected.encoder == crop_model_config(name).encoder_name
        assert expected.epochs == crop_model_config(name).epochs


def test_crop_checkpoint_paths_are_distinct() -> None:
    paths = [PurePosixPath(e.weights_path) for e in CROP_CHECKPOINTS.values()]
    assert len({path.name for path in paths}) == len(paths)
    assert len({path.parent.name for path in paths}) == len(paths)


def test_unknown_crop_checkpoint_is_refused_and_the_names_are_listed() -> None:
    with pytest.raises(ValueError, match="expected one of.*r34"):
        crop_checkpoint("resnet50")


@pytest.mark.parametrize("name", ["r50", "effb3"])
def test_new_encoders_are_unpinned_placeholders(name: str) -> None:
    expected = crop_checkpoint(name)
    assert expected.tensor_sha256 is None
    assert expected.holdout_pq is None
    good = {
        "epochs": expected.epochs,
        "encoder": expected.encoder,
        "tensor_sha256": "ab",
    }
    # An unpinned entry verifies on epochs and encoder and says the hash was not checked.
    assert verify_checkpoint(good, expected=expected) == []
    assert "ab" in unpinned_warnings(good, expected=expected)[0]


@pytest.mark.parametrize("name", ["r50", "effb3"])
def test_unpinned_entry_still_refuses_another_encoder_or_epochs(name: str) -> None:
    expected = crop_checkpoint(name)
    r34_file = {
        "epochs": R34_CROPS.epochs,
        "encoder": R34_CROPS.encoder,
        "tensor_sha256": R34_CROPS.tensor_sha256,
    }
    with pytest.raises(CheckpointMismatchError):
        verify_checkpoint(r34_file, expected=expected)
    short_run = {
        "epochs": expected.epochs - 1,
        "encoder": expected.encoder,
        "tensor_sha256": "ab",
    }
    with pytest.raises(CheckpointMismatchError):
        verify_checkpoint(short_run, expected=expected)


def test_r34_records_the_pq_the_notebooks_cross_check() -> None:
    assert R34_CROPS.holdout_pq == 0.3831
    assert R34_CROPS_40EP.holdout_pq == 0.3675


def test_pinned_hash_is_enforced_for_custom_expectation() -> None:
    pinned = ExpectedCheckpoint("x", 5, "resnet34", "aa" * 32, "/p")
    bad = {"epochs": 5, "encoder": "resnet34", "tensor_sha256": "bb" * 32}
    with pytest.raises(CheckpointMismatchError, match="Expected 'x' at /p"):
        verify_checkpoint(bad, expected=pinned)
    assert verify_checkpoint(bad, expected=pinned, allow_unverified=True)


def test_verify_and_report_messages() -> None:
    lines: list[str] = []
    verify_and_report(GOOD, emit=lines.append)
    assert len(lines) == 1 and lines[0].startswith("checkpoint verified")

    lines.clear()
    fingerprint = {"epochs": 80, "encoder": "resnet34", "tensor_sha256": "cd" * 32}
    verify_and_report(
        fingerprint,
        expected=dataclasses.replace(R34_CROPS, tensor_sha256=None),
        emit=lines.append,
    )
    assert (
        len(lines) == 1 and lines[0].startswith("WARNING:") and "not pinned" in lines[0]
    )

    lines.clear()
    verify_and_report({**GOOD, "epochs": 3}, allow_unverified=True, emit=lines.append)
    assert any("unverified checkpoint" in line for line in lines)
    with pytest.raises(CheckpointMismatchError):
        verify_and_report({**GOOD, "epochs": 3}, emit=lines.append)


def test_tensor_sha256_matches_documented_recipe_and_ignores_order() -> None:
    import numpy as np

    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    state = {"b": _FakeTensor(array), "a": 7}
    expected = hashlib.sha256()
    expected.update(b"a\0" + repr(7).encode() + b"\0")
    expected.update(b"b\0" + str((2, 3)).encode() + b"\0")
    expected.update(str(array.dtype).encode() + b"\0" + array.tobytes() + b"\0")
    assert tensor_sha256(state) == expected.hexdigest()
    assert tensor_sha256(dict(reversed(list(state.items())))) == expected.hexdigest()
    assert tensor_sha256({"a": 8}) != tensor_sha256({"a": 7})


def test_checkpoint_fingerprint_reads_fields() -> None:
    fp = checkpoint_fingerprint(
        {"epochs": 40, "encoder_name": "resnet34", "state_dict": {"a": 1}}
    )
    assert fp["epochs"] == 40 and fp["encoder"] == "resnet34"
    assert fp["tensor_sha256"] == tensor_sha256({"a": 1})
    empty = checkpoint_fingerprint({})
    assert empty == {"epochs": None, "encoder": None, "tensor_sha256": None}
    assert checkpoint_problems(empty, expected=R34_CROPS)  # all three missing


def test_hostile_values_do_not_crash_or_flood() -> None:
    long = "x" * 5000
    problems = checkpoint_problems(
        {"epochs": long, "encoder": long, "tensor_sha256": long}
    )
    assert problems and all(len(p) < 400 for p in problems)
    assert checkpoint_problems({"epochs": "1e400", "tensor_sha256": "a"})
    assert checkpoint_problems({"epochs": -1, "tensor_sha256": "a"})


ABSURD_EPOCHS = [
    "1e400",
    "-1e400",
    "1e309",
    "9" * 5000,
    "-" + "9" * 5000,
    "1" + "0" * 400 + ".0",
    10**400,
    10**5000,  # repr() of this int raises ValueError (digit limit)
    -(10**5000),
    "-0",
    "-0.0",
    "+0",
    float("inf"),
    float("-inf"),
    1e308,
    "0x10",
    " ",
    "\x00",
    "1_0",
    ["12"],
    {"epochs": 12},
    object(),
]


@pytest.mark.parametrize(
    "epochs",
    ABSURD_EPOCHS,
    ids=[f"case{index}" for index in range(len(ABSURD_EPOCHS))],  # repr() can raise
)
def test_absurd_epochs_never_raise(epochs: object) -> None:
    from filament.checkpoint import _as_epoch_count

    assert _as_epoch_count(epochs) in {None, 0} or isinstance(
        _as_epoch_count(epochs), int
    )
    for expected in (R18_12EP, R34_CROPS):
        fingerprint = {
            "epochs": epochs,
            "encoder": "resnet18",
            "tensor_sha256": EXPECTED_TENSOR_SHA256,
        }
        problems = checkpoint_problems(fingerprint, expected=expected)
        assert problems  # none of these is 12 or 80
        assert all(len(problem) < 400 for problem in problems)
        with pytest.raises(CheckpointMismatchError):
            verify_checkpoint(fingerprint, expected=expected)
        lines: list[str] = []
        verify_and_report(
            fingerprint, expected=expected, allow_unverified=True, emit=lines.append
        )
        assert lines


def test_minus_zero_is_zero_epochs_not_a_crash() -> None:
    from filament.checkpoint import _as_epoch_count

    assert _as_epoch_count("-0") == 0
    assert _as_epoch_count("-0.0") == 0
    assert checkpoint_problems({"epochs": "-0", "tensor_sha256": "a"}) == [
        "epochs=0, expected 12",
        f"tensor_sha256=a, expected {EXPECTED_TENSOR_SHA256}",
    ]


@pytest.mark.parametrize(
    ("epochs", "expected"),
    [(12, 12), ("12", 12), (" 12.0 ", 12), (12.0, 12), (0, 0), (1_000_000, 1_000_000)],
)
def test_plausible_epochs_still_parse(epochs: object, expected: int) -> None:
    from filament.checkpoint import _as_epoch_count

    assert _as_epoch_count(epochs) == expected


@pytest.mark.parametrize("epochs", [-1, 1_000_001, "1000001", 12.5, True, "nan"])
def test_implausible_epochs_are_not_a_count(epochs: object) -> None:
    from filament.checkpoint import _as_epoch_count

    assert _as_epoch_count(epochs) is None


def test_d2_is_pinned_to_its_kaggle_run() -> None:
    from filament.checkpoint import R34_V2_AGREE

    assert crop_checkpoint("r34_v2_agree") is R34_V2_AGREE
    assert R34_V2_AGREE.tensor_sha256 == (
        "3ae546d82f1a88dfad210d90287035d843c6db224b5619a2f31c5a2f54610364"
    )
    assert R34_V2_AGREE.epochs == 80 and R34_V2_AGREE.encoder == "resnet34"
    assert R34_V2_AGREE.holdout_pq == 0.3875
