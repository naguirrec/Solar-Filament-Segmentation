"""Check that a U-Net checkpoint is the one the submission is meant to use.

The first U-Net submission scored LB 0.04 because it loaded 3-epoch weights
(Kaggle dataset ``submissionunetv4/unet.pt``) instead of the 12-epoch ones.
This module holds the pure comparison so a notebook can refuse the wrong
file before it predicts anything.

What a checkpoint must match is an :class:`ExpectedCheckpoint`:
:data:`R18_12EP` (the 512 px ResNet-18 weights, the default) and
:data:`R34_CROPS` (the native-resolution ResNet-34 crop run, 80 epochs;
:data:`R34_CROPS_40EP` is the earlier 40-epoch run). :data:`CROP_CHECKPOINTS`
holds the crop models by name, for the notebooks: :func:`crop_checkpoint`.

Nothing here imports torch. :func:`tensor_sha256` only calls
``detach().cpu().contiguous().numpy()`` on tensor-like values, so it works on
a real ``state_dict`` and on test doubles. The notebook builds the
fingerprint (stored ``epochs``, stored encoder, tensor hash) and passes it in
as a plain mapping.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedCheckpoint:
    """What the checkpoint a notebook loads has to be.

    ``tensor_sha256`` is ``None`` while the hash is not known (it is only
    known after the training run). The guard then checks epochs and encoder
    only and says so; see :func:`unpinned_warnings`.

    ``holdout_pq`` is the pooled PQ the training notebook recorded at threshold
    0.5, ``close_size`` 0, ``min_area`` 400. The evaluation notebooks compare
    their own row with it; ``None`` until the run is recorded.
    """

    label: str
    epochs: int
    encoder: str
    tensor_sha256: str | None
    weights_path: str
    holdout_pq: float | None = None


# Output of the Kaggle notebook ``train-unet-r18-12ep-2026-10-31``.
EXPECTED_EPOCHS = 12
EXPECTED_TENSOR_SHA256 = (
    "70c341c95596a62160983e335d176bad14e29e0755d9bda3e66787d7b9079a04"
)
EXPECTED_WEIGHTS_PATH = (
    "/kaggle/input/notebooks/ignacioaguirrecandia/"
    "train-unet-r18-12ep-2026-10-31/unet.pt"
)

R18_12EP = ExpectedCheckpoint(
    label="12-epoch ResNet-18, 512 px",
    epochs=EXPECTED_EPOCHS,
    encoder="resnet18",
    tensor_sha256=EXPECTED_TENSOR_SHA256,
    weights_path=EXPECTED_WEIGHTS_PATH,
)

# Output of the Kaggle notebook ``train-unet-r34-crops-80ep``
# (``notebooks/train_unet_crops.ipynb`` with ``EPOCHS = 80``). The epoch count
# is the default ``EPOCHS`` of that notebook; if the wall-clock guard stops the
# run earlier, the stored epochs differ and the guard says so. The hash is the
# one printed by the 80-epoch run of 2026-11-02 (see docs/RESULTS.md).
R34_CROPS = ExpectedCheckpoint(
    label="ResNet-34 native-resolution crops, 80 epochs",
    epochs=80,
    encoder="resnet34",
    tensor_sha256="3ac77e982e589dabcd56cbfaf1fa54c4adddb52bbeab28757639ef164c67ba5a",
    weights_path=(
        "/kaggle/input/notebooks/ignacioaguirrecandia/"
        "train-unet-r34-crops-80ep/unet_r34_crops.pt"
    ),
    holdout_pq=0.3831,
)

# The earlier 40-epoch run of the same recipe (Kaggle notebook
# ``train-unet-r34-crops``, 2026-11-02, LB 0.32). Kept so a notebook can still
# check those weights with ``expected=R34_CROPS_40EP``.
R34_CROPS_40EP = ExpectedCheckpoint(
    label="ResNet-34 native-resolution crops, 40 epochs",
    epochs=40,
    encoder="resnet34",
    tensor_sha256="c8b3052443ebcac956e0e650d93010790ee7109670f17be6899d065c3257fc0c",
    weights_path=(
        "/kaggle/input/notebooks/ignacioaguirrecandia/"
        "train-unet-r34-crops/unet_r34_crops.pt"
    ),
    holdout_pq=0.3675,
)

# Not trained yet. ``tensor_sha256`` and ``holdout_pq`` are placeholders: after
# the run, copy both from the training notebook's output, and set ``epochs`` to
# the stored number if the wall-clock guard stopped the run early.
R50_CROPS = ExpectedCheckpoint(
    label="ResNet-50 native-resolution crops, 120 epochs",
    epochs=120,
    encoder="resnet50",
    tensor_sha256=None,
    weights_path=(
        "/kaggle/input/notebooks/ignacioaguirrecandia/"
        "train-unet-r50-crops/unet_r50_crops.pt"
    ),
)

EFFB3_CROPS = ExpectedCheckpoint(
    label="EfficientNet-B3 native-resolution crops, 100 epochs",
    epochs=100,
    encoder="efficientnet-b3",
    tensor_sha256=None,
    weights_path=(
        "/kaggle/input/notebooks/ignacioaguirrecandia/"
        "train-unet-effb3-crops/unet_effb3_crops.pt"
    ),
)

# The recipe v2 runs of docs/ROADMAP_V2.md (D1 and D2). Not trained yet, so the
# same placeholders apply. ``state_dict`` holds the weight average of the last
# epoch.
R34_V2 = ExpectedCheckpoint(
    label="ResNet-34 crops, recipe v2 (AdamW, cosine, EMA), 80 epochs",
    epochs=80,
    encoder="resnet34",
    tensor_sha256=None,
    weights_path=(
        "/kaggle/input/notebooks/ignacioaguirrecandia/train-unet-r34-v2/unet_r34_v2.pt"
    ),
)

# D2, trained on Kaggle on 2026-11-03 (80 of 80 epochs, status complete). The hash
# is the one the training notebook printed, recomputed from the checkpoint file.
R34_V2_AGREE = ExpectedCheckpoint(
    label="ResNet-34 crops, recipe v2 with agreement weights, 80 epochs",
    epochs=80,
    encoder="resnet34",
    tensor_sha256="3ae546d82f1a88dfad210d90287035d843c6db224b5619a2f31c5a2f54610364",
    holdout_pq=0.3875,
    weights_path=(
        "/kaggle/input/notebooks/ignacioaguirrecandia/"
        "train-unet-r34-v2-agree/unet_r34_v2_agree.pt"
    ),
)

# The same names as ``filament.crop_config.CROP_MODELS``.
CROP_CHECKPOINTS: dict[str, ExpectedCheckpoint] = {
    "r34": R34_CROPS,
    "r50": R50_CROPS,
    "effb3": EFFB3_CROPS,
    "r34_v2": R34_V2,
    "r34_v2_agree": R34_V2_AGREE,
}

# Hash of the 3-epoch weights that produced LB 0.04 (submissionunetv4/unet.pt).
THREE_EPOCH_TENSOR_SHA256 = (
    "5ed568146b35ee9380f417f8ae05ec234aaab16d8364bd1f9532348cfc288c0c"
)

# Anything longer than this is not a real epoch count or digest; it is cut
# before it goes into a message.
_MAX_SHOWN = 80
# No run has more than this many epochs. Larger values are treated as noise.
_MAX_EPOCHS = 1_000_000
# float() on a longer string cannot be a sane count.
_MAX_EPOCH_TEXT = 64


class CheckpointMismatchError(RuntimeError):
    """The loaded checkpoint is not the expected U-Net."""


def crop_checkpoint(name: str) -> ExpectedCheckpoint:
    """What the named crop model's file must match, or ``ValueError`` listing the names."""
    if name not in CROP_CHECKPOINTS:
        raise ValueError(
            f"unknown crop model {name!r}, expected one of {sorted(CROP_CHECKPOINTS)}"
        )
    return CROP_CHECKPOINTS[name]


def tensor_sha256(state_dict: Mapping[str, object]) -> str:
    """Hash the tensors of a ``state_dict`` so a re-zip of the same weights matches.

    For each key in sorted order: the key, then for a tensor its shape, its
    dtype string and its bytes (``detach().cpu().contiguous().numpy()``), or
    ``repr(value)`` for anything else, each part ended by a NUL byte. This is
    the function the notebooks used to carry inline; the digests are
    unchanged (the R18 pin ``70c341c9…9a04`` was made with it).
    """
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        digest.update(str(key).encode())
        digest.update(b"\0")
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint: Mapping[str, object]) -> dict[str, object]:
    """The fields :func:`checkpoint_problems` compares, read from a loaded file.

    ``epochs`` and ``encoder`` are what the file stores (``epochs``,
    ``encoder_name``); ``tensor_sha256`` is computed from its ``state_dict``.
    A missing field is ``None``.
    """
    state = checkpoint.get("state_dict")
    return {
        "epochs": checkpoint.get("epochs"),
        "encoder": checkpoint.get("encoder_name"),
        "tensor_sha256": tensor_sha256(state) if isinstance(state, Mapping) else None,
    }


def checkpoint_problems(
    fingerprint: Mapping[str, object],
    *,
    expected: ExpectedCheckpoint = R18_12EP,
    expected_epochs: int | None = None,
    expected_tensor_sha256: str | None = None,
) -> list[str]:
    """Return one message per way ``fingerprint`` differs from ``expected``.

    ``fingerprint`` has the keys ``epochs``, ``tensor_sha256`` and, optionally,
    ``encoder``. A missing ``epochs`` or ``tensor_sha256`` key, ``None``, or the
    text ``not stored`` counts as missing. The ``encoder`` key is checked only
    when it is present in the mapping (older callers do not pass it); present
    but ``None`` or ``not stored`` is a problem, and so is any other name.

    When ``expected.tensor_sha256`` is ``None`` the hash is not compared (see
    :func:`unpinned_warnings`). ``expected_epochs`` and
    ``expected_tensor_sha256`` override those two fields of ``expected``.
    An empty list means the checkpoint matches.
    """
    want_epochs = expected.epochs if expected_epochs is None else expected_epochs
    want_hash = (
        expected.tensor_sha256
        if expected_tensor_sha256 is None
        else expected_tensor_sha256
    )
    problems: list[str] = []

    epochs = fingerprint.get("epochs")
    if _is_missing(epochs):
        problems.append("epochs is not stored in the checkpoint")
    else:
        found = _as_epoch_count(epochs)
        if found is None:
            problems.append(f"epochs={_short(epochs, quote=True)} is not an integer")
        elif found != want_epochs:
            problems.append(f"epochs={found}, expected {want_epochs}")

    if "encoder" in fingerprint:
        encoder = fingerprint["encoder"]
        if _is_missing(encoder):
            problems.append("encoder is not stored in the checkpoint")
        elif str(encoder).strip() != expected.encoder:
            problems.append(f"encoder={_short(encoder)}, expected {expected.encoder}")

    raw_digest = fingerprint.get("tensor_sha256")
    if _is_missing(raw_digest):
        problems.append("tensor_sha256 is not available")
    elif want_hash is not None:
        digest = str(raw_digest).strip().lower()
        if digest != want_hash.strip().lower():
            note = (
                " (the 3-epoch weights)" if digest == THREE_EPOCH_TENSOR_SHA256 else ""
            )
            problems.append(
                f"tensor_sha256={_short(digest)}{note}, expected {want_hash}"
            )
    return problems


def unpinned_warnings(
    fingerprint: Mapping[str, object],
    *,
    expected: ExpectedCheckpoint = R18_12EP,
) -> list[str]:
    """Warn when ``expected`` has no pinned hash, so the hash was not checked.

    Epochs and encoder can match while the weights are still the wrong file,
    for example an earlier run of the same recipe. The message carries the
    computed hash to pin. Empty when the hash is pinned.
    """
    if expected.tensor_sha256 is not None:
        return []
    computed = fingerprint.get("tensor_sha256")
    shown = "not available" if _is_missing(computed) else _short(computed)
    return [
        (
            f"tensor_sha256 is not pinned for '{expected.label}': epochs and "
            "encoder were checked, the weights themselves were not. "
            f"This checkpoint's tensor_sha256 is {shown}. "
            "[VERIFY] after the training run, compare it with the hash printed "
            "there and pin it in filament.checkpoint."
        )
    ]


DEFAULT_MISMATCH_REMEDY = (
    "Set WEIGHTS_PATH to that file, or set ALLOW_UNVERIFIED_WEIGHTS = True to "
    "predict with this checkpoint anyway."
)


def verify_checkpoint(
    fingerprint: Mapping[str, object],
    *,
    allow_unverified: bool = False,
    expected: ExpectedCheckpoint = R18_12EP,
    expected_epochs: int | None = None,
    expected_tensor_sha256: str | None = None,
    remedy: str = DEFAULT_MISMATCH_REMEDY,
) -> list[str]:
    """Raise :class:`CheckpointMismatchError` unless the fingerprint matches.

    With ``allow_unverified=True`` the mismatch is not raised; the problems
    are returned so the caller can print them. Returns ``[]`` on a match.
    ``remedy`` is the last sentence of the error: a notebook that has no
    ``ALLOW_UNVERIFIED_WEIGHTS`` setting names what it does offer.
    """
    problems = checkpoint_problems(
        fingerprint,
        expected=expected,
        expected_epochs=expected_epochs,
        expected_tensor_sha256=expected_tensor_sha256,
    )
    if problems and not allow_unverified:
        detail = "; ".join(problems)
        raise CheckpointMismatchError(
            f"Wrong U-Net checkpoint: {detail}. Expected '{expected.label}' at "
            f"{expected.weights_path}. {remedy}"
        )
    return problems


def verify_and_report(
    fingerprint: Mapping[str, object],
    *,
    allow_unverified: bool = False,
    expected: ExpectedCheckpoint = R18_12EP,
    expected_epochs: int | None = None,
    expected_tensor_sha256: str | None = None,
    emit: Callable[[str], None] = print,
    remedy: str = DEFAULT_MISMATCH_REMEDY,
) -> list[str]:
    """:func:`verify_checkpoint`, then print what the guard did and did not check.

    Prints ``WARNING:`` lines for every tolerated mismatch and for an unpinned
    hash, or one ``checkpoint verified`` line when everything matched and the
    hash was pinned. Raises exactly when :func:`verify_checkpoint` does.
    """
    problems = verify_checkpoint(
        fingerprint,
        allow_unverified=allow_unverified,
        expected=expected,
        expected_epochs=expected_epochs,
        expected_tensor_sha256=expected_tensor_sha256,
        remedy=remedy,
    )
    pinned = (
        expected.tensor_sha256
        if expected_tensor_sha256 is None
        else expected_tensor_sha256
    )
    warnings: list[str] = []
    if problems:
        warnings.append(f"unverified checkpoint: {'; '.join(problems)}")
    if pinned is None:
        warnings.extend(unpinned_warnings(fingerprint, expected=expected))
    for line in warnings:
        emit(f"WARNING: {line}")
    if not warnings:
        emit(
            f"checkpoint verified: '{expected.label}' epochs="
            f"{expected.epochs if expected_epochs is None else expected_epochs} "
            f"encoder={expected.encoder} tensor_sha256={pinned}"
        )
    return problems


def _as_epoch_count(value: object) -> int | None:
    """Return ``value`` as an int, or ``None`` if it is not a whole number.

    Accepts ints, integer-valued floats such as ``12.0``, and text such as
    ``"12"`` or ``"12.0"``. Rejects ``12.5``, ``nan``, ``inf``, bools
    (``True`` or ``"True"``), negative counts, counts above one million, and
    text too long to be a count (``"1e400"``, a 5000-digit string). Never
    raises.
    """
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, int):
            number = value
        else:
            text = str(value).strip()
            if len(text) > _MAX_EPOCH_TEXT:
                return None
            as_float = float(text)
            if not as_float.is_integer():  # also False for nan and inf
                return None
            number = int(as_float)
    except (ValueError, OverflowError, TypeError):
        return None
    # "-0" parses to 0. A negative count is not a count.
    if not 0 <= number <= _MAX_EPOCHS:
        return None
    return number


def _short(value: object, *, quote: bool = False) -> str:
    """``value`` as text cut to a length a message can carry. Never raises.

    ``quote=True`` uses ``repr`` (a string keeps its quotes); otherwise a
    string is shown as it is.
    """
    try:
        text = repr(value) if quote or not isinstance(value, str) else value
    except (ValueError, TypeError):
        text = f"<{type(value).__name__}>"
    if len(text) > _MAX_SHOWN:
        return f"{text[:_MAX_SHOWN]}…({len(text)} chars)"
    return text


def _is_missing(value: object) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip() in ("", "not stored")
    )
