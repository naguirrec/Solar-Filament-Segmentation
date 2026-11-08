"""Torch side of the crop run. Skipped when torch or smp is not installed (CI)."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from filament import crop_training as ct
from filament.checkpoint import (
    R34_CROPS,
    checkpoint_fingerprint,
    checkpoint_problems,
    crop_checkpoint,
    tensor_sha256,
)
from filament.crop_config import (
    CROP_MODELS,
    CropTrainConfig,
    crop_model_config,
    scheduled_lr,
)
from filament.crop_data import agreement_mask

ROOT = Path(__file__).resolve().parents[1]


def _legacy_tensor_sha256(state_dict: dict) -> str:
    """The function ``notebooks/infer_unet.ipynb`` carried before it moved to src."""
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


def test_tensor_sha256_equals_the_legacy_notebook_function() -> None:
    torch.manual_seed(0)
    model = ct.build_model("resnet18", None)
    state = model.state_dict()  # includes int64 num_batches_tracked
    assert any(v.dtype == torch.int64 for v in state.values())
    assert tensor_sha256(state) == _legacy_tensor_sha256(state)
    mixed = {"a": torch.arange(6, dtype=torch.float16).reshape(2, 3), "b": 3}
    assert tensor_sha256(mixed) == _legacy_tensor_sha256(mixed)


def test_normalize_gray_matches_the_numpy_recipe() -> None:
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(2, 8, 8), dtype=np.uint8)
    out = ct.normalize_gray(torch.from_numpy(gray)).numpy()
    scaled = gray.astype(np.float32) / 255.0
    rgb = np.stack((scaled,) * 3, axis=1)
    mean = np.array(ct.IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(ct.IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    assert out.shape == (2, 3, 8, 8)
    assert np.allclose(out, (rgb - mean) / std, atol=1e-6)


def test_tiled_prediction_equals_whole_frame_for_a_pointwise_model() -> None:
    # A 1x1 convolution has no spatial context, so tiling must not change it.
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, size=(100, 90), dtype=np.uint8)
    device = torch.device("cpu")
    tiled = ct.predict_probability(
        model, gray, device, tile_size=32, overlap=8, batch_size=3, use_amp=False
    )
    with torch.no_grad():
        whole = torch.sigmoid(model(ct.normalize_gray(torch.from_numpy(gray)[None])))[
            0, 0
        ].numpy()
    assert tiled.shape == gray.shape
    assert np.allclose(tiled, whole, atol=1e-6)


def _tiny_data(count: int = 3, side: int = 64):
    rng = np.random.default_rng(0)
    images = [
        rng.integers(0, 256, size=(side, side), dtype=np.uint8) for _ in range(count)
    ]
    masks = []
    for image in images:
        mask = np.zeros_like(image)
        mask[20:30, 10:50] = 1
        masks.append(mask)
    return images, masks


def _tiny_config(**changes) -> CropTrainConfig:
    values = {
        "encoder_name": "resnet34",
        "encoder_weights": None,
        "epochs": 3,
        "crops_per_epoch": 4,
        "crop_size": 32,
        "batch_size": 2,
        "use_amp": False,
    }
    values.update(changes)
    return CropTrainConfig(**values)


def test_training_runs_and_stores_completed_epochs(tmp_path: Path) -> None:
    images, masks = _tiny_data()
    path = tmp_path / "unet_r34_crops.pt"
    model, result = ct.train_model(
        images,
        masks,
        _tiny_config(epochs=2),
        torch.device("cpu"),
        checkpoint_path=path,
        extra_metadata={"tile_size": 32},
        log=lambda _: None,
    )
    assert result.epochs_completed == 2 and not result.stopped_early
    assert len(result.epoch_losses) == len(result.epoch_seconds) == 2
    assert all(np.isfinite(result.epoch_losses))
    stored = ct.load_checkpoint(path)  # weights_only=True must accept the metadata
    assert stored["epochs"] == 2
    assert stored["status"] == "complete"
    assert stored["encoder_name"] == "resnet34"
    assert stored["crop_size"] == 32
    assert stored["training_resolution"] == "native"
    assert stored["native_shape"] == [64, 64]
    assert stored["seed"] == 0 and stored["batch_size"] == 2
    assert stored["loss"] == "dice+bce"
    assert stored["tile_size"] == 32
    assert stored["smp_version"] == ct.smp.__version__
    fingerprint = checkpoint_fingerprint(stored)
    assert fingerprint["tensor_sha256"] == tensor_sha256(model.state_dict())
    expected = _expected(2, tensor_sha256(model.state_dict()))
    assert checkpoint_problems(fingerprint, expected=expected) == []


def _expected(epochs: int, digest: str):
    """R34_CROPS with the epochs and hash of the tiny synthetic model."""
    return dataclasses.replace(R34_CROPS, epochs=epochs, tensor_sha256=digest)


def test_wall_clock_guard_stops_and_records_completed_epochs(tmp_path: Path) -> None:
    images, masks = _tiny_data()
    ticks = iter(range(0, 10**6, 1000))  # every clock() call advances 1000 s
    lines: list[str] = []
    path = tmp_path / "unet.pt"
    _model, result = ct.train_model(
        images,
        masks,
        _tiny_config(epochs=5, max_train_hours=0.5),  # 1800 s
        torch.device("cpu"),
        checkpoint_path=path,
        log=lines.append,
        clock=lambda: float(next(ticks)),
    )
    assert result.stopped_early
    assert result.epochs_completed == 1  # the second epoch would pass 1800 s
    stored = ct.load_checkpoint(path)
    assert stored["epochs"] == 1
    assert stored["epochs_requested"] == 5
    assert stored["stopped_early"] is True
    assert stored["status"] == "stopped_early"
    note = [line for line in lines if line.startswith("NOTE: stopping")]
    assert len(note) == 1
    assert "Completed epochs: 1 of 5" in note[0] and "epochs=1" in note[0]
    # The guard of the inference notebook sees the shortfall.
    problems = checkpoint_problems(checkpoint_fingerprint(stored), expected=R34_CROPS)
    assert problems and "epochs=1, expected 80" in problems[0]


def test_same_seed_gives_the_same_weights() -> None:
    images, masks = _tiny_data()
    config = _tiny_config(epochs=1)
    first, _ = ct.train_model(images, masks, config, torch.device("cpu"), log=print)
    second, _ = ct.train_model(images, masks, config, torch.device("cpu"), log=print)
    assert tensor_sha256(first.state_dict()) == tensor_sha256(second.state_dict())


@pytest.mark.parametrize("name", sorted(CROP_MODELS))
def test_every_crop_model_round_trips_through_its_checkpoint(
    name: str, tmp_path: Path
) -> None:
    """The encoder exists in smp, trains a step, and its file reloads and matches."""
    preset = crop_model_config(name)
    config = dataclasses.replace(
        preset,
        encoder_weights=None,
        epochs=1 + preset.warmup_epochs,
        crops_per_epoch=2,
        crop_size=64,
        batch_size=2,
        use_amp=False,
    )
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (96, 96), dtype=np.uint8)]
    # A stored agreement weight, or the plain 1: the kind this recipe trains on.
    positive = 255 if preset.agreement_weights else 1
    masks = [(images[0] > 128).astype(np.uint8) * positive]
    path = tmp_path / "weights.pt"
    model, _ = ct.train_model(
        images,
        masks,
        config,
        torch.device("cpu"),
        checkpoint_path=path,
        log=lambda _: None,
    )
    stored = ct.load_checkpoint(path)
    fingerprint = checkpoint_fingerprint(stored)
    assert fingerprint["encoder"] == crop_checkpoint(name).encoder
    assert fingerprint["tensor_sha256"] == tensor_sha256(model.state_dict())
    rebuilt = ct.model_from_checkpoint(stored, torch.device("cpu"))
    assert tensor_sha256(rebuilt.state_dict()) == fingerprint["tensor_sha256"]


# --- recipe v2 -----------------------------------------------------------------


def _v2_config(**changes) -> CropTrainConfig:
    return _tiny_config(
        **{
            "learning_rate": 2e-4,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
            "lr_schedule": "cosine",
            "warmup_epochs": 1,
            "grad_clip_norm": 1.0,
            "ema_decay": 0.9,
            **changes,
        }
    )


def test_optimizer_settings_reach_the_optimizer() -> None:
    model = torch.nn.Conv2d(1, 1, 1)
    adamw = ct.build_optimizer(model, _v2_config())
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.param_groups[0]["lr"] == 2e-4
    # AdamW's own default is 1e-2, a hundred times the recipe's value.
    assert adamw.param_groups[0]["weight_decay"] == 1e-4
    adam = ct.build_optimizer(model, _tiny_config())
    assert type(adam) is torch.optim.Adam
    assert adam.param_groups[0]["lr"] == 1e-4
    assert adam.param_groups[0]["weight_decay"] == 0


def test_stored_weights_are_the_exact_running_average(tmp_path: Path) -> None:
    # Deleting the update would leave the initial weights in the file, so this
    # recomputes the average from every raw state the optimizer produced.
    images, masks = _tiny_data()
    decay = 0.9
    raw_states: list[dict[str, torch.Tensor]] = []
    initial: dict[str, torch.Tensor] = {}
    real_init, real_update = ct.WeightEma.__init__, ct.WeightEma.update

    def spy_init(self, model, decay):
        initial.update({k: v.clone() for k, v in model.state_dict().items()})
        real_init(self, model, decay)

    def spy_update(self, model):
        raw_states.append({k: v.clone() for k, v in model.state_dict().items()})
        real_update(self, model)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ct.WeightEma, "__init__", spy_init)
        patch.setattr(ct.WeightEma, "update", spy_update)
        path = tmp_path / "unet.pt"
        ct.train_model(
            images,
            masks,
            _v2_config(epochs=2, ema_decay=decay),
            torch.device("cpu"),
            checkpoint_path=path,
            log=lambda _: None,
        )
    assert len(raw_states) == 4  # 2 epochs of 2 steps
    stored = ct.load_checkpoint(path)["state_dict"]
    for key, value in stored.items():
        if value.is_floating_point():
            expected = initial[key].double()
            for state in raw_states:
                expected = decay * expected + (1 - decay) * state[key].double()
            assert torch.allclose(value.double(), expected, atol=1e-5, rtol=1e-5), key
        else:
            assert torch.equal(value, raw_states[-1][key]), key
    moved = [k for k, v in stored.items() if not torch.equal(v, initial[k])]
    assert moved  # the file is not the initial weights


def test_agreement_weights_reach_the_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    # A later refactor that dropped the weight would turn D2 into a second D1.
    images, _ = _tiny_data(count=1)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:30, 10:50] = 178  # 0.4 + 0.6 * 1/2, stored as a uint8
    seen: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    real_forward = ct.DiceBceLoss.forward

    def spy(self, logits, target, weight=None):
        seen.append((target.detach().clone(), None if weight is None else weight))
        return real_forward(self, logits, target, weight)

    monkeypatch.setattr(ct.DiceBceLoss, "forward", spy)
    config = _v2_config(epochs=2, agreement_weights=True, p_fg=1.0)
    ct.train_model(images, [mask], config, torch.device("cpu"), log=lambda _: None)
    assert len(seen) == config.total_steps
    for target, weight in seen:
        assert weight is not None
        assert weight.dtype == torch.float32 and weight.shape == target.shape
        assert set(target.unique().tolist()) <= {0.0, 1.0}
        positive = target == 1
        assert positive.any()
        assert torch.allclose(weight[positive], torch.tensor(178 / 255.0))
        assert torch.all(weight[~positive] == 1.0)


def test_plain_recipe_passes_no_weight_to_the_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images, masks = _tiny_data()
    seen: list[torch.Tensor | None] = []
    real_forward = ct.DiceBceLoss.forward

    def spy(self, logits, target, weight=None):
        seen.append(weight)
        return real_forward(self, logits, target, weight)

    monkeypatch.setattr(ct.DiceBceLoss, "forward", spy)
    ct.train_model(
        images, masks, _v2_config(epochs=2), torch.device("cpu"), log=lambda _: None
    )
    assert seen and all(weight is None for weight in seen)


def test_training_refuses_a_cache_built_for_the_other_recipe() -> None:
    images, masks = _tiny_data()
    with pytest.raises(ValueError, match="agreement_weights=True"):
        ct.train_model(
            images,
            masks,
            _v2_config(agreement_weights=True),
            torch.device("cpu"),
            log=lambda _: None,
        )
    stored = [mask * 178 for mask in masks]
    with pytest.raises(ValueError, match="agreement_weights=False"):
        ct.train_model(
            images, stored, _v2_config(), torch.device("cpu"), log=lambda _: None
        )


def _clip_norms_and_calls(
    monkeypatch: pytest.MonkeyPatch, *, enabled: bool
) -> tuple[list[float], list[str]]:
    """Train with a scaler at 2**16 or off. Returns the norms clipping saw, and the order."""
    images, masks = _tiny_data()
    calls: list[str] = []
    norms: list[float] = []

    class Recording(torch.amp.GradScaler):
        def unscale_(self, optimizer):
            calls.append("unscale")
            return super().unscale_(optimizer)

        def step(self, optimizer, *args, **kwargs):
            calls.append("step")
            return super().step(optimizer, *args, **kwargs)

        def update(self, new_scale=None):
            calls.append("update")
            return super().update(new_scale)

    monkeypatch.setattr(
        ct,
        "_grad_scaler",
        lambda device, _amp: Recording(
            device.type, enabled=enabled, init_scale=2.0**16
        ),
    )
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(parameters, max_norm, **kwargs):
        calls.append("clip")
        norm = real_clip(parameters, max_norm, **kwargs)
        norms.append(float(norm))
        return norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    ct.train_model(
        images, masks, _v2_config(epochs=2), torch.device("cpu"), log=lambda _: None
    )
    return norms, calls


def test_gradients_are_unscaled_before_they_are_clipped() -> None:
    # On a T4 the scaler is on, and clipping scaled gradients would clip almost
    # everything. An enabled scaler on the CPU checks this without a GPU: the norm
    # clipping sees has to be the one it sees with no scaler at all.
    with pytest.MonkeyPatch.context() as patch:
        scaled, calls = _clip_norms_and_calls(patch, enabled=True)
    assert calls == ["unscale", "clip", "step", "update"] * 4
    with pytest.MonkeyPatch.context() as patch:
        plain, _ = _clip_norms_and_calls(patch, enabled=False)
    assert len(scaled) == len(plain) == 4
    assert scaled == pytest.approx(plain, rel=1e-3)


def _recorded_pre_clip_norms(
    monkeypatch: pytest.MonkeyPatch, clip: float, epochs: int = 3
) -> tuple[list[float], ct.TrainResult]:
    """Train and return the norm clipping saw at every step, and the result."""
    images, masks = _tiny_data()
    norms: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, **kwargs):
        norm = real_clip(parameters, max_norm, **kwargs)
        norms.append(float(norm))  # the norm before clipping
        return norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    _, result = ct.train_model(
        images,
        masks,
        _v2_config(epochs=epochs, grad_clip_norm=clip),
        torch.device("cpu"),
        log=lambda _: None,
    )
    return norms, result


def test_logged_grad_norms_are_the_per_epoch_means_of_the_real_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = 1.0
    norms, result = _recorded_pre_clip_norms(monkeypatch, clip)
    steps = 2  # 4 crops per epoch, batch of 2
    assert len(norms) == 3 * steps
    assert max(norms) > clip  # not vacuous: clipping really fires on this data
    expected_norms, expected_shares = [], []
    for epoch in range(3):
        chunk = norms[epoch * steps : (epoch + 1) * steps]
        expected_norms.append(sum(chunk) / steps)
        expected_shares.append(sum(n > clip for n in chunk) / steps)
    assert result.epoch_grad_norms == pytest.approx(expected_norms)
    assert result.epoch_clipped_fractions == pytest.approx(expected_shares)
    # The norm is read before clipping: after it, every step would read 1.0.
    assert result.epoch_grad_norms != pytest.approx([clip] * 3, rel=1e-3)


def test_a_clip_that_never_fires_logs_no_clipped_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    norms, result = _recorded_pre_clip_norms(monkeypatch, clip=1e9)
    assert max(norms) < 1e9
    assert result.epoch_clipped_fractions == [0.0, 0.0, 0.0]
    assert result.epoch_grad_norms == pytest.approx(
        [sum(norms[i : i + 2]) / 2 for i in (0, 2, 4)]
    )


def test_skipped_steps_do_not_enter_the_logged_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images, masks = _tiny_data()
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = {"count": 0}

    def spy(parameters, max_norm, **kwargs):
        calls["count"] += 1
        norm = real_clip(parameters, max_norm, **kwargs)
        # Report the first step of epoch 1 as a scaler overflow.
        return torch.tensor(float("inf")) if calls["count"] == 1 else norm

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    _, result = ct.train_model(
        images, masks, _v2_config(epochs=2), torch.device("cpu"), log=lambda _: None
    )
    assert all(np.isfinite(result.epoch_grad_norms))
    assert all(0.0 <= share <= 1.0 for share in result.epoch_clipped_fractions)


def test_each_epoch_records_how_much_clipping_did() -> None:
    images, masks = _tiny_data()
    lines: list[str] = []
    _, result = ct.train_model(
        images, masks, _v2_config(epochs=2), torch.device("cpu"), log=lines.append
    )
    assert len(result.epoch_grad_norms) == 2
    assert all(norm > 0 for norm in result.epoch_grad_norms)
    assert len(result.epoch_clipped_fractions) == 2
    assert all(0.0 <= share <= 1.0 for share in result.epoch_clipped_fractions)
    epoch_lines = [line for line in lines if line.startswith("epoch 1/")]
    assert "grad_norm=" in epoch_lines[0] and "clipped=" in epoch_lines[0]


def test_stored_grad_norms_survive_the_checkpoint(tmp_path: Path) -> None:
    images, masks = _tiny_data()
    path = tmp_path / "unet.pt"
    _, result = ct.train_model(
        images,
        masks,
        _v2_config(epochs=2),
        torch.device("cpu"),
        checkpoint_path=path,
        log=lambda _: None,
    )
    stored = ct.load_checkpoint(path)
    assert stored["epoch_grad_norms"] == pytest.approx(result.epoch_grad_norms)
    assert stored["epoch_clipped_fractions"] == pytest.approx(
        result.epoch_clipped_fractions
    )


def test_a_run_without_clipping_records_no_norms() -> None:
    images, masks = _tiny_data()
    lines: list[str] = []
    _, result = ct.train_model(
        images, masks, _tiny_config(epochs=1), torch.device("cpu"), log=lines.append
    )
    assert result.epoch_grad_norms == [] and result.epoch_clipped_fractions == []
    assert not any("grad_norm=" in line for line in lines)


def test_weight_ema_averages_floats_and_copies_integers() -> None:
    model = torch.nn.BatchNorm2d(1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.running_mean.fill_(1.0)
    ema = ct.WeightEma(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.running_mean.fill_(3.0)
        model.num_batches_tracked.fill_(5)
    ema.update(model)
    assert ema.state["weight"].item() == pytest.approx(0.9 * 1.0 + 0.1 * 2.0)
    assert ema.state["running_mean"].item() == pytest.approx(0.9 * 1.0 + 0.1 * 3.0)
    assert ema.state["num_batches_tracked"].item() == 5
    assert model.weight.item() == 2.0  # the model itself is left alone


def test_weight_ema_does_not_alias_the_model() -> None:
    model = torch.nn.Conv2d(1, 1, 1)
    ema = ct.WeightEma(model, decay=0.5)
    before = ema.state["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    assert torch.equal(ema.state["weight"], before)


def test_ema_run_stores_the_average_and_returns_what_it_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    images, masks = _tiny_data()
    path = tmp_path / "unet.pt"
    last_raw: dict[str, torch.Tensor] = {}
    real_update = ct.WeightEma.update

    def spy(self, model):
        last_raw.clear()
        last_raw.update({k: v.clone() for k, v in model.state_dict().items()})
        real_update(self, model)

    monkeypatch.setattr(ct.WeightEma, "update", spy)
    model, _ = ct.train_model(
        images,
        masks,
        _v2_config(epochs=2),
        torch.device("cpu"),
        checkpoint_path=path,
        log=lambda _: None,
    )
    stored = ct.load_checkpoint(path)
    assert stored["status"] == "complete"
    assert stored["ema_decay"] == 0.9
    assert stored["optimizer"] == "adamw"
    assert stored["lr_schedule"] == "cosine"
    assert stored["warmup_epochs"] == 1
    assert stored["grad_clip_norm"] == 1.0
    assert stored["weight_decay"] == 1e-4
    assert stored["loss"] == "dice+bce"
    digest = checkpoint_fingerprint(stored)["tensor_sha256"]
    assert digest == tensor_sha256(model.state_dict())
    assert digest != tensor_sha256(last_raw)  # the file is not the last raw weights
    rebuilt = ct.model_from_checkpoint(stored, torch.device("cpu"))
    assert tensor_sha256(rebuilt.state_dict()) == digest


def _recording_optimizer(rates: list[float]):
    class Recording(torch.optim.AdamW):
        def step(self, closure=None):
            rates.append(self.param_groups[0]["lr"])
            return super().step(closure)

    return Recording


def test_optimizer_steps_follow_the_schedule_and_the_gradients_are_clipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images, masks = _tiny_data()
    config = _v2_config(epochs=3)
    rates: list[float] = []
    norms: list[float] = []
    monkeypatch.setattr(torch.optim, "AdamW", _recording_optimizer(rates))
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(parameters, max_norm, **kwargs):
        norms.append(max_norm)
        return real_clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    ct.train_model(images, masks, config, torch.device("cpu"), log=lambda _: None)
    assert config.total_steps == 6
    assert rates == [pytest.approx(scheduled_lr(config, s)) for s in range(6)]
    assert norms == [1.0] * 6


def test_default_recipe_does_not_clip_or_average(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    images, masks = _tiny_data()

    def no_clip(*_args, **_kwargs):
        raise AssertionError("the default recipe must not clip")

    def no_ema(*_args, **_kwargs):
        raise AssertionError("the default recipe must not average")

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", no_clip)
    monkeypatch.setattr(ct.WeightEma, "__init__", no_ema)
    path = tmp_path / "unet.pt"
    model, _ = ct.train_model(
        images,
        masks,
        _tiny_config(epochs=1),
        torch.device("cpu"),
        checkpoint_path=path,
        log=lambda _: None,
    )
    stored = ct.load_checkpoint(path)
    assert stored["optimizer"] == "adam"
    assert stored["ema_decay"] is None
    assert stored["learning_rate"] == 1e-4
    assert tensor_sha256(stored["state_dict"]) == tensor_sha256(model.state_dict())


def test_weighted_loss_changes_only_the_bce_term() -> None:
    loss_fn = ct.DiceBceLoss()
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    plain = loss_fn(logits, target)
    assert loss_fn(logits, target, torch.ones_like(target)).item() == pytest.approx(
        plain.item()
    )
    # Zero weight everywhere leaves the Dice term alone.
    dice = loss_fn.dice(logits, target).item()
    zeroed = loss_fn(logits, target, torch.zeros_like(target)).item()
    assert zeroed == pytest.approx(dice)
    # Weights are averaged over all pixels, not over their sum.
    half = loss_fn(logits, target, torch.full_like(target, 0.5)).item()
    assert half - dice == pytest.approx(0.5 * (plain.item() - dice))


def test_agreement_run_trains_and_records_the_weighted_loss(tmp_path: Path) -> None:
    from filament.annotations import FilamentAnnotation, TrainImage

    polygon = (10.0, 20.0, 50.0, 20.0, 50.0, 30.0, 10.0, 30.0)
    annotation = FilamentAnnotation(
        id=1, image_id=0, category_id=1, spine=None, segmentation=(polygon,)
    )
    image = TrainImage(
        id=0,
        file_name="a.png",
        date_captured=None,
        height=64,
        width=64,
        annotations=(annotation,),
        views=((annotation,), ()),
    )
    stored_mask = agreement_mask(image)
    images, _ = _tiny_data(count=1)
    path = tmp_path / "unet.pt"
    _, result = ct.train_model(
        images,
        [stored_mask],
        _v2_config(epochs=2, agreement_weights=True),
        torch.device("cpu"),
        checkpoint_path=path,
        log=lambda _: None,
    )
    assert all(np.isfinite(result.epoch_losses))
    assert ct.load_checkpoint(path)["loss"] == "dice+weighted_bce"


def test_guard_says_when_the_schedule_did_not_finish() -> None:
    images, masks = _tiny_data()
    ticks = iter(range(0, 10**6, 1000))
    lines: list[str] = []
    ct.train_model(
        images,
        masks,
        _v2_config(epochs=5, max_train_hours=0.5),
        torch.device("cpu"),
        log=lines.append,
        clock=lambda: float(next(ticks)),
    )
    assert any("schedule did not finish" in line for line in lines)


def test_require_cuda_fails_early_and_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        ct.require_cuda()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=4 * 1024**3),
    )
    with pytest.raises(RuntimeError, match="4.0 GiB"):
        ct.require_cuda(6.0)
    assert ct.require_cuda(3.0).type == "cuda"


def test_model_from_checkpoint_rejects_another_smp_version() -> None:
    model = ct.build_model("resnet18", None)
    checkpoint = {
        "state_dict": model.state_dict(),
        "encoder_name": "resnet18",
        "smp_version": "0.0.1",
    }
    with pytest.raises(RuntimeError, match="0.0.1"):
        ct.model_from_checkpoint(checkpoint, torch.device("cpu"))


def _load_smoke():
    spec = importlib.util.spec_from_file_location(
        "smoke_crop_train", ROOT / "scripts" / "smoke_crop_train.py"
    )
    assert spec is not None and spec.loader is not None
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    return smoke


def test_holdout_sweep_scores_a_known_good_pointwise_model(tmp_path: Path) -> None:
    # The synthetic filaments are gray 60 on a disk of 190. A 1x1 convolution
    # that fires below gray 125 finds them exactly, so the tiled holdout
    # evaluation and the pooled PQ must come out high. An untrained network
    # scores about 0 in the smoke test, which would hide a broken scorer.
    smoke = _load_smoke()
    records = smoke.make_records(tmp_path)
    holdout = records[-2:]
    cut = ((125 / 255) - ct.IMAGENET_MEAN[0]) / ct.IMAGENET_STD[0]
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 0, 0, 0] = -50.0
        model.bias.fill_(50.0 * cut)
    sweep = ct.run_holdout_sweep(
        model,
        holdout,
        tmp_path,
        torch.device("cpu"),
        min_areas=(10, 100),
        out_csv=tmp_path / "sweep.csv",
        tile_size=64,
        overlap=16,
        infer_batch_size=4,
        use_amp=False,
    )
    scores = {row["min_area"]: float(row["pooled_pq"]) for row in sweep.rows}
    assert scores["10"] > 0.7
    assert sweep.best is not None and float(sweep.best["pooled_pq"]) == max(
        scores.values()
    )


def test_smoke_script_runs_the_agreement_recipe(tmp_path: Path) -> None:
    outcome = _load_smoke().run(tmp_path, "r34_v2_agree")
    assert outcome["result"].epochs_completed == 2
    stored = outcome["checkpoint"]
    assert stored["loss"] == "dice+weighted_bce"
    assert stored["ema_decay"] == 0.999


def test_smoke_script_runs_end_to_end(tmp_path: Path) -> None:
    outcome = _load_smoke().run(tmp_path)

    assert outcome["weights_path"].is_file()
    assert outcome["result"].epochs_completed == 1
    csv_lines = outcome["sweep_path"].read_text().splitlines()
    assert csv_lines[0] == (
        "predictor,min_area,close_size,pooled_pq,holdout_files,annotator_views"
    )
    assert len(csv_lines) == 3  # header plus two min_area values
    assert csv_lines[1].startswith("unet_r34_crops,10,0,")
    assert csv_lines[1].split(",")[4:] == ["2", "2"]  # 2 holdout files, 2 views
    assert outcome["sweep"].inference_seconds > 0
