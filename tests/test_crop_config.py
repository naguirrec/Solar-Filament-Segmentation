"""Crop-run defaults and the wall-clock guard. Pure Python."""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest

from filament.crop_config import (
    CROP_MODELS,
    DEFAULT_CROP_MODEL,
    ESTIMATED_EPOCH_SECONDS,
    LONG_RUN_MAX_TRAIN_HOURS,
    CropTrainConfig,
    crop_model_config,
    estimated_train_hours,
    next_epoch_fits,
    scheduled_lr,
)

ENCODER_RUNS = ("r34", "r50", "effb3")


def test_defaults_are_the_agreed_run() -> None:
    config = CropTrainConfig()
    assert config.encoder_name == "resnet34"
    assert config.encoder_weights == "imagenet"
    assert config.epochs == 80
    assert config.crops_per_epoch == 4520  # 8 crops x 565 train files
    assert config.crops_per_epoch == 8 * 565
    assert config.crop_size == 512
    assert config.batch_size == 8
    assert config.use_amp is True
    assert config.max_train_hours == 5.0
    assert config.seed == 0
    assert config.learning_rate == 1e-4
    assert config.optimizer == "adam"
    assert config.weight_decay == 0.0
    assert config.lr_schedule == "constant"
    assert config.warmup_epochs == 0
    assert config.grad_clip_norm is None
    assert config.ema_decay is None
    assert config.agreement_weights is False


@pytest.mark.parametrize(
    "bad",
    [
        {"epochs": 0},
        {"crop_size": -1},
        {"batch_size": True},
        {"crops_per_epoch": 1.5},
        {"p_fg": 1.2},
        {"p_fg": -0.1},
        {"learning_rate": 0.0},
        {"max_train_hours": 0},
        {"optimizer": "sgd"},
        {"lr_schedule": "step"},
        {"optimizer": "adamw", "weight_decay": -1.0},
        {"weight_decay": 1e-4},  # Adam would turn it into an L2 penalty
        {"warmup_epochs": 1},  # warmup without a cosine
        {"lr_schedule": "cosine", "warmup_epochs": -1},
        {"lr_schedule": "cosine", "warmup_epochs": 1.5},
        {"lr_schedule": "cosine", "warmup_epochs": True},
        {"lr_schedule": "cosine", "warmup_epochs": 80},  # no room for the cosine
        {"grad_clip_norm": 0.0},
        {"ema_decay": 1.0},
        {"ema_decay": 0.0},
    ],
)
def test_invalid_settings_are_rejected(bad: dict) -> None:
    with pytest.raises(ValueError):
        CropTrainConfig(**bad)


def test_first_epoch_always_runs() -> None:
    assert next_epoch_fits(0.0, [], 5.0)
    assert next_epoch_fits(10**9, [], 5.0)


def test_guard_uses_the_slowest_completed_epoch() -> None:
    hour = 3600.0
    # 4 h elapsed, epochs took 600 s and 1200 s: 4 h + 20 min <= 5 h.
    assert next_epoch_fits(4 * hour, [600.0, 1200.0], 5.0)
    # 4 h 50 min elapsed, slowest 1200 s: 5 h 10 min > 5 h.
    assert not next_epoch_fits(4 * hour + 3000.0, [600.0, 1200.0], 5.0)


def test_guard_boundary_is_inclusive() -> None:
    assert next_epoch_fits(3600.0 - 100.0, [100.0], 1.0)
    assert not next_epoch_fits(3600.0 - 99.0, [100.0], 1.0)


def test_r34_stays_the_default_and_equals_the_agreed_run() -> None:
    assert DEFAULT_CROP_MODEL == "r34"
    assert crop_model_config() == CropTrainConfig()
    assert crop_model_config("r34") == CropTrainConfig()


def test_each_name_selects_its_encoder() -> None:
    assert crop_model_config("r34").encoder_name == "resnet34"
    assert crop_model_config("r50").encoder_name == "resnet50"
    assert crop_model_config("effb3").encoder_name == "efficientnet-b3"


@pytest.mark.parametrize("name", ["resnet50", "R50", "", "r18"])
def test_unknown_model_is_refused_and_the_names_are_listed(name: str) -> None:
    with pytest.raises(ValueError, match="expected one of.*r34"):
        crop_model_config(name)


@pytest.mark.parametrize("name", ENCODER_RUNS)
def test_encoder_runs_change_only_encoder_epochs_and_cap(name: str) -> None:
    config = crop_model_config(name)
    same = dataclasses.replace(
        CropTrainConfig(),
        encoder_name=config.encoder_name,
        epochs=config.epochs,
        max_train_hours=config.max_train_hours,
    )
    assert config == same


def test_recipe_v2_keeps_the_data_loss_and_epochs_of_r34() -> None:
    base, v2 = crop_model_config("r34"), crop_model_config("r34_v2")
    assert v2 == dataclasses.replace(
        base,
        learning_rate=2e-4,
        optimizer="adamw",
        weight_decay=1e-4,
        lr_schedule="cosine",
        warmup_epochs=1,
        grad_clip_norm=1.0,
        ema_decay=0.999,
    )
    assert v2.epochs == base.epochs == 80
    assert v2.seed == base.seed and v2.crops_per_epoch == base.crops_per_epoch


def test_agreement_run_adds_only_the_weights() -> None:
    v2, agree = crop_model_config("r34_v2"), crop_model_config("r34_v2_agree")
    assert agree == dataclasses.replace(v2, agreement_weights=True)
    assert v2.agreement_weights is False


def test_steps_count_the_smaller_last_batch() -> None:
    config = CropTrainConfig(crops_per_epoch=10, batch_size=4, epochs=3)
    assert config.steps_per_epoch == 3
    assert config.total_steps == 9
    assert CropTrainConfig().steps_per_epoch == 565


def test_constant_schedule_never_changes() -> None:
    config = CropTrainConfig(epochs=2, crops_per_epoch=8, batch_size=2)
    rates = [scheduled_lr(config, step) for step in range(config.total_steps)]
    assert rates == [config.learning_rate] * 8


def _cosine(**changes) -> CropTrainConfig:
    values = {
        "epochs": 5,
        "crops_per_epoch": 8,
        "batch_size": 2,
        "learning_rate": 1.0,
        "lr_schedule": "cosine",
        "warmup_epochs": 1,
    }
    values.update(changes)
    return CropTrainConfig(**values)


def test_warmup_rises_linearly_to_the_peak() -> None:
    config = _cosine()
    rates = [scheduled_lr(config, step) for step in range(4)]
    assert rates == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert scheduled_lr(config, 4) == pytest.approx(1.0)  # first cosine step


def test_cosine_falls_monotonically_and_stays_positive() -> None:
    config = _cosine()
    rates = [scheduled_lr(config, step) for step in range(4, config.total_steps)]
    assert all(a > b for a, b in itertools.pairwise(rates))
    assert rates[-1] > 0
    assert rates[-1] < 0.05
    assert max(rates) == pytest.approx(config.learning_rate)
    progress = 8 / 16
    expected = 0.5 * (1 + math.cos(math.pi * progress))
    assert scheduled_lr(config, 4 + 8) == pytest.approx(expected)


def test_schedule_without_warmup_starts_at_the_peak() -> None:
    config = _cosine(warmup_epochs=0)
    assert scheduled_lr(config, 0) == pytest.approx(1.0)


def test_v2_schedule_keeps_the_mean_rate_of_the_constant_run() -> None:
    config = crop_model_config("r34_v2")
    rates = [scheduled_lr(config, step) for step in range(config.total_steps)]
    assert max(rates) == pytest.approx(2e-4)
    assert sum(rates) / len(rates) == pytest.approx(1e-4, rel=1e-3)
    # The moving average mostly covers the last 1000 steps, at under 0.2% of the peak.
    assert max(rates[-1000:]) < 0.002 * max(rates)


@pytest.mark.parametrize("step", [-1, 8, 1.0, True])
def test_schedule_rejects_a_step_outside_the_run(step: object) -> None:
    config = CropTrainConfig(epochs=2, crops_per_epoch=8, batch_size=2)
    with pytest.raises(ValueError):
        scheduled_lr(config, step)


@pytest.mark.parametrize("name", sorted(CROP_MODELS))
def test_planned_epochs_fit_the_wall_clock_cap(name: str) -> None:
    low, high = estimated_train_hours(name)
    assert 0 < low <= high
    # Even the pessimistic estimate finishes before the guard would stop it.
    assert high <= crop_model_config(name).max_train_hours


def test_long_runs_stay_inside_the_kaggle_session() -> None:
    assert LONG_RUN_MAX_TRAIN_HOURS <= 9.0
    for name in CROP_MODELS:
        assert crop_model_config(name).max_train_hours <= LONG_RUN_MAX_TRAIN_HOURS


def test_every_model_has_an_estimate() -> None:
    assert set(ESTIMATED_EPOCH_SECONDS) == set(CROP_MODELS)
    with pytest.raises(ValueError, match="expected one of"):
        estimated_train_hours("nope")
