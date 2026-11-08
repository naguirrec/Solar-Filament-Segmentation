"""Settings for the native-resolution crop run, and its wall-clock guard.

Pure Python: no torch. The notebook builds a :class:`CropTrainConfig` from
its constants, and the tests can read the same defaults without a GPU stack.

None of these values is tuned. ``epochs``, ``crops_per_epoch`` and
``batch_size`` come from a timing estimate for one Kaggle T4. The other
fields follow the 512 px ResNet-18 trainer (``notebooks/train_unet.ipynb``)
so the comparison with it changes resolution, encoder and sampling only.

:data:`CROP_MODELS` names the runs the notebooks can train. ``r34``, ``r50``
and ``effb3`` change only the encoder, the epochs and the wall-clock cap.
``r34_v2`` keeps the data, loss and epochs of ``r34`` and changes the
optimization recipe; ``r34_v2_agree`` adds agreement weights to the BCE.

:func:`scheduled_lr` is the learning rate of one optimizer step. It is pure
Python, so the schedule is tested without torch.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

# 8 random crops for each of the 565 train files.
DEFAULT_EPOCHS = 80
DEFAULT_CROPS_PER_EPOCH = 4520
DEFAULT_CROP_SIZE = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_TRAIN_HOURS = 5.0

# Inference. Stride is tile size minus overlap: 512 - 128 = 384, which
# gives a 5 x 5 grid of 25 tiles on a 2048 x 2048 frame.
DEFAULT_TILE_OVERLAP = 128
DEFAULT_INFER_BATCH_SIZE = 8
DEFAULT_THRESHOLD = 0.5

OPTIMIZERS = ("adam", "adamw")
LR_SCHEDULES = ("constant", "cosine")


@dataclass(frozen=True, slots=True)
class CropTrainConfig:
    """Everything :func:`filament.crop_training.train_model` reads.

    ``crops_per_epoch`` crops are drawn each epoch, spread evenly over the
    train files. ``p_fg`` is the chance that a crop is centered on a
    foreground pixel instead of a uniform position. ``max_train_hours`` caps
    the training loop only (not data loading or the holdout evaluation).

    The last group is the optimization recipe. Its defaults are the recipe of
    the ``r34``, ``r50`` and ``effb3`` runs: Adam at a constant rate, no
    clipping, no weight average, plain BCE. ``learning_rate`` is the peak of a
    ``"cosine"`` schedule, reached after ``warmup_epochs``. ``ema_decay`` turns
    on an exponential moving average of the weights, and the checkpoint then
    stores the average. ``agreement_weights`` weights the BCE by how many
    annotators marked each pixel (:mod:`filament.crop_data`).
    """

    encoder_name: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    epochs: int = DEFAULT_EPOCHS
    crops_per_epoch: int = DEFAULT_CROPS_PER_EPOCH
    crop_size: int = DEFAULT_CROP_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = 1e-4
    p_fg: float = 0.67
    augment: bool = True
    seed: int = 0
    use_amp: bool = True
    max_train_hours: float = DEFAULT_MAX_TRAIN_HOURS
    optimizer: str = "adam"
    weight_decay: float = 0.0
    lr_schedule: str = "constant"
    warmup_epochs: int = 0
    grad_clip_norm: float | None = None
    ema_decay: float | None = None
    agreement_weights: bool = False

    def __post_init__(self) -> None:
        for name in ("epochs", "crops_per_epoch", "crop_size", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if not 0.0 <= self.p_fg <= 1.0:
            raise ValueError(f"p_fg must be in [0, 1], got {self.p_fg!r}")
        if not self.learning_rate > 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if not self.max_train_hours > 0:
            raise ValueError(
                f"max_train_hours must be > 0, got {self.max_train_hours!r}"
            )
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}"
            )
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(
                f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}"
            )
        if not self.weight_decay >= 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay!r}")
        if self.optimizer == "adam" and self.weight_decay != 0:
            raise ValueError("weight_decay needs optimizer 'adamw'")
        warmup = self.warmup_epochs
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError(f"warmup_epochs must be an int >= 0, got {warmup!r}")
        if warmup and self.lr_schedule != "cosine":
            raise ValueError("warmup_epochs needs lr_schedule 'cosine'")
        if warmup >= self.epochs:
            raise ValueError(
                f"warmup_epochs {warmup} must be below epochs {self.epochs}"
            )
        if self.grad_clip_norm is not None and not self.grad_clip_norm > 0:
            raise ValueError(
                f"grad_clip_norm must be > 0 or None, got {self.grad_clip_norm!r}"
            )
        if self.ema_decay is not None and not 0.0 < self.ema_decay < 1.0:
            raise ValueError(
                f"ema_decay must be in (0, 1) or None, got {self.ema_decay!r}"
            )

    @property
    def steps_per_epoch(self) -> int:
        """Optimizer steps in one epoch. The last batch may be smaller."""
        return -(-self.crops_per_epoch // self.batch_size)

    @property
    def total_steps(self) -> int:
        """Optimizer steps in the whole run if the wall-clock guard does not fire."""
        return self.epochs * self.steps_per_epoch


def next_epoch_fits(
    elapsed_seconds: float,
    epoch_seconds: Sequence[float],
    max_hours: float,
) -> bool:
    """Return True when one more epoch is expected to end inside the budget.

    ``elapsed_seconds`` is the training time so far. ``epoch_seconds`` holds
    the duration of every completed epoch. The next epoch is assumed to take
    as long as the slowest one so far, which errs toward stopping early. With
    no completed epoch there is nothing to estimate from, so the first epoch
    always runs.
    """
    if not epoch_seconds:
        return True
    expected = elapsed_seconds + max(epoch_seconds)
    return expected <= max_hours * 3600.0


def scheduled_lr(config: CropTrainConfig, step: int) -> float:
    """Learning rate of optimizer step ``step``, counted from 0.

    ``"constant"`` is ``learning_rate`` at every step. ``"cosine"`` rises
    linearly to ``learning_rate`` over ``warmup_epochs``, then follows half a
    cosine toward 0. The first step after the warmup is at the peak, and the
    last step of the run keeps a small positive rate. With few warmup steps
    the mean over the run is about half the peak.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be an int >= 0, got {step!r}")
    if step >= config.total_steps:
        raise ValueError(f"step {step} is past the {config.total_steps} of this run")
    if config.lr_schedule == "constant":
        return config.learning_rate
    warmup_steps = config.warmup_epochs * config.steps_per_epoch
    if step < warmup_steps:
        return config.learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (config.total_steps - warmup_steps)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# Long runs: a Kaggle session ends at 12 h. Data prep and the holdout took about
# 13 min for the ResNet-34, so this cap keeps the whole notebook under about 9 h.
LONG_RUN_MAX_TRAIN_HOURS = 8.5

DEFAULT_CROP_MODEL = "r34"

# Recipe v2 on the ResNet-34 run: same data, loss, seed and 80 epochs. The peak
# is the middle of the 1e-4 to 3e-4 range of docs/ROADMAP_V2.md: a cosine has
# half its peak as its mean, so 2e-4 keeps the mean rate of the constant 1e-4.
# Untuned, and the 0.999 decay is the roadmap's value.
R34_V2_CONFIG = CropTrainConfig(
    learning_rate=2e-4,
    optimizer="adamw",
    weight_decay=1e-4,
    lr_schedule="cosine",
    warmup_epochs=1,
    grad_clip_norm=1.0,
    ema_decay=0.999,
)

CROP_MODELS: dict[str, CropTrainConfig] = {
    "r34": CropTrainConfig(),
    "r50": CropTrainConfig(
        encoder_name="resnet50",
        epochs=120,
        max_train_hours=LONG_RUN_MAX_TRAIN_HOURS,
    ),
    "effb3": CropTrainConfig(
        encoder_name="efficientnet-b3",
        epochs=100,
        max_train_hours=LONG_RUN_MAX_TRAIN_HOURS,
    ),
    "r34_v2": R34_V2_CONFIG,
    "r34_v2_agree": dataclasses.replace(R34_V2_CONFIG, agreement_weights=True),
}

# Seconds per epoch on one Kaggle T4 at batch 8, as (low, high). The ResNet-34 is
# measured (126 s). The others are estimates scaled from it and not measured on a
# T4 [VERIFY]; see docs/kaggle-gpu.md for how they were derived. The v2 recipes run
# the same network, plus the weight average and the clipping at every step.
ESTIMATED_EPOCH_SECONDS: dict[str, tuple[float, float]] = {
    "r34": (126.0, 126.0),
    "r50": (140.0, 240.0),
    "effb3": (130.0, 280.0),
    "r34_v2": (126.0, 145.0),
    "r34_v2_agree": (126.0, 150.0),
}


def crop_model_config(name: str = DEFAULT_CROP_MODEL) -> CropTrainConfig:
    """The training settings of the named model, or ``ValueError`` listing the names."""
    if name not in CROP_MODELS:
        raise ValueError(
            f"unknown crop model {name!r}, expected one of {sorted(CROP_MODELS)}"
        )
    return CROP_MODELS[name]


def estimated_train_hours(name: str = DEFAULT_CROP_MODEL) -> tuple[float, float]:
    """``(low, high)`` hours of training for the named model's configured epochs."""
    config = crop_model_config(name)
    low, high = ESTIMATED_EPOCH_SECONDS[name]
    return config.epochs * low / 3600.0, config.epochs * high / 3600.0
