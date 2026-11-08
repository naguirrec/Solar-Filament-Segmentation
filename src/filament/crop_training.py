"""Torch side of the native-resolution crop run.

Everything that needs torch or segmentation-models-pytorch lives here:
model, loss, the training loop, checkpoint files, and the tiled predictor
that the holdout evaluation and the test-set notebook share. The notebooks
and ``scripts/smoke_crop_train.py`` import it; CI does not (it has no torch).
The sampling, tiling, scoring and checkpoint-guard logic it calls is pure
numpy and is tested without torch.

Recipe, relative to the 512 px ResNet-18 trainer (``notebooks/train_unet.ipynb``):

* a bigger ImageNet encoder instead of ``resnet18``: ``resnet34`` by default,
  or another named model of :data:`filament.crop_config.CROP_MODELS`;
* random 512 px crops of the native 2048 px frame instead of whole frames
  downscaled to 512, with foreground-biased positions;
* Dice + BCE-with-logits loss and Adam, learning rate 1e-4, as before;
* mixed precision on CUDA, with the loss computed in float32.

:class:`filament.crop_config.CropTrainConfig` can change the optimization
recipe: AdamW, a cosine schedule with warmup, gradient clipping, an exponential
moving average of the weights (:class:`WeightEma`) and agreement weights on the
BCE. The defaults leave all of them off, so the recipe above is unchanged.

Frames go to the GPU as uint8 and are normalized there (divide by 255,
repeat to 3 channels, ImageNet mean/std), the same arithmetic as the
512 px trainer.

cuDNN is set to deterministic and benchmark off, as in the 512 px trainer.
Seeds are fixed, but mixed-precision GPU runs are not promised to be
bit-identical.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

from filament.annotations import TrainImage
from filament.checkpoint import tensor_sha256
from filament.crop_config import (
    DEFAULT_INFER_BATCH_SIZE,
    DEFAULT_THRESHOLD,
    DEFAULT_TILE_OVERLAP,
    CropTrainConfig,
    next_epoch_fits,
    scheduled_lr,
)
from filament.crop_data import check_mask_kind, decode_agreement_mask, read_gray
from filament.crops import CropSampler
from filament.ensemble import resize_probability
from filament.sweep import (
    best_row,
    count_views,
    format_postproc_table,
    format_table,
    postproc_rows,
    sweep_min_area,
    sweep_postproc_grid,
    sweep_rows,
    write_postproc_rows,
    write_rows,
)
from filament.tiling import predict_tiled
from filament.train_data import downscale_pair

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PREDICTOR_NAME = "unet_r34_crops"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_cuda(min_gib: float = 6.0) -> torch.device:
    """Return the CUDA device, or raise early with a clear message.

    The run is sized for one Kaggle T4 (16 GB). A GPU below ``min_gib`` GiB,
    such as the 4 GB laptop card, is refused.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. On Kaggle set Accelerator to GPU T4 "
            "(a single T4) in the notebook settings and run again."
        )
    gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gib < min_gib:
        raise RuntimeError(
            f"This GPU has {gib:.1f} GiB, below the {min_gib:.0f} GiB this run "
            "needs. Use a Kaggle T4."
        )
    return torch.device("cuda")


def normalize_gray(batch: torch.Tensor) -> torch.Tensor:
    """``(B, H, W)`` uint8 gray frames to ``(B, 3, H, W)`` float32 network input."""
    x = batch.to(torch.float32) / 255.0
    x = x.unsqueeze(1).expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=x.device)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=x.device)
    return (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)


class DiceBceLoss(torch.nn.Module):
    """Sum of binary Dice and BCE-with-logits, as in the 512 px trainer.

    ``weight`` is an optional per-pixel weight, applied to the BCE only: the
    weighted terms are averaged over all pixels. The Dice term stays unweighted.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weight is None:
            bce = self.bce(logits, target)
        else:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, weight=weight
            )
        return self.dice(logits, target) + bce


class WeightEma:
    """Exponential moving average of a model's ``state_dict``.

    After each optimizer step, ``average = decay * average + (1 - decay) * value``
    for every floating-point entry, batch-norm statistics included. Integer
    entries (``num_batches_tracked``) copy the current value. The average starts
    from the model as it is when this object is built.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for key, value in model.state_dict().items():
            average = self.state[key]
            if average.is_floating_point():
                average.lerp_(value, 1.0 - self.decay)
            else:
                average.copy_(value)


def build_model(encoder_name: str, encoder_weights: str | None) -> torch.nn.Module:
    """``smp.Unet`` with 3 input channels and one output class."""
    try:
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not build the {encoder_name} U-Net with "
            f"encoder_weights={encoder_weights!r}. For 'imagenet', smp "
            "downloads the weights from the Hugging Face Hub, so Internet has "
            "to be On. Do not switch to ssl or swsl weights. "
            f"Original error: {exc}"
        ) from exc


@dataclass(slots=True)
class TrainResult:
    """What a training run did. ``epochs_completed`` is what the file stores."""

    epochs_requested: int
    epochs_completed: int = 0
    stopped_early: bool = False
    epoch_losses: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    train_seconds: float = 0.0
    # With clipping: the mean gradient norm before clipping, and the share of steps
    # it clipped. Skipped steps of the scaler (inf or NaN) do not count in the mean.
    epoch_grad_norms: list[float] = field(default_factory=list)
    epoch_clipped_fractions: list[float] = field(default_factory=list)


def checkpoint_dict(
    model: torch.nn.Module,
    config: CropTrainConfig,
    result: TrainResult,
    *,
    native_shape: tuple[int, int],
    train_files: int,
    status: str,
    extra: dict | None = None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> dict:
    """The dict written to disk. ``epochs`` is the number of COMPLETED epochs.

    ``state_dict`` replaces the model's own weights, for a weight average.
    """
    data = {
        "state_dict": model.state_dict() if state_dict is None else dict(state_dict),
        "encoder_name": config.encoder_name,
        "encoder_weights": config.encoder_weights,
        "smp_version": smp.__version__,
        "in_channels": 3,
        "classes": 1,
        "epochs": result.epochs_completed,
        "epochs_requested": result.epochs_requested,
        "stopped_early": result.stopped_early,
        "status": status,
        "crop_size": config.crop_size,
        "training_resolution": "native",
        "native_shape": [int(native_shape[0]), int(native_shape[1])],
        "crops_per_epoch": config.crops_per_epoch,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "weight_decay": config.weight_decay,
        "lr_schedule": config.lr_schedule,
        "warmup_epochs": config.warmup_epochs,
        "grad_clip_norm": config.grad_clip_norm,
        "ema_decay": config.ema_decay,
        "p_fg": config.p_fg,
        "augment": config.augment,
        "use_amp": config.use_amp,
        "seed": config.seed,
        "loss": "dice+weighted_bce" if config.agreement_weights else "dice+bce",
        "train_files": int(train_files),
        "epoch_losses": [float(v) for v in result.epoch_losses],
        "epoch_grad_norms": [float(v) for v in result.epoch_grad_norms],
        "epoch_clipped_fractions": [float(v) for v in result.epoch_clipped_fractions],
    }
    if extra:
        data.update(extra)
    return data


def save_checkpoint(path: Path, data: dict) -> None:
    """Write ``data`` atomically (temp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict:
    """Read a checkpoint written by :func:`save_checkpoint`."""
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or "state_dict" not in loaded:
        raise RuntimeError(f"{path} is not a crop-run checkpoint (no state_dict).")
    return loaded


def model_from_checkpoint(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    """Rebuild the U-Net a checkpoint describes and load its weights.

    ``encoder_weights`` is ``None`` here: the ``state_dict`` already holds the
    encoder, so nothing is downloaded.
    """
    saved_smp = checkpoint.get("smp_version")
    if saved_smp is not None and saved_smp != smp.__version__:
        raise RuntimeError(
            f"The checkpoint was saved with segmentation-models-pytorch "
            f"{saved_smp}; this session has {smp.__version__}."
        )
    model = build_model(str(checkpoint["encoder_name"]), None)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model


def whole_frame_model_from_checkpoint(
    checkpoint: dict, device: torch.device, *, side: int = 512
) -> torch.nn.Module:
    """Rebuild a model trained on whole frames downscaled to ``side`` px.

    This is the 512 px ResNet-18 of ``notebooks/train_unet.ipynb``. Its
    checkpoint stores ``side``; a different value is refused, because the
    network would then see frames at a size it was not trained on.
    """
    saved_side = checkpoint.get("side")
    if saved_side is None or int(saved_side) != side:
        raise RuntimeError(
            f"The checkpoint records side={saved_side!r}, expected {side}. "
            "It is not a whole-frame model of that size."
        )
    return model_from_checkpoint(checkpoint, device)


def _grad_scaler(device: torch.device, enabled: bool):
    """``torch.amp.GradScaler`` (torch 2.3+), else the older ``torch.cuda.amp`` one."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(
    model: torch.nn.Module, config: CropTrainConfig
) -> torch.optim.Optimizer:
    """Adam or AdamW over every parameter, as ``config.optimizer`` says."""
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def _batch_targets(
    batch_masks: np.ndarray, config: CropTrainConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(B, 1, S, S)`` float32 target and BCE weight (``None`` for plain BCE)."""
    if not config.agreement_weights:
        target = torch.from_numpy(batch_masks).to(device, non_blocking=True)
        return target.unsqueeze(1).to(torch.float32), None
    target_array, weight_array = decode_agreement_mask(batch_masks)
    target = torch.from_numpy(target_array).to(device, non_blocking=True)
    weight = torch.from_numpy(weight_array).to(device, non_blocking=True)
    return target.unsqueeze(1).to(torch.float32), weight.unsqueeze(1)


def train_model(
    images: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    config: CropTrainConfig,
    device: torch.device,
    *,
    checkpoint_path: Path | None = None,
    extra_metadata: dict | None = None,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[torch.nn.Module, TrainResult]:
    """Train on random native-resolution crops and return the model.

    Each epoch draws ``config.crops_per_epoch`` crops spread evenly over the
    images. Before each epoch after the first, the wall-clock guard
    (:func:`filament.crop_config.next_epoch_fits`) stops the loop if the next
    epoch is expected to end past ``config.max_train_hours``. The stored
    ``epochs`` is then the number of COMPLETED epochs.

    With ``checkpoint_path`` the checkpoint is rewritten after every epoch
    (status ``in_progress``) and once more at the end (``complete`` or
    ``stopped_early``), so a killed session still leaves usable weights.

    With ``config.ema_decay`` every checkpoint stores the weight average, and
    the returned model holds the average too, so it is what the file holds. The
    average of the last epoch is the final model: no epoch is picked on
    validation. With a cosine schedule a run stopped by the guard ends before
    the rate has decayed; the log says so.
    """
    check_mask_kind(masks, config.agreement_weights)
    set_seed(config.seed)
    amp = bool(config.use_amp and device.type == "cuda")
    model = build_model(config.encoder_name, config.encoder_weights).to(device)
    loss_fn = DiceBceLoss()
    optimizer = build_optimizer(model, config)
    scaler = _grad_scaler(device, amp)
    ema = None if config.ema_decay is None else WeightEma(model, config.ema_decay)
    scheduled = config.lr_schedule != "constant"
    global_step = 0
    sampler = CropSampler(
        images,
        masks,
        config.crop_size,
        p_fg=config.p_fg,
        augment=config.augment,
    )
    native_shape = images[0].shape
    result = TrainResult(epochs_requested=config.epochs)
    steps_per_epoch = config.steps_per_epoch
    log(
        f"training {config.encoder_name} epochs={config.epochs} "
        f"crops_per_epoch={config.crops_per_epoch} ({config.crops_per_epoch / len(images):.1f} "
        f"per file) crop={config.crop_size} batch={config.batch_size} "
        f"steps/epoch={steps_per_epoch} amp={amp} device={device}"
    )
    log(
        f"recipe optimizer={config.optimizer} lr={config.learning_rate:g} "
        f"weight_decay={config.weight_decay:g} schedule={config.lr_schedule} "
        f"warmup_epochs={config.warmup_epochs} clip={config.grad_clip_norm} "
        f"ema={config.ema_decay} agreement_weights={config.agreement_weights}"
    )

    def _save(status: str) -> None:
        if checkpoint_path is None:
            return
        save_checkpoint(
            checkpoint_path,
            checkpoint_dict(
                model,
                config,
                result,
                native_shape=native_shape,
                train_files=len(images),
                status=status,
                extra=extra_metadata,
                state_dict=None if ema is None else ema.state,
            ),
        )

    start = clock()
    for epoch in range(1, config.epochs + 1):
        elapsed = clock() - start
        if not next_epoch_fits(elapsed, result.epoch_seconds, config.max_train_hours):
            result.stopped_early = True
            log(
                f"NOTE: stopping before epoch {epoch}. The next epoch is expected "
                f"to take about {max(result.epoch_seconds):.0f} s and "
                f"{elapsed:.0f} s have passed, which would exceed "
                f"MAX_TRAIN_HOURS={config.max_train_hours}. "
                f"Completed epochs: {result.epochs_completed} of {config.epochs}. "
                f"The checkpoint stores epochs={result.epochs_completed}."
            )
            if scheduled:
                log(
                    "NOTE: the learning-rate schedule did not finish, so these "
                    "weights are not comparable with a complete run."
                )
            break
        epoch_start = clock()
        model.train()
        rng = np.random.default_rng([config.seed, epoch])
        total = torch.zeros((), device=device)
        seen = 0
        norm_total = 0.0
        norm_steps = 0
        clipped_steps = 0
        for step, (batch_images, batch_masks) in enumerate(
            sampler.epoch_batches(config.crops_per_epoch, config.batch_size, rng),
            start=1,
        ):
            x = torch.from_numpy(batch_images).to(device, non_blocking=True)
            y, weight = _batch_targets(batch_masks, config, device)
            if scheduled:
                rate = scheduled_lr(config, global_step)
                for group in optimizer.param_groups:
                    group["lr"] = rate
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                logits = model(normalize_gray(x))
            loss = loss_fn(logits.float(), y, weight)
            scaler.scale(loss).backward()
            if config.grad_clip_norm is not None:
                # Clip the real gradients, not the ones the scaler multiplied.
                scaler.unscale_(optimizer)
                norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip_norm
                    )
                )
                if np.isfinite(norm):
                    norm_total += norm
                    norm_steps += 1
                    clipped_steps += norm > config.grad_clip_norm
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            total += loss.detach() * x.shape[0]
            seen += int(x.shape[0])
            if epoch == 1 and step % 100 == 0:
                log(
                    f"  epoch 1 step {step}/{steps_per_epoch} "
                    f"{clock() - epoch_start:.0f} s"
                )
        mean_loss = float(total) / max(seen, 1)
        seconds = clock() - epoch_start
        result.epochs_completed = epoch
        result.epoch_losses.append(mean_loss)
        result.epoch_seconds.append(seconds)
        result.train_seconds = clock() - start
        clipping = ""
        if config.grad_clip_norm is not None:
            mean_norm = norm_total / max(norm_steps, 1)
            share = clipped_steps / max(norm_steps, 1)
            result.epoch_grad_norms.append(mean_norm)
            result.epoch_clipped_fractions.append(share)
            clipping = f"grad_norm={mean_norm:.2f} clipped={share:.0%} "
        log(
            f"epoch {epoch}/{config.epochs} mean_loss={mean_loss:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} {clipping}"
            f"time={seconds:.0f} s elapsed={result.train_seconds / 60:.1f} min"
        )
        if epoch == 1 and device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            log(f"  peak GPU memory allocated in epoch 1: {peak:.1f} GiB")
        _save("in_progress")

    result.train_seconds = clock() - start
    _save("stopped_early" if result.stopped_early else "complete")
    if ema is not None:
        model.load_state_dict(ema.state)
    return model, result


def make_predict_fn(
    model: torch.nn.Module,
    device: torch.device,
    *,
    use_amp: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``fn(tiles)``: ``(B, S, S)`` uint8 to ``(B, S, S)`` float32 probabilities."""
    amp = bool(use_amp and device.type == "cuda")
    model.eval()

    def predict(tiles: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(tiles)).to(device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                logits = model(normalize_gray(x))
            probability = torch.sigmoid(logits.float())
        return probability[:, 0].cpu().numpy()

    return predict


def predict_probability(
    model: torch.nn.Module,
    gray: np.ndarray,
    device: torch.device,
    *,
    tile_size: int,
    overlap: int = DEFAULT_TILE_OVERLAP,
    batch_size: int = DEFAULT_INFER_BATCH_SIZE,
    use_amp: bool = True,
) -> np.ndarray:
    """Full-frame foreground probability by overlapped, averaged tiles.

    The holdout evaluation and the test-set notebook both call this.
    """
    return predict_tiled(
        gray,
        make_predict_fn(model, device, use_amp=use_amp),
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
    )


def predict_probability_resized(
    model: torch.nn.Module,
    gray: np.ndarray,
    device: torch.device,
    *,
    side: int = 512,
) -> np.ndarray:
    """Full-frame probability from a model that sees the whole frame at ``side`` px.

    This is how the 512 px ResNet-18 (``notebooks/train_unet.ipynb``) predicts:
    the frame is area-averaged to ``side`` with
    :func:`filament.train_data.downscale_pair`, divided by 255 and normalized
    with the ImageNet mean and standard deviation, in float32 (no mixed
    precision, as in training). The probability map is then resized back to
    the frame with bilinear interpolation (:func:`filament.ensemble.resize_probability`)
    so it can be averaged with a native-resolution model. The notebook that
    submitted this model upsampled the thresholded mask with nearest
    neighbour instead; the bilinear resize is only for ensembling.
    """
    small, _unused = downscale_pair(
        gray, np.zeros(gray.shape, dtype=np.uint8), side=side
    )
    model.eval()
    with torch.inference_mode():
        batch = normalize_gray(torch.from_numpy(small).to(device).unsqueeze(0))
        probability = torch.sigmoid(model(batch).float())[0, 0].cpu().numpy()
    return resize_probability(probability, gray.shape)


def tiled_predictor(
    model: torch.nn.Module,
    device: torch.device,
    *,
    tile_size: int,
    overlap: int = DEFAULT_TILE_OVERLAP,
    batch_size: int = DEFAULT_INFER_BATCH_SIZE,
    use_amp: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """``fn(gray)`` for a native-resolution crop model (the ResNet-34 run)."""

    def predict(gray: np.ndarray) -> np.ndarray:
        return predict_probability(
            model,
            gray,
            device,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            use_amp=use_amp,
        )

    return predict


def resized_predictor(
    model: torch.nn.Module, device: torch.device, *, side: int = 512
) -> Callable[[np.ndarray], np.ndarray]:
    """``fn(gray)`` for the whole-frame 512 px model (the ResNet-18 run)."""

    def predict(gray: np.ndarray) -> np.ndarray:
        return predict_probability_resized(model, gray, device, side=side)

    return predict


def semantic_from_probability(
    probability: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> np.ndarray:
    """Binary uint8 mask: probability at or above ``threshold``."""
    return (probability >= threshold).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class HoldoutSweep:
    rows: list[dict[str, str]]
    best: dict[str, str] | None
    inference_seconds: float
    total_seconds: float


def run_holdout_sweep(
    model: torch.nn.Module,
    holdout: Sequence[TrainImage],
    image_dir: Path,
    device: torch.device,
    *,
    min_areas: Sequence[int],
    out_csv: Path,
    tile_size: int,
    overlap: int = DEFAULT_TILE_OVERLAP,
    infer_batch_size: int = DEFAULT_INFER_BATCH_SIZE,
    use_amp: bool = True,
    threshold: float = DEFAULT_THRESHOLD,
    close_size: int = 0,
    predictor_name: str = PREDICTOR_NAME,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> HoldoutSweep:
    """Score the holdout at full resolution and write the sweep CSV.

    Each holdout frame is predicted once with tiled inference; the thresholded
    mask is reused for every ``min_area``. Scoring is pooled per-view PQ from
    :mod:`filament.sweep` (``fast_view_counts`` / ``PqCounts``). Returns the
    rows (also written to ``out_csv`` with the columns of
    ``results/postprocess_sweep_2026-11-01.csv``), the best row, and timings.
    ``predictor_name`` fills the ``predictor`` column of those rows.
    """
    image_dir = Path(image_dir)
    inference_seconds = 0.0

    def _read(image: TrainImage) -> np.ndarray:
        return read_gray(image_dir / image.file_name, (image.height, image.width))

    def _predict(gray: np.ndarray) -> np.ndarray:
        nonlocal inference_seconds
        begin = clock()
        probability = predict_probability(
            model,
            gray,
            device,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=infer_batch_size,
            use_amp=use_amp,
        )
        inference_seconds += clock() - begin
        return semantic_from_probability(probability, threshold)

    def _progress(index: int, total: int, image: TrainImage) -> None:
        if index == 1 or index % 20 == 0 or index == total:
            log(f"  holdout {index}/{total} {image.file_name}")

    begin = clock()
    totals = sweep_min_area(
        holdout, _read, _predict, min_areas, close_size=close_size, progress=_progress
    )
    rows = sweep_rows(
        predictor_name,
        totals,
        close_size=close_size,
        holdout_files=len(holdout),
        views=count_views(holdout),
    )
    write_rows(out_csv, rows)
    return HoldoutSweep(
        rows=rows,
        best=best_row(rows),
        inference_seconds=inference_seconds,
        total_seconds=clock() - begin,
    )


@dataclass(frozen=True, slots=True)
class PostprocGrid:
    rows: list[dict[str, str]]
    inference_seconds: float
    total_seconds: float


def run_postproc_grid(
    model: torch.nn.Module,
    holdout: Sequence[TrainImage],
    image_dir: Path,
    device: torch.device,
    *,
    thresholds: Sequence[float],
    close_sizes: Sequence[int],
    min_areas: Sequence[int],
    out_csv: Path,
    tile_size: int,
    overlap: int = DEFAULT_TILE_OVERLAP,
    infer_batch_size: int = DEFAULT_INFER_BATCH_SIZE,
    use_amp: bool = True,
    predictor_name: str = PREDICTOR_NAME,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
) -> PostprocGrid:
    """Score the holdout over a threshold / close_size / min_area grid.

    Each holdout frame goes through the network once; the probability map is
    reused for every grid point (:func:`filament.sweep.sweep_postproc_grid`).
    The rows are written to ``out_csv`` sorted by pooled PQ, best first.
    ``predictor_name`` fills the ``predictor`` column of those rows.
    """
    image_dir = Path(image_dir)
    inference_seconds = 0.0

    def _read(image: TrainImage) -> np.ndarray:
        return read_gray(image_dir / image.file_name, (image.height, image.width))

    def _predict(gray: np.ndarray) -> np.ndarray:
        nonlocal inference_seconds
        begin = clock()
        probability = predict_probability(
            model,
            gray,
            device,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=infer_batch_size,
            use_amp=use_amp,
        )
        inference_seconds += clock() - begin
        return probability

    def _progress(index: int, total: int, image: TrainImage) -> None:
        if index == 1 or index % 10 == 0 or index == total:
            log(f"  holdout {index}/{total} {image.file_name}")

    begin = clock()
    totals = sweep_postproc_grid(
        holdout,
        _read,
        _predict,
        thresholds,
        close_sizes,
        min_areas,
        progress=_progress,
    )
    rows = postproc_rows(
        predictor_name,
        totals,
        holdout_files=len(holdout),
        views=count_views(holdout),
    )
    write_postproc_rows(out_csv, rows)
    return PostprocGrid(
        rows=rows,
        inference_seconds=inference_seconds,
        total_seconds=clock() - begin,
    )


def describe_postproc_grid(grid: PostprocGrid, out_csv: Path, top: int = 15) -> str:
    """The text the post-processing notebook prints after the grid."""
    return "\n".join(
        [
            format_postproc_table(grid.rows, limit=top),
            f"wrote {len(grid.rows)} rows to {out_csv} (top {top} shown)",
        ]
    )


def describe_sweep(sweep: HoldoutSweep, out_csv: Path) -> str:
    """The text the notebooks print after the sweep."""
    lines = [format_table(sweep.rows), f"wrote {len(sweep.rows)} rows to {out_csv}"]
    if sweep.best is None:
        lines.append("best row: none (every pooled_pq is nan)")
    else:
        best = sweep.best
        lines.append(
            f"best row: min_area={best['min_area']} close_size={best['close_size']} "
            f"pooled_pq={best['pooled_pq']}"
        )
    return "\n".join(lines)


__all__ = [
    "PREDICTOR_NAME",
    "DiceBceLoss",
    "HoldoutSweep",
    "PostprocGrid",
    "TrainResult",
    "WeightEma",
    "build_model",
    "build_optimizer",
    "checkpoint_dict",
    "describe_postproc_grid",
    "describe_sweep",
    "load_checkpoint",
    "make_predict_fn",
    "model_from_checkpoint",
    "normalize_gray",
    "predict_probability",
    "predict_probability_resized",
    "require_cuda",
    "resized_predictor",
    "run_holdout_sweep",
    "run_postproc_grid",
    "save_checkpoint",
    "semantic_from_probability",
    "set_seed",
    "tensor_sha256",
    "tiled_predictor",
    "train_model",
    "whole_frame_model_from_checkpoint",
]
