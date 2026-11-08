"""End-to-end smoke test of the native-resolution crop run, on CPU.

Runs the same ``filament`` functions that ``notebooks/train_unet_crops.ipynb``
calls, in the same order, on a few synthetic 256 x 256 frames with fake
filaments: split by date, decode into RAM, train a ResNet-34 U-Net on 64 px crops
for one epoch, or for two when the recipe has a warmup (``encoder_weights=None``,
so nothing is downloaded; ``--model`` picks another run of
``filament.crop_config.CROP_MODELS``), write and re-read the checkpoint, check its fingerprint, run tiled holdout
inference, and write the ``min_area`` sweep CSV. It needs torch and
segmentation-models-pytorch, never a GPU, and no MAGFiLO data. The numbers it
prints say nothing about model quality.

    PYTHONPATH=src python scripts/smoke_crop_train.py [--out DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch

from filament.annotations import FilamentAnnotation, TrainImage
from filament.checkpoint import (
    checkpoint_fingerprint,
    crop_checkpoint,
    tensor_sha256,
    verify_and_report,
)
from filament.crop_config import (
    CROP_MODELS,
    DEFAULT_CROP_MODEL,
    crop_model_config,
)
from filament.crop_data import load_train_cache
from filament.crop_training import (
    describe_sweep,
    load_checkpoint,
    model_from_checkpoint,
    run_holdout_sweep,
    train_model,
)
from filament.train_data import split_training_images

SIDE = 256
FILES = 10  # ten dates: the two newest are the holdout
CROP = 64


def make_records(image_dir: Path, *, seed: int = 0) -> list[TrainImage]:
    """Write ``FILES`` synthetic frames to ``image_dir`` and return their records.

    Each frame is a bright noisy disk on dark sky with two dark stripes (the
    fake filaments). A stripe is both drawn and annotated as a polygon.
    """
    rng = np.random.default_rng(seed)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = np.ogrid[:SIDE, :SIDE]
    disk = (rows - SIDE / 2) ** 2 + (cols - SIDE / 2) ** 2 <= (SIDE * 0.47) ** 2
    records: list[TrainImage] = []
    for index in range(FILES):
        gray = np.where(disk, 190.0, 8.0) + rng.normal(0, 4, (SIDE, SIDE))
        annotations = []
        for number in range(2):
            x0 = int(rng.integers(60, 150))
            y0 = int(rng.integers(60, 190))
            length = int(rng.integers(40, 80))
            thickness = int(rng.integers(8, 14))
            polygon = [
                x0, y0,
                x0 + length, y0 + thickness // 2,
                x0 + length, y0 + thickness // 2 + thickness,
                x0, y0 + thickness,
            ]  # fmt: skip
            points = np.array(polygon, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(gray, [points], 60.0)
            annotations.append(
                FilamentAnnotation(
                    id=index * 10 + number,
                    image_id=index,
                    category_id=1,
                    spine=None,
                    segmentation=(tuple(float(v) for v in polygon),),
                )
            )
        name = f"202001{index + 1:02d}120000Aa.jpeg"
        ok = cv2.imwrite(str(image_dir / name), np.clip(gray, 0, 255).astype(np.uint8))
        if not ok:
            raise RuntimeError(f"could not write {name}")
        records.append(
            TrainImage(
                id=index,
                file_name=name,
                date_captured=f"2020-01-{index + 1:02d} 12:00:00",
                height=SIDE,
                width=SIDE,
                annotations=tuple(annotations),
            )
        )
    # A second annotator on the oldest file, who drew only its first filament,
    # so the train side has a file with two views.
    oldest = records[0]
    records.insert(
        1,
        dataclasses.replace(
            oldest, id=FILES, annotations=oldest.annotations[:1], views=()
        ),
    )
    return records


def run(out_dir: Path, model_name: str = DEFAULT_CROP_MODEL) -> dict:
    """Run the whole path and return what the checks need."""
    started = time.monotonic()
    expected = crop_checkpoint(model_name)
    image_dir = out_dir / "images"
    weights_path = out_dir / Path(expected.weights_path).name
    sweep_path = out_dir / f"holdout_sweep_{model_name}.csv"
    device = torch.device("cpu")

    records = make_records(image_dir)
    split = split_training_images(records)
    print(f"train files={len(split.train)} holdout files={len(split.holdout)}")

    preset = crop_model_config(model_name)
    cache = load_train_cache(
        split.train,
        image_dir,
        progress_every=4,
        agreement_weights=preset.agreement_weights,
    )
    config = dataclasses.replace(
        preset,
        encoder_weights=None,
        epochs=1 + preset.warmup_epochs,
        crops_per_epoch=8,
        crop_size=CROP,
        batch_size=2,
        use_amp=False,
        max_train_hours=1.0,
    )
    model, result = train_model(
        cache.images,
        cache.masks,
        config,
        device,
        checkpoint_path=weights_path,
        extra_metadata={"tile_size": CROP, "tile_overlap": 16, "threshold": 0.5},
    )

    checkpoint = load_checkpoint(weights_path)
    fingerprint = checkpoint_fingerprint(checkpoint)
    print(f"tensor_sha256={fingerprint['tensor_sha256']}")
    if fingerprint["tensor_sha256"] != tensor_sha256(model.state_dict()):
        raise RuntimeError("the checkpoint on disk differs from the trained model")
    # A pinned hash belongs to the real Kaggle run, not to this tiny synthetic
    # model: check the encoder against the named entry, and the epochs and the
    # hash against the model just trained.
    verify_and_report(
        fingerprint,
        expected=dataclasses.replace(
            expected,
            epochs=config.epochs,
            tensor_sha256=tensor_sha256(model.state_dict()),
        ),
    )

    loaded = model_from_checkpoint(checkpoint, device)
    sweep = run_holdout_sweep(
        loaded,
        split.holdout,
        image_dir,
        device,
        min_areas=(10, 40),
        out_csv=sweep_path,
        tile_size=CROP,
        overlap=16,
        infer_batch_size=4,
        use_amp=False,
        predictor_name=Path(expected.weights_path).stem,
    )
    print(describe_sweep(sweep, sweep_path))
    print(
        f"epochs_completed={result.epochs_completed} "
        f"train_seconds={result.train_seconds:.1f} "
        f"holdout_inference_seconds={sweep.inference_seconds:.1f}"
    )
    print(f"smoke test finished in {time.monotonic() - started:.1f} s")
    return {
        "weights_path": weights_path,
        "sweep_path": sweep_path,
        "checkpoint": checkpoint,
        "result": result,
        "sweep": sweep,
        "holdout": split.holdout,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--model",
        choices=sorted(CROP_MODELS),
        default=DEFAULT_CROP_MODEL,
        help="named crop model whose encoder and recipe are used",
    )
    args = parser.parse_args(argv)
    if args.out is not None:
        run(args.out, args.model)
    else:
        with tempfile.TemporaryDirectory(prefix="smoke_crop_") as directory:
            run(Path(directory), args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
