# Train a U-Net on a Kaggle GPU

Step-by-step notebook: [`notebooks/train_unet.ipynb`](../notebooks/train_unet.ipynb).

This is the first training run, for someone who has not used a Kaggle GPU before. It fits a semantic mask. On the date holdout it prints semantic Dice and panoptic quality. It does not write a submission CSV and it does not submit to Kaggle.

The week-1 CPU baseline ([`docs/kaggle-setup.md`](kaggle-setup.md)) leaves the accelerator off. This page is the one that turns the GPU on.

512×512 is an untuned starting size so a free session can finish. Frames on disk are 2048×2048. The mask is downscaled with nearest neighbor, so it stays binary and thin barbs can disappear. The image uses area averaging (512 divides 2048). Both choices are untuned.

The GitHub repository stays private. Do not make it public.

## Settings

1. Create a notebook from the competition so the data is already attached.
2. Accelerator: **GPU T4**. One T4 is enough. A second GPU spends the same free weekly quota faster.
3. Internet: **On**. The notebook installs `segmentation-models-pytorch==0.6.0` (MIT). Kaggle already has PyTorch, so the notebooks do not install it. Locally it is the optional `gpu` extra of `pyproject.toml` (`docs/reproduce.md`); CI does not install it.
4. If that install fails, stop. Do not switch to an encoder weight whose license is research-only.
5. The encoder is `resnet18` with `encoder_weights="imagenet"`. smp 0.6.0 pins resnet18/imagenet to Hugging Face repo `smp-hub/resnet18.imagenet` at revision `3f2325ff978283d47aa6a1d6878ca20565622683`, file `model.safetensors` (Hub license field `other`, not BSD-3; the model card names `https://download.pytorch.org/models/resnet18-5c106cde.pth` as the original file, and byte-identity with that `.pth` was not checked [VERIFICAR]), and an unauthenticated Hub download is expected. Do not use `ssl` or `swsl`.
6. Before the session ends, download `/kaggle/working/unet.pt` from the Output panel. The session disk is wiped when the session stops. Do not commit the weights.

## Data

Train images, 2048×2048 JPEG:

`/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images`

Train JSON:

`/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`

Do not train on `test/test_images`.

All four categories are one filament class. The target is the union of every polygon on the image. `category_id` is not read.

## Private code

Zip this repo without `data/`, `.env`, or `kaggle.json`. Upload the zip as a **private** Kaggle dataset and attach it. The notebook copies `src` to `/kaggle/working/src` and puts that directory on `PYTHONPATH`.

## Run

Loss is Dice + BCE. Seed is 0. Three epochs and batch size 4 are untuned; batch size 4 fits a T4 at 512.

The holdout is the most recent 20% of calendar dates from `filament.split.split_by_date`. A date stays whole. Pixels are not split. `date_captured` wins; a blank date falls back to `file_name`.

Panoptic quality calls `filament.train_data.instance_pq`. That scores the
prediction against each annotator view of the file and pools those views.
It does not call `score_instances`, and it does not use the stand-in list
that keeps one overlapping mask. The area cut is still
`labels_from_semantic`'s default: 400 at 2048 px and 25 at 512 px. The close
size is that module's default, 0 (no closing). Both come from the 2026-11-01
holdout sweep (see [RESULTS.md](RESULTS.md)). The notebook does not pass
another cutoff or kernel.
A mean of those per-file scores is not the competition PQ.
`filament.baseline_holdout.score_holdout` pools every view of every
holdout image into one PQ. The training notebook still prints one
`instance_pq` per holdout file; this page does not change that notebook.

Weights are written only to `/kaggle/working/unet.pt`. The file is a dict with `state_dict`, `encoder_name="resnet18"`, and `encoder_weights="imagenet"`.

## Colab backup

Use a GPU runtime and the same notebook. Unzip the private repo under `/content` and `cd` into it, or leave that folder directly under `/content`. When `/kaggle/working` is missing, the path cell reads:

`/content/filament-data/train/train_images`

`/content/filament-data/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`

and writes `/content/unet.pt`. Change those three lines if the upload sits somewhere else. The Asus laptop (about 4 GB of VRAM) is not the training machine.

## Native-resolution ResNet-34 crop run

Two notebooks, run in this order. Neither needs a cell edited, and neither submits to Kaggle.

| Step | Notebook | Kaggle name | Produces |
| --- | --- | --- | --- |
| 1 | [`notebooks/train_unet_crops.ipynb`](../notebooks/train_unet_crops.ipynb) | `train-unet-r34-crops-80ep` | `unet_r34_crops.pt`, `holdout_sweep_r34.csv` |
| 2 | [`notebooks/infer_unet_crops.ipynb`](../notebooks/infer_unet_crops.ipynb) | any | `submission.csv` (not submitted) |

It differs from the 512 px ResNet-18 run above in three ways: encoder `resnet34`; random 512 px crops of the native 2048 px frame instead of whole frames downscaled to 512 (so thin barbs are not lost to resizing); and overlapped tiled inference at native resolution. Loss (Dice + BCE), optimizer (Adam, 1e-4), seed and the date holdout are the same, so the comparison changes resolution, encoder and sampling only. None of the settings is tuned. See the first cells of each notebook for the full table.

### 1. Train

1. Upload a **new version** of the `ignacioaguirrecandia/solar-code` dataset from the merged repo. The notebook looks for `src/filament/crop_training.py` and stops with a message if the attached version predates it. [VERIFY] this has to be done after the merge.
2. In Kaggle, **Import Notebook** → `train_unet_crops.ipynb`. Name it `train-unet-r34-crops-80ep` (the inference notebook assumes that slug).
3. Add inputs: the competition `filament-segmentation-2026` and the dataset `ignacioaguirrecandia/solar-code`. Nothing else.
4. Settings: Accelerator **GPU T4** (a single T4, not T4 x2), Internet **On**.
5. **Save & Run All**. It fails in the first code cells, before reading any data, if there is no CUDA GPU or the GPU has under 6 GiB.

Defaults: `EPOCHS = 80`, `CROPS_PER_EPOCH = 4520` (8 crops for each of the 565 train files), `CROP_SIZE = 512`, `BATCH_SIZE = 8`, mixed precision on. `EPOCHS`, `CROPS_PER_EPOCH` and `BATCH_SIZE` come from a timing estimate for one T4 that is not recorded in the repo [VERIFY]. The wall time on Kaggle has not been measured; the target is at most about 6 h in total. The training loop is capped at `MAX_TRAIN_HOURS = 5.0`: if the next epoch is expected to end past it (the slowest completed epoch is the estimate), training stops, the notebook prints a note, and the checkpoint stores the number of **completed** epochs in `epochs`. The checkpoint is rewritten after every epoch, so a killed session still leaves usable weights (`status` is `in_progress` until the end).

The last cells print the timings (data prep, each epoch, total training, holdout inference), the checkpoint fingerprint, and the sweep table with its best row. The holdout is the same 142 files / 231 annotator views as `results/postprocess_sweep_2026-11-01.csv`, scored with the same pooled per-view PQ (`filament.sweep` reuses `fast_view_counts` and `PqCounts`). Each holdout frame is predicted once, and that mask is reused for every `min_area`.

Afterwards, by hand and in a pull request:

- Pin `tensor_sha256` in `R34_CROPS` (`src/filament/checkpoint.py`) from the printed value. It is `None` until then, and the inference notebook prints a `WARNING` that it only checked epochs and encoder. If the guard stopped training early, also set `R34_CROPS.epochs` to the stored number.
- Add a row to [`RESULTS.md`](RESULTS.md) and commit `holdout_sweep_r34.csv` as `results/<date>_holdout_sweep_r34.csv`. Do not commit the weights.

### 2. Infer

1. Import `infer_unet_crops.ipynb`. Inputs: the competition, `ignacioaguirrecandia/solar-code`, and the **output of the notebook `train-unet-r34-crops-80ep`** (Add Input → Notebook Output). Accelerator GPU T4, Internet On (for the `pip install`).
2. **Save & Run All**. It loads `/kaggle/input/notebooks/ignacioaguirrecandia/train-unet-r34-crops-80ep/unet_r34_crops.pt` (the path is `R34_CROPS.weights_path`, written once in `filament.checkpoint`). If that file is missing it searches for `unet_r34_crops.pt`, and re-zips a Kaggle-extracted archive the way `infer_unet.ipynb` does.
3. The checkpoint guard refuses a file whose stored `epochs` or encoder differ from `R34_CROPS`, and, once pinned, one whose `tensor_sha256` differs.
4. It predicts every test image with the same tiled inference as the holdout evaluation, uses `MIN_AREA = 400` and `CLOSE_SIZE = 0` (the ResNet-18 optimum; change them only if you decide to follow the new sweep), and writes `/kaggle/working/submission.csv`. Submitting that file is a separate step that needs an explicit OK.

### Other encoders

`MODEL` in the settings cell of each crop notebook picks a name of `filament.crop_config.CROP_MODELS`. The default `r34` is the run above. Each entry in this table changes only the encoder, the epochs and the wall-clock cap, and has an entry in `filament.checkpoint.CROP_CHECKPOINTS` with the same name. The recipe v2 runs have their own section below.

| `MODEL` | Encoder | Save the training notebook as | Epochs | `MAX_TRAIN_HOURS` | s/epoch on a T4 | Training |
| --- | --- | --- | --- | --- | --- | --- |
| `r34` | `resnet34` | `train-unet-r34-crops-80ep` | 80 | 5.0 | 126 (measured) | 2.8 h (measured) |
| `r50` | `resnet50` | `train-unet-r50-crops` | 120 | 8.5 | 140 to 240 (estimate) | 4.7 to 8.0 h |
| `effb3` | `efficientnet-b3` | `train-unet-effb3-crops` | 100 | 8.5 | 130 to 280 (estimate) | 3.6 to 7.8 h |

**How the estimates were made.** None was measured on a T4 [VERIFY]; the first epoch of the run prints the real speed and peak GPU memory. The start point is the measured 126 s of the ResNet-34. The cost relative to it, for a U-Net with 3 input channels at 512 px:

| Encoder | Parameters | GMACs per crop | Activations, batch 8, fp16 | CPU training step (batch 2, 256 px) |
| --- | --- | --- | --- | --- |
| `resnet34` | 24.4 M | 31.3 (×1.00) | 1.9 GiB (×1.00) | 0.54 s (×1.0) |
| `resnet50` | 32.5 M | 42.5 (×1.36) | 3.7 GiB (×1.89) | 0.53 to 0.61 s (×1.0 to ×1.1) |
| `efficientnet-b3` | 13.2 M | 10.4 (×0.33) | 4.0 GiB (×2.06) | 1.12 to 1.19 s (×2.1 to ×2.2) |

The ranges take the lowest and the highest of those ratios that apply: ×1.1 to ×1.9 for `resnet50`, ×1.0 to ×2.2 for `efficientnet-b3` (its low MAC count is not a low cost: the CPU step is twice as long, and depthwise convolutions use a GPU poorly). The CPU timing comes from one laptop run with few steps, and an earlier run under memory pressure was discarded, so it only orders the encoders. The epochs are chosen so that the pessimistic end still finishes before the cap. If the run is slower, the wall-clock guard stops it, and the checkpoint stores the completed epochs.

**Memory.** Batch 8 and mixed precision stay as they are for every model. The sum of all layer outputs is an upper bound for what a forward pass holds; with the same again for the gradients and about 0.5 GiB for the weights and the Adam state, `resnet50` and `efficientnet-b3` stay at or below about 8.5 GiB of the 16 GB of a T4. An out-of-memory error would come in the first step, because all crops have the same size.

The epochs differ from the ResNet-34's, so a gain from `r50` is not due to the encoder alone: 40 to 80 epochs of the ResNet-34 already moved the holdout from 0.3675 to 0.3831.

After a run, set `tensor_sha256` and `holdout_pq` of that entry in `src/filament/checkpoint.py` from the notebook's output, and `epochs` (with its `label` and the `CROP_MODELS` entry) if the run stopped early. Until then the evaluation and inference notebooks print a `WARNING` that only epochs and encoder were checked.

### Recipe v2 (`r34_v2`, `r34_v2_agree`)

These are D1 and D2 of [`ROADMAP_V2.md`](ROADMAP_V2.md). They keep the ResNet-34, the data, Dice + BCE, the seed and the 80 epochs of `r34`, so the comparison with its 0.3831 (threshold 0.5, `close_size` 0, `min_area` 400, no TTA) changes only the optimization recipe.

| `MODEL` | Save the training notebook as | Checkpoint | s/epoch on a T4 | Training |
| --- | --- | --- | --- | --- |
| `r34_v2` | `train-unet-r34-v2` | `unet_r34_v2.pt` | 126 to 145 (estimate) | 2.8 to 3.2 h |
| `r34_v2_agree` | `train-unet-r34-v2-agree` | `unet_r34_v2_agree.pt` | 126 to 150 (estimate) | 2.8 to 3.3 h |

| Setting | `r34` | `r34_v2` | `r34_v2_agree` |
| --- | --- | --- | --- |
| Optimizer | Adam | AdamW, weight decay 1e-4 | the same as `r34_v2` |
| Learning rate | constant 1e-4 | peak 2e-4, 1 warmup epoch, cosine toward 0 | the same as `r34_v2` |
| Gradient clipping | none | norm 1.0 | the same as `r34_v2` |
| Stored weights | last epoch | moving average (decay 0.999) of the last epoch | the same as `r34_v2` |
| BCE | plain | plain | weighted: `0.4 + 0.6 * k / R` on the union, 1 elsewhere |

The roadmap gives 1e-4 to 3e-4 as the rate. 2e-4 is the middle, and a cosine has half its peak as its mean, so the mean rate stays about the constant 1e-4 of `r34`. It is not tuned. The estimates take the measured 126 s of `r34` as the low end; the high end is a guess for the weight average and clipping at every step [VERIFY], and the first epoch prints the real speed.

During the last 1000 steps, the span the average mostly covers at decay 0.999, the rate is under 0.2% of its peak. So the average should be very close to the last weights, and a gain cannot be credited to it. The weight decay is close to a no-op too: at these rates it shrinks the weights by a factor of about 0.99955 over the whole run, so AdamW here behaves almost like Adam. That leaves the schedule and the clipping as the changes that can matter. D1 changes the recipe as a whole, without separating its parts.

The same seed draws the same crops: the sampler only reads which mask pixels are nonzero. So `r34`, `r34_v2` and `r34_v2_agree` see the same crop positions and flips, and `load_train_cache` stops if a file's annotator views do not add up to its union mask.

To run one:

1. Upload a new version of the `ignacioaguirrecandia/solar-code` dataset from the merged `main`.
2. **Import Notebook** → `train_unet_crops.ipynb`. In the settings cell change `MODEL = "r34"` to the name in the table, and save the notebook under the name in the table.
3. Add inputs: the competition `filament-segmentation-2026` and the dataset `ignacioaguirrecandia/solar-code`. Accelerator **GPU T4** (a single T4), Internet **On**.
4. **Save & Run All**.

Afterwards, in a pull request: set `tensor_sha256` and `holdout_pq` of that entry in `src/filament/checkpoint.py` from the output (and `epochs` if the guard stopped the run; such a run is not comparable, because its schedule did not finish), add a row to [`RESULTS.md`](RESULTS.md), and commit `holdout_sweep_<MODEL>.csv` under `results/`. Do not commit the weights.

### Export probability maps

[`notebooks/export_probs.ipynb`](../notebooks/export_probs.ipynb) predicts each holdout and test frame once per flip and stores the mean and the standard deviation of the flips, so post-processing sweeps, the bootstrap and the component selector run on CPU. Run it once per model, on a T4.

1. **Import Notebook** → `export_probs.ipynb`. In the settings cell set `MODEL` if it is not `r34`.
2. Add inputs: the competition, `ignacioaguirrecandia/solar-code`, and the output of that model's training notebook. Accelerator **GPU T4**, Internet **On**.
3. **Save & Run All**. The last cell scores the cached holdout maps at a few settings; for `r34`, threshold 0.45, `close_size` 11, `min_area` 400 should give about 0.409 (Nacho's notes, not a committed CSV [VERIFY]). The manifest says `cross_check: ok` or `FAILED`, by a floor against the checkpoint's recorded PQ; the 0.409 itself has no tolerance.
4. On the Output tab choose **New Dataset**, name it `solar-probs-<MODEL>`, and keep it private. Do not commit the maps.

A split is one directory with `mean.npy`, `std.npy` and `files.json` (`filament.probcache`), about 2.2 GiB for the holdout and 2.8 GiB for 180 test frames [VERIFY]. The maps are stacked per split instead of one file per frame, so an export leaves 7 files (8 if the checkpoint has to be re-zipped): older forum posts report a limit of 500 output files per notebook, which is not confirmed for today's Kaggle [VERIFY] and would stop 644 files. The time per image is not measured yet [VERIFY]; the notebook prints it, and the manifest stores it.

### Analyze the gap between holdout and leaderboard

[`notebooks/analyze_holdout_gap.ipynb`](../notebooks/analyze_holdout_gap.ipynb) runs on CPU from the exported maps and measures hypotheses H1 (selection) and H2 (annotator mix) of [`ROADMAP_V2.md`](ROADMAP_V2.md) section 3. Attach the competition, `ignacioaguirrecandia/solar-code` and the private dataset `solar-probs-<MODEL>`, leave the accelerator off, and **Save & Run All**. Download `holdout_gap_<MODEL>.csv` and `holdout_per_file_<MODEL>.csv` from the Output tab: both are small and hold no image or mask data, so they go in `results/`.

### Sweep the object filter

[`notebooks/sweep_object_filter.ipynb`](../notebooks/sweep_object_filter.ipynb) runs on CPU from the exported maps, with the same inputs as the gap analysis. It drops components by their probability (mean, maximum or 90th percentile), then moves `min_area` and the pixel threshold, one axis at a time. Its last row is the verdict, by a cross-fit over 5 partitions of the dates against the current setting. Download `holdout_object_filter_<MODEL>.csv`: it is small and holds no image or mask data, so it goes in `results/`.

### Sweep the component selector

[`notebooks/sweep_selector.ipynb`](../notebooks/sweep_selector.ipynb) runs on CPU from the exported maps, with the same inputs as the gap analysis. It fits a small model on the features of each candidate component and emits a candidate when its predicted matched IoU per view is above PQ / 2. All scores are out of fold, by date. Read the `verdict` rows of `holdout_selector_<MODEL>.csv`. It saves no selector, because a selector only holds for one exact checkpoint and is only worth keeping if one is adopted. It uses the scikit-learn that Kaggle has; the repo pins 1.9.1, and another version can give slightly different numbers.

### Instance detector (phase F): train, export, sweep

Three notebooks, in this order. None needs a cell edited and none submits to Kaggle. **The first run of each is a pilot** (`PILOT = True` in its settings cell). Only the train notebook has had its pilot on Kaggle (a T4, 2 epochs, version 3 completed); the export and sweep notebooks have not run on Kaggle yet. Everything marked [VERIFY] in them is an assumption about `rfdetr` 1.12.0 that a pilot confirms or breaks.

| Step | Notebook | Save as | Hardware | Inputs besides the competition and `solar-code` | Produces |
| --- | --- | --- | --- | --- | --- |
| 1 | [`train_detector.ipynb`](../notebooks/train_detector.ipynb) | `train-detector` | GPU T4, Internet On | none | `detector_<MODEL_CLASS>_<RESOLUTION>.pth`, `detector_train_manifest.json` |
| 2 | [`export_detections.ipynb`](../notebooks/export_detections.ipynb) | `export-detections` | GPU T4, Internet On | output of `train-detector` | `detections_holdout.jsonl.gz`, `detections_test.jsonl.gz`, `detector_export_manifest.json` |
| 3 | [`sweep_detector.ipynb`](../notebooks/sweep_detector.ipynb) | any | none (CPU), Internet Off | output of `export-probs-r34`, output of `export-detections` | `holdout_detector_sweep.csv` and a last line `VERDICT <config> pq=... diff=... fraction_better=... adopted=k/5 base_fixed_in_sample=...` |

Before step 1, upload a new version of `ignacioaguirrecandia/solar-code` from the merged repo (the notebooks stop with a message if the version predates `filament.detector_train`, `detector_export` or `detector_sweep`).

- **Train.** `RFDETRSegLarge` at `RESOLUTION = 768` (the Large default is 504, so it is passed explicitly and checked against `patch_size * num_windows` read from the model's config). The splits are `build_detector_splits` with its defaults; the holdout is never written to disk. A pilot uses 40 train files, 10 validation files and 2 epochs, stops within 20 minutes, and prints seconds per epoch, peak GPU memory and the projected time of the full run (train and validation files scaled separately, validation at no cost, so the longest projection). A pilot at batch 4 measured 109 s per epoch, which gives `EPOCHS = 16` (about 6.3 h at 23.6 min per epoch, a projection at batch 4). A later pilot ran out of memory at batch 4 on the T4's 14.56 GiB; batch 2 peaked at 8.49 GiB reserved, so the settings are `BATCH_SIZE = 2` with `GRAD_ACCUM_STEPS = 2`, whose product keeps the library's default (`batch_size` 4, `grad_accum_steps` 1 in rfdetr 1.12.0). Batch 2 speed was not measured before the full run. `MAX_TRAIN_HOURS = 7.0` is a wall-clock guard; `PILOT_STOP_AFTER_MINUTES` (default `None`) makes it fire after that many minutes, so a pilot of about 3 minutes can exercise the guard once the first validation has written a checkpoint. The library writes `checkpoint_best_total.pth`, `checkpoint_best_ema.pth` and `last_ema.pth` to the scratch disk (no per-epoch `checkpoint.pth`: its interval is 10). A watcher copies the current best checkpoint to the saved output whenever it changes (temp file plus `os.replace`), and the train cell survives `KeyboardInterrupt`, `SystemExit` and exceptions, so the checkpoint cell still runs; it raises if training left no checkpoint. The checkpoint handed on is the library's best on the detector validation folder (`CHECKPOINT_CHOICE = "best"`) or the last EMA weights, `last_ema.pth` (`"last"`), printed with its SHA-256. The dataset and the run directory live on the scratch disk, not in the saved output.
- **Export.** Each of the 4 windows of a frame is predicted at score 0.01 (untuned; low on purpose so later cuts are free). Masks are mapped to the frame with `filament.detector_data` and stored as RLE with the score and window index, one line per frame. The `test` split only produces detections: nothing is fitted on it. The manifest has the checkpoint hash, the resolution and seconds per image.
- **Sweep.** From the U-Net probability cache (no U-Net run) and the detections, one axis at a time: score cut (raw, then cross-fit calibrated), dilation, `emit_unclaimed`. The base is the current pipeline (pooled PQ 0.4255 in sample, cross-fit 0.4210; the notebook stops if it cannot reproduce 0.4255). The verdict is the cross-fit by date over 5 partitions with the `axis_report` adoption rule; the paired bootstrap of every row, the detector-only PQ and the number of detections that deduplication drops across windows are in the output. **To submit without Nacho's OK, the cross-fit `diff` must exceed +0.01, at least 4 of the 5 partitions must adopt (`adopted=k/5` on the VERDICT line), and the run must not be a pilot.** `diff` is measured against the base at its fixed setting, in sample (0.4255), which is stricter than the cross-fitted 0.4210. The `on_disk` gate of the pipeline is applied in the claim (`claim_pixels(..., disk=...)`).

### Instance detector, second try: merge only

Phase F's claim fusion lost clearly (best pooled PQ 0.3603 against the base's 0.4255 on the holdout) and the cause was not measured. [`notebooks/sweep_detector_merge.ipynb`](../notebooks/sweep_detector_merge.ipynb) tests the opposite use of the detector: the base labels stay, and `filament.detector_fusion.merge_components` lets a detection (score >= `cut`, mask dilated by 5 px) **join** the base components it covers (at least `coverage` of the component's area inside). It never splits a component, drops one or changes a pixel, so the pixel set equals the base's by construction.

- **Run.** Same inputs and hardware as `sweep_detector.ipynb` (step 3 above: output of `export-probs-r34`, output of `export-detections`, CPU, Internet Off), plus a `solar-code` version that contains `filament.detector_merge_sweep`. Run `PILOT = True` first (its numbers are not a result). It does not train, read a test image, write a submission or submit.
- **Settings, one axis at a time** (untuned placeholders, raw scores, no calibration): `cut` 0.2, 0.3, 0.5, 0.7, 0.9; `coverage` 0.5, 0.8; deduplication off or on. That is 7 distinct settings around the defaults `cut` 0.5, `coverage` 0.5, dedup off.
- **Gate.** The same as the claim sweep: cross-fit by date over 5 partitions, adopted in at least 4 of 5, `diff` above +0.01 against the base at its fixed setting (0.4255 in sample), not a pilot. The notebook stops if the base does not reproduce 0.4255 within 0.002. The last line is the `VERDICT` line; the output is `holdout_detector_merge_sweep.csv`.
- **Diagnostics.** The base, the phase F claim fusion at `DEFAULT_CONFIG` and every merge setting get pooled PQ, TP, FP, FN, SQ (mean IoU of matches) and RQ (`TP / (TP + FP/2 + FN/2)`), printed and written in the `note` column of the CSV; the merge settings also get the number of merges and of frames changed. That says whether claiming lost RQ or SQ. One-to-many and many-to-one counts are not reported: `src` has no helper for them and none was written.
- **Running time** is not measured [VERIFY]. A frame where a setting joins nothing reuses the base counts, so only changed frames are rescored.

### Final evaluation: numbers and figures for the report

[`notebooks/evaluate_final.ipynb`](../notebooks/evaluate_final.ipynb) scores the fixed final pipeline (U-Net r34 80 epochs, flip TTA, threshold 0.45, close_size 11, min_area 400, disk gate, object filter mean >= 0.8) on the date holdout with what the final evaluation asks for: PQ and mean Dice, the IoU and Dice distributions, the one-to-many and many-to-one relations, and overlay figures. The definitions are those of the organizers' self-evaluation notebook; the docstring of `filament.final_metrics` says which function of it each rule comes from, and what it does not define (SQ and RQ, the quantiles, a mean Dice of matched pairs only, and views with no ground truth), with the definition used.

- **Run.** CPU, Internet Off. Attach the competition, `ignacioaguirrecandia/solar-code` (a version with `filament.final_metrics`, `final_eval` and `final_figures`) and the output of `export-probs-r34`. Run `PILOT = True` first (6 holdout files; the numbers are not a result), then **Save & Run All** with `PILOT = False`. The notebook stops if the pooled PQ differs from 0.4255 by more than 0.002.
- **Output.** `holdout_final_metrics.csv` (`metric,value,source`; numbers only, so it goes in `results/`) and small PNGs: `iou_distribution.png`, `dice_distribution.png`, `relations.png` and `overlay_<kind>.png` for good match, fragmentation, over-merge, miss and false positive. The examples are chosen by a fixed rule from the match degrees and `SEED`, not by hand. The figures show dataset frames: do not publish them before the Rules and Data tabs have been read, and do not commit them. They carry "GONG data, NSO/AURA/NSF".
- **Measured.** It ran on Kaggle as `evaluate-final-r34` from a6b78a2 (142 files, 231 views, about 68 s for the frames on a CPU), imported `filament.final_figures` with the Kaggle matplotlib, and reproduced the pooled PQ of 0.4255. The tests run its compute cells on a synthetic holdout.
- It reads no test image, writes no submission and does not submit.

### Check the code without a GPU

CI has no torch, so it tests the sampling, tiling, scoring and guard logic (pure numpy) and skips the torch tests. With a Python that has torch and `segmentation-models-pytorch==0.6.0`:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python scripts/smoke_crop_train.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_crop_training_torch.py
```

The smoke script runs the notebook's code path on a few synthetic 256 px frames on the CPU: split, decode, training (one epoch, two with a warmup), checkpoint write and read-back, tiled holdout inference and the sweep CSV. `--model r34_v2_agree` runs it with the recipe v2 and the agreement weights. It downloads nothing and its numbers say nothing about model quality.
