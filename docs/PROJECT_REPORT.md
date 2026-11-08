# Project report: Solar Filament Segmentation Challenge 2026

Status as of 2026-11-02, on `main` @ `faa4ffa` (41 merged PRs). The competition closes 2026-11-15 (23:59 CDT as stated on the Overview page; see `AGENTS.md`).

This report is built from `git log`, the 41 merged pull requests (descriptions and review comments), `docs/RESULTS.md` and the rest of `docs/`. A number that none of those sources contains is marked **[UNKNOWN]**. Nothing here is estimated unless it says so.

## 1. Task and setup

- **Task.** Per-filament instance masks on 2048 × 2048 H-alpha GONG frames (MAGFiLO data). The Kaggle leaderboard metric is Panoptic Quality (PQ); the organizers also say that end-to-end computational efficiency counts.
- **Data** (`docs/train-annotations.md`). A COCO-style train JSON with 1154 image entries and 8199 polygon annotations, 4 categories (Left, Right, Unidentifiable, Ambiguous). All categories are treated as one filament class. There is no mask folder; masks are rasterized from polygons. 180 test images [VERIFICAR] (the figure comes from the inference notebook text; the code does not check it, and the organizers' number was never read).
- **Local validation.** The most recent 20% of calendar dates are held out. After collapsing repeated file names, that is 565 train files and a 142-file holdout with 231 annotator views. Scores are one **pooled per-view PQ**: TP, FP, FN and IoU sums are added over every annotator view of every holdout file and one PQ is read at the end (`filament.metrics.fast_view_counts`, `PqCounts`).
- **Pipeline on `main`.** Frame → foreground probability (U-Net, tiled at native resolution) → optional flip TTA → threshold → solar-disk gate → connected components (optionally after an elliptical closing that only decides the labels) → drop components below `min_area` → one RLE per component (`pycocotools`) → `submission.csv` → local validator.
- **Process.** Every change is a PR against `main`. From PR #29 (2026-11-01) a PR merges when `./scripts/check.sh` passes in CI and a different agent has approved it. No data, weights or submission CSVs are committed; agents may submit to Kaggle without asking only when the expected gain in validation is more than +0.01 against the best submission; otherwise a submission needs Nacho's OK (rule in `AGENTS.md`, confirmed 2026-11-06).

## 2. Summary table

Holdout PQ is the pooled per-view PQ on the 142-file date holdout unless noted. LB is the public leaderboard (about 50% of the test set, 2 decimals).

| # | Date | Change | Holdout PQ | Public LB |
| --- | --- | --- | --- | --- |
| 1 | 2026-10-27 | Classic baseline: dark-side adaptive threshold (block 101, offset 10), `min_area` 2000 | not recorded | 0.02 |
| 2 | 2026-10-27 | Baseline after PR #8 (block 151, offset 5), `min_area` 2000 | not recorded | 0.06 |
| 3 | 2026-10-28 | U-Net ResNet-18, 512 px whole frame, 3 epochs | Dice 0.5023; pooled PQ [UNKNOWN] (an old per-image PQ of 0.0374 is not comparable) | not recorded |
| 4 | 2026-10-31 | Same, 12 epochs | Dice 0.6339; pooled PQ in rows 5 and 7 (an old per-image PQ of 0.1506 is not comparable) | not recorded |
| 5 | 2026-11-01 | Post-processing sweep, `min_area` × `close_size` | baseline best 0.0734 (`min_area` 1200); r18 12 ep best 0.2886 (`min_area` 400) | n/a |
| 6 | 2026-11-01 | First U-Net submission, `min_area` 240. **3-epoch weights loaded by mistake** | [UNKNOWN] for the 3-epoch weights (0.275 is the 12-epoch model at `min_area` 240) | 0.04 |
| 7 | 2026-11-01 | Corrected r18, 12 epochs, `min_area` 400, `close_size` 0 | 0.2886 | 0.23 |
| 8 | 2026-11-02 | ResNet-34 on native-resolution 512 px crops, 40 epochs, thr 0.5, `close_size` 0, `min_area` 400 (submission 57889475) | 0.3675 | 0.32 |
| 9 | [UNKNOWN] | Row 8 weights + thr 0.45 / `close_size` 15 / `min_area` 400 | [UNKNOWN] | 0.33 |
| 10 | [UNKNOWN] | Row 9 + 4-view flip TTA | 0.403 | 0.34 |
| 11 | 2026-11-02 | ResNet-34 crops, 80 epochs, thr 0.5 / 0 / 400, no TTA | 0.3831 | not submitted |
| 12 | [UNKNOWN] | ResNet-34 crops 80 epochs + TTA, thr 0.45 / `close_size` 11 / `min_area` 400 | 0.409 | 0.34 |
| – | 2026-11-02 | r18 + r34 probability ensemble (weights 0.5/0.5 and 0.3/0.7), with and without TTA | below r34 alone; numbers [UNKNOWN] | not submitted |

Rows 9, 10 and 12 come from Nacho's notes and are not in `docs/RESULTS.md`. Rows 9 and 10 are taken to use the 40-epoch weights because they are listed as increments of row 8. Their dates are after row 8 (2026-11-02) and otherwise [UNKNOWN]; the notebooks and submission IDs are [UNKNOWN] too. The grids and CSVs that chose the settings of rows 9, 10 and 12 (`holdout_postproc_r34.csv`, `holdout_tta_ensemble.csv`) are not committed.

## 3. Chronology

### Phase 0: repository and tooling (2026-10-26 to 10-27)

| PR | What | Why / expected gain |
| --- | --- | --- |
| #1 | `AGENTS.md` (project prompt), placeholder file removed | One source of project rules for coding agents |
| #2 | `scripts/check.sh` (ruff, format, pytest), one-job GitHub Actions, review rule | Same check locally and in CI |
| #3 | `filament.runtime` profiles (local, Kaggle, Colab), Kaggle setup guide, `data/` and `outputs/` git-ignored | Keep competition data out of git |
| #4 | RLE encode/decode and submission CSV writer (`pycocotools`) | The submission format |
| #5 | Pinned runtime dependencies, `uv.lock`, `requirements.txt` | Reproducibility, required by the final delivery |

### Phase 1: classic baseline and the metric (2026-10-27)

| PR | What | Why / expected gain |
| --- | --- | --- |
| #6, #7 | Untuned CPU baseline (dark-side adaptive threshold, 3×3 opening, connected components) and a notebook; formatting fix | First valid submission and a PQ reference |
| #8 | One label image instead of one full-size mask per component; off-disk sky gate | The first version could emit thousands of 4 MB masks on a full-disk frame and run out of memory |
| #9 | Kirillov PQ (greedy one-to-one, IoU > 0.5) | A local metric to compare models without spending submissions |
| #10 | Loader for the train JSON, polygons rasterized in memory | Training targets |
| #11 | Validation split by observation date | A random split would leak neighbouring frames |
| #12 | Baseline `min_area` 128 → 2000 | A probe on 10 test images showed 200–360 components per frame at 128 and a mean of 7.2 at 2000 (train mean 7.1). This was a heuristic, later removed (see #24) |
| #13 | Semantic mask → instances → CSV; `score_instances` | The U-Net needs a path from pixels to filaments |

### Phase 2: first U-Net (2026-10-27 to 10-28)

| PR | What | Why / expected gain |
| --- | --- | --- |
| #14 | Kaggle T4 notebook: `smp` U-Net, ResNet-18 ImageNet, whole frame downscaled to 512 px, Dice + BCE, 3 epochs, batch 4 | The simplest deep baseline. Expected to beat the classic method; untuned |
| #15–#17 | Category 4 renamed `Ambiguous`; private dataset `src` found at any depth; ImageNet-weight text corrected (weights come from `smp-hub/resnet18.imagenet`, Hub license field `other`) | Fixes found when the first notebooks met the real Kaggle layout |
| #18 | `EPOCHS` setting, default 12 | The 3-epoch loss was still falling (1.4370, 1.2121, 1.0906) |
| #19 | `score_holdout`: baseline and U-Net scored at 2048 on the same holdout | Comparable numbers |

### Phase 3: inference, data quirks, metric fidelity (2026-10-31)

| PR | What | Why / expected gain |
| --- | --- | --- |
| #20 | `infer_unet.ipynb`: predict the test JPEGs and write `submission.csv`; it never submits | A reproducible path to a submission |
| #21 | Repeated `file_name`s in the train JSON collapse into one sample (semantic target = union of all polygons) | A file would otherwise sit on both sides of the split. Whether repeats are separate annotators was [VERIFICAR] in the PR |
| #22 | `validate_submission`: header, unique ids, RLE decode, non-empty and disjoint masks | Catch a malformed CSV before upload |
| #23 | Pooled per-view PQ: every pair with IoU > 0.5 counts, one PQ over all views | The holdout metric had to follow the annotator-view structure. Equality with the organizer notebook is still [VERIFICAR] in the PR |
| #24 | Post-processing sweep notebook; the test-derived 2000 px cut and the opening are removed | The cut came from counting components on test images. `min_area` is now chosen on the holdout |

### Phase 4: sweeps, results log, wrong-weights incident (2026-11-01)

| PR | What | Why / expected gain |
| --- | --- | --- |
| #25, #27 | Explicit `WEIGHTS_PATH`; wider `min_area` grid (80–3000); Kaggle-extracted `.pt` directories are re-zipped | A lone `unet_12ep.pt` was ignored. Kaggle unpacks `.pt` zips into folders |
| #26 | `docs/RESULTS.md` and `results/` | One row per experiment |
| #28 | Print a fingerprint of the loaded checkpoint (path, size, MD5, epochs, encoder, tensor hash) | The first U-Net submission scored 0.04 and the loaded weights were suspected |
| #29 | Merge rule: green CI + approval by a different agent | Speed, without self-approval |
| #30 | Defaults `min_area` 400, `close_size` 0; checkpoint guard (`filament.checkpoint`) stops on wrong epochs or hash | Fix for the wrong-weights submission |
| #31 | Record the corrected submission (LB 0.23) | The holdout 0.289 now tracks the LB reasonably |

### Phase 5: ResNet-34 native-resolution crops (2026-11-01)

A five-PR series (#32–#37) plus a preparatory one. The idea: the ResNet-18 sees a 4× area-averaged frame, so thin barbs can be lost. Train on native-resolution 512 px crops and predict with overlapped tiles.

| PR | What | Expected gain |
| --- | --- | --- |
| #32 | `ExpectedCheckpoint` presets for R18 and R34; tensor hash and fingerprint move into `filament.checkpoint` | The same guard protects both models |
| #33 | Foreground-biased crop sampler (`p_fg` 0.67), flip/rotation augmentation, in-RAM cache (about 4.5 GiB), wall-clock guard | Native resolution without a `DataLoader` |
| #34 | Sliding-window tiled inference (tile 512, overlap 128, 25 tiles per frame) | Full-frame prediction by a crop model |
| #35 | Holdout sweep that predicts each frame once and scores every `min_area` | Cheap post-processing sweeps |
| #36 | Torch training loop (Dice + BCE, Adam 1e-4, AMP, atomic per-epoch checkpoints), holdout evaluation, CPU smoke script | Only resolution, encoder and sampling differ from the 512 px trainer |
| #37 | `train_unet_crops.ipynb`, `infer_unet_crops.ipynb`, `docs/kaggle-gpu.md`, notebook tests | Runnable on Kaggle |

### Phase 6: results and post-processing (2026-11-02)

| PR | What | Expected gain |
| --- | --- | --- |
| #38 | Pin the 40-epoch r34 hash `c8b30524…fc0c`; record 0.3675 / LB 0.32 | Guard enforced for r34 |
| #39 | Grid sweep of threshold × `close_size` × `min_area` from one prediction per frame (`labels_for_min_areas` labels once) | Post-processing is free accuracy; the net is not retrained |
| #40 | `filament.ensemble`: flip TTA, r18 + r34 averaging, multi-config sweep notebook, same flags in `infer_unet_crops.ipynb` | TTA and ensembling without retraining |
| #41 | Pin the 80-epoch hash `3ac77e98…ba5a` (the 40-epoch pin stays as `R34_CROPS_40EP`); default `EPOCHS` 80 | Longer training; record 0.3831 |

## 4. Results

**Models** (pooled per-view PQ, 142 holdout files, 231 views, thr 0.5, `close_size` 0).

| Model | `min_area` 80 | 140 | 240 | 400 | 700 | 1200 | 2000 | 3000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r34 crops, 40 epochs (about 130 s/epoch on a T4) | 0.330 | 0.342 | 0.356 | **0.368** | 0.346 | 0.281 | 0.201 | 0.125 |
| r34 crops, 80 epochs (about 127 s/epoch, 168.7 min of training, 3.03 h notebook) | 0.354 | 0.362 | 0.377 | **0.383** | 0.368 | 0.306 | 0.230 | 0.131 |

For r18 (12 epochs) only 240 / 400 / 700 are in `RESULTS.md`: 0.2750 / 0.2886 / 0.2783. The baseline's best is 0.0734. `min_area` 400 is the optimum for all three U-Nets.

**Increments.** Differences below are plain subtractions of the figures in the summary table.

| Step | Holdout | LB |
| --- | --- | --- |
| r18 → r34 crops (40 epochs) | 0.2886 → 0.3675 (+0.079) | 0.23 → 0.32 |
| 40 → 80 epochs (no TTA, same post-processing) | 0.3675 → 0.3831 (+0.016) | not submitted |
| r34 40 epochs: + post-processing 0.45/15/400 | [UNKNOWN] | 0.32 → 0.33 |
| + 4-view flip TTA | → 0.403 | 0.33 → 0.34 |
| 80 epochs + TTA, 0.45/11/400, versus row 10 | 0.403 → 0.409 (+0.006) | 0.34 → 0.34 |

## 5. What worked and what did not

**Worked**
- **Native-resolution crops with a bigger encoder.** The largest single gain: +0.079 holdout and +0.09 LB over the 512 px ResNet-18. The PR series changed resolution, encoder and sampling together, so the share of each is not separated.
- **More epochs.** 40 → 80 epochs moved every `min_area` point by 0.006 to 0.029 in the sweep (0.016 at the optimum). The loss curve and the learning-rate schedule were not examined here; the rate is constant.
- **Tuning `min_area` on the holdout.** 400 beat the earlier 2000 and 240 for every U-Net.
- **Post-processing and flip TTA.** Each added about 0.01 LB at 2 decimals.
- **Checkpoint guard.** After it, no wrong-weights run has been recorded.

**Did not work, or is not shown to work**
- **r18 + r34 ensemble.** Below r34 alone on the holdout [numbers UNKNOWN]. The r18 is a 12-epoch, 512 px whole-frame model (0.2886) with no flip augmentation and was ensembled after a bilinear resize of its map. It is not a crop model, and the cause of the drop was not isolated. It has not been submitted.
- **`close_size` 3.** It did nothing for the U-Net (a 1 px gap becomes 4 px after nearest upsampling from 512) and hurt the classic baseline at every `min_area`.
- **Small holdout gains do not show on the LB.** Row 12 is 0.006 above row 10 on the holdout and both are 0.34 on the LB. With a public score of 2 decimals on about half of the test set, a 0.005 difference is not visible, and the sweeps compare 90 to 168 settings on one holdout, so a gain of a few thousandths can be selection noise.
- **Holdout is above LB by about 0.05 to 0.07 in every paired case.** r18: 0.289 vs 0.23. r34 40 epochs: 0.368 vs 0.32. Row 10: 0.403 vs 0.34. Row 12: 0.409 vs 0.34. The cause is [UNKNOWN]. The LB covers about half the test set; the local metric has no efficiency term and its equality with the organizer notebook is unconfirmed.

## 6. Problems found and fixes

| Problem | How it showed | Fix |
| --- | --- | --- |
| **Wrong weights submitted**: the first U-Net run used 3-epoch weights (`submissionunetv4/unet.pt`) instead of the 12-epoch ones | LB 0.04 against a holdout of 0.275 | #28 fingerprint, #30 guard on epochs and tensor SHA-256, #32 presets per model; corrected LB 0.23 (#31) |
| **Hash not pinned** for a new model | #32 review: with `tensor_sha256 = None` any resnet34 file of the right epoch count passes with a warning | Pinned after each run: #38 (40 epochs), #41 (80 epochs) |
| **Pinned hash broke the torch smoke tests** | Two tests failed once `R34_CROPS` held the real hash, because a synthetic model was checked against it | #41: the tests check the synthetic model against its own hash |
| **Notebook weight lookup** | A lone `unet_12ep.pt` was ignored (#25); Kaggle unpacks uploaded `.pt` into folders (#27); the default path depends on the exact slug of the training notebook | Explicit `WEIGHTS_PATH`; if the default path is missing the notebook searches by file name, picks one match, otherwise stops and lists candidates; folders are re-zipped |
| **Stale `solar-code` dataset** | Every notebook needs the repo zip re-uploaded after a merge; an old version fails at import | Notebooks check for the new function and say so (#39, #40). Re-upload stays manual |
| **Full-disk memory blow-up** (#8) | Thousands of full-size masks per frame | One label image, RLE per component |
| **Test-set-derived area cut** (#12) | `min_area` 2000 came from counting components on test images | Removed in #24; `min_area` is chosen on the date holdout |
| **Train JSON quirks** | Category 4 is `Ambiguous` (#15); repeated `file_name`s (#21) | Loader fix; collapse to one sample, annotator entries kept as views (#23) |
| **Metric mismatch** | The greedy per-image PQ (#9) did not follow the annotator-view structure | Pooled per-view PQ (#23) |
| **Vacuous test** | #35 review (REQUEST_CHANGES): the equivalence test had TP 0 at every `min_area`, so it never checked the IoU sum | Reworked with a mask that matches ground truth (tp > 0), then approved |
| **CI red on `main`** | `ruff format --check` failed on a notebook (#7) and an uploaded note (#21) | Reformatted |
| **Even `close_size`** | 10 and 20 were requested; `label_foreground` accepts only 0 or odd | 11 and 21 used (#40) |
| **Silent CPU fallback** | #37 review nit: `infer_unet_crops.ipynb` only prints a warning if there is no GPU | **Not fixed** |
| **Review notes** | #32: `checkpoint_fingerprint` was unused by the notebook; #33: `jitter` not validated against `crop_size`, last batch of size 1 not excluded; #34: NaN scores pass silently | Recorded as non-blocking; not confirmed fixed |

## 7. Not verified

- AMP on a T4: PR #36 states the CUDA/AMP path was not run before merge. The Kaggle training runs recorded in `RESULTS.md` are the only evidence that it works. The sweep, TTA and ensemble notebooks (#39, #40) were also not executed before merge; their Kaggle runs are known only through Nacho's notes.
- The pooled PQ rule against the organizer self-evaluation notebook ([VERIFICAR] in #9, #10 and #23).
- The Hub license of `smp-hub/resnet18.imagenet` and `resnet34.imagenet` (r18: field `other`; r34: not checked).
- End-to-end inference time, which the organizers count in the score. Only the per-epoch training time is recorded.

## 8. Next steps

1. **Bigger encoder.** PR #42 (open, not merged) adds `MODEL = "r50"` (120 epochs) and `"effb3"` (100 epochs) to the training, sweep and inference notebooks. Its time (4.7 to 8.0 h for r50) and memory figures are estimates scaled from the measured 126 s/epoch of r34 and have not been run on a T4. Epochs differ from r34's, so a gain is not due to the encoder alone.
2. **Longer or better-scheduled training.** 40 → 80 epochs still gained +0.016. The learning rate is constant and only the last epoch is saved.
3. **Larger context or input scale.** Training crops are already at native resolution, so "higher resolution" would mean an upscaled input or larger crops (for example 1024 px) to give the network more context for thin barbs. Both are untested and cost more per epoch.
4. **A crop-trained second model for the ensemble.** The existing ensemble member is a 512 px whole-frame r18.
5. **A less noisy selection.** Evaluate on more than one date split or fold, and prefer a setting that is good across the grid rather than at its best cell.
6. **Delivery items** from `AGENTS.md`: the 4-page report, a public repo with a notebook that runs the whole pipeline from weights stored in the repo, and `requirements.txt`. Committing weights and making the repo public need Nacho's explicit OK, and the weights are not in the repo today.
