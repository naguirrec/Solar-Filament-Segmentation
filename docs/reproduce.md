# Reproducing the final predictions

`notebooks/reproduce.ipynb` runs the whole final pipeline from the model weights, without retraining. It is the notebook the final delivery asks for (`AGENTS.md`, `docs/ROADMAP_V2.md` phase G).

**The pipeline is fixed** (`docs/RESULTS.md`): U-Net with a ResNet-34 encoder, 80 epochs on native-resolution crops (tensor SHA-256 starts `3ac77e98`); tiled inference (overlap 128) with flip TTA (four views); threshold 0.45; `close_size` 11; solar-disk gate; `min_area` 400; object filter, mean probability of a component at least 0.8. The detector of phase F is not part of it.

What the notebook does, in order:

1. Stops if there is no GPU (unless `ALLOW_CPU`).
2. Finds the weights and verifies them (a wrong hash stops the run).
3. Re-scores the date holdout (142 files, 231 views) with the existing pooled PQ code and compares it with the documented **0.4255**; it stops outside ±0.002 (not in a `PILOT`). This happens before any CSV is written.
4. Shows 6 overlays of masks on holdout images (fixed seed) and saves `overlays_holdout.png`.
5. Predicts the 180 test images, writes `submission.csv` and validates it with `filament.validate`.
6. Prints and saves (`timing.json`) the seconds per image, for inference alone and for the whole pipeline.

It never submits. There is no Kaggle submit call.

All the logic is in `src/filament` (`reproduce.py`, `weights.py`, plus the modules the other notebooks use). The cells only set options and call it.

## Settings

Everything is in the first code cell. The ones to know:

| Setting | Default | Meaning |
| --- | --- | --- |
| `PILOT` | `False` | Run `PILOT_FRAMES` (4) holdout frames and test images only, to check the plumbing. Not a result: the PQ is not compared with 0.4255 and the timing is not a measurement. |
| `ALLOW_CPU` | `False` | Without a GPU the notebook stops. `True` runs on the CPU (very slow; use it with `PILOT`). |
| `DATA_ROOT` | `None` | Folder with `train/` and `test/`. `None`: `FILAMENT_DATA_ROOT`, the Kaggle path, then `data/MAGFiLO_1.0_Kaggle_2026` and `data/` under the repository root (not the working directory, which is `notebooks/` in Jupyter). |
| `OUTPUT_DIR` | `None` | `/kaggle/working` on Kaggle, `outputs/reproduce/` locally. |
| `WEIGHTS_FILE` | `unet_r34_crops_fp16.pt` | The file looked for in `weights/`. |

`THRESHOLD`, `CLOSE_SIZE`, `MIN_AREA`, `OBJECT_FILTER`, `USE_TTA` are the final pipeline. Do not change them for a reproduction: a test keeps them equal to the documented values, and the PQ check fails if they differ.

## On Kaggle

1. Import `notebooks/reproduce.ipynb`. Settings: Accelerator **GPU T4**, Internet **On** (only for `pip install segmentation-models-pytorch==0.6.0`; Kaggle's own PyTorch is used).
2. Attach the competition `filament-segmentation-2026` and **this public repository as a dataset of your own**: on Kaggle, Datasets → New Dataset, upload the repository folder or the zip from GitHub (the notebook finds `src` at any depth, and the version must contain `filament/reproduce.py`). Nothing private is needed.
3. The weights: `weights/unet_r34_crops_fp16.pt` is committed, so it is in that dataset, and nothing else is attached.
4. Save & Run All. The notebook prints which weights it used (`repo` or `kaggle`).

The author's own route replaces step 2 with the private dataset `ignacioaguirrecandia/solar-code`. A jury account cannot attach it, so it is not the route to follow.

Outputs: `/kaggle/working/submission.csv`, `overlays_holdout.png`, `timing.json`.

## Locally

From the repository root, with a GPU:

```bash
uv sync --locked --extra gpu            # or: pip install -r requirements.txt
```

Then open `notebooks/reproduce.ipynb` with a Jupyter that uses this environment (`ipykernel` is pinned in `requirements.txt` for the kernel; the Jupyter front end is not) and run all cells. Jupyter starts in `notebooks/`; the notebook finds the repository from there. Put the competition data in `data/MAGFiLO_1.0_Kaggle_2026/` (with `train/` and `test/`), or set `DATA_ROOT` or `FILAMENT_DATA_ROOT`. The data is not in the repository and must not be committed.

The lock has no Intel macOS (`x86_64`) wheel for torch 2.11.0, so `pip install -r requirements.txt` fails on an Intel Mac. Use Linux or Windows, an Apple-silicon Mac (CPU or MPS, with `ALLOW_CPU` for a pilot), or Kaggle.

`requirements.txt` is `uv export --frozen --no-dev --no-emit-project --no-hashes --extra gpu -o requirements.txt`. The `gpu` extra of `pyproject.toml` pins `torch==2.11.0` and `segmentation-models-pytorch==0.6.0`; `torchvision` (0.26.0) and `timm` (1.0.31) follow from the lock. CI does not install the extra: `./scripts/check.sh` runs without torch and skips the torch tests.

## Weights: how they get into the repo

The owner gave the OK on 2026-11-07 and `weights/unet_r34_crops_fp16.pt` is committed. `.gitignore` still ignores `*.pt` and `*.pth`, so any other weights file needs the owner's OK and `git add -f`.

How the file was made:

```bash
# the checkpoint is unet_r34_crops.pt from the training notebook train-unet-r34-crops-80ep
uv run --extra gpu python scripts/export_weights.py path/to/unet_r34_crops.pt \
    --name unet_r34_crops_fp16.pt
git add -f weights/unet_r34_crops_fp16.pt weights/MANIFEST.json
```

The script writes the `state_dict` in fp16 (about 49 MB) with the fields needed to rebuild the network, fails if the file is over 100 MB, and records in `weights/MANIFEST.json`: the file SHA-256, the tensor SHA-256 of the fp16 state, the tensor SHA-256 of the original (it has to be the pin `3ac77e98…` of `filament.checkpoint.R34_CROPS`), size, encoder and epochs. A backup copy as a GitHub Release is still to be decided.

fp16 changes the values slightly. The hashes say which weights these are; only the holdout PQ check of the notebook shows that the conversion is harmless.

## What is verified and what is not

Verified on real data, once, on a Kaggle Tesla T4 (`reproduce-final-r34`, code at main @ 966b1ab, the notebook unchanged, the weights file of this repository): holdout PQ 0.4255 against the documented 0.4255, 180 test images, a validated `submission.csv` of 1258 rows, 1.08 s per image for inference and 1.40 s for the whole pipeline. That run took the code and the weights from the author's private `solar-code` dataset, not from an upload of this repository. Kaggle did not unpack the `.pt`, and its PyTorch was 2.11.0+cu130.

The fp16 masks are not byte-identical to the fp32 submission 57954296 (0.35 on the public leaderboard): 1258 rows against 1256, 22 of the 176 frames with masks identical, foreground IoU 0.99948, 1526 pixels differing. The leaderboard score of the fp16 CSV has not been measured.

Verified also (CI and local runs, synthetic data):

* The weights logic: manifest, file and tensor hashes, the descent from the pinned fp32 hash, lookup order (repo before Kaggle), the Kaggle re-zip of an extracted `.pt`.
* The export script on a tiny synthetic checkpoint, including the 100 MB limit (with torch installed locally; skipped in CI), and that an fp16 file loads into the fp32 ResNet-34 U-Net and predicts the same to within 0.05 on random input.
* The post-processing of a frame, the submission rows and their validation, the pooled-PQ scoring path, the PQ check, the overlays and the timing, with a stand-in for the network.
* The structure of the notebook (settings in one cell, `PILOT = False`, no submit call, no input prompts, order of the steps) and its cells after model loading, run on synthetic data.
* `requirements.txt` agrees with `uv.lock` (a test), and the lock resolves with the `gpu` extra.

Not verified:

* The jury route on an account other than the author's. On the author's account it was run once (`reproduce-jury-route`, 2026-11-08): the GitHub zip of main @ 0286670 uploaded as a dataset, the notebook unchanged, PQ 0.4255, and a `submission.csv` identical to the first run's.
* The local route with a GPU and Jupyter.
* Whether Kaggle's PyTorch stays 2.11.0 (the notebook warns if it does not).
* `requirements.txt` on Linux pulls CUDA 13 wheels for torch 2.11.0 (from the lock). Whether they run on a given machine depends on its NVIDIA driver; not tested. On Kaggle the notebook does not install torch.
* Whether Kaggle unpacks a `.pt` in a dataset made another way (it did not with a zip upload). `weights.py` takes only a file in `weights/`, and the failure message names this possibility.
