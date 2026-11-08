# Kaggle setup

First notebook for the Solar Filament Segmentation Challenge 2026.
Do not publish the notebook, the dataset, or this repository. The GitHub
repository stays private. Do not make it public.

## Kaggle

1. Join [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026) and accept the rules.
2. Create a notebook with **New Notebook**.
3. Add the competition data.
4. Set the accelerator to **None** (CPU). Do not enable the GPU.
5. Zip this repository without `data/`, `.env`, or `kaggle.json`. Upload that ZIP as a **private** Kaggle dataset and attach it to the notebook. Do not make the GitHub repository public.
6. Copy `src` into `/kaggle/working/src` and put that directory on `PYTHONPATH` so `import filament` works. The week-1 notebook does this when it finds `src` in the private dataset.
7. Set `FILAMENT_DATA_ROOT` to the **test** image folder. A sibling `train` directory exists; do not point the submission at train.

`/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images`

Test files are `.jpeg`, with names such as `20220621185332Bh.jpeg`. Baseline defaults target a full-disk frame on the order of 2k pixels and are not fitted on MAGFiLO. `require_data_root()` raises an error that names `FILAMENT_DATA_ROOT` when that variable is unset.

```python
import os
import sys

os.environ["FILAMENT_DATA_ROOT"] = (
    "/kaggle/input/competitions/filament-segmentation-2026/"
    "MAGFiLO_1.0_Kaggle_2026/test/test_images"
)

# Copied private-dataset src. Not the train/ directory.
sys.path.insert(0, "/kaggle/working/src")

from filament.runtime import resolve_runtime

runtime = resolve_runtime()
print(runtime.profile)  # kaggle
print(runtime.device)  # cpu
print(runtime.output_dir)  # /kaggle/working
print(runtime.require_data_root())
```

Kaggle is selected when `KAGGLE_KERNEL_RUN_TYPE` is set or `/kaggle/working` exists. Output on this profile is `/kaggle/working`. Set `FILAMENT_OUTPUT_DIR` when you need a different directory. Do not paste tokens into the notebook.

### GPU

Leave the accelerator on **None**. Do not enable the GPU. This baseline is CPU only, so the free weekly GPU quota stays unused.

## Colab backup

Same code, on a CPU runtime. Set the profile and the data folder yourself:

```python
import os

os.environ["FILAMENT_PROFILE"] = "colab"
os.environ["FILAMENT_DATA_ROOT"] = "/content/filament-data"  # folder you uploaded

from filament.runtime import resolve_runtime

runtime = resolve_runtime()
```

`FILAMENT_PROFILE=colab` selects this profile even when Colab's own variables are missing. Output defaults to `/content/outputs`. The device stays `cpu`. Upload the competition data to the folder you named; do not commit it.

## Local machine

Development machine: a Linux laptop with 16 GB of RAM and about 4 GB of VRAM. Use it for development and CPU smoke tests. It is not a place to train a U-Net at 2048.

With no environment variables, the profile is `local`, the data directory is `data`, the output directory is `outputs`, and the device is `cpu`. The same three variables override those defaults: `FILAMENT_PROFILE`, `FILAMENT_DATA_ROOT`, and `FILAMENT_OUTPUT_DIR`.

## Secrets and data

Never commit MAGFiLO images or labels, `kaggle.json`, or `.env`.

`data/` and `outputs/` ignore every file except their own `.gitignore`, so a download or a run cannot be committed by accident.

`.env` and `kaggle.json` are listed in the root `.gitignore`. Keep them untracked anyway.

## Week-1 notebook

[`notebooks/week1_submission.ipynb`](../notebooks/week1_submission.ipynb) runs the classic CPU baseline and writes `submission.csv` through `filament.submission` when `require_data_root()` returns a directory. The baseline builds one label image and encodes each component to RLE on its own, so a full-disk frame does not allocate a list of full-size masks.

Keep the accelerator on **None**. Do not enable the GPU. Set `FILAMENT_DATA_ROOT` to the test image folder above, not train. The notebook still runs a synthetic 64×64 demo when that folder is not a directory. That demo passes a small `block_size` and `min_area`. Defaults used on the test images are an untuned starting point for a full-disk frame on the order of 2k pixels. They are not fitted on MAGFiLO.
