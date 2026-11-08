# Solar Filament Segmentation Challenge 2026

Our solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) (Kaggle, IEEE BigData Cup 2026): one mask per solar filament in 2048 × 2048 GONG H-alpha images (MAGFiLO dataset).

## Result

| | Holdout PQ | Public leaderboard | Seconds per image (T4) |
| --- | --- | --- | --- |
| Final pipeline | 0.4255 (0.4210 cross-fitted) | 0.35 | 1.40 |

The holdout is the most recent 20% of observation dates in the training data (142 files, 231 annotator views). [`docs/RESULTS.md`](docs/RESULTS.md) has every experiment, including the ones that did not help.

## Pipeline

1. U-Net with a ResNet-34 encoder (`segmentation-models-pytorch` 0.6.0), trained on native-resolution 512 × 512 crops for 80 epochs.
2. Tiled inference over the full frame (tiles of 512, overlap 128) with flip test-time augmentation (4 views).
3. Threshold 0.45, solar-disk gate, an elliptical closing of size 11 that only decides which pixels form one instance, connected components.
4. Drop components under 400 pixels and components with mean probability below 0.8.
5. One RLE per component (`pycocotools`), validated before the CSV is written.

## Reproduce the predictions without retraining

[`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb) loads the weights in [`weights/`](weights/) (fp16, 49 MB, SHA-256 in `weights/MANIFEST.json`), checks that the holdout PQ is 0.4255 ± 0.002, and writes `submission.csv` for the 180 test images. Step-by-step instructions for Kaggle and for a local run: [`docs/reproduce.md`](docs/reproduce.md).

On Kaggle: upload this repository as a dataset of your own, attach it and the competition to the notebook, choose a T4 GPU with Internet on, and run all cells.

Locally, with a GPU:

```bash
uv sync --locked --extra gpu   # or: pip install -r requirements.txt
```

then open `notebooks/reproduce.ipynb`. The competition data goes in `data/MAGFiLO_1.0_Kaggle_2026/` and is not part of this repository.

## Repository layout

- `src/filament/`: all the logic (data loading, training, tiled inference, post-processing, metrics, RLE, validation). The notebooks only call it.
- `notebooks/`: Kaggle notebooks for training (`train_unet_crops.ipynb`), probability export, the sweeps behind each decision, the final evaluation (`evaluate_final.ipynb`) and the reproduction.
- `tests/`: unit tests on synthetic data. `./scripts/check.sh` runs ruff and pytest, as CI does (CI does not install torch).
- `results/`: small CSVs of the experiments in `docs/RESULTS.md`.
- `docs/`: results log, reproduction guide, Kaggle setup and project reports.

## Data and attribution

The competition data (MAGFiLO) is not included. GONG data, NSO/AURA/NSF. Dataset: Ahmadzadeh et al., *A dataset of manually annotated filaments from H-alpha observations*, Scientific Data 11, 1031 (2024), doi:10.1038/s41597-024-03876-y.
