# Train annotations

Verified on the Kaggle competition folder. The public MAGFiLO paper is not
the source for this layout or for these category names.

There is no mask folder and no sample submission CSV. The competition
folder contains only train images, one train JSON file, and test images.
Test images are not for training.

## Paths

Train images, 2048×2048 JPEG, names like `20220621185332Bh.jpeg`:

`/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images`

Train annotations:

`/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`

Test images (do not train on these):

`/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images`

## Schema

The train file is a COCO-style JSON object.

| Key | What Nacho counted |
| --- | --- |
| `info`, `licenses` | present |
| `categories` | 4 objects |
| `images` | 1154 objects |
| `annotations` | 8199 objects |

`categories` entries have `supercategory`, `id`, and `name`. Every
`supercategory` is `filament`.

| id | name |
| --- | --- |
| 1 | Left |
| 2 | Right |
| 3 | Unidentifiable |
| 4 | Ambiguous |

`images` keys: `license`, `file_name`, `url`, `height`, `width`,
`date_captured`, `id`. The JPEG files are 2048×2048. The loader uses the
`height` and `width` on each image record.

`annotations` keys: `segmentation`, `area`, `iscrowd`, `spine`,
`image_id`, `bbox`, `category_id`, `id`.

`segmentation` is a list of polygons, not compressed RLE. Each polygon is
a flat COCO list of `x, y` coordinates. `rasterize_polygons` in
`src/filament/annotations.py` is the only decoder: `pycocotools`
`frPyObjects`, then `merge` when an annotation has more than one polygon,
then `decode`. Pieces of one annotation are unioned into one uint8 mask.
The loader does not write mask files. Masks are built when
`TrainImage.masks` is called, so a full 2048×2048 stack is not allocated
at load time.

All four category ids are one filament class. The loader keeps every
annotation and leaves `category_id` on the record. The submission CSV
stays `filament_id` and `segmentation_rle`, with no class column.

`spine` is stored as the JSON value. It is not parsed.

`date_captured` and `file_name` are on each image. `date_split_key`
returns `date_captured` when it is a non-blank string, and `file_name`
when the date is missing or blank. This module does not assign folds.

The loader checks the category names above. It does not require the file
to contain 1154 images or 8199 annotations, so a smaller JSON with the
same schema can be read.

## [VERIFICAR]

- `spine` format. The value is kept raw until the layout is known.
- Whether the official metric ignores chirality. Check the competition
  Evaluation page. This note does not decide that, and that page was not
  opened for this loader.

The train JSON repeats some `file_name` values. Annotation counts on those
repeats are close, and not always equal. [VERIFICAR] whether that means the
same image was annotated more than once. `split_training_images` keeps one
sample per `file_name` before the date split. Training still sees one image
per file. The semantic target is the union of that file's polygons. A union
training target is allowed; the score is separate from it.

The collapsed `annotations` list is not the score. It still drops an exact
duplicate polygon, and when two entries of the same file overlap with IoU
above 0.5 it keeps one mask (the other polygons stay on `semantic_only` so
the union still covers them). That "keep one mask" list was a stand-in
panoptic ground truth. The official score does not use it. Each original
entry is stored on `views` and scored on its own: one annotator's instance
masks, not a union and not a consensus. TP, FP, FN, and matched IoU are
summed over every view, then one PQ is computed. The match is every pair
with IoU > 0.5, not a one-to-one greedy match, so overlapping predictions
can inflate TP. A zero denominator is NaN, not 1. There is no per-image
average.

[VERIFICAR] that this pooled count matches the live organizer
self-evaluation notebook byte for byte. That notebook was not downloaded.
Overlapping instances from the same entry stay separate on that entry
unless the polygons are exactly the same. If `date_captured` disagrees
inside a file, the split uses the date in `file_name`.
