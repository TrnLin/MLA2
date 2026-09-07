# New Usage training dataset

**Ready:** `data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv`.

This version has the 38,612 teacher rows, the earlier 120 additions, and the new 567 images:
**39,299 total rows**. It uses the same nine Usage classes. No new NA images were added.
There are 33,459 labelled Usage development images; one additional development row has no Usage
label and is correctly excluded by `get_samples(..., target="usage")`.

## Train and validation

Use the same five-fold method as before: four folds train, one validates. Related products stay
together. All teacher rows and the earlier 120 added rows keep their original IDs and folds.

| Validation fold | Training images | Validation images | New images in training | New images in validation |
|---|---:|---:|---:|---:|
| 0 | 26,751 | 6,708 | 436 | 131 |
| 1 | 26,771 | 6,688 | 458 | 109 |
| 2 | 26,771 | 6,688 | 457 | 110 |
| 3 | 26,772 | 6,687 | 456 | 111 |
| 4 | 26,771 | 6,688 | 461 | 106 |

The new collection has 440 linked families. Smart Casual includes a conservative 49-image
family, so its class counts cannot be exactly 80/20 in every fold. Keeping relatives together
takes priority over exact percentages. All five class/fold tables are saved with the dataset.
The 5,778 holdout and 61 quarantine rows keep their existing membership and blank target fields.

## Rare-class totals

These are development counts, before choosing a validation fold.

| Usage | Teacher | Earlier additions | New additions | Total |
|---|---:|---:|---:|---:|
| Home | 1 | 16 | 120 | 137 |
| Party | 12 | 97 | 127 | 236 |
| Smart Casual | 47 | 6 | 156 | 209 |
| Travel | 22 | 1 | 164 | 187 |
| NA | 61 | 0 | 0 | 61 |

The other four classes are unchanged. All added files are RGB PNGs, width 60 × height 80.
New labels retain their product-text or retailer-collection evidence; they are not presented
as teacher annotations. [Source evidence](../rare_expansion_20260906/README.md) records the
visual review, duplicate checks, photo/source differences and rights limits.

## Use the new split

Extract `teacher_plus_rare_usage_v2.zip` into the project root, keeping its folder layout.
It includes the versioned split, all 687 external images, source evidence, source-border
geometry, all five fold ID lists, and validation results. Reuse the existing teacher image
files from the project or `task3-data.zip`; the ZIP does not duplicate those images.

```python
from fashion.data import get_cv_split, get_samples, load_splits

splits = load_splits("data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv")
train, validation = get_cv_split(splits, validation_fold=0)
train = get_samples(train, target="usage")
validation = get_samples(validation, target="usage")
```

Use folds 0 through 4 for five-fold training. The full `splits.csv` is the authority;
`train.csv`, `validation.csv`, and `cv/fold_*/` are derived views. The two full CSV exports
use fold 0. `added_images.csv` contains all 687 external images; `new_images.csv` contains
only the 567 added in this version.

Use the existing `usage` label map and normal `FashionDataset`. Other target labels on added
images are blank, with false masks. `image_geometry.csv` holds all 687 source-border rectangles
for later training-fold-only channel statistics. Do not reuse statistics or class weights from
the old dataset. Future training runs must select this version explicitly and log its split hash.

Use the new [v2 training notebook and package](../usage_expanded_v2_e8_20260906/README.md)
to train this dataset with the same E8 recipe. The existing
`task3_training/usage_expanded_e8.ipynb` remains frozen to the earlier 120-image dataset.
The dataset-only package does not launch training or change earlier results.

## Checks

- All 39,299 image files decoded through the normal RGB 60×80 loader and matched their hashes.
- A mixed batch of teacher, earlier external, and new external images passed.
- Every earlier row and field was preserved. Source membership and target masks passed.
- All five splits have zero ID, image-hash, duplicate, normalized-name or family crossings.
- The nine-class map and original teacher/prediction/earlier split hashes are unchanged.
- 58 focused data tests passed. Formatting and lint checks passed.
- No model was fitted and no learned preprocessing was fitted.

See [validation.json](validation.json), [family links](linked_external_families.csv),
and [split summary](split_summary.png). The combined dataset also contains its own
`validation.json`, counts and geometry. Its split SHA-256 is
`0bd490672543b7022fa7a883be9b006d63681d7c168e497e0388616836696776`.

## Rebuild locally

```bash
./.venv/bin/python reports/task3/usage_expanded_v2_20260906/audit_family_links.py
./.venv/bin/python reports/task3/usage_expanded_v2_20260906/build_dataset.py
./.venv/bin/python reports/task3/usage_expanded_v2_20260906/package_dataset.py
```

The build requires the existing teacher files, reviewed source caches, and original 10 MB
collection package. It checks the source inputs before creating the versioned dataset.
No commit, push, public upload or training run was made.
