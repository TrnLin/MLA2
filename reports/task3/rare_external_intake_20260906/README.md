# Teacher + rare-class dataset

**38,612 teacher rows + 120 selected images = 38,732 rows.**
Use this combined dataset to train the **same nine-class Usage model** with the normal project
loader. The added images directly supply `usage` labels. No separate model or output head is needed.

## Files to use

Dataset folder: `data/processed/teacher_plus_rare_usage_20260906/`.

- [Combined splits](../../../data/processed/teacher_plus_rare_usage_20260906/splits.csv).
- [Training rows](../../../data/processed/teacher_plus_rare_usage_20260906/train.csv) and
  [validation rows](../../../data/processed/teacher_plus_rare_usage_20260906/validation.csv) for fold 0.
- [All five fold ID lists](../../../data/processed/teacher_plus_rare_usage_20260906/cv).
- [Existing label map](../../../data/processed/teacher_plus_rare_usage_20260906/label_maps.json).
- [Added images and their source records](../../../data/processed/teacher_plus_rare_usage_20260906/added_images.csv).
- [Validation checks](combined_validation.json), [loading example](USAGE.md),
  [image previews](GALLERY.md) and [dataset package](teacher_plus_rare_usage.zip).

The full `splits.csv` is the authority. It works with `load_splits`, `get_cv_split`, `get_samples`
and `FashionDataset`. The CSVs for training and validation are exports of fold 0 using valid Usage
labels. Other folds are selected from the full split with `get_cv_split(splits, fold)`.

## What was added

Counts below are for development data, where training and validation take place.

| Usage class | Teacher | Added | Combined |
|---|---:|---:|---:|
| Home | 1 | 16 | 17 |
| Party | 12 | 97 | 109 |
| Smart Casual | 47 | 6 | 53 |
| Travel | 22 | 1 | 23 |
| NA | 61 | 0 | 61 |

The added labels map from exact seller occasion/style fields to the same-named Usage class.
Their source text, IDs, links and mapping basis stay in the dataset. The nine-class label order
stays unchanged. Added rows have no invented gender, season or articleType labels.

## Splits and checks

There are **32,893 development rows**, **5,778 holdout rows** and **61 quarantined rows**.
One existing teacher development row has no Usage label, so Usage training uses 32,892 rows.
Fold 0 exports **26,315 training images** and **6,577 validation images**, including 96 and 24
added images respectively. All 120 added images take part in the five-fold development dataset.

Every teacher row retains its original partition and fold. Added product families stay together;
their five-fold allocation uses seed 2753. Holdout and quarantine labels remain sealed. The
teacher-only baseline split and official prediction data are unchanged.

All **38,732 images** passed the ordinary dataset loader, file-hash checks and the common RGB
60×80 transform. A mixed teacher/external batch passed with shape `(8, 3, 80, 60)` and eight
indices from the same nine-class label map. Inputs use float32 values in [0, 1]. Fit any optional
channel standardization on the combined training fold; do not reuse the source-only statistics.
No model was trained. PyTorch is not installed locally; the checks used the framework-neutral loader.

The 120 stored external PNGs are already 60 pixels wide × 80 pixels high. All were visually
inspected beside their originals. The earlier audit compared 264 candidates against 44,441
teacher-role images and withheld 144 candidates. Duplicate checks are heuristic. Smart Casual
still adds only two product families, and Travel adds one. Keep source metadata when examining errors.

## Rebuild

From the repository root:

```bash
./.venv/bin/python reports/task3/rare_external_intake_20260906/build_combined_dataset.py
```

This is an offline data build and check. The package contains combined manifests, the 120 added
PNGs, loader/build code and evidence. It reuses the teacher images already present in this project.
Extract it at the repository root of another copy with the same teacher data.

The old files under `data/external/rare_usage_20260906/` are the source acquisition record and
image cache. Their original source-only partitions are superseded for training by the combined
split linked above. See [source attribution](ATTRIBUTION.md) and
[the dataset contract](../../../docs/decisions/0018-task3-external-source-label-intake.md).
