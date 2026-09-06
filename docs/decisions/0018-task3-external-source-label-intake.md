# 0018 — Teacher + rare-class Usage dataset

- Status: Accepted
- Date: 2026-09-06
- Amends: the Task 3 teacher-only restriction in [0015](0015-teacher-only-shared-image-preparation.md)
  and the sole-split wording in [0014](0014-development-holdout-cv-boundary.md) for this named dataset

## Decision

Create `data/processed/teacher_plus_rare_usage_20260906/splits.csv` by adding the 120 admitted
images to all 38,612 teacher rows. Train the **same nine-class `usage` target** with the normal
`FashionDataset`, `load_splits` and `get_cv_split` APIs. This dataset needs no separate model,
auxiliary target or extra output head.

Map exact source Home, Party, Smart Casual and Travel occasion/style labels to the same-named
Usage labels. Keep the original source evidence and the explicit mapping in each added row.
Other targets stay blank with false validity masks; missing source fields never become NA.
Keep the teacher class order, including NA, unchanged.

Preserve every original teacher field, partition and CV fold. Keep holdout and quarantine labels
blank in the runtime dataset. Give the 120 added development images unique IDs starting at
1,000,000,000, and place their 109 linked product families into five folds. Balance added label
counts deterministically with seed 2753; never divide a linked family across folds. All 120
images participate in the combined development data, including the original intake's source
validation rows. The earlier source-only 80/20 allocation is not a training contract for this
combined dataset.

The versioned directory contains the full split, the unchanged label map, added-image rows,
fold-0 train/validation tables, all five fold ID lists, class counts and validation results.
The full split is the authority; exported tables and ID lists are derived from it. The original
teacher-only `data/processed/splits.csv` stays available for earlier experiments and comparisons.
Models using the added data must select the new dataset path and record its split hash.

Admitted source images remain RGB PNGs, width 60 × height 80. The ordinary image transform loads
teacher and added images as RGB at the same size, with float values in [0, 1]. Do not use the
old source-only normalization statistics for the combined data. If a model standardizes channels,
fit its statistics on that model's combined training fold only.

## Admission and validation

Retain the completed source, image and identity audit. It decoded 264 candidates, withheld weak
labels, unsuitable photos and repeated images, and compared originals, resized images and
foreground views against 44,441 teacher-role images without reading protected labels. No accepted
source image met its teacher-overlap acceptance rule. These heuristic comparisons reduce duplicate
risk but do not prove that every related product has been found.

The combined build checks the saved audit's input hashes, every current image hash, intact teacher
rows, target masks, source membership, class mapping and zero family crossings. It exercises the
ordinary dataset loader for every image and a mixed teacher/external minibatch with one Usage label
index per image. This is data preparation; no model fit is performed.

## Use

Train from scratch and register each future fit. Use the same Usage loss and output classes for
both sources. Preserve source metadata for error analysis: seller labels can differ in practice,
and Smart Casual has only two added product families. Report the expanded dataset's split hash
and compare teacher-image results against a matched teacher-only run. Shared Task 4 indexing and
other tasks do not switch datasets automatically.

The old `data/external/rare_usage_20260906/` manifests remain acquisition evidence and image storage.
Their legacy source-only loader is not the training interface for this dataset. Source URLs,
attribution and rights evidence remain attached to the intake.

## Evidence

- [Combined builder](../../src/fashion/data/expanded_usage.py)
- [Combined data tests](../../tests/data/test_expanded_usage.py)
- [Original image-only overlap audit](../../src/fashion/data/external_usage_audit.py)
- [E8 training and run instructions](../../reports/task3_usage_expanded_e8_20260906/README.md)

The full intake report and combined validation stay in the local
`reports/task3_rare_external_intake_20260906/` directory. The training archive carries
the reviewed dataset, image provenance, validation and source evidence needed by Colab.
