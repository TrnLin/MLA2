# 0020 — Freeze the Task 3 gender SAM25 model

- Status: Superseded by [0025](0025-task3-gender-sam25-refit-final-model.md) for the final Gender artifact; the five-fold evidence remains historical
- Date: 2026-09-06
- Amends: [0014](0014-development-holdout-cv-boundary.md), for the final gender artifact only

## Context

The gender investigation moved from a scratch SmallCNN through pooling,
augmentation, label review, MixUp, and sharpness-aware minimization (SAM).
The fixed SAM25 recipe has now completed all five canonical folds. The five
saved models were also evaluated together on the reserved holdout and teacher
test images. The owner has chosen to freeze this model after reviewing those
results. Usage remains a separate, unfinished model decision.

## Decision

- Select experiment `t3_gender_name_truth_mixup_alpha020_sam005_epoch25_cv`.
- Freeze the five epoch-25 checkpoints listed in
  `reports/task3/gender_sam25_cv_result_20260906/model_manifest.json`.
  Their SHA-256 hashes, configurations, and fold normalizations define the
  artifact. Do not replace them with the earlier two-fold screen checkpoints.
- Keep the scratch 390,181-parameter SmallCNN with GeM pooling (`p=3`),
  dropout 0.30, the recorded translation/darkening/grayscale recipe,
  MixUp alpha 0.20, SAM rho 0.05, and seed 2753. Stop at epoch 25 with the
  original cosine schedule `T_max=30`; do not select a best epoch afterwards.
- Use the task-specific name-corrected gender labels recorded in the manifest.
  Keep canonical labels and `data/processed/splits.csv` unchanged. Report the
  original-label scores beside corrected-label diagnostics.
- For a new image, use each fold's own normalization, average all five softmax
  probability vectors equally, then take the largest value. Class order is
  `Boys, Girls, Men, Unisex, Women`. Inputs remain teacher RGB images at 60×80
  pixels. The higher-resolution dataset supplied evaluation labels only.
- For development OOF scores, use only the model that held out each row.
  Never average all five models on development rows.
- For gender only, this five-model ensemble replaces the all-development
  single-model refit prescribed by 0014. No additional refit is selected.
  All other tasks retain their existing decision rules.
- Keep the evaluated model fixed. Holdout and recovered test labels are now
  known; they cannot serve as fresh selection data for later changes.

## Why

The chosen trade-off is stable fold results and a smaller training-to-validation
gap, rather than the highest observed screen score. SAM25 scores 79.73% pooled
five-fold macro-F1 on its corrected-label basis, with a mean 10.15-point gap.
On the same two screen folds, MixUp alone scores higher (80.89% versus 79.86%),
but has a larger gap (12.91 versus 9.93 points). This choice does not establish
SAM25 as the highest-scoring model or prove a causal improvement on every step.

## Consequences

Original-label ensemble results are 89.82% accuracy / 77.14% macro-F1 on 5,778
holdout images, and 92.02% / 61.85% on 5,829 teacher test images. The corrected
holdout diagnostic is 80.45% macro-F1; it must not replace the primary score.
The test is 76.72% Women and detects only 19 of 84 Unisex products. These are
material limits. There are also 797 test names that match development names;
the test is not established as fully independent at product-family level.

The recipe and prediction files were fixed before evaluation labels were
opened, but this final acceptance was made after the evaluation was reviewed.
Do not describe final model selection itself as blind to holdout/test results.
The five-fold average has a higher inference cost than a single model.
The gender-only CSV is not the complete four-target submission.

## Evidence

- [Fixed five-fold training](../task3_gender_sam25_five_fold.md)
- [Five-fold review](../../reports/task3/gender_sam25_cv_result_20260906/README.md)
- [Holdout and test review](../../reports/task3/gender_sam25_holdout_test_20260906/README.md)
- [Model manifest](../../reports/task3/gender_sam25_cv_result_20260906/model_manifest.json)
- [Frozen predictions](../../reports/task3/gender_sam25_holdout_test_20260906/prediction_freeze.json)
- [Main report notebook](../../notebooks/06_task3_part1_gender_usage.ipynb)
