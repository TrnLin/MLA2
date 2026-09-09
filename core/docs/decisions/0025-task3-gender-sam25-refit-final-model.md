# 0025 — Freeze the Task 3 Gender SAM25 refit

- Status: Accepted
- Date: 2026-09-07
- Supersedes: [0020](0020-task3-gender-sam25-final-model.md), the final Gender artifact
- Amends: [0014](0014-development-holdout-cv-boundary.md), the recorded acceptance timing

## Context

The owner accepted the completed single Gender SAM25 refit after reviewing its
reserved-holdout comparison with the earlier five-model average. The method was
chosen through the previous development investigation. The later refit uses the
same recipe with fresh weights and all development rows.

## Decision

Freeze run
`t3_gender_name_truth_mixup_alpha020_sam005_epoch25_refit_20260907T061926Z_db0fc1ee`.
The authoritative handoff is
`reports/task3/gender_final_sam25_refit_20260907/model_manifest.json`.
Its checkpoint SHA-256 is
`41a5f5ea027805e8edbbf667d080564776ca9873d9146e0dec15c0d4bec2b272`.

This is one scratch 390,181-parameter SmallCNN with GeM pooling (p=3), trained
on all 32,773 development images. Use the saved task-only name-corrected labels;
canonical labels and `data/processed/splits.csv` remain unchanged. The recipe is
25 fixed epochs, seed 2753, batch 128, AdamW at 0.001, weight decay 0.0001,
dropout 0.30, MixUp alpha 0.20 and SAM rho 0.05. Keep the original cosine
schedule T_max=30 and minimum rate 0.00001. Epoch 25 is final, not a best epoch.
The saved translation, mild-darkening and grayscale augmentation remains fixed.
No validation, early stopping or holdout-based epoch selection occurred in this fit.

Inference uses this checkpoint's full-development normalization, RGB height 80
and width 60, EXIF orientation, aspect-preserving LANCZOS resizing and centred
letterboxing with normalized padding set to zero. Use evaluation mode, softmax
then argmax, with class order `Boys, Girls, Men, Unisex, Women`. No augmentation
or fold averaging applies at inference. Retain the saved normalization rather
than recomputing it on evaluation or user images.

Notebook 4 contains development evidence, refit training analysis and the frozen
recipe. `07_task3_part2_final_evaluation.ipynb` contains the reserved-holdout results and final judgement.
Original teacher labels remain primary; the fixed name-rule diagnostic remains
separate. The teacher test set is prediction-only in the submitted analysis.

## Why

The completed refit gives 90.00% accuracy and 77.44% macro-F1 on the same 5,778
reserved holdout images, against 89.82% and 77.14% for the earlier average.
It fixes 65 old errors and creates 55 new ones. This small observed gain supports
accepting the simpler single model, but does not prove superiority across seeds.
Unisex remains weak, Boys F1 falls slightly, and mean class recall falls.

The earlier 79.73% corrected-label OOF macro-F1 belongs to the five fold models,
not the new refit. There is no independent development validation score for a
model trained on all development images.

## Consequences

The five-model manifest and predictions remain historical evidence with their
original bytes. The refit's original training-only manifest is also preserved;
the separate acceptance manifest records this later decision. Acceptance followed
holdout review. Do not backdate it, call this a newly blind selection, or treat
the already opened holdout as fresh tuning data.

One forward pass replaces five. Application latency and refit corruption
robustness were not benchmarked. The evidence supports catalogue audience-tag
suggestions with review for ambiguous products. It does not make a product's
intended audience reliably visible in every image.

Usage's E8 refit decision and other task boundaries remain unchanged. This freeze
does not deploy the app or generate a submission. The eventual official file
must preserve `id,gender,articleType,season,usage`.

## Evidence

- [Main notebook](../../notebooks/06_task3_part1_gender_usage.ipynb)
- [Final evaluation notebook](../../notebooks/07_task3_part2_final_evaluation.ipynb)
- [Accepted model manifest](../../reports/task3/gender_final_sam25_refit_20260907/model_manifest.json)
- [Completed training notebook](../../notebooks/task3_training/gender_sam25_refit.ipynb)
- [Refit contract](../task3_gender_sam25_refit.md)
- [Reserved-holdout comparison](../../reports/task3/gender_sam25_refit_holdout_20260907/README.md)
