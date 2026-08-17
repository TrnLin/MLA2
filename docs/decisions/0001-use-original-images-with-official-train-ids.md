# 0001 — Use Original Images Only for Official Training IDs

**Status:** Accepted  
**Date:** 2026-08-17

## Context

The teacher dataset is a split of the original Fashion Product Images dataset:

- 38,617 IDs are teacher training rows;
- 5,829 IDs are the official test rows; and
- together they exactly match all 44,446 original metadata IDs.

Labels on every shared row are identical. The original images have much higher resolution,
but the original metadata also exposes labels for all official test IDs.

## Decision

Use the official teacher-train IDs as the allowed training population. Join those IDs to
the original high-resolution images and metadata when needed.

Before any cleaning, sampling, feature selection, or modelling:

1. load the IDs from `data/raw/teacher/test/styles_prediction.csv`;
2. remove those IDs from the original metadata;
3. exclude the five train metadata rows with no image in either source; and
4. retain the test IDs as a permanent exclusion list.

Never use recovered official test labels for training, tuning, model selection, or error
analysis.

## Consequences

- Training can use higher-quality source images without changing the official population.
- The official teacher test remains a valid held-out set.
- The five metadata-only IDs `12347`, `39401`, `39403`, `39410`, and `39425` are unusable
  for image models.
- Code must filter by test ID before exposing target columns from the original metadata.

## Evidence

See [`../dataset-quality-comparison.md`](../dataset-quality-comparison.md) and
[`../../notebooks/00_dataset_comparison.ipynb`](../../notebooks/00_dataset_comparison.ipynb).
