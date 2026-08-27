# Task 4 Image Preprocessing Design

## Status

Approved in the 2026-08-27 design interview. This design implements Task 4
Milestone 3 without opening the holdout or choosing a learned model.

## Goal

Freeze one clear image-input contract for teacher, V1, and arbitrary user
images. Choose its size with development-only retrieval evidence, then create
reusable local caches and fold-safe normalization evidence.

## Safety boundary

- `data/processed/splits.csv` remains the only split.
- Ordinary comparison uses validation fold `1`; training rows are folds `0`,
  `2`, `3`, and `4`.
- The top two sizes receive the five-fold stability check required by ADR 0019.
- Only rows with `partition == "development"` may have pixels opened.
- Holdout and official teacher-test pixels remain sealed.
- V1 inherits teacher IDs and folds. It is never independently split.
- No learned model, pretrained model, or final retrieval baseline is introduced.

## Candidate contracts

Compare these `(width, height)` sizes:

1. `60×80`
2. `96×128`
3. `240×320`

Code continues to spell array shapes as `(height, width)`.

Every candidate:

- applies EXIF orientation;
- composites transparency onto white and converts grayscale or other modes to
  RGB;
- preserves aspect ratio;
- resizes with Pillow LANCZOS;
- centres content on a white canvas;
- returns a content mask so padding can be excluded from fitted statistics and
  suitable hand-crafted features;
- emits `uint8` RGB for lossless caching and scales to `[0, 1]` at model input.

After learned-model scaling, RGB mean and standard deviation are fitted only
from the current round's training folds. The fitted values from the model's
training source are reused for every query and gallery image; incoming images
never refit or select their own statistics. Standardized padding is set to
zero, the fitted channel mean.

## Fixed retrieval probe

The probe is deliberately untrained so the experiment measures preprocessing
without beginning model selection.

- Divide each transformed image into a fixed `4×4` grid.
- Extract a spatial HSV colour histogram and a spatial gradient-orientation
  histogram from real content pixels only.
- L2-normalize the colour block and edge block separately.
- Give the two blocks equal weight and L2-normalize the joined descriptor.
- Rank by exact cosine distance.
- Break ties through the existing protocol: distance, then numeric product ID.

The probe has no tuned weights, bins, or feature selection on fold `1`.

## Experiment matrix

For every candidate size, extract teacher and V1 descriptors once and evaluate:

- teacher query → teacher gallery;
- V1 query → V1 gallery;
- teacher query → V1 gallery;
- V1 query → teacher gallery.

Use all eligible queries and galleries from Protocol A and Protocol B. Protocol
A mean per-query linear nDCG@10 remains the primary retrieval score. Protocol B,
cross-source directions, ties, throughput, memory, and storage are supporting
evidence.

For size selection, average the teacher→teacher and V1→V1 Protocol A
query-mean nDCG@10 values with equal source weight. Cross-source directions do
not enter this selection value. The highest value selects the shared size. The
two highest sizes are then evaluated with every development fold held out in
turn; report their unweighted mean and standard deviation. The later learned
model study still decides whether teacher or V1 should supply the deployed
index.

## Arbitrary-query checks

Unit tests cover EXIF orientation, grayscale, transparency, wide inputs, tall
inputs, output shape, dtype, and masks.

For retrieval evidence, place clean fold queries on deterministic wide and tall
white canvases without cropping or degrading their content. Compare their
nDCG@10 and Top-10 overlap with the clean-query rankings. This milestone does
not test blur, compression, crop loss, or background replacement; those remain
for final robustness and failure analysis.

## Local cache

After the winning size is known, build versioned, lossless `uint8` caches for
both teacher and V1 development images:

- sorted product IDs;
- memory-mappable RGB arrays;
- compact content bounds from which masks can be reconstructed;
- metadata containing the contract, source paths, row count, ID digest, source
  fingerprint, and array shape.

The cache lives below `data/processed/task4/`, is ignored by Git, and is reused
only when its metadata exactly matches. It never contains holdout, quarantine,
or official teacher-test pixels.

## Outputs

Reusable logic lives in focused modules under `src/fashion/retrieval/`. Tests
live under `tests/`.

The main Task 4 notebook replaces preprocessing placeholders with:

- the frozen input and leakage contract;
- a concise experiment matrix;
- a compact quality-and-cost table;
- visible teacher/V1 and odd-aspect examples;
- one report-useful comparison figure;
- a short statement of what the probe proves and does not prove.

Tracked evidence is written below `results/evidence/task4/`; the figure is
written below `results/figures/task4/`. ADR 0020 freezes the result and
`notebooks/task-4/PROGRESS.md` marks Milestone 3 complete only after the
experiment, cache validation, notebook execution, tests, and lint all pass.
