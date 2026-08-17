# 0003 — Separate Raw Data by Provenance and Use Processed Manifests

**Status:** Accepted and implemented  
**Date:** 2026-08-17

## Context

The repository currently has original, teacher-train, and teacher-test files under names
that do not clearly describe provenance. Training also needs paired image paths without
copying the 14 GiB original collection.

Raw data must remain reproducible and must not be silently cleaned or overwritten.

## Decision

Use this layout:

```text
data/
├── raw/
│   ├── original/
│   │   ├── styles.csv
│   │   ├── images.csv
│   │   ├── images/
│   │   └── styles/
│   └── teacher/
│       ├── train/
│       │   ├── styles_train.csv
│       │   └── images_train/
│       └── test/
│           ├── styles_prediction.csv
│           └── images_test/
└── processed/
    ├── train_manifest.csv
    ├── splits.csv
    └── label_review.csv
```

Rules:

1. Treat everything below `data/raw/` as immutable.
2. Keep one row per official train ID in `train_manifest.csv`.
3. Store paths to high- and low-resolution files; do not copy images into `processed/`.
4. Build the manifest only after excluding every official test ID.
5. Keep all train/validation assignment in `data/processed/splits.csv`.
6. Record reviewed label concerns in `label_review.csv` rather than changing raw CSV files.

## Implementation

The raw folders were moved into this layout on 2026-08-17. Post-move checks reproduced:

- 44,446 original metadata rows, 44,441 original JPGs, and 44,446 JSON records;
- 38,617 teacher-train metadata rows and 38,612 train JPGs; and
- 5,829 teacher-test rows and 5,829 test JPGs.

The old `data/train/`, `data/train-old/`, and `data/test/` paths no longer exist.

## Consequences

- Dataset provenance is visible from each path.
- Processed files can be deleted and rebuilt.
- The original image collection is stored only once.
- All notebooks and loaders must read shared path constants rather than hard-coded old
  directories.
- Moving the existing raw folders requires updating `fashion.config` and re-running the
  dataset audits.

## Evidence

This layout supports decisions
[`0001`](0001-use-original-images-with-official-train-ids.md) and
[`0002`](0002-treat-resolutions-as-paired-views.md) while preserving the single-split
contract in `AGENTS.md`.
