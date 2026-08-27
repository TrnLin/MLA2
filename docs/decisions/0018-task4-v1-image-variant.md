# 0018 — Task 4 V1 is an image variant

- Status: Accepted
- Date: 2026-08-25

## Context

Task 4 may compare the optional Fashion Product Images V1 collection with the
supplied 60×80 teacher images. A separate train/test split would conflict with the
project's single-split rule and could expose official prediction products.

## Decision

- Treat V1 as an ID-keyed high-resolution image variant, not as independent data.
- Never make a new V1 train/test split. Valid teacher rows inherit `partition`,
  `cv_fold`, duplicate groups, and family groups from `data/processed/splits.csv`.
- Keep the V1 audit in `notebooks/task-4/01_v1_eda.ipynb`; shared Notebook 01 stays
  teacher-only.
- Use only development rows for V1 pixel distributions and paired image analysis.
- Read only V1 `images.csv` and `images/`. Do not read external style or label files.
- Leave the decision to use V1 in a search model open until a controlled Task 4
  quality-and-cost comparison is run.

## Why

The V1 catalogue has exactly the same 44,446 IDs as teacher train plus teacher test.
It adds larger image files, not new products or labels. Reusing the canonical split
prevents leakage and keeps every model comparison consistent.

## Consequences

V1 cannot be claimed as independent external evaluation data. Holdout, quarantine,
and official prediction labels stay protected. Task 4 may compare teacher and V1
pixels, but it must fit any learned preprocessing inside the current training folds.

## Evidence

- `results/evidence/task4/external_provenance.json`
- `results/evidence/task4/external_reconciliation.csv`
- `results/figures/task4/external_geometry.png`
- `results/figures/task4/same_id_agreement.png`
