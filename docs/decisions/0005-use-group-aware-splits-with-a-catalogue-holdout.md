# 0005 — Use Group-Aware Splits with a Catalogue Holdout

**Status:** Accepted  
**Date:** 2026-08-19

## Context

Exact duplicate images can leak pixels across partitions, while the product mix changes
across catalogue ID ranges. A fully mixed split would support stable model development but
could overstate performance on later catalogue products. A strict year split is unsuitable
because several year groups contain only 2–20 products.

## Decision

Store all partition assignments in the only allowed split file,
`data/processed/splits.csv`.

Build the split with these rules:

1. keep every paired high- and low-resolution view for an ID in the same partition;
2. place each confirmed exact-image group wholly in one partition;
3. make the development training and validation partitions rare-class-aware;
4. reserve official-training IDs 46,919–51,999 as an untouched catalogue-shift holdout;
5. tune only on validation and use the catalogue holdout once for final stress evaluation;
6. report holdout class support and mark unseen or unsupported classes; and
7. never expose official test labels.

Near-duplicate hashes are review warnings, not grouping rules. After creating the split,
review only strong near matches that cross a boundary, using original images. Join pairs
only when human review confirms the same product or photo.

## Consequences

- Duplicate pixels cannot inflate validation or holdout results.
- The catalogue holdout measures robustness to a later product mix.
- Some holdout classes may lack earlier support; this is reported as a dataset limitation,
  not hidden by moving rows.
- All notebooks, training runs, search evaluation, and the app must read the shared split
  file rather than creating local splits.

## Evidence

See [`../eda-problem-review.md`](../eda-problem-review.md), `target-skew.csv`,
`exact-duplicates.csv`, `near-duplicates.csv`, and the drift evidence under
`results/figures/eda/`.
