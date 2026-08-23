# 0014 — Development, holdout, and cross-validation boundary

- Status: Accepted
- Date: 2026-08-23
- Supersedes: partition details in 0002, train/validation scope in 0003, fixed-partition
  counts in 0011, and partition wording in 0012

## Context

The existing holdout has stayed sealed, but the old train and validation labels have already been
used during development. Moving those rows into a new holdout would no longer give an independent
test. A permanent validation partition also makes later cross-validation harder to compare fairly.

## Decision

- Keep the existing 5,778 holdout IDs and 61 quarantine IDs unchanged.
- Merge the old train and validation IDs into one 32,773-row `development` partition.
- Give every development row one deterministic `cv_fold` from 0 to 4 using seed `2753`.
- Allocate the atomic `product_family_group`, not individual rows. Exact hashes, accepted visual
  groups, and normalized product-name groups must also have zero fold crossings.
- Permit descriptive EDA on all development rows. Fit learned preprocessing and models only on the
  four training folds of an evaluation round.
- Let each task owner predeclare either one fixed validation fold or all five folds. Models compared
  inside one task must use the same choice. Reporting only the best observed fold is forbidden.
- Refit the frozen final method on all development rows before unlocking holdout exactly once.

The effective shares in one five-fold round are about 68% training, 17% validation, and 15%
independent holdout. Group safety takes priority over exact percentages.

## Consequences

`data/processed/splits.csv` remains the only split. Its canonical partitions are `development`,
`holdout`, and `quarantine`; only development rows have `cv_fold`. Holdout and quarantine labels
remain blank in persisted artifacts. No nested cross-validation is required by default.

Twenty percent holdout is rejected because it would not improve five-fold validity, would remove
about 1,932 more rows from development, and would require contaminating or replacing a protected
boundary whose membership is already valid.

## Verification

- Development IDs equal the old train/validation union.
- Holdout and quarantine ID-set SHA-256 digests do not change.
- Every development family belongs to exactly one fold.
- Protected target values cannot affect fold allocation or public development evidence.
