# 0010 — Official outputs and supported metrics

- Status: Superseded by 0016
- Date: 2026-08-21

## Context

Classes with fewer than three independent training product families do not give
enough independent evidence for a fair primary macro metric. The taxonomy must
not use validation or sealed holdout outcomes to decide that scope.

`Suits` occurs only on metadata row 12347, whose image is absent. It is retained in
the raw audit but cannot be learned from this image dataset.

## Decision

Use two explicit label views:

1. **Official output:** every label observed in the training partition. The
   one-time split builder uses raw labels for deterministic family-level
   stratification before partitions are sealed. Train-fitted
   label maps and submission heads cover this full space: 124 `articleType`, 4
   `season`, 5 `gender`, and 9 `usage` labels. Literal `NA` and `Home` remain real
   output labels.
2. **Supported primary metric:** train-fitted labels with at least three
   independent training families. Apply that frozen map to validation and
   holdout. Use this slice for fair macro-F1 comparison. Report validation gaps
   honestly; do not inspect protected holdout target coverage.

For compatibility, the existing `*_deployed` columns now hold the full official
output labels. New `*_supported` columns identify the primary-metric slice.

An unexpected label absent from the train-fitted official map receives index
`-1`, is masked, and is reported. This is a safety fallback, not the expected
path: validation reports it without changing the frozen map.

## Consequences

The model and fixed prediction manifest can represent every trainable dataset
label. Primary model ranking remains fair. Supported-only results must be named
as such; they are not a replacement label taxonomy. Any output-head change must
rebuild the split, label maps, normalization, and EDA evidence.

No protected holdout outcome was used to choose this policy.

## Evidence

- The generated taxonomy table records 124/4/5/9 official output classes and
  100/4/5/8 supported primary-metric classes.
- The 24 low-support article types and `usage=Home` stay in the official maps;
  none is renamed or merged.
- The split builder performs raw-label family stratification before sealing.
  Taxonomy, label maps, supported scope, and normal EDA are train-fitted and do
  not read sealed class names or counts.
- Tests cover all official labels in train-fitted maps plus the safe `-1`/mask
  path for an unexpected train-absent label.
