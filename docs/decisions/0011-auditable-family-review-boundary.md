# 0011 - Automatic family safety boundary

- Status: Superseded in part by 0014 and amended by 0017
- Date: 2026-08-22

## Context

Leakage safety must be automatic and repeatable. It cannot depend on a person completing a form.
Normalized names are also broad: they can join different designs and reduce the number of
independent products available for splitting and evaluation.

## Decision

- Quarantine every labelled product in a visual component that also contains an official
  prediction image.
- Quarantine exact-image groups with conflicting target labels.
- Keep each remaining normalized name, exact hash, and accepted automatic visual component in one
  partition.
- Never read a human decision during preparation, cached validation, notebook Run All, or tests.
- Keep unresolved questions only in `docs/reviews/open_decisions.md`. They are non-blocking.
- Keep IDs `41303/43587` in train as a predeclared visual-family example so no validation or holdout
  outcome selects the example after splitting.

## Consequences

The rule is conservative. A broad family can reduce effective sample size, but it cannot cross an
active partition. The final active data has 38,551 product rows and 27,009 family units, a reduction
of 11,542 units or 29.9%. There are 4,567 multi-row families and the largest contains 80 rows.

ADR 0017 later treats literal product-name `NA` as missing and records the resulting family and CV
refreeze changes. The counts in this historical record describe the earlier family graph.

All active exact-hash, automatic-visual, normalized-name, and family blocks have zero crossings.
The fixed partitions contain 26,992 train, 5,781 validation, 5,778 holdout, and 61 quarantine rows.
