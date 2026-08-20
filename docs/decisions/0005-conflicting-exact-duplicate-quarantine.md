# 0005 — Conflicting exact-duplicate quarantine

- Status: Accepted
- Date: 2026-08-20

## Context

Exact-image groups can carry different valid labels. Keeping such a group in
training, validation, or holdout would make identical pixels support contradictory
outcomes and could make evaluation depend on an arbitrary metadata row. This is
separate from the unresolved broader product-group leakage question in `0004`.

## Decision

For each exact SHA-256 group, compare the distinct valid values of
`articleType`, `season`, `gender`, and `usage`. If any target has more than one
valid value, quarantine every labelled row in the group before partition
allocation. A missing label remains masked and does not conflict with a valid
label. Record the conflicting target names and validate them from the underlying
labels whenever the split is loaded or rebuilt.

## Why

Whole-group quarantine avoids selecting one disputed label, leaking identical
pixels into evaluation, or silently training against contradictory targets. It
also leaves missing-label handling unchanged.

## Consequences

The reviewed data quarantines 20 conflicting exact-image groups containing 41
labelled rows. Train-only normalization and all downstream evidence must be rebuilt
after applying the policy. Aggregate conflict counts may be reported as structural
evidence, but protected target distributions remain closed.

## Evidence

- Exact SHA groups and target masks in `data/processed/splits.csv`.
- Conflict discovery and validation in `fashion.data.splits`, enforced for
  downstream consumers by `fashion.data.dataset.load_splits`.
- Quarantine reconciliation in `fashion.eda.scope`.
