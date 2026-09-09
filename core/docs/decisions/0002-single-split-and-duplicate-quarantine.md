# 0002 — Single split and exact-duplicate quarantine

- Status: Superseded in part by 0014
- Date: 2026-08-20

## Context

Independent splits would make model comparisons unfair and could put exact image
twins in different partitions or into the official prediction boundary.

## Decision

Use `data/processed/splits.csv` as the only split. Allocate exact SHA-256 groups as
atomic units with seed `2753`. Put every labelled byte-identical twin of an official
prediction image in `quarantine`.

## Why

One file gives every task the same examples. Atomic groups prevent direct visual
leakage. Quarantine prevents training or tuning on known official-test twins.

## Consequences

No notebook, model, retrieval index, or app may call `train_test_split`. Changing
the policy requires a new decision and a complete rebuild of downstream evidence.

## Evidence

- Repository hard constraint on `data/processed/splits.csv`.
- Exact duplicate and cross-role checks in `fashion.data.splits`.
- Assignment requirement for fair analysis and independent evaluation.
