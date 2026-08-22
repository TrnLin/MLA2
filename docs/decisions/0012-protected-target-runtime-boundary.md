# 0012 — Protected target runtime boundary

- Status: Accepted
- Date: 2026-08-21

## Context

The split builder needs target labels to detect conflicts and prove taxonomy
coverage. Normal EDA and model setup need only IDs, hashes, families, partitions,
and development labels. Rechecking holdout target coverage during every load
opens protected outcomes too early.

## Decision

Split creation may use raw labels once for deterministic group allocation. After
partition sealing, taxonomy and sensitivity are fitted on train only. The persisted
split itself blanks target values and masks in holdout and quarantine rows. Full
targets are joined from the local raw teacher CSV only after an explicit final-
evaluation unlock.

Generated EDA may publish the build audit’s pass/fail contract, but never holdout
class names, counts, distributions, examples, or outcomes.

## Consequences

Changing only protected target values cannot change the lean cache, normal EDA,
taxonomy, or development diagnostics. The holdout remains independent until final
evaluation.

## Evidence

- Runtime validation checks partition values, IDs, hashes, quarantine structure,
  and group crossings without reading protected target columns.
- `splits.csv` stores blank protected targets, false masks, and a `protected` status.
- `csv_summary.json` hashes only headers, IDs, and permitted development targets;
  its `protected_target_values_hashed` count is zero.
- A sentinel test changes raw, split, and staging protected values, regenerates every
  public table and cache fingerprint, and proves byte-for-byte equality.
- The final loader requires an explicit unlock and then joins targets from the local
  raw teacher CSV; those values are not delivered in Git.
