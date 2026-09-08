# Final fix report

## Changes

- Sealed final-eligible Task 1 runs to the approved optimization settings,
  model, canonical split, and two approved preprocessing configurations.
- Added pooled OOF macro-F1, weighted F1, Top-1, and Top-5 metrics and persisted
  them as `results/evidence/task1/oof_metrics.csv`.
- Updated the Task 1 notebook to display its schedule, results, and registry
  fields.
- Added tests for the new guards, pooled metrics, unique folds, and confusion
  figure output.

## Verification

- Ruff: passed.
- Focused Task 1 tests: 37 passed.
- Full suite: 132 passed, 1 pre-existing prepared-data cache fingerprint
  failure, 2 warnings.
- The cache failure is caused by the local raw image inventory differing from
  the delivered cache (38,612 vs 38,613 files); no canonical split path is
  missing.

## Review note

The delegated final-fix and re-review agents were unavailable because of the
host usage limit. The changes were manually reviewed against the final review
findings; no additional regression was found.

Commit: `066b6a0`.
