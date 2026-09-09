# 0016 — Development label scope

- Status: Accepted
- Date: 2026-08-23
- Supersedes: 0010 and the earlier supported/deployed policy in 0007

## Context

The shared split currently carries raw, deployed, and supported versions of every target. The extra
views hide rare classes before task owners have selected an evaluation method. They also make it hard
to tell whether a count describes teacher metadata, image-backed rows, all development rows, or the
training side of one fold.

## Decision

- Keep one canonical raw column for each target plus `has_<target>_label`.
- Build label maps from every nonblank label observed on image-backed development rows.
- Treat literal `NA` as a valid label. Only a truly blank value is missing.
- Never drop, merge, or mask a class because it has few products or families.
- Report product count, family count, folds present, and training-complement support for each class.
- If a class is absent from the training complement of a fold, keep its validation rows and report an
  untrainable-class limitation for that fold.
- Keep raw teacher metadata findings separate from image-backed, development-observed, and fold-level
  trainable findings.

## Consequences

The canonical schema no longer has `*_deployed`, `*_supported`, deployment status, evaluation status,
or minimum-support masking fields. Imbalance handling is a task-level experiment choice. Natural
validation and holdout distributions are not resampled.

The label map is stable across all development folds. A protected label seen only after holdout unlock
does not change the map during development and must be reported honestly at final evaluation.

## Verification

- Label maps contain every nonblank development-observed class.
- Low-support classes remain in `splits.csv` and the class summary.
- No supported/deployed field or minimum-family cutoff remains in shared runtime code.
- Missing-image metadata labels are audited but are not described as trainable image classes.
