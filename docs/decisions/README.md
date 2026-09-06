# Project Decisions

Store choices here when they will constrain later notebooks, models, data
processing, evaluation, or the app.

## Before making a decision

1. Read every accepted record below.
2. Check the assignment and rubric.
3. Gather evidence.
4. Copy `TEMPLATE.md` to the next four-digit number.
5. Explain the choice and its trade-offs.

Use names like `0001-short-decision-name.md`.

Do not silently rewrite an accepted decision. If it changes, write a new record,
mark the old one as superseded, and link both records.

## Accepted decisions

- `0001-data-roles-and-raw-immutability.md`
- `0005-conflicting-exact-duplicate-quarantine.md`
- `0006-python-constraints-workflow.md`
- `0015-teacher-only-shared-image-preparation.md`
- `0016-development-label-scope.md`
- `0017-product-name-na-and-cv-refreeze.md`
- `0018-task3-external-source-label-intake.md` — combined teacher + rare-class Task 3 dataset;
  direct Usage labels, unchanged teacher folds, grouped folds for added images.
- `0019-task3-text-supported-usage-expansion.md` — a new combined version adds 567 reviewed
  product-text/collection-labelled images while preserving all earlier rows and folds.
- `0020-task3-gender-sam25-final-model.md` — freeze the five SAM25 gender checkpoints
  and their equal-probability ensemble; record evaluation limits and label bases.
- `0021-task3-usage-gap-replacements.md` — replace 130 outside images in a new v3 dataset;
  fill reviewed type/photo gaps while preserving class totals and retained rows/folds.
- [0022-task3-usage-e1-final-model.md](0022-task3-usage-e1-final-model.md) — freeze the five
  teacher-only E1 Usage checkpoints and equal probability average; record rare-class failures.

## Superseded decisions

- `0004-product-group-leakage-check.md` — superseded by 0011.
- `0007-supported-deployment-taxonomy.md` — superseded by 0010.
- `0003-protected-eda-and-train-only-statistics.md` — superseded by 0014.
- `0008-aspect-preserving-image-contract.md` — superseded by 0015.
- `0013-high-resolution-training-variants.md` — superseded by 0015.
- `0010-official-output-and-supported-metrics.md` — superseded by 0016.

## Accepted with later amendments

- `0014-development-holdout-cv-boundary.md` — Task 3 combined-data exception in 0018;
  Gender and Usage final artifacts are five-fold ensembles under 0020 and 0022.
  Other boundaries remain active.
- `0002-single-split-and-duplicate-quarantine.md` — partition details superseded by 0014;
  the sole-split and duplicate-quarantine rules remain active.
- `0011-auditable-family-review-boundary.md` — fixed partition counts superseded by 0014 and the
  product-name `NA` rule amended by 0017; the automatic family safety rule remains active.
- `0012-protected-target-runtime-boundary.md` — partition wording superseded by 0014;
  the protected runtime boundary remains active.
- `0009-task4-retrieval-isolation.md` — the fixed shared query/gallery and image-variant protocol is
  superseded by 0015; the rule against retrieval self-match leakage remains active.

## Proposed decisions and open gates

None for the EDA/data-preparation phase.
