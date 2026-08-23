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
- `0008-aspect-preserving-image-contract.md`
- `0009-task4-retrieval-isolation.md`
- `0010-official-output-and-supported-metrics.md`
- `0014-development-holdout-cv-boundary.md`

## Superseded decisions

- `0004-product-group-leakage-check.md` — superseded by 0011.
- `0007-supported-deployment-taxonomy.md` — superseded by 0010.
- `0003-protected-eda-and-train-only-statistics.md` — superseded by 0014.

## Accepted with later amendments

- `0002-single-split-and-duplicate-quarantine.md` — partition details superseded by 0014;
  the sole-split and duplicate-quarantine rules remain active.
- `0011-auditable-family-review-boundary.md` — fixed partition counts superseded by 0014;
  the automatic family safety rule remains active.
- `0012-protected-target-runtime-boundary.md` — partition wording superseded by 0014;
  the protected runtime boundary remains active.

## Proposed decisions and open gates

None for the EDA/data-preparation phase.
