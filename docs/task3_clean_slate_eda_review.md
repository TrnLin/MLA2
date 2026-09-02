# Task 3 clean-slate EDA final review

Date: 2026-09-02

## Verdict

**Pass. No automatable blocker remains.**

The automated EDA code, saved evidence, notebook, and figures are current and consistent.
The only unfinished step is the planned human review: two people must fill the two blank forms
independently.

No model was trained. No implementation file was edited during this review.

## Two former blockers

### 1. A shortened review scope is now rejected

The canonical sample is built in
[`task3_clean_slate_eda.py:308`](../src/fashion/data/task3_clean_slate_eda.py#L308).
Analysis rebuilds it from the current teacher split, seed, and `sample_per_class`, then requires an
exact ordered match at
[`task3_clean_slate_eda.py:571`](../src/fashion/data/task3_clean_slate_eda.py#L571).

I repeated the old attack: I removed the same row from the saved scope and both forms. Analysis
rejected it as different from the canonical Task 3 sample.

### 2. Human evidence is now bound to the current audit

The lock stores and checks both `audit_contract_hash` and `task3_label_scope_digest` at
[`task3_clean_slate_eda.py:595`](../src/fashion/data/task3_clean_slate_eda.py#L595).
Pack rebuilding removes a stale key, lock, and all three analysis files when either value changes at
[`task3_clean_slate_eda.py:437`](../src/fashion/data/task3_clean_slate_eda.py#L437).
Completion checks both current hashes and every protected human artifact hash at
[`task3_clean_slate_eda.py:2024`](../src/fashion/data/task3_clean_slate_eda.py#L2024).

Two temporary attacks passed safely:

- Changing only a teacher image changed the audit hash and removed the stale human outputs.
- Changing a Task 3 label changed the label digest, removed the stale human outputs, and made
  completion false.

## Rebuilt evidence check

- Scope is teacher-only. All 32,773 development teacher images were rehashed. The fresh contract
  exactly matches the saved contract. Audit hash:
  `da18800eda834813de427f2288ac2a383a5d0eca3c03e3d82cb617bd09d29232`.
- All 44 manifest entries have the right path, size, and SHA-256 hash.
- The review scope exactly matches the current 367-row canonical sample. Both reviewer forms are
  blank and label-blind. The key, lock, and analysis files do not exist yet, as expected.
- All 16 current feature caches pass their hash, shape, type, row, and ID checks.
- Probe batch `ff6d4df37815663b` has exactly 80 complete registered runs. Every run is debug,
  scratch, and submission-ineligible.
- All eight neighbourhood artifacts pass. The 17,440 neighbour rows have no query/reference family
  overlap.
- Literal usage label `NA` is preserved and checked.
- Notebook code cells 1-9 ran in order with no errors, warnings, or stderr. Its dynamic batch ID
  matches the current evidence.
- All five embedded figures byte-match the saved PNGs. They were visually checked and are readable
  with no clipped labels.
- `pip check` passes. Ruff passes. The focused tests pass 10/10. The full suite passes 111 tests,
  with 1 expected skip.

## Only remaining action

Two people must complete these files independently:

- `results/evidence/task3/clean_slate_eda/observability_review/observability_reviewer_1.csv`
- `results/evidence/task3/clean_slate_eda/observability_review/observability_reviewer_2.csv`

Then run the existing review analysis and rebuild the notebook evidence. This is a human gate, not
an automated defect.
