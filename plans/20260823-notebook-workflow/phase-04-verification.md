# Phase 04 - Verify and reconcile notebook contracts

## Ownership

- Run structural notebook validation, lint, targeted tests, and saved-output checks.
- Review all changed files for leakage, stale paths, duplicate logic, and unsupported claims.
- Reconcile plan checkboxes and status after evidence is available.

## Verification matrix

- Notebook JSON validates under `nbformat`.
- `00_problem_definition.ipynb` and scaffolds have no stale execution outputs.
- `01_data_preparation.ipynb` executes from a fresh kernel in cached mode.
- Relevant notebook tests pass, including the clean-package/full-build path where practical.
- All tracked references use the new notebook names.
- Figures and HTML contain the required headings and no visible helper-code guides.
- Git diff contains no secrets, local paths in portable artifacts, or protected target values.
- Commit messages follow conventional format and contain no AI attribution.

## Todo

- [ ] Run notebook structure checks.
- [ ] Run targeted pytest and Ruff checks.
- [ ] Run independent code review.
- [ ] Fix critical findings and repeat verification.
- [ ] Mark all completed plan items and set final plan status.

## Success criteria

- Relevant checks pass or a documented platform-only limitation is reported with evidence.
- Git history shows separate reviewable commits for the requested work boundaries.

