# Phase 04 - Verify and reconcile notebook contracts

## Ownership

- Run structural notebook validation, lint, targeted tests, and saved-output checks.
- Review all changed files for leakage, stale paths, duplicate logic, and unsupported claims.
- Reconcile plan checkboxes and status after evidence is available.

## Verification matrix

- Notebook JSON validates under `nbformat`.
- `00_problem_definition.ipynb` and scaffolds have no stale execution outputs.
- `01_data_preparation.ipynb` executes from a fresh kernel in cached mode twice with stable hashes.
- A full build runs in a child process and never exposes protected targets to the notebook kernel.
- Development IDs equal the old train/validation union; holdout and quarantine ID digests stay fixed.
- Every development row has one fold and no family, SHA, duplicate, or normalized-name block crosses.
- Shared preparation passes when the external image folder is absent and never parses its `images.csv`.
- Relevant notebook tests pass, including the clean-package/full-build path where practical.
- All tracked references use the new notebook names.
- Figures and HTML contain the required headings and no visible helper-code guides.
- Git diff contains no secrets, local paths in portable artifacts, or protected target values.
- Commit messages follow conventional format and contain no AI attribution.

## Todo

- [ ] Run split migration, leakage, deterministic-fold, and external-isolation tests.
- [ ] Run notebook structure, execution, provenance, and output-adjacency checks.
- [ ] Run targeted pytest, full pytest, Ruff, dependency, and secret checks.
- [ ] Run independent code review and fix critical findings.
- [ ] Mark all completed plan items and set final plan status.

## Success criteria

- Relevant checks pass or a documented platform-only limitation is reported with evidence.
- Git history shows separate reviewable commits for the requested work boundaries.
