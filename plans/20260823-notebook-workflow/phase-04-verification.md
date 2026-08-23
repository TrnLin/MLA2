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

- [x] Run split migration, leakage, deterministic-fold, and external-isolation tests.
- [x] Run notebook structure, execution, provenance, and output-adjacency checks.
- [x] Run targeted pytest, full pytest, Ruff, dependency, and secret checks.
- [x] Run a fresh final code review and fix critical findings.
- [x] Mark all completed plan items and set final plan status.

## Completion evidence

- Full teacher-only build completed in a child process.
- Two cached Run All output manifests matched at SHA-256
  `a54ce37adf08e4e5c72e2ba70a30e57de70d5c6bc8672dfc33588d2ff9a8cc13`.
- A cached Run All also passed while `data/raw/external/` was temporarily absent.
- The artifact registry contains 40 files and every byte hash matches.
- All seven notebooks pass `nbformat`; saved HTML has 18 sections and no visible code input.
- Ruff passed, `pip check` reported no broken requirements, and full pytest reported 45 passed.
- Secret, stale-contract, external-input, and portable-path scans passed.
- External migration retained 177,778 files and 31,422,558,264 bytes. The canonical
  before/after manifest digest is
  `5d22d856f2e8f9a17ae361837d8c4eaaf49dcf00b5f1839b14b84401867401ea`.
- `results/runs.csv`, model files, and checkpoints were unchanged.

## Success criteria

- Relevant checks pass or a documented platform-only limitation is reported with evidence.
- Git history shows separate reviewable commits for the requested work boundaries.
