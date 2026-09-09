# Verification

Production training has not started. The user will run the notebook in VS Code with a GPU kernel.

- 63 focused tests passed, including numerical weighted MixUp/SAM gradient checks, two fresh
  synthetic CPU model fits, per-fold class weights, receipt verification, resume behavior,
  exact teacher-image comparisons, and the new notebook's actual result/plot cells.
- The planned-notebook-name check passed separately: 64 passing checks total.
- Two existing notebook tests were excluded because the user's earlier notebooks contain
  real training outputs; those tests require empty outputs. Their related code tests passed.
  The existing output-bearing notebooks were preserved.
- Ruff formatting and lint checks passed for the changed training code, new tests and build tools.
- `git diff --check` passed.
- The bundle build hashed and decoded all 33,459 eligible development images, preserved the
  approved split files, verified both completed v2 reference folds, and verified every archive hash.
- The initial Colab setup simulation reached the two-fold training call, verified both packed
  baseline references and passed a setup rerun. Its `colab_bootstrap_check.json` records the
  initial tested bundle hash. The final bundle adds the last notebook-output test and updates
  only the launch prose to name VS Code; its current hash is in `bundle_receipt.json`.
- The rendered notebook's title, recipe table and setup text were visually inspected.

The CPU fits use tiny synthetic fixtures in temporary directories. They do not claim production
GPU performance and do not write runs to the project or Drive experiment registries.
