# Fashion package

Put reusable project code in this package.

Keep modules focused on one job. Notebooks and scripts should import this code
instead of copying it.

- `data/` owns raw audits, manifest construction, the sole shared split, image
  transforms, and train-only statistics. Downstream code must load partitioned
  data through `fashion.data.dataset.load_splits` so split invariants are checked.
- `eda/` owns scope protection, calculations, diagnostics, provenance, and plots.
