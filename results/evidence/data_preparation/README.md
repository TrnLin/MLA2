# Data-preparation evidence

Notebook 01 writes these small teacher-only tables during Run All. They cover raw
inventory and hashing, exact ID reconciliation, partitions and `cv_fold` balance,
duplicate/family boundaries, appendix NMI, pixel diagnostics, transform risk, lineage,
and the artifact registry. Tables used only to draw the family, threshold-review,
fold-support, metadata-pattern, and shortcut figures stay visible in Notebook 01 and are
not duplicated as standalone CSV files.

The evidence uses development labels only. Holdout and quarantine target cells are not
aggregated. No model was trained and no metric, transform, normalization, or Task 4
protocol was selected.

`artifact_registry.csv` records repository path, byte size, row count when useful, and
SHA-256 for every main processed file, evidence table, and displayed figure. The current
CV assignment digest is also stored in the split/CV summaries and notebook provenance.
