# Data-preparation evidence

Notebook 01 writes these small teacher-only tables during Run All. They cover raw
inventory and hashing, exact ID reconciliation, partitions and `cv_fold` balance,
duplicate/family boundaries, development class support, NMI, image quality, transform
risk, lineage, and the artifact registry.

The evidence uses development labels only. Holdout and quarantine target cells are not
aggregated. No model was trained and no metric, transform, normalization, or Task 4
protocol was selected.

`artifact_registry.csv` records repository path, byte size, row count when useful, and
SHA-256 for every main processed file, evidence table, and displayed figure.
