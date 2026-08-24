# Data-preparation evidence

Notebook 01 writes these small teacher-only tables during Run All. They cover raw
inventory and hashing, exact ID reconciliation, partitions and `cv_fold` balance,
duplicate/family boundaries, appendix NMI, pixel diagnostics, transform risk, lineage,
and the artifact registry. `acquisition_shortcut_summary.csv` stores the development-only
headline checks behind the acquisition-risk figure. `broad_name_family_review_index.csv`
stores the fixed selection rule, 24 shown IDs, names, family IDs, group sizes, and hashes.
Tables used only to draw the family, threshold-review,
fold-support, metadata-pattern, and shortcut figures stay visible in Notebook 01 and are
not duplicated as standalone CSV files.

The evidence uses development labels only. Holdout and quarantine target cells are not
aggregated. No model was trained and no metric, transform, normalization, or Task 4
protocol was selected.

`artifact_registry.csv` records repository path, byte size, row count when useful, and
SHA-256 for every main processed file, evidence table, and displayed figure. The current
CV assignment digest is also stored in the split/CV summaries and notebook provenance.
The value 22,905 means conservative split groups, not verified independent products.
