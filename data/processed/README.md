# Processed data

Store rebuildable audits, manifests, label maps, normalization statistics, and
`splits.csv` here.

`splits.csv` is the sole split for training, validation, holdout, quarantine, the
search index, and the app. Quarantine includes cross-role exact duplicates and
exact-image groups with conflicting valid target labels. Generated files are
ignored by Git.
