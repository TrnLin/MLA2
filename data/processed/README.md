# Processed data

Store rebuildable audits, manifests, label maps, normalization statistics, and
`splits.csv` here.

`splits.csv` is the sole partition file for training, validation, holdout,
quarantine, retrieval, and the app. The scored Task 4 development gallery uses
train rows only; validation rows are queries and never enter that gallery.

Whole product families stay together. Quarantine includes exact and every
cross-role automatic visual match, plus exact-image groups with
conflicting valid labels. Label maps and `taxonomy.json` are fitted on train after
splitting. The frozen mapping is then applied to validation and protected rows.
Separate `*_supported` columns mark the fair primary-metric slice with at least
three independent **training** families. A validation label unknown to train is
masked and reported; holdout target coverage is not audited in public EDA.

Holdout and quarantine targets are blank in the persisted `splits.csv`, not merely
hidden after loading. Full target values require the explicit final-evaluation loader,
which joins them from the local raw teacher CSV only after evaluation is unlocked.

`training_image_variants.csv.gz` contains exactly two rows for every train,
validation, and holdout product: `original` and `high_resolution`. The rows share
one ID and group, and their weights sum to one. Official prediction IDs and
quarantine are excluded.

`paired_normalization.json` is the doubled-policy default. It uses original and
high-resolution training rows at weight 0.5 each. `normalization_original_only.json`
is only the controlled low-resolution baseline.

`preparation_cache.json` stores cheap source-inventory fingerprints and full hashes
for the lean protected-safe artifact pack. Large repeated tables are deterministic
gzip CSV files that pandas reads directly. No target-bearing training manifest is
delivered; training uses `splits.csv`. Normal **Run All** rejects missing or changed
artifacts. Raw image content assurance is narrower: every path and size plus a fixed
hash sample.

Set `FASHION_EDA_MODE=full` before opening the notebook for a slow forensic rebuild
after raw inputs change. Full mode requires `data/fashion-dataset/`, rebuilds the
doubled image manifest, regenerates label-free low/high alignment evidence, and
recomputes both normalization policies. There is no EDA preparation script.
