# Processed data

This folder holds the teacher-only shared data contract. Everything here is rebuildable
from `data/raw/teacher/` by Notebook 01.

`splits.csv` is the only partition file. It contains:

- `development`: 32,773 rows with one deterministic `cv_fold` from 0 to 4;
- `holdout`: 5,778 rows with blank protected targets and false label masks;
- `quarantine`: 61 rows with blank protected targets and false label masks.

Families and accepted duplicate groups do not cross partitions or development folds.
The current development ID/fold assignment SHA-256 is
`bad7bc4ae65fbbfd815567f4ccfa308d6e57dc650bc15c0b8e798867a335f2fd`; ADR 0017 records
the one-time refreeze after fixing missing product-name `NA` values.
Task code must use `load_splits`, `get_cv_split`, or `iter_cv_folds` from
`fashion.data.dataset`. Only the explicit final-evaluation loader may join protected
targets, after methods are frozen.

Main handoff files include:

- `splits.csv`, `split_summary.json`, and `cv_fold_summary.json`;
- `prediction_manifest.csv`;
- `taxonomy.json` and `label_maps.json`;
- `development_class_summary.csv`;
- `development_image_profile.json`;
- `audit/` tables for inputs, images, duplicates, and families;
- `preparation_cache.json`.

`development_image_profile.json` is descriptive only. It records
`allowed_for_model_fit: false`. No learned image statistics, selected transform, model, or
Task 4 protocol is stored here.

Normal Notebook 01 Run All validates `preparation_cache.json`. Full mode rebuilds the
contract from teacher files:

```bash
FASHION_DATA_PREPARATION_MODE=full ./.venv/bin/python -m jupyter lab
```

Optional data under `data/raw/external/` is outside this contract and outside Git.
