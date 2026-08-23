# 0013 - Every model product has both resolution variants

- Status: Superseded by 0015
- Date: 2026-08-22

## Decision

The low- and high-resolution files are two image variants of the same Fashion Product Images
product. They are not separate semantic datasets. `data/processed/splits.csv` remains the only
split. Both variants inherit the same ID, partition, exact-image group, and product-family group.

The Kaggle source is image-only at runtime. Catalogue and manifest code may read `images/` and the
safe `filename` column of `images.csv`. It must never open `styles.csv` or `styles/*.json`, because
those files contain protected targets for holdout and official prediction products.

Every eligible ID in train, validation, and holdout must have exactly two rows in
`data/processed/training_image_variants.csv.gz`: `original` and `high_resolution`. The build fails
closed if either row is missing. Quarantine and every official prediction ID are excluded.

Both rows use the same `sample_group` and `independence_group`. Each has
`per_product_weight = 0.5`, so weights sum to one per product. Sampling, counts, and metrics must be
reported at product level. Two resolutions do not count as two independent products or two pieces
of evidence.

`fashion.data.variants.load_image_variants` always returns both variants. It does not expose a
single-resolution modelling choice. Before returning rows, it checks full split coverage, exact
pairs, half weights, groups, safe existing paths, the official prediction IDs, and the saved
manifest/split/source hashes. Holdout targets are blank in the persisted split unless
`evaluation_unlocked=True` explicitly joins them from the local raw teacher CSV.
That unlock is rejected for train and validation.

`FashionDataset` returns the variant key, variant name, product/sample group, independence/family
group, duplicate group, and half weight. Training can use the weight for loss reduction. Counts and
metrics must aggregate by `sample_group`, so two image sizes never become two products.

Main-policy normalization uses only paired training rows. Each original and high-resolution row
contributes weight 0.5, giving each training product total weight 1. Validation, holdout,
quarantine, and prediction pixels never enter the fit. The old low-only statistic remains a named
comparison baseline; it is not the statistic for the doubled policy.

The geometry audit found 36 same-ID pairs with aspect-ratio changes below 1% (maximum 0.906%).
These are consistent with minor pixel rounding during resizing. Geometry and perceptual hashes are
strong automatic evidence, not proof of matching product content. The 84 high-distance same-ID
pairs are non-blocking items in `docs/reviews/open_decisions.md`.

## Rebuild

Run the catalogue and merge functions from `notebooks/01_data_preparation.ipynb`. They are reusable library
code, not standalone EDA scripts:

```python
from fashion.data.variants import (
    build_training_variant_manifest,
    catalogue_high_resolution_dataset,
)

catalogue_high_resolution_dataset()
build_training_variant_manifest()
```

The catalogue reads only IDs from the official prediction CSV. Kaggle metadata target files are
not opened at all. Their target values are never accessed, parsed, compared, or used.
