# Data-preparation evidence

The official `notebooks/01_data_preparation.ipynb` writes a scoped JSON summary and flat CSV
evidence here during **Run All**.
The tables cover partitions, family-block trade-offs, full official outputs, the supported
primary-metric slice, product and family validation coverage, train-only image quality and
normalization, paired high-resolution coverage, transforms, and the product-level Task 4 metadata
proxy contract. Raw metadata is file-fingerprinted, while the notebook loads only its header and
IDs. The evidence does not aggregate protected holdout or
quarantine target outcomes. The former
`data_reconciliation.json` file was removed because it was stale and outside the
official evidence graph.
