# Notebooks

Use notebooks to tell the project story and show results.

`01_data_preparation.ipynb` is the one official shared data-preparation workflow. It is
intentionally self-contained: a fresh-kernel **Run All** validates or rebuilds the shared
data contract, performs focused train-only EDA checks, and saves report evidence and figures.
Its wider job includes raw hashing, image/metadata reconciliation, duplicate and family
control, the sole split, protected-label checks, shared transforms, normalization, and the
Task 4 query/gallery boundary. EDA is one part of this workflow, not the whole notebook.
Each analysis block presents its purpose, focused evidence, finding, and downstream
consequence. Code stays in the notebook for auditability but is collapsed by default.
Every code cell has an immediately preceding markdown guide. These guides stay in the
notebook but are removed from the code-free HTML report. The notebook must not import a
project data-preparation helper module.

Every count names its unit. Product IDs are used for label distributions and final
metrics. The paired loader uses two image-input rows per active product, so train,
validation, and holdout input counts are twice their product counts.

Cached validation is the default. It fully hashes the lean protected-safe prepared
pack and checks raw path/size inventory plus a fixed content sample. Use
`FASHION_DATA_PREPARATION_MODE=full` for the slow forensic rebuild after raw inputs change.

Keep reusable loading, split, image-variant, training, retrieval-metric, and
evaluation contracts in `src/fashion/`. Both image variants share one product ID
and never count as independent evidence. Number later notebooks in reading order.
