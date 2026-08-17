# Ground-Up Exploratory Data Analysis Design

**Date:** 2026-08-17  
**Status:** Approved

## Purpose

Build a new exploratory data analysis from scratch for two goals:

1. understand the selected fashion dataset in detail; and
2. produce evidence that can be cited in the assignment report.

The EDA describes problems and likely modelling effects. It does not clean labels,
choose a rare-class policy, create a split, or make final modelling decisions.

## Dataset boundary

The unit of analysis is one product ID. The population starts with the 38,617 official
teacher-training IDs and joins their original metadata, original image, and 60×80 teacher
image. The five IDs without usable paired images are reported and excluded, leaving 38,612
usable products.

All 5,829 official test IDs remain quarantined. Their IDs may be used to prove exclusion,
but their labels must never enter the EDA. Paired high- and low-resolution images are two
views of one product and must never be counted as separate observations.

CSV loading must preserve literal `usage == "NA"` separately from a blank value and repair
the two comma-spill columns into `productDisplayName`.

## Cleanup and module design

The old EDA is replaced rather than extended. Before removing it, the four primitives used
by `fashion.dataset_comparison` move to a small shared data-audit module:

- CSV audit and product-name repair;
- hierarchy-conflict detection;
- difference hashing; and
- Hamming distance.

The new code uses focused modules:

- `fashion.data_audit` owns shared raw-data and image-hash primitives.
- `fashion.eda` exposes the small notebook-facing interface that builds the selected
  population and runs the complete analysis.
- Internal helpers calculate pure tables, image measurements, and plots without putting
  analysis logic in notebook cells.

The obsolete EDA plan, obsolete EDA design, old overview figure, stale README references,
and old notebook content are removed. A new `notebooks/00_eda.ipynb` is created from an
empty notebook.

## Analysis

### Metadata and target distributions

Analyse every usable metadata field: `id`, `gender`, `masterCategory`, `subCategory`,
`articleType`, `baseColour`, `season`, `year`, `usage`, and `productDisplayName`.

For the four targets, report counts, shares, blanks, literal values, majority share,
imbalance ratio, normalized entropy, effective class count, Gini impurity, and top-class
concentration. Show every `articleType` class on long-tail, log-scale, and cumulative
views. Summarize class support in `1`, `2`, `3–4`, `5–9`, and `10+` bands.

The product-name audit covers missing names, name length, common words, and deterministic
name-versus-label contradiction candidates. It is not a language-modelling task.

### Relationships and drift

Use Cramér's V, not ordinary numeric correlation, for categorical fields. Show one matrix
for all useful categorical pairs. Detailed row-normalized heatmaps cover the strongest
association and the task-relevant `gender × usage` relationship.

Measure how category shares change across `year` and deterministic ID-range bins. ID is
described as catalogue order, not time. These are descriptive drift measures only.

Audit the `masterCategory → subCategory → articleType` hierarchy and retain conflict IDs.

### Images and duplicates

Measure every usable 60×80 image. Measure a deterministic, proportionally stratified
sample of up to 2,048 original images with seed `2753`, while ensuring every observed
`articleType` can contribute an example. Report width, height, mode, file size, brightness,
contrast, colourfulness, saturation, and sharpness. Compare paired high- and low-resolution
measurements without treating views as extra products.

Use full-dataset content hashes for exact duplicates. Use difference hashes on a
deterministic sample of up to 2,048 products for near-duplicate candidates. A perceptual
match is only a review candidate and never an automatic merge.

Fixed-seed image grids show common, rare, unusual, grayscale, exact-duplicate, and
near-duplicate examples.

## Evidence and notebook

`notebooks/00_eda.ipynb` is a short narrative. It imports the implementation, runs or loads
the analysis, displays the detailed tables and plots, explains what each result means, and
ends with:

- factual findings;
- likely modelling effects; and
- open decisions for later work.

It must not decide how to clean labels, treat rare classes, or design the split.

Generated evidence lives under `results/figures/eda/`:

- one four-panel report figure covering target imbalance, the `articleType` long tail,
  the Cramér's V matrix, and image quality;
- detailed CSV tables;
- a machine-readable JSON summary;
- cached image measurements; and
- deterministic review grids.

The report uses the combined figure. Detailed plots remain available in the notebook.

## Statistical framing

The EDA uses descriptive measures and effect sizes. It does not use hypothesis tests or
p-values. With 38,612 products, tiny unimportant differences can easily appear
statistically significant.

Macro-F1 is the primary classification framing. Majority accuracy is shown only as a
sanity baseline.

## Safety and failure handling

- Use `./.venv/bin/python` and seed `2753`.
- Never call `train_test_split`.
- Never create or modify `data/processed/splits.csv`.
- Never expose official test labels.
- Fail clearly when required raw paths or columns are missing.
- Record unreadable images, unmatched IDs, malformed rows, blanks, and unusual modes
  rather than hiding them.
- Cache files include enough provenance to reject stale results when inputs or settings
  change.
- Notebook execution must be reproducible from a clean kernel.

## Verification

Automated tests cover CSV semantics, population construction, skew metrics, support bands,
Cramér's V, deterministic sampling, image measurements, duplicate candidates, and output
contracts. Verification also:

1. proves dataset comparison still works after shared helpers move;
2. executes the new notebook from top to bottom;
3. reconciles the expected 38,617 source IDs and 38,612 usable products;
4. confirms official test IDs are absent;
5. confirms evidence files and the combined figure exist; and
6. confirms no split or other processed dataset was created.

## Acceptance criteria

- The old EDA implementation and stale artifacts are gone.
- Dataset comparison remains functional.
- The new notebook is built from scratch and contains no analysis logic.
- Every metadata field, target skew, categorical relationship, drift view, image property,
  duplicate risk, and requested visual review is present.
- All numerical claims are generated by code.
- The final summary reports facts and open questions without silently choosing policies.
