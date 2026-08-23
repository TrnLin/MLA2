# Phase 02 - Restructure the data-preparation notebook

## Ownership

- Restructure the renamed `notebooks/01_data_preparation.ipynb` workflow.
- Preserve reusable code cells and current trusted evidence wherever possible.
- Update direct tests, saved HTML, documentation paths, and provenance references owned by the rename.

## Required narrative order

1. Scope, data contract, full-build mode, and cached-review mode.
2. Raw source inventory and immutable SHA-256 hashing.
3. Image decoding, geometry, CSV/image ID reconciliation, and visual tag sanity examples.
4. Exact duplicates, perceptual candidates, product families, and quarantine.
5. Canonical manifests, sole split, protected labels, and image-variant counting rules.
6. Train-only missingness, taxonomy, class imbalance, and target association.
7. Train-only image quality, numeric correlation, and low/high alignment.
8. Shared deterministic transform and train-only normalization.
9. Task 4 query/gallery and relevance-proxy data contract.
10. Output hashes, provenance, limitations, decisions, and completion gate.

## New evidence required

- Raw hashing coverage table that distinguishes raw SHA-256 from perceptual and output hashes.
- CSV metadata ID to valid-image ID reconciliation table.
- Deterministic train-only contact sheet with all four target tags.
- Plain-language NMI explanation for categorical association.
- Spearman correlation for numeric image-quality features.
- Explicit imbalance policy: no SMOTE on pixels, no validation/holdout rebalancing, compare ordinary
  loss with one train-only imbalance-aware method later.
- Findings to decisions to future experiments table.

## Safety rules

- Do not expose holdout or quarantine target values.
- Do not use human review decisions as preparation inputs.
- Do not silently correct semantic labels from the contact sheet.
- Do not create another split.
- Do not count two resolution variants as independent products.
- Keep cached mode fast, but show the full logical build order before cached validation details.

## Todo

- [x] Preserve and remap useful code cells.
- [x] Add or revise markdown headers and explanations.
- [x] Add focused evidence cells for the identified gaps.
- [x] Execute the notebook and refresh deterministic evidence, figures, and HTML.
- [x] Update affected tests and documentation references.
- [x] Commit the completed phase separately.

## Success criteria

- Fresh full build follows audit -> manifest -> perceptual audit -> family -> split -> train-only stats.
- Routine cached run validates the prepared contract without hiding the logical build process.
- Required outputs are reproducible and protected targets never enter notebook state.
- Every main finding names its downstream modelling action.
