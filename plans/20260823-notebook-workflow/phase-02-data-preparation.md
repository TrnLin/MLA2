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
5. Canonical manifests, sole split, protected labels, and group-safe five-fold allocation.
6. Development-only missingness, raw taxonomy, class imbalance, and fold support.
7. Development-only target association and numeric image-quality correlation.
8. Deterministic image/tag examples and transformation-risk demonstrations.
9. Output hashes, provenance, limitations, teammate handoff, and completion gate.

## New evidence required

- Raw hashing coverage table that distinguishes raw SHA-256 from perceptual and output hashes.
- CSV metadata ID to valid-image ID reconciliation table.
- Deterministic train-only contact sheet with all four target tags.
- Plain-language NMI explanation for categorical association.
- Spearman correlation for numeric image-quality features.
- Explicit imbalance boundary: diagnose only; task owners choose any loss or sampling comparison.
- Exact development/holdout/quarantine counts, five-fold coverage, and preserved ID-set digests.
- An artifact registry naming each real input and output without inventing file I/O for simple cells.
- Findings to decisions to future experiments table.

## Safety rules

- Do not expose holdout or quarantine target values.
- Do not use human review decisions as preparation inputs.
- Do not silently correct semantic labels from the contact sheet.
- Do not create another split.
- Do not read the optional external image collection from the shared workflow.
- Do not select a shared resize, crop, padding, normalization, augmentation, or retrieval protocol.
- Fit every learned preprocessing value on the training folds of the current evaluation round later.
- Keep cached mode fast, but show the full logical build order before cached validation details.

## Todo

- [x] Migrate the canonical split to development/holdout/quarantine plus `cv_fold`.
- [x] Remove supported/deployed label masking and paired-image defaults.
- [x] Make full preparation teacher-only and preserve the existing protected membership.
- [x] Rebuild the notebook with local outputs and tailored explanations.
- [x] Execute the notebook and refresh deterministic evidence, figures, and HTML.
- [x] Update affected tests and documentation references.
- [x] Commit each completed implementation slice separately.

## Success criteria

- Fresh full build follows audit -> manifest -> perceptual audit -> family -> protected split ->
  development-only descriptive evidence.
- Routine cached run validates the prepared contract without hiding the logical build process.
- Required outputs are reproducible and protected targets never enter notebook state.
- Every main finding names its downstream handoff without choosing a task result.
