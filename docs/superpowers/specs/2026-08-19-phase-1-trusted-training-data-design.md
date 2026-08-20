# Phase 1 Trusted Training Data Design

**Date:** 2026-08-19  
**Status:** Approved  
**Scope:** Build the reviewed training manifest and the project's only split.

## Purpose

Phase 1 turns the accepted dataset decisions into three checked artifacts:

- `data/processed/label_review.csv`;
- `data/processed/train_manifest.csv`; and
- `data/processed/splits.csv`.

The work is completed in six steps. After each step, the agent reports the result and waits
for user approval before continuing.

## Safety boundary

Official test data is quarantined before any allowed target label is loaded:

1. read only the `id` column from
  `data/raw/teacher/test/styles_prediction.csv`;
2. require 5,829 unique integer test IDs;
3. read only IDs from the teacher-training and original metadata to prove that the official
  populations are disjoint and reconcile to 44,446 products;
4. load target labels only from
  `data/raw/teacher/train/styles_train.csv`; and
5. use the original collection only for the matching sharp image paths.

Phase 1 never loads target columns from `data/raw/original/styles.csv`. This makes the
quarantine stronger than loading all original labels and filtering them afterwards.
Existing EDA population code will be changed to use the same safe population loader so
there is not a second, weaker path.

The five official-training IDs with no image in either source are excluded before the
manifest is built: `12347`, `39401`, `39403`, `39410`, and `39425`. This leaves 38,612
usable products before any reviewed scope exclusions.

## Architecture

Focused modules under `src/fashion/data/` own separate jobs:

- population loading and quarantine;
- review candidate creation and approved review-table application;
- manifest construction;
- duplicate-group construction and split assignment; and
- artifact validation and deterministic writing.

A small command-line script calls those modules. It supports the six user checkpoints but
also supports one clean full rebuild after all review choices are approved. A short
`notebooks/01_preprocessing.ipynb` explains the process and displays checks; it contains no
cleaning or split logic.

## Six-step flow



### 1. Quarantine official test IDs

Load only official test IDs. Check their count, uniqueness, integer form, and disjointness
from the teacher-training IDs. No test target column is requested or retained.

### 2. Reconcile the allowed population

Parse teacher-training metadata while preserving literal `usage == "NA"` and repairing the
two trailing spill columns. Match each allowed ID to one sharp original image and one blurry
teacher image. Check path uniqueness and image readability. Report the 38,617 source rows,
the five known missing-image IDs, and the 38,612 usable products.

### 3. Review corrections, masks, and exclusions

Create targeted review material from the EDA evidence. It covers:

- the approved corrections for IDs `45824` and `38223`;
- the 20 blank season labels, one blank usage label, and 71 literal `"NA"` usage labels;
- 392 gender-name candidates, reviewed in related product groups;
- uncommon category-hierarchy mappings;
- 22 exact-image groups with conflicting targets;
- the confirmed blank image at ID `44998`, which is excluded as unusable;
- visually suspicious non-fashion products; and
- all 343 blurry grayscale images compared with their sharp originals.

Text rules only create candidates. They never correct labels automatically. Clear mistakes
are corrected, unresolved target labels are masked, readable fashion and beauty products
are kept, and only visually confirmed non-fashion or unusable products are excluded.

`label_review.csv` is long-form: one row per reviewed action. Its stable columns are:

`id,field,action,old_value,new_value,reason,evidence,review_status`.

Allowed actions are `keep`, `correct`, `mask`, and `exclude`. Final artifact generation
fails if a required candidate is still pending. Raw files are never edited.

### 4. Build the manifest

`train_manifest.csv` has one row per usable official-training product before scope
filtering. Rows are sorted by integer ID. It contains:

- the cleaned metadata and four target labels;
- repository-relative sharp and blurry image paths;
- one stable duplicate-group ID;
- `is_in_scope`; and
- one validity flag for each target.

A missing or unresolved target changes only its own validity flag. The product remains
available to other targets and visual search when it is otherwise in scope.

### 5. Build the only split

Each manifest ID receives exactly one value in `splits.csv`:

- `train`;
- `validation`;
- `catalogue_holdout`; or
- `excluded`.

The file has stable columns `id,partition,group_id` and is sorted by ID.

Exact low-resolution image copies are joined into atomic groups before any assignment.
Confirmed near copies are also joined, but a hash match alone is never enough.

Every in-scope group containing an ID from `46,919–51,999` goes into the catalogue
holdout. Three exact-copy groups cross that boundary, so IDs `41236`, `46732`, and `46833`
also enter the holdout. This keeps every required high ID and prevents identical pixels
from appearing in development.

The remaining development groups use seed `2753`. Five deterministic
`StratifiedGroupKFold` candidates make article type rare-class-aware while keeping whole
groups together. The selected validation fold minimizes:

1. distance from 20% of development products; and
2. total-variation distance between development and validation class shares for each
  labelled target.

Missing or masked labels are omitted only from their target's balance score. Ties are
resolved by fold number. Natural class prevalence is not changed by deletion,
oversampling, or generated images.

After the provisional split, every sampled near-duplicate warning with dHash distance 0 or
1 that crosses a boundary is reviewed using sharp originals. Confirmed copies are merged
into groups and the split is rebuilt from the same seed. Visually similar but different
products remain separate. Each confirmed or rejected pair is stored in `label_review.csv`
as a `duplicate_group` review action, so the final grouping can be rebuilt without relying
on an untracked review page.

### 6. Verify and rebuild exactly

The final gate checks:

- 5,829 official test IDs are absent from every processed artifact;
- the source, missing-image, usable, reviewed-exclusion, and partition counts reconcile;
- IDs and image paths are unique where required;
- every usable product has both image views in one row and one partition;
- no duplicate group crosses a partition;
- all required reviews are resolved;
- target masks match missing and unresolved labels;
- partition values are disjoint and complete;
- the development split is the closest feasible grouped 80/20 split;
- validation and holdout target shares are reported without changing them; and
- two clean builds produce byte-identical CSV files.

CSV output uses fixed column order, integer-ID order, UTF-8, `\n` line endings, no index,
and no runtime timestamp. Rebuild checks compare SHA-256 hashes.

## Version control

`label_review.csv` and `splits.csv` are versioned decisions and must be committed.
`train_manifest.csv` is generated and remains ignored. `.gitignore` will allow only the two
versioned processed CSV files under `data/processed/`; raw data and all other generated data
remain ignored.

No commit is made automatically at a user checkpoint. The final Phase 1 commit happens
only after the user approves the safety and rebuild results.

## Testing

Tests are written before implementation. Small fixtures cover quarantine ordering, literal
`"NA"` handling, spill-column repair, missing image pairs, review actions, target-specific
masks, duplicate-group merging, holdout expansion, grouped split selection, and stable CSV
bytes.

An integration test builds all three artifacts from a miniature raw-data tree. Dataset
verification then runs against the real files with `./.venv/bin/python`. The existing EDA
tests must still pass after population loading moves to the shared safe module.

## Completion

Phase 1 is complete only when the three artifacts rebuild exactly, all counts reconcile,
test labels remain quarantined, split-safety checks pass, the user approves Step 6, and the
shared split is committed.