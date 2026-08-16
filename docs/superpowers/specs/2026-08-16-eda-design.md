# Exploratory Data Analysis Design

**Date:** 2026-08-16  
**Status:** Approved for implementation planning

## Purpose

`notebooks/00_eda.ipynb` will describe the supplied data as it exists and provide
reproducible evidence for cleaning, splitting, and modelling decisions. It will not clean
the dataset, assign partitions, or write processed CSV files.

The notebook is complete only when every cleaning and split choice required by
`01_preprocessing.ipynb` has an evidence reference or is explicitly marked unresolved.

## Scope

Create only the foundation needed for EDA:

- `pyproject.toml` with minimal project metadata and the EDA dependencies;
- a repository `.gitignore`;
- `notebooks/00_eda.ipynb`;
- `results/figures/`;
- `src/fashion/config.py` for repository-relative paths and the analysis seed; and
- `src/fashion/eda.py` for focused, side-effect-free audit helpers.

The complete `src/fashion/` package remains deferred. The EDA environment requires
`pandas`, `numpy`, `matplotlib`, `pillow`, and `jupyter`; PyTorch is out of scope.

The repository has been initialized with Git. Committing implementation work is not part
of this design decision and requires a separate request.

## Architecture and data flow

The notebook is the narrative entry point. It reads raw inputs through helpers, explains
why each audit matters, displays compact evidence, and records the resulting decision.
Reusable mechanics live in `src/fashion/eda.py`, keeping notebook cells short and
reviewable.

Data flows in one direction:

1. raw CSV and image files;
2. validated audit summaries and identified image groups;
3. statistical and visual evidence;
4. split constraints and evidence for the three open modelling questions; and
5. a written decision ledger consumed by notebook 01.

The EDA may write one report-candidate composite figure to `results/figures/`. It must not
create `data/processed/clean.csv`, `data/processed/splits.csv`, or any equivalent split
assignment.

## Audit sequence

### 1. Raw schema and label semantics

Read `styles_train.csv` without converting the literal usage label `"NA"` into a missing
value. Report the physical CSV shape, named columns, two trailing phantom columns,
duplicate IDs, blanks, and literal values. Reconcile the metadata row count with the image
count and identify every unmatched ID in both directions.

### 2. Image integrity

Attempt to open and verify every training image. Report unreadable or corrupt files,
dimensions, and colour modes without silently dropping any finding.

### 3. Duplicate risk

Use content hashes to identify exact duplicate files. Use perceptual hashes to generate
candidate near-duplicate groups, then display candidates for visual confirmation. A
perceptual-hash threshold creates candidates only; it is not sufficient evidence to merge
items automatically.

Accepted duplicate groups become a split constraint: members must remain in the same
partition.

### 4. Target, hierarchy, and association evidence

For `articleType`, `season`, `gender`, and `usage`, report class support, imbalance ratios,
majority-class baselines, long tails, and missing values. Establish macro-F1 as the primary
classification framing.

Audit label co-occurrence and consistency across `masterCategory`, `subCategory`, and
`articleType`. Measure normalized gender/usage association. Association can motivate a
shared-backbone experiment, but cannot demonstrate that multi-task learning will improve
generalization; that decision remains conditional on a controlled model comparison.

### 5. Stratified visual review

With a fixed seed, display compact image grids covering common, rare, ambiguous,
suspicious, exact-duplicate, and near-duplicate examples. Commentary must distinguish
likely annotation noise from genuine visual difficulty.

### 6. Split requirements

Without assigning rows to partitions, identify:

- duplicate groups that must remain atomic;
- targets requiring balance checks;
- classes too small for fair stratified evaluation;
- minimum support constraints; and
- unresolved policies that notebook 01 must not guess.

### 7. Open decisions and final evidence

Gather evidence for:

1. rare-`articleType` handling: keep, drop from that task, or merge through a defensible
   hierarchy;
2. whether gender/usage association justifies testing a shared backbone against independent
   models; and
3. candidate relevance definitions for visual-search evaluation where no ground-truth
   similarity labels exist.

Produce one report-worthy composite dataset figure. End with a decision ledger containing
the decision, evidence reference, confidence, downstream consequence, and unresolved
follow-up for every cleaning, splitting, and modelling choice.

## Failure handling

- Missing required paths and unusable CSV schemas fail immediately with actionable errors.
- Corrupt images are collected and reported as findings.
- Unexpected dimensions, colour modes, unmatched IDs, duplicate IDs, blanks, and literal
  `"NA"` values remain visible; the EDA does not clean them.
- Sampling and candidate ordering use a fixed seed.
- Audit summaries retain counts and IDs so totals can be reconciled.

## Verification

Verification will:

1. import the EDA package in the project environment;
2. run focused smoke checks of helper outputs;
3. execute the notebook from top to bottom in a clean kernel;
4. assert that key audit totals reconcile;
5. confirm the composite figure exists; and
6. confirm that no processed data or split assignments were created.

There will be no standalone `tests/` directory. Every numerical claim in the completed
notebook must come from a fresh execution rather than the handoff or prior documentation.

## Documentation correction

Fresh EDA evidence must also correct the existing repository-design fact table: it currently
records 38,612 training images but omits that `styles_train.csv` contains 38,617 metadata
rows, leaving five label rows without image files. Update both the Markdown design note and
its manually synchronized HTML view once the notebook reproduces the finding.

## Acceptance criteria

- All seven required audits are present and reproducible.
- Literal `"NA"` and genuine blanks remain distinguishable.
- CSV/image discrepancies and corrupt files are explicitly identified.
- Exact duplicates and visually reviewable near-duplicate candidates are reported.
- All target distributions, hierarchy checks, and relevant associations are measured.
- Split constraints are stated without creating a split.
- The three open modelling questions receive evidence without overstating conclusions.
- One report-candidate composite figure and a complete decision ledger are produced.
- A clean-kernel execution succeeds without creating processed artifacts.
