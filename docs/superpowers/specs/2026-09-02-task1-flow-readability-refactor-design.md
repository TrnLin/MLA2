# Task 1 Flow and Readability Refactor Design

**Date:** 2026-09-02  
**Scope:** Task 1 `articleType` code and Notebook 02  
**Change type:** Behaviour-preserving refactor followed by fresh Task 1 runs

## Goal

Make Task 1 easy to read from top to bottom. A reader should see the dataset problem,
understand why each preprocessing and model choice fits that problem, run the models,
inspect the evidence, and reach a final decision without jumping between large mixed-purpose
files.

The refactor preserves public imports, split rules, label order, metric definitions, artifact
schemas, and run-registry rules. Existing Task 1 results may be removed after verification
because the group will perform a fresh full run.

## Why this refactor is needed

The current implementation works, but three files carry too many responsibilities:

- `src/fashion/task1/experiments.py` has 774 lines covering CNN orchestration, classical
  tuning, selection provenance, evidence writing, and plots.
- `src/fashion/task1/classical.py` has 747 lines covering configuration, HOG extraction,
  caching, KNN, Linear SVM, registry handling, and artifact writing.
- `src/fashion/task1/training.py` has 449 lines covering validation, data loading, the CNN
  epoch loop, checkpoint selection, artifact writing, and registry handling.

Notebook 02 explains the right topics, but the run story is interrupted by long blocks of
policy text. Its CNN controller defaults to smoke while its classical controller currently
starts at final. The later error-analysis and final-decision sections describe future work
instead of displaying completed evidence.

## Fixed project constraints

- `data/processed/splits.csv` remains the only split source.
- Holdout and quarantine labels remain sealed until the final-evaluation notebook.
- Submitted models train from scratch. Pretrained systems remain comparison benchmarks only.
- Every physical training run is recorded through `fashion.train.registry`.
- The fixed 124-class label order and metric definitions do not change.
- Reusable logic stays in `src/fashion/`; Notebook 02 stays narrative and thin.
- The prediction format `id,gender,articleType,season,usage` does not change.
- Shared prepared data, including `splits.csv` and label maps, is never deleted by this work.

## Refactor approach

Use small modules with one clear job. Keep `fashion.task1`,
`fashion.task1.classical`, and `fashion.task1.experiments` as stable import surfaces so
notebooks and external callers do not need a breaking migration.

### Shared image contract

Create `src/fashion/task1/image_contract.py` to own the common deterministic image geometry:

- image size `(80, 60)`, representing the existing 60-by-80 canvas;
- white padding `(255, 255, 255)`;
- EXIF orientation correction, RGB conversion, and aspect-preserving placement through the
  existing `transform_image_with_mask` helper.

Both CNN preprocessing and HOG extraction consume these shared values. Their model-specific
branches remain different:

```text
shared deterministic image preparation
|- CNN: RGB statistics -> optional train-only augmentation -> tensor normalization
`- classic: grayscale -> HOG -> KNN or Linear SVM
```

The serialized preprocessing and HOG dictionaries keep their existing fields and values so
configuration and artifact schemas remain understandable.

### CNN training flow

Create `src/fashion/task1/cnn_engine.py` for the pure training mechanics:

- deterministic data-loader construction;
- one epoch of optimization;
- validation prediction and metric calculation;
- best-state selection;
- an internal result containing the model, history, best epoch, metrics, and predictions.

Keep `src/fashion/task1/training.py` as the high-level registered fold runner. Its visible flow
becomes:

1. validate the requested run;
2. validate the canonical split for reportable runs;
3. create the registry record;
4. prepare fold data and train-only normalization;
5. call the CNN engine;
6. write and hash checkpoint, history, and predictions;
7. finalize the registry and return `Task1FoldResult`.

`Task1TrainConfig`, `Task1FoldResult`, `select_training_device`, and `train_task1_fold` keep
their public names and behaviour.

### Classical model flow

Split the current `classical.py` responsibilities into:

- `src/fashion/task1/classical_features.py`: HOG specifications, deterministic extraction,
  cache identity, cache validation, and aligned feature loading.
- `src/fashion/task1/classical_models.py`: KNN and Linear SVM configurations, fixed grids,
  neighbour voting, score expansion, and estimator fitting.
- `src/fashion/task1/classical_training.py`: classical run configuration and result types,
  one registered fold execution, artifact writing, and exact completed-result reuse.

Reduce `src/fashion/task1/classical.py` to a documented compatibility facade that re-exports
the existing public classical API. Private tests move to the module that owns the behaviour.

### Experiment controller flow

Split orchestration into:

- `src/fashion/task1/cnn_experiments.py`: CNN smoke/full schedules, fold aggregation, OOF
  checks, and CNN evidence writing.
- `src/fashion/task1/classical_experiments.py`: classical smoke/tune/final schedules, HOG and
  model selection, selection provenance, five-fold aggregation, and classical evidence writing.
- `src/fashion/task1/plotting.py`: comparison and confusion-matrix figures.

Reduce `src/fashion/task1/experiments.py` to a documented compatibility facade. Existing
public names such as `run_task1_experiment`, `run_task1_classical_experiment`, and both result
dataclasses remain importable from their current locations and from `fashion.task1`.

Dependency injection for fold runners remains available. It keeps controller tests fast and
allows failures to be tested without training real models.

### EDA and problem evidence

Create `src/fashion/task1/analysis.py` with small, read-only helpers that turn prepared evidence
into a Task 1 problem profile and a decision-evidence table. The helpers use existing prepared
data and do not create a new split or inspect protected labels.

The problem profile exposes the evidence already found by Notebook 01:

- 32,773 image-backed development products;
- 124 `articleType` classes;
- class support from 1 to 5,748 products;
- 26 classes with fold-support warnings;
- 12 classes untrainable in at least one fold;
- 294 grayscale images;
- 12 images outside the usual 60-by-80 geometry;
- duplicate-family risk handled by the saved family-safe folds.

Notebook prose must connect evidence to choices:

| Evidence | Problem | Choice to test |
|---|---|---|
| Tiny images and unusual dimensions | Stretching or cropping can change/remove shape | Aspect-preserving 60-by-80 white canvas |
| Grayscale images | Input channel formats are inconsistent | Deterministic RGB conversion |
| Very long class tail | Accuracy can hide rare-class failure | Fixed-124-class macro-F1 and per-class evidence |
| Missing fold support for rare classes | Some classes cannot be learned in every fold | Keep all classes visible and report the limitation |
| Product shape is important to article type | A simple shape baseline is useful | Compare two HOG scales with KNN and Linear SVM |
| Local visual patterns may exceed hand-made shape features | HOG may miss colour and texture | Compare against a scratch CNN |
| Small training images may overfit | Mild transformations might help | Controlled no-augmentation versus mild-augmentation CNN comparison |

These links are hypotheses, not automatic winners. The notebook must use generated comparison
tables to accept or reject each hypothesis. For example, the current evidence gives higher
primary macro-F1 to no augmentation than mild augmentation, so the current run must not be
described as proof that augmentation wins. Fresh runs replace the old conclusion.

## Notebook 02 flow

Reorder and shorten `notebooks/02_task1_article_type.ipynb` into this story:

1. **Problem:** input, output, 124-class task, and final use.
2. **EDA evidence:** display the Task 1 problem profile and a small evidence-to-choice table.
3. **Safety contract:** fixed split, protected data, training-only fitted values, scratch models.
4. **Evaluation:** explain macro-F1, supporting metrics, five folds, and OOF coverage using the
   long-tail evidence.
5. **Candidate hypotheses:** explain CNN, HOG + KNN, and HOG + Linear SVM using the product-image
   problem rather than generic model descriptions.
6. **Controlled preprocessing:** show shared image preparation and the model-specific branches;
   compare CNN augmentation as one isolated change.
7. **Run plan:** display the exact smoke, tune, and full schedules before execution.
8. **Run controllers:** use safe defaults (`RUN_MODE="smoke"` and
   `CLASSICAL_STAGE="smoke"`) and call only reusable package functions.
9. **Results:** display generated five-fold, OOF, and per-class evidence when it exists.
10. **Failure analysis:** show rare classes and important confusion pairs with representative
    examples when final evidence exists.
11. **Decision:** state which hypotheses passed, which failed, and why the chosen model fits the
    real problem.
12. **Handoff:** remain not ready until complete registered five-fold evidence exists.

Run All must be safe. Smoke mode may write non-final smoke artifacts, but it must not start full
CNN training, classical tuning, or classical final evaluation by default.

## Evidence and artifact compatibility

The refactor preserves these evidence schemas and paths:

- CNN: `fold_metrics.csv`, `comparison.csv`, `oof_metrics.csv`, and per-class files;
- classical: `classical_tuning.csv`, `classical_selection.json`,
  `classical_fold_metrics.csv`, `classical_comparison.csv`, `classical_oof_metrics.csv`, and
  per-class files;
- figures under `results/figures/task1/`;
- physical model artifacts under `results/task1/<run_id>/`;
- registry rows in `results/runs.csv`.

Implementation hashes must include every new module that can change a trained artifact or its
selection. Old artifacts are not resumed across a changed implementation identity.

## Task 1 output cleanup

After the refactor passes tests, remove only generated Task 1 state before the fresh run:

- `results/task1/`;
- `results/evidence/task1/`;
- `results/figures/task1/`;
- `data/processed/task1_hog_cache/`;
- rows with `task == "task1"` from `results/runs.csv`.

Before deletion, resolve and verify each directory is exactly inside the repository and exactly
matches one of the paths above. Rewrite `results/runs.csv` atomically while preserving its header
and every non-Task-1 row. Do not remove shared data-preparation evidence, shared figures,
`splits.csv`, label maps, or raw images.

The cleanup happens after code verification and before real smoke/full reruns. It is not performed
by an ordinary notebook Run All.

## Testing strategy

Start with characterization tests that lock the current external contract:

- public imports and call signatures;
- shared 60-by-80 geometry and white padding;
- exact CNN and classical schedules;
- registry status and `final_eligible` rules;
- prediction and evidence column schemas;
- five-fold and OOF completeness rules;
- artifact and selection provenance checks.

Then move focused private tests to the new owning modules and add tests for:

- both model families consuming the same shared image contract;
- CNN engine output and best-state selection;
- HOG extraction/cache independent from estimator fitting;
- KNN/SVM fixed-class outputs independent from registry writing;
- controller stage validation without physical training;
- EDA profile values and evidence-to-choice mapping;
- Notebook 02 section order, safe smoke defaults, forbidden split/pretrained tokens, and absence
  of embedded training loops.

Verification finishes with:

1. focused Task 1 tests;
2. the complete project test suite;
3. Ruff on changed Python files;
4. notebook-format validation;
5. one real CNN smoke run and one real classical smoke stage;
6. registry and artifact checks for those smoke runs.

Full CNN, classical tuning, and selected five-fold runs are handed back to the user and are not
started automatically during this refactor.

## Error handling

- Invalid stages, folds, class maps, score widths, cache contents, and artifact hashes fail with
  clear errors before aggregate evidence is written.
- Registry contexts continue to record completed, failed, and interrupted physical runs.
- Aggregate evidence is written only after every required fold and prediction table validates.
- A failed refactor test does not trigger deletion of old Task 1 results.
- Missing final evidence makes Notebook 02 show a clear "not ready" message instead of failing
  while trying to display a missing table.

## Non-goals

- No new model family, loss, sampler, or hyperparameter grid.
- No change to the saved split, label taxonomy, evaluation formulas, or final prediction format.
- No pretrained submitted model.
- No full training run during the refactor.
- No report-writing work beyond making Notebook 02 produce clear, report-ready evidence.

## Success criteria

The refactor is complete when:

1. A reader can follow Notebook 02 from EDA evidence to model decision without reading source
   internals.
2. Every preprocessing and model hypothesis is tied to a visible dataset problem and later judged
   using generated results.
3. Shared image geometry has one source of truth while CNN and HOG keep correct model-specific
   branches.
4. The large mixed-purpose modules are replaced by focused modules and stable compatibility
   facades.
5. Public Task 1 imports, output schemas, registry rules, split safety, and metric semantics remain
   stable.
6. Focused tests, the full suite, lint, notebook validation, and both real smoke paths pass.
7. Old Task 1-only results can be safely cleared without touching shared prepared data.
