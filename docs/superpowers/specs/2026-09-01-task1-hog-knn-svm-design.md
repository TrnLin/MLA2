# Task 1 HOG + KNN/SVM Design

Date: 2026-09-01

## Goal

Add two classic machine-learning baselines for Task 1 (`articleType`):

- Histogram of Oriented Gradients (HOG) with K-Nearest Neighbours (KNN).
- HOG with a linear Support Vector Machine (SVM).

The baselines provide algorithm comparison evidence beside the Task 1 CNN. They are
not assumed to be the final Task 1 model. The shared Task 1 evaluation framework will
decide how well they perform.

## Fixed project constraints

- Read `data/processed/splits.csv` only through the existing dataset APIs.
- Use only the development partition and its saved folds for selection and evaluation.
- Never use holdout, quarantine, or official prediction images for tuning.
- Preserve the fixed 124-class label order, including when a training fold is missing a
  class.
- Append every evaluated model/fold configuration to `results/runs.csv` through
  `fashion.train.registry`.
- Keep the notebook narrative. Put reusable work in `src/fashion/`.
- Train from scratch. HOG, KNN, and Linear SVM do not use pretrained weights.

## Scope

Included:

- Two grayscale HOG settings.
- KNN and Linear SVM tuning on saved fold 0.
- Five-fold evaluation of the selected KNN and selected SVM.
- Shared Task 1 metrics, out-of-fold predictions, per-class evidence, and a short
  rare-class failure analysis.
- Safe HOG feature caching and resumable experiment stages.

Not included:

- Raw-pixel comparison.
- PCA or another learned feature reduction step.
- RGB or HSV HOG.
- RBF SVM or probability calibration.
- Speed or model-size comparison in the report.
- Augmentation for classic models.

The run registry may still record runtime because it is part of the common registry
schema. Runtime will not be used to rank or compare these models.

## Selected experiment approach

Use a staged search:

1. Run a smoke test that proves the full path works on a small, non-reportable sample.
2. On saved fold 0, compare two HOG settings with one default KNN and one default
   Linear SVM.
3. Select one shared HOG setting using the average of the two default models' fold-0
   macro-F1 scores.
4. Tune KNN and Linear SVM separately on that selected HOG matrix and fold 0.
5. Select one KNN and one Linear SVM using fold-0 macro-F1. Break a metric tie
   deterministically using weighted-F1, top-1 accuracy, top-5 accuracy, then canonical
   configuration ID.
6. Evaluate both selected configurations on folds 0, 1, 2, 3, and 4. Reuse an exact
   matching fold-0 result instead of rerunning it.
7. Rank and discuss final candidates using the mean of their five fold-level macro-F1
   scores. Report the sample standard deviation (`ddof=1`) and the other shared Task 1
   metrics as supporting evidence.

This approach gives five-fold evidence for both algorithms without running the full
parameter grid five times.

## Image and HOG contract

Use the existing shape-safe image loader to apply EXIF orientation, convert the source
to RGB, and produce a deterministic 80-by-60 image. Convert that result to grayscale
before HOG. Do not augment images or fit CNN-style RGB normalisation.

Hold these HOG settings fixed:

- `orientations=9`
- `cells_per_block=(2, 2)`
- `block_norm="L2-Hys"`
- `transform_sqrt=True`
- `feature_vector=True`
- `channel_axis=None`

Compare only `pixels_per_cell`:

- `(16, 16)`, producing 288 features for an 80-by-60 image.
- `(10, 10)`, producing 1,260 features for an 80-by-60 image.

Varying one HOG setting keeps the comparison easy to explain. The two cell sizes test
coarse shape against finer detail. The report will justify HOG through this ablation and
published support rather than a raw-pixel baseline.

## Model search

### HOG selection defaults

- KNN: `n_neighbors=5`, `weights="distance"`, Euclidean distance.
- Linear SVM: `C=1`, `class_weight=None`.

Run both defaults on both HOG matrices for fold 0. For each HOG setting, average the
KNN and SVM macro-F1 values. Select the HOG setting with the higher average. Break an
exact tie in favour of `(16, 16)` because it is smaller.

### KNN tuning

Use `sklearn.neighbors.KNeighborsClassifier` with:

- `n_neighbors` in `{3, 5, 11}`.
- `weights` in `{"uniform", "distance"}`.
- Euclidean distance.

The search therefore has six configurations. KNN remains exact even when it is slow.
Validation queries should run in bounded batches to control memory. Neighbour distances
and indexes for `k=11` may be reused to derive the smaller `k` results, but tests must
prove that reused uniform and distance votes match scikit-learn behaviour.

### Linear SVM tuning

Use `sklearn.svm.LinearSVC` with:

- `C` in `{0.1, 1, 10}`.
- `class_weight` in `{None, "balanced"}`.
- Fixed seed 2753.
- `dual="auto"`, `tol=1e-4`, and `max_iter=5000`.

The search therefore has six configurations. Use a documented finite iteration limit.
A convergence warning makes that run failed and ineligible for selection; it must not
be silently accepted. Do not use RBF SVM or `SVC(probability=True)`.

HOG already applies local L2-Hys normalisation. Do not add `StandardScaler`, because it
would add a learned preprocessing choice that is outside this design.

## Fixed-class model outputs

The evaluation framework requires a finite score matrix with shape `(rows, 124)`.

For KNN:

- Map `predict_proba()` or equivalent reused-neighbour votes from the fitted model's
  observed `classes_` into the fixed global class order.
- Fill classes absent from the training fold with zero.
- Require every row to sum to one within numerical tolerance.

For Linear SVM:

- Apply a stable row-wise softmax to the `decision_function()` columns for the fitted
  model's observed classes.
- Map those observed-class values into the fixed global class order.
- Fill classes absent from the training fold with zero and require every row to sum to
  one within numerical tolerance.

The softmax values are ranking scores for common evaluation. They are not claimed to be
calibrated probabilities.

## Evaluation contract

Reuse the Task 1 evaluation framework without redefining metrics. Each fold produces:

- `macro_f1`
- `weighted_f1`
- `top1_accuracy`
- `top5_accuracy`

The primary comparison value is the mean of exactly five fold-level `macro_f1` scores.
Also report the sample standard deviation. Pooled out-of-fold metrics are supporting
evidence and do not replace the five-fold mean.

Each prediction artifact contains:

- Product ID and true/predicted indexes and labels.
- `prob_000` through `prob_123` in fixed label-map order.

After five folds, validate exactly one out-of-fold prediction for every eligible
development ID. Produce the existing per-class precision, recall, F1, and support
table. The notebook will use it for a short rare-class failure analysis and will also
show important confusion pairs. Rare-class analysis is interpretation of the shared
evidence, not a new selection metric.

## Components and file boundaries

### `src/fashion/task1/classical.py`

Owns:

- Immutable HOG specification and cache identity.
- Deterministic image-to-HOG extraction.
- Cache validation and loading.
- KNN fitting and fixed-class score construction.
- Linear SVM fitting and fixed-class score construction.
- Small pure helpers for missing-class expansion and stable softmax.

It does not choose folds, rank experiments, write report tables, or contain notebook
presentation code.

### `src/fashion/task1/experiments.py`

Adds one `run_task1_classical_experiment()` orchestration entry point. It owns:

- `smoke`, `tune`, and `final` stages.
- Existing split and label-map loading.
- Configuration enumeration and deterministic IDs.
- Run-registry lifecycle and provenance.
- Exact-result reuse and resume checks.
- Prediction and metric artifact writing.
- HOG selection, model selection, and five-fold aggregation.

Classic evidence filenames must be separate from existing CNN evidence so one path
cannot overwrite the other.

### `notebooks/02_task1_article_type.ipynb`

The notebook calls the runner and tells the experiment story. It shows:

- Why HOG is a suitable shape descriptor for `articleType`.
- The two-cell-size comparison.
- Fold-0 KNN and SVM tuning tables.
- Five-fold comparison against the CNN through the shared evaluation framework.
- Short rare-class and confusion-pair analysis.
- An honest final judgement about where classic features work and fail.

### Tests and dependencies

Add `tests/task1/test_classical.py` and extend focused experiment/notebook contract tests
where necessary. Add a reproducible `scikit-image` dependency to `pyproject.toml` and
`requirements/constraints-py312.txt`; KNN and Linear SVM continue to come from
scikit-learn.

## HOG cache design

Store derived HOG arrays in an ignored local cache under
`data/processed/task1_hog_cache/`. One cache entry contains:

- Ordered product IDs.
- Float32 HOG matrix.
- Complete HOG and preprocessing specification.
- Schema version.
- Hashes of the split, source image inventory, and configuration.

The cache is valid only when all IDs, shapes, settings, versions, and hashes match.
Reject and rebuild incomplete, stale, or malformed entries. Write cache files
atomically so interruption cannot create an apparently valid cache. HOG is
non-learned, so one development-wide feature matrix is safe; fold boundaries are
applied only when fitting models.

## Run identity and artifacts

Every distinct configuration/fold evaluation gets one append-once registry row,
including smoke, failed, and interrupted runs. Use:

- `task="task1"`
- A clear classic-ML `model_family`.
- `primary_metric_name="macro_f1_124"`
- `scratch=True`
- `benchmark_only=False`, because these are scratch-trained candidates rather than
  pretrained comparison-only models.
- `final_eligible=False` for smoke runs and `final_eligible=True` for complete
  canonical fold runs. This permits an exact fold-0 tuning result to join the selected
  candidate's final five-fold evidence without rerunning it.

Config hashes include the image transform, HOG spec, model settings, fold, seed, and
relevant software versions. Exact completed results may be reused only when identity
and artifact hashes match.

Store generated run artifacts below the already ignored `results/task1/` tree. Write
classic comparison evidence under `results/evidence/task1/` as
`classical_tuning.csv`, `classical_fold_metrics.csv`, `classical_comparison.csv`,
`classical_oof_metrics.csv`, and `per_class_classical_<candidate-id>.csv`. Never
replace the existing CNN files `fold_metrics.csv`, `comparison.csv`,
`oof_metrics.csv`, or its per-class files.

## Failure handling

- A bad image reports its product ID and path.
- Non-finite or wrong-width features and scores fail before metric calculation.
- Cache validation failure triggers a safe rebuild; repeated extraction failure stops
  the run.
- SVM convergence warnings mark the run failed and exclude it from selection.
- Missing fold classes are handled through fixed-class expansion rather than failing.
- Interrupted runs remain interrupted in the registry and may be resumed with a new
  append-once run identity.
- Smoke evidence is clearly marked and can never enter final five-fold tables.

## Verification

Unit tests will cover:

- Exact feature lengths for both HOG settings.
- Deterministic extraction and grayscale handling.
- Cache hit, stale-cache rejection, and interrupted-write safety.
- Fixed 124-column expansion for KNN and SVM when training classes are missing.
- Stable finite SVM softmax scores.
- Reused KNN votes matching scikit-learn for every `k` and weight mode.
- Registry success, failure, interruption, and resume behaviour.
- No holdout, quarantine, or official-test access.

An integration smoke test will run both algorithms on a small fold subset and prove
that the common metric and prediction artifacts are valid. Before report use, the
final runner must prove that each selected candidate has folds 0 through 4 exactly
once and complete out-of-fold coverage.

## Success criteria

The work is complete when:

1. Both HOG settings can be extracted, cached, and validated.
2. Fold-0 HOG selection and both six-setting model searches complete through one
   runner.
3. The selected KNN and selected Linear SVM each have valid five-fold evidence.
4. All rows and score columns satisfy the shared Task 1 evaluation contract.
5. Every evaluated setting appears in the run registry.
6. The notebook presents the comparison and a short rare-class failure analysis.
7. Tests pass and no classic artifact overwrites CNN evidence.
