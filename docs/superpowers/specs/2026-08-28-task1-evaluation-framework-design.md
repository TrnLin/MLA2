# Task 1 Evaluation Framework Design

**Date:** 2026-08-28  
**Status:** Approved in chat; awaiting review of this written design  
**Scope:** Task 1 `articleType` development evaluation only

## 1. Goal

Build a leakage-safe evaluation framework for Task 1 before choosing image
preprocessing or training substantive models. The same framework must evaluate a
majority-class smoke model, the later HOG baseline, and later PyTorch CNNs without
copying fold, metric, registry, or evidence code.

The framework must keep nearly all Task 1 work in Task 1 folders to reduce merge
conflicts. The only shared runtime component is the run registry required by the
project contract. Dependency declarations and their resolved constraints are also
shared because Python packages are installed for the repository as a whole.

## 2. Non-goals

This work does not select an image size, resize method, normalization rule,
augmentation, loss, sampler, architecture, optimizer, or winning model. It does not
implement HOG or a CNN. It does not unlock holdout labels or generate official test
predictions.

The included majority model exists only to prove that all five folds, metrics,
artifacts, and registry states work end to end. Its real comparison and analysis may
be reused later, but framework completion does not claim that the Task 1 baseline
investigation is complete.

## 3. Project constraints

- `data/processed/splits.csv` is the only split.
- Fold access uses `fashion.data.dataset.iter_cv_folds`.
- All five saved folds are evaluated. The best observed fold is never selected.
- Holdout and quarantine rows stay outside development evaluation.
- All 124 development-observed `articleType` labels stay in the output and metric
  space, including labels with little or no training support in a fold.
- Values learned from data are fitted on the current fold's training rows only.
- Every run starts a row in `results/runs.csv` through
  `fashion.train.registry` and finishes as `complete` or `failed`.
- Task 1 uses teacher images only.
- Submitted learned models are trained from scratch. Pretrained weights are not part
  of this framework.

## 4. Alternatives considered

### 4.1 Chosen: isolated Task 1 evaluator with a small shared registry

Task-specific contracts, evaluation, metrics, slices, artifacts, smoke code, and
tests live in Task 1 folders. A small model protocol lets later scikit-learn and
PyTorch models use the same evaluator. The shared registry owns only generic run-row
state and has no Task 1 metric logic.

This gives Task 1 one clear owner and keeps merge conflicts small while still meeting
the shared registry rule.

### 4.2 Rejected: generic framework for all classification tasks now

A generic multi-task framework could reduce later duplication, but its interfaces
would be designed before Tasks 2 and 3 have evidence about their needs. It would also
make several teammates edit the same files. The small Task 1 model protocol can be
generalised later only if another task proves that the same abstraction is useful.

## 5. File ownership

### 5.1 Task 1 source

```text
src/fashion/task1/
├── __init__.py
├── contracts.py
├── evaluator.py
├── metrics.py
├── slices.py
├── artifacts.py
├── reproducibility.py
├── smoke.py
└── models/
    ├── __init__.py
    └── majority.py
```

- `contracts.py` defines immutable evaluation configuration, model interfaces, fold
  outputs, and validation rules.
- `evaluator.py` owns the five-fold lifecycle and contains no model-specific code.
- `metrics.py` calculates fixed-label classification metrics.
- `slices.py` assigns predeclared analysis slices and summarises them.
- `artifacts.py` writes deterministic local run evidence.
- `reproducibility.py` sets seeds and records software, Git, and device information.
- `smoke.py` runs the majority model through the framework.
- `models/majority.py` implements the model protocol without reading image pixels.

### 5.2 Task 1 tests

```text
tests/task1/
├── test_contracts.py
├── test_metrics.py
├── test_slices.py
├── test_evaluator.py
├── test_artifacts.py
├── test_registry.py
└── test_smoke.py
```

Registry tests stay in `tests/task1/` because Task 1 introduces and owns the first
registry contract. Other tasks may add their own contract tests without editing these
files.

### 5.3 Shared files

```text
src/fashion/train/__init__.py
src/fashion/train/registry.py
pyproject.toml
requirements/constraints-py312.txt
```

The package initializer only exposes the stable registry API. The registry does not
import Task 1. Dependency changes add pinned, mutually compatible versions of
PyTorch, torchvision, scikit-learn, and scikit-image. The full CPython 3.12
constraints file is re-resolved rather than hand-editing only four lines.

### 5.4 Runtime and report outputs

```text
tmp/task1/runs/<run_id>/
tmp/task1/checkpoints/<run_id>/
results/runs.csv
results/evidence/task1/
results/figures/task1/
```

Raw per-run predictions, score matrices, and model-specific checkpoint files remain
local under the ignored `tmp/task1/` tree. This supports PyTorch `.pt` files and later
scikit-learn checkpoint formats without adding generated files to Git. Only compact,
deliberately selected tables and figures needed by the report are copied to the Task
1 evidence and figure folders.

## 6. Evaluation contract

`Task1EvaluationConfig` is an immutable dataclass with these values:

- `task_id = "task1"`
- `target = "articleType"`
- `folds = (0, 1, 2, 3, 4)`
- `seed = 2753`
- `primary_metric = "macro_f1_124"`
- `top_k = 5`
- `zero_division = 0`
- `high_confidence_threshold = 0.80`
- paths for the canonical split, label map, local artifact root, model root, and run
  registry

The ordered class vocabulary comes from the development-built `articleType` entry in
`data/processed/label_maps.json`. The evaluator requires exactly 124 unique labels and
requires the canonical ordered-vocabulary SHA-256
`2403d44bd056ec52cc30bf8839573d722d73522e7db18431ecd7f8873a68b1d1`.
The hash is calculated from compact UTF-8 JSON for the ordered `classes` list and is
stored in every run configuration. A run stops if the vocabulary count or hash
changes.

The primary score for a run is the arithmetic mean of the five fold-level
`macro_f1_124` values. Standard deviation uses the sample definition (`ddof=1`). Each
fold-level macro-F1 passes all 124 class indices to scikit-learn with
`zero_division=0`. Therefore a class with no true or predicted validation members in a
fold contributes zero for that fold. This is strict, matches the notebook's frozen
wording, and is supplemented by class support and per-class results so its limitation
is visible.

Supporting metrics are:

- weighted F1;
- top-1 accuracy;
- top-5 accuracy;
- per-class precision, recall, F1, and support;
- a 124 by 124 out-of-fold confusion matrix;
- pooled out-of-fold versions of macro-F1, weighted F1, top-1, and top-5, labelled as
  supporting rather than replacing the five-fold primary result.

## 7. Model interface

The evaluator receives a factory, not an already-created model. The factory creates a
fresh model for one fold so learned state cannot cross fold boundaries.

```python
class Task1FoldModel(Protocol):
    def fit(self, training_rows: pd.DataFrame) -> FitSummary: ...

    def predict_scores(self, validation_rows: pd.DataFrame) -> ScoreBatch: ...

    def save_checkpoint(self, path: Path) -> Path | None: ...
```

`ScoreBatch` contains the prediction product IDs in returned order and a finite
`float32` or `float64` score matrix with shape `(len(validation_rows), 124)`. This lets
the evaluator prove that scores correspond to the expected validation IDs instead of
assuming array order. Larger values mean stronger preference. The evaluator uses
rankings for top-1 and top-5. Models that claim probabilities must validate that rows
are non-negative and sum to one within a documented tolerance; raw-logit models are
also allowed and are marked as logits in `FitSummary`.

The framework passes only training rows to `fit` and only validation rows to
`predict_scores`. A later HOG pipeline will fit its feature scaling and classifier
inside `fit`. A later CNN pipeline will fit normalization, model weights, and any
training-only sampling values inside `fit`.

`FitSummary` records duration, model parameter or coefficient count, score kind,
checkpoint support, and model-specific notes. It does not carry validation metrics.

## 8. Five-fold flow

For one run, `evaluate_task1_cv` performs these steps:

1. Validate the configuration and ordered class map.
2. Call `start_run` before model work begins.
3. Load the redacted canonical split with `load_splits`.
4. Reject any non-development input to the fold loop.
5. Iterate the five official folds with `iter_cv_folds`.
6. Filter training and validation rows with `has_articleType_label`.
7. Build a new fold model from the factory using the fold number and seed.
8. Call `fit` with a defensive copy of training rows.
9. Call `predict_scores` with a defensive copy of validation rows.
10. Validate ID order, score shape, score finiteness, and class count.
11. Calculate fold metrics, per-class rows, predictions, and error slices.
12. Save the optional fold checkpoint and local fold artifacts.
13. After all folds, require every eligible development ID exactly once in the
    out-of-fold predictions.
14. Calculate the five-fold summary and pooled supporting evidence.
15. Write deterministic run artifacts.
16. Call `finish_run` with summary fields and artifact paths.

Any exception after `start_run` calls `fail_run` with the exception type and a short
sanitised message, then re-raises the exception. Raw image data, labels, environment
secrets, and full tracebacks are not written to `results/runs.csv`.

## 9. Predetermined error slices

Class-support slices are calculated separately in every fold from that fold's
training rows:

- `unseen`: 0 training products;
- `very_rare`: 1 to 9 training products;
- `rare`: 10 to 99 training products;
- `common`: at least 100 training products.

Structural slices use prepared, non-target fields:

- family size `single` or `multi` within development;
- image mode `RGB`, `grayscale`, or `other`;
- geometry `standard_60x80` or `unusual`.

High-confidence mistakes are incorrect top-1 predictions whose normalised confidence
is at least `0.80`. They are selected only for models that return probabilities. Logit
models report this slice as unavailable instead of applying a softmax inside the
evaluator and silently changing the score meaning.

Every slice table includes product count, correct count, top-1 accuracy, macro-F1 over
labels present in that slice, and a clear availability field. Slice macro-F1 is
supporting error analysis and is not used to rank models.

## 10. Artifacts

Each local run directory contains:

```text
tmp/task1/runs/<run_id>/
├── config.json
├── environment.json
├── fold_metrics.csv
├── summary.json
├── predictions.csv
├── scores.npz
├── per_class_metrics.csv
├── confusion_matrix.csv
└── error_slices.csv
```

`predictions.csv` contains run ID, fold, product ID, true label/index, predicted
label/index, ordered top-5 labels, top-1 score, correctness, score kind, and slice
keys. `scores.npz` stores the full 124-column score matrix plus product IDs and class
indices without expanding it into a large CSV.

CSV files use stable column order, UTF-8, Unix newlines, and deterministic row sorting.
JSON uses sorted keys. Atomic temporary-file replacement prevents a partially written
artifact from looking complete.

## 11. Run registry

The shared registry exposes:

```python
start_run(record: RunStart, path: Path = RUNS_CSV) -> None
finish_run(run_id: str, completion: RunCompletion, path: Path = RUNS_CSV) -> None
fail_run(run_id: str, failure: RunFailure, path: Path = RUNS_CSV) -> None
```

`start_run` appends one uniquely identified row with status `running`. `finish_run` or
`fail_run` updates only that run ID through an atomic rewrite. Starting an existing run
ID or finishing a non-running run is rejected.

The fixed generic columns are:

- schema version, run ID, task ID, target, status;
- UTC start and finish timestamps;
- model ID, preprocessing ID, seed, and fold policy;
- canonical configuration hash and class-map hash;
- primary metric name, mean, and standard deviation;
- JSON-encoded supporting metric summary;
- wall-clock training and inference seconds;
- parameter count and checkpoint pattern;
- local artifact directory and promoted evidence paths;
- Git commit, dirty-worktree flag, Python version, package versions, device;
- failure type and sanitised failure message.

Registry writes use a small lock file next to `results/runs.csv` plus atomic
replacement, so two teammates cannot silently overwrite each other's local rows. The
registry CSV remains generated and ignored by Git, avoiding merge conflicts.

## 12. Reproducibility and devices

The Task 1 seed helper sets Python, NumPy, and PyTorch seeds. Each fold derives its seed
deterministically from seed `2753` and the fold number. The run records whether PyTorch
deterministic algorithms were enabled and records warnings when the chosen operation or
device cannot promise exact repeatability.

Device selection is explicit: requested CUDA, MPS, or CPU must exist, while `auto`
chooses CUDA, then MPS, then CPU. The resolved device is recorded. Tests use CPU and do
not require GPU hardware.

## 13. Dependency policy

`pyproject.toml` adds direct, pinned versions of PyTorch, torchvision,
scikit-learn, and scikit-image. Versions must support CPython 3.12, Apple Silicon, the
existing NumPy pin, and one another. Dependency resolution updates the entire
`requirements/constraints-py312.txt` file in a clean CPython 3.12 environment, then
checks `pip check` before project tests. No package version is guessed or added only to
the constraints file.

## 14. Test strategy

### 14.1 Contracts and leakage

- Accept exactly folds 0 through 4 and reject missing, repeated, or extra folds.
- Require the 124-class ordered map and a stable class-map hash.
- Prove the evaluator passes training IDs only to `fit` and validation IDs only to
  `predict_scores` using a spy model.
- Reject holdout and quarantine rows even if a caller constructs a bad frame.
- Prove a new model instance is created for every fold.

### 14.2 Metrics

- Compare macro-F1, weighted F1, top-1, top-5, and per-class values with hand-computed
  tiny examples.
- Prove absent labels remain in strict fixed-label macro-F1.
- Test fewer-than-five examples, ties with stable class-index ordering, wrong score
  shapes, `NaN`, infinity, and unknown target labels.

### 14.3 Out-of-fold coverage and slices

- Prove every eligible development ID appears exactly once.
- Reject missing and duplicate predictions.
- Test every support boundary: 0, 1, 9, 10, 99, and 100.
- Test family, mode, geometry, and probability-only confidence slices.

### 14.4 Artifacts and registry

- Compare deterministic artifact bytes from repeated equivalent writes.
- Prove atomic writes do not leave a completed-looking partial artifact.
- Test registry start, finish, fail, duplicate ID rejection, illegal state changes,
  stable columns, and two-process write locking.
- Prove a failed evaluator run leaves a `failed` registry row.

### 14.5 Smoke evaluation

- Run a small five-fold fixture through the majority model on CPU.
- Confirm five fold rows, one summary, complete out-of-fold coverage, valid evidence,
  and a completed registry row.
- The smoke test does not decode images or train a CNN.

## 15. Completion gate

The framework is complete only when all of these are true:

1. `./.venv/bin/python -m pytest tests/task1 -q` passes.
2. Existing project tests pass.
3. Ruff checks the new source and tests without errors.
4. `pip check` passes in the resolved CPython 3.12 environment.
5. The five-fold smoke evaluation completes on CPU.
6. Every eligible fixture development ID has one out-of-fold prediction.
7. A holdout-injection test fails closed.
8. A completed run and a deliberately failed run have correct registry states.
9. Notebook 02 can import the evaluator and shared registry without copied logic.

Passing this gate means the evaluation machinery is ready. It does not mean Task 1
preprocessing, baselines, model comparisons, error analysis, or final judgement are
complete.
