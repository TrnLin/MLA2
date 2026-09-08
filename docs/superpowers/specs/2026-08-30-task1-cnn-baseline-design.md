# Task 1 Small-CNN Baseline Design

**Date:** 2026-08-30
**Scope:** Task 1 `articleType` only

## Goal

Add a small convolutional neural network (CNN) baseline inspired by the Kaggle
CIFAR-10 notebook. The model will train from scratch on the assignment images.
It will use the project's existing Task 1 preprocessing, saved cross-validation
folds, fixed 124-class label map, and shared run registry.

The baseline is a comparison anchor. It is not a preselected final model.

## Hard boundaries

- Read split assignments only from `data/processed/splits.csv` through the
  existing safe dataset helpers.
- Never call `train_test_split`.
- Use development folds 0--4 for model comparison. Keep holdout and quarantine
  labels sealed.
- Train all submitted-model candidates from scratch. This baseline uses no
  pretrained weights.
- Append every physical training run, including smoke runs and failures, through
  `fashion.train.registry` into `results/runs.csv`.
- Keep reusable logic in `src/fashion/`. The notebook controls runs and explains
  results.

## Model

Create a `Task1SmallCNN` in `src/fashion/task1/models.py`.

The network accepts a float tensor shaped `(batch, 3, 80, 60)` and returns raw
class scores shaped `(batch, 124)`. It follows the Kaggle model's basic pattern:
convolution, ReLU, pooling, and dense classification. It fixes the original
notebook's fragile flattening and accidental convolution reuse.

Proposed layers:

1. 5x5 convolution from 3 to 32 channels, ReLU.
2. 5x5 convolution from 32 to 64 channels, ReLU, 2x2 max pooling.
3. 5x5 convolution from 64 to 64 channels, ReLU, 2x2 max pooling.
4. Two separate 5x5 convolutions from 64 to 64 channels, each followed by ReLU,
   then 2x2 max pooling.
5. Adaptive average pooling to `(4, 3)`.
6. Flatten the resulting 768 values, apply a 128-unit dense layer with ReLU,
   then a 124-unit output layer.

Adaptive pooling makes the classifier head independent of a hand-calculated
flattened image size. The model exposes its configuration and parameter count for
run records and checkpoints.

## Components

### `src/fashion/task1/dataset.py`

- Wrap one fold's metadata rows as a PyTorch dataset.
- Resolve image paths from the project root.
- Apply the existing Task 1 training or validation transform.
- Encode `articleType` with the existing development-defined label map.
- Reject missing paths, missing labels, unknown labels, duplicate IDs, and
  protected partitions.
- Return the tensor, integer label, and product ID needed for prediction evidence.

### `src/fashion/task1/training.py`

- Build deterministic data loaders for one validation fold.
- Fit normalization on the four training folds only.
- Train one fold and evaluate the held-out development fold.
- Save the best checkpoint by fixed-label macro-F1.
- Save epoch history and row-level validation predictions.
- Calculate macro-F1, weighted F1, Top-1 accuracy, and Top-5 accuracy.
- Record runtime, parameter count, configuration hashes, artifact hashes, and run
  status through the shared registry.
- Leave aggregation across folds separate from one-fold training.

### `src/fashion/train/`

Bring the shared registry and its required support code from
`origin/feature/task2-season` into the Task 1 branch. Reuse its schema and
lifecycle instead of creating a second Task 1-only registry.

### `notebooks/02_task1_article_type.ipynb`

The notebook will:

- select smoke or full mode;
- show the frozen configuration before training;
- call reusable functions from `src/fashion/`;
- display registered run IDs and statuses;
- aggregate five-fold results;
- compare no augmentation with mild augmentation;
- produce concise report-ready tables and figures;
- update the existing baseline, experiment, evidence, and handoff sections without
  embedding a second training implementation.

## Training configuration

The starting baseline configuration is deliberately close to the Kaggle example:

- seed: project seed `2753`;
- epochs: 20 for full runs;
- batch size: 128, with a clear configuration override for limited hardware;
- loss: unweighted cross-entropy;
- optimizer: Adam;
- maximum learning rate: `1e-3`;
- weight decay: `1e-5`;
- scheduler: OneCycleLR;
- gradient clipping: maximum gradient norm `1.0`, enabled and recorded;
- checkpoint rule: highest validation macro-F1;
- no early stopping, so preprocessing variants receive the same training budget.

Training augmentation must receive the current epoch so its seeded transformation
changes by epoch but remains reproducible. Validation is always deterministic.

## Run modes

### Smoke mode

- Use validation fold 0 only.
- Use batch size 16 and limit training and validation to two batches each.
- Train for one epoch.
- Exercise data loading, forward and backward passes, evaluation, checkpointing,
  artifact writing, and registry finalization.
- Record `stage="smoke"` and `final_eligible=false`.

Smoke results are checks, not report evidence.

### Full mode

Run both controlled preprocessing candidates across validation folds 0--4:

1. `task1_rgb_60x80_no_aug_v1`;
2. `task1_rgb_60x80_mild_aug_v1`.

This creates ten physical training runs. Each run receives the same seed, model,
loss, optimizer, epoch budget, metric rule, and checkpoint rule. Only preprocessing
changes.

The notebook reports the arithmetic mean and sample standard deviation (`ddof=1`)
of fold macro-F1. It also builds all out-of-fold supporting metrics and one
confusion matrix without replacing the frozen five-fold primary comparison.

## Artifacts

Each run writes beneath `results/task1/<run_id>/`:

- `checkpoint.pt` with model state, configuration, label-map identity,
  normalization, fold, seed, and best epoch;
- `history.csv` with one row per epoch;
- `predictions.csv` with product ID, true index, predicted index, and class
  probabilities needed for Top-K metrics.

The full notebook writes cited comparison figures to `results/figures/`. It does
not hand-type experimental scores into the report.

Artifact files are written safely and hashed before the registry run is finalized.
A failed or interrupted run keeps its status and cannot be aggregated as completed
evidence.

## Metrics

The primary metric is macro-F1 over the fixed list of all 124 classes with
`zero_division=0`. Supporting metrics are weighted F1, Top-1 accuracy, Top-5
accuracy, per-class precision/recall/F1/support, and the out-of-fold confusion
matrix.

The Kaggle notebook's binary-style AUC calculation will not be copied because it
does not correctly describe this 124-class task.

## Error handling

- Reject any training or normalization input containing the active validation
  fold.
- Reject holdout and quarantine rows in development training code.
- Reject non-finite losses and non-finite prediction scores.
- Reject mismatched class counts and label-map hashes when loading checkpoints.
- Record exceptions and keyboard interruptions in the registry before re-raising
  them.
- Never treat smoke, failed, interrupted, or incomplete runs as final evidence.

## Tests

Add focused tests for:

- model input and output shapes and parameter counting;
- dataset label encoding, product IDs, and protected-row rejection;
- fold separation and training-only normalization;
- fixed-124-class macro-F1 and Top-5 calculations;
- deterministic loading and repeatable smoke training;
- best-checkpoint selection and artifact contents;
- completed, failed, and interrupted registry lifecycle;
- one tiny end-to-end run using generated images.

The existing full test suite must still pass. A real-data smoke run is required
before any full five-fold experiment starts.

## Acceptance criteria

- The notebook can run a one-epoch smoke test and show a completed, non-final run.
- Full mode can produce ten completed registered fold runs without touching
  protected labels.
- Every checkpoint is a scratch-trained 124-class Task 1 CNN.
- Both preprocessing candidates use identical settings except augmentation.
- Results include five-fold mean and sample standard deviation of macro-F1.
- Saved predictions cover every eligible development product exactly once per
  preprocessing candidate.
- Tests prove split safety, metric semantics, artifact creation, and registry
  lifecycle.
