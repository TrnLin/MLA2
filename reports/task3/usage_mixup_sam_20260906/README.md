# Usage MixUp + SAM: fresh weights, folds 0 and 4

This trial trains two new SmallCNN models from random weights. It loads no Gender model
or earlier Usage checkpoint. The completed expanded v2 E8 models supply comparison scores only.

It uses the existing main EDA family folds and the approved
`data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv` dataset: teacher images plus
687 admitted Usage images. All nine Usage classes remain in the task.

| Control | Value |
|---|---|
| Validation folds | 0 and 4 only; other choices are rejected |
| Training data per run | Other four saved development folds |
| Initial weights | Fresh random weights for each fold |
| Model | Original Usage SmallCNN, 80 × 60 RGB |
| Epochs and saved checkpoint | 30, final epoch |
| MixUp | Alpha 0.2, every training batch |
| SAM | Rho 0.05, AdamW, two gradient passes and one update |
| Class loss weights | Recomputed from training labels, beta 0.999, cap 5 |
| Other E8 controls | Batch 128, seed 2753, translation 2 px with probability 0.5 |
| Learning rate | 0.001, cosine schedule over 30 epochs, minimum 0.00001 |

The class loss weights are not learned model weights. They give rare labels more influence.
MixUp uses `lambda * weighted_CE(y) + (1-lambda) * weighted_CE(partner_y)`. Each term divides
by the sum of class weights in that batch. A permutation preserves this denominator, so
the loss agrees with explicitly weighted soft targets. SAM uses the exact same mixed batch
for both passes; it retains one BatchNorm running-statistics update. Any Dropout mask is replayed.

Mixed training inputs do not have ordinary training F1. Final clean training predictions
are saved separately, with teacher and external scores. SAM loss summaries use the same
class-weight denominator as the training loop.

## Run in VS Code

1. Put `usage_mixup_sam_training.zip` beside the existing `task3-data.zip` in
   `MyDrive/MLA2/data/`.
2. Open [the notebook](../../../notebooks/task3_training/usage_mixup_sam_screen.ipynb) in VS Code,
   select a fresh Colab GPU kernel, and choose **Run All**.

The notebook verifies its bundled code, data and baseline artifacts. All training pixels
are hashed and decoded before fitting. Teacher and external product families keep their saved
folds. Protected holdout/test images are not used for training or selection.

Outputs go to `MyDrive/MLA2/task3_usage_mixup_sam/`. Each run is registered through
`fashion.train.registry` in the screen's own
`experiments/t3_usage_expanded_v2_mixup_sam/usage/results/runs.csv`. A separate local mirror is
also written. The global repository registry and previous experiment logs are preserved.

A rerun verifies and reuses a completed fold. An interrupted fold starts a new run from
random weights. It does not load an unfinished checkpoint. The two-fold limit is enforced
in both the screen runner and the shared fold trainer.

## Read the result

`aggregate_folds_0_4/teacher_comparison.json` compares the candidate with the completed v2 E8
baseline on exactly the same teacher images from folds 0 and 4. It includes per-fold F1,
clean teacher training gaps, rare-class F1 and recall, and separate source scores.

Use teacher validation F1 and rare-class recall to judge the trial. A smaller training gap
alone is not a win. Both methods change together, so this experiment cannot isolate MixUp's
effect from SAM's. Tiny teacher rare-class counts limit certainty. Earlier holdout/test results
already exist; this does not create a new untouched test.

No other folds are launched and no model is promoted automatically.

## Rebuild the bundle

```bash
./.venv/bin/python reports/task3/usage_mixup_sam_20260906/build_bundle.py
```

PyTorch is required. The build verifies all development images and both saved baseline
folds. Its manifest hashes every archived file, including the new code and notebook.
