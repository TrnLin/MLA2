# Usage E8 training on the new dataset

This trains the existing E8 SmallCNN recipe from scratch on the **teacher data plus 687 added
images**: the earlier 120 and the new 567. It uses the current main EDA's saved teacher folds.
All teacher and earlier external IDs and folds stay fixed.

## Run in Colab

1. Open `notebooks/task3_training/usage_expanded_v2_e8.ipynb` in Colab.
2. Select a GPU runtime.
3. Upload `teacher_plus_rare_usage_v2_training.zip` to `MyDrive/MLA2/data/`.
4. Keep the existing `task3-data.zip` in that folder, then choose **Run All**.

The training ZIP contains the new code, versioned dataset, all 687 added images and the saved
reference evidence. It reuses teacher images from `task3-data.zip`. The separate dataset-only ZIP
is not needed. No Git commit or push is required.

Results go to `MyDrive/MLA2/task3_usage_expanded_v2_e8/`. The dedicated run log is:

```text
MyDrive/MLA2/task3_usage_expanded_v2_e8/experiments/t3_usage_expanded_v2_e8/usage/results/runs.csv
```

Every training execution uses `fashion.train.registry`, with a local `results/runs.csv` mirror.
The new notebook does not write to the earlier Usage or Gender logs.

Run All trains five models sequentially, one per saved fold. Each starts with fresh weights and
runs **30 epochs**. After a disconnect, complete folds are reused only after their recipe,
dataset, registry, checkpoints, predictions and diagnostic files are verified. An unfinished
fold restarts from scratch under a new run ID; it is not treated as complete.

## Fixed experiment

- Dataset: `data/processed/teacher_plus_rare_usage_v2_20260906/splits.csv`.
- Target: the original nine Usage classes, including literal `NA`.
- Images: RGB, width 60 × height 80.
- Model: E8 SmallCNN, widths 32/64/128/256, average pooling, fresh scratch weights.
- Batch 128; 30 epochs; AdamW; learning rate 0.001; weight decay 0.0001.
- Cosine schedule ending at 0.00001; seed 2753; FP32; final-epoch checkpoint.
- Effective-number weighted cross-entropy: beta 0.999, cap 5.
- Training-only translations up to 2 pixels, probability 0.5.
- Normalization and class weights are fitted on that fold's training rows only.
- Added white borders are excluded from channel statistics using the saved source geometry.

The trainer changes the data, keeping the E8 model and training recipe fixed. No pretrained or
reference checkpoint weights are loaded into the model.

| Validation fold | Training images | Validation images |
|---|---:|---:|
| 0 | 26,751 | 6,708 |
| 1 | 26,771 | 6,688 |
| 2 | 26,771 | 6,688 |
| 3 | 26,772 | 6,687 |
| 4 | 26,771 | 6,688 |

All 33,459 valid Usage development images receive one validation prediction across the five
folds. The first fold is slightly larger because related external products stay together.
The new Smart Casual intake includes a conservative 49-image family.

## Read the results

Use the **same teacher validation images** to compare original E8, the first 120-image expansion,
and this 687-image expansion. Both reference sets are checked against their completed run logs,
recipes, saved file hashes and exact validation IDs. Original E8 teacher macro-F1 is 0.4194;
the first expansion's teacher macro-F1 is 0.4086. The new score does not exist until training runs.

The trainer also saves separate scores for teacher images, the earlier additions, the new
additions, all external images and the combined set. External scores have a different source and
class mix. They explain behavior; they do not replace the teacher-only comparison.

Each fold saves `final_epoch.pt`, `config.json`, `normalization.json`, `history.csv`,
`oof_predictions.csv`, `training_predictions.csv`, `source_metrics.json`, `metrics.json`
and `robustness.csv`. The aggregate directory adds `teacher_comparison.json`,
`teacher_per_class.csv`, pooled predictions, class scores, a confusion matrix and failure rows.
The notebook saves the learning-curve figure under its output folder's
`results/figures/task3/usage_expanded_v2_e8_learning_curves.png`.

This notebook uses development data only. Earlier holdout and teacher-test evaluations already
exist; a future evaluation must not be described as an untouched first test.

## Code and checks

- Trainer: `src/fashion/train/task3_usage_expanded_v2.py`.
- Shared optimizer loop: `src/fashion/train/task3_baseline.py`.
- Tests: `tests/train/test_task3_usage_expanded_v2.py`.
- Package builder: `reports/task3/usage_expanded_v2_e8_20260906/build_bundle.py`.
- Data and reference checks: `preflight.json`.
- Archive hashes and size: `bundle_receipt.json`.

The focused tests cover a real synthetic CPU training step, old/new data routing, fold-only
statistics and weights, protected-row exclusion, registry mirroring, completed-fold reuse,
corrupt-checkpoint rejection and comparison on matching teacher IDs. Production GPU training is
started by the notebook; preparing this package does not launch it.

Rebuild the package with:

```bash
./.venv/bin/python reports/task3/usage_expanded_v2_e8_20260906/build_bundle.py
```

For a local GPU run using the saved local references:

```bash
./.venv/bin/python -m fashion.train.task3_usage_expanded_v2 \
  --output-root results/task3_usage_expanded_v2_e8 \
  --e8-directory results/evidence/task3/experiments/t3_usage_e8_translation/usage \
  --source-registry results/runs.csv \
  --previous-directory results/evidence/task3/usage_expanded_e8_20260906 \
  --previous-registry results/evidence/task3/usage_expanded_e8_20260906/results/runs.csv \
  --local-registry results/task3_usage_expanded_v2_e8_local/results/runs.csv
```

The reference registry is read-only. The local mirror belongs only to this new experiment;
the trainer refuses to overwrite a log containing other experiments.
