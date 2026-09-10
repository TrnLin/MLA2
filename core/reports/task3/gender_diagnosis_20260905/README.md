# Saved gender model diagnosis

The saved-evidence checks are complete. GPU inference checks are still pending.

Four source bundles (G2 and CompactBlurCNN, folds 0 and 4) passed registry, artifact hash, canonical validation ID and saved-metric checks. No training or checkpoint inference was performed here.

| Model | Fold | Clean training F1 | Validation F1 | Gap |
|---|---:|---:|---:|---:|
| G2 | 0 | 0.993147 | 0.760498 | 0.232649 |
| G2 | 4 | 0.994583 | 0.730461 | 0.264122 |
| CompactBlurCNN | 0 | 0.872132 | 0.717581 | 0.154551 |
| CompactBlurCNN | 4 | 0.885618 | 0.708529 | 0.177088 |

CompactBlurCNN reduces the gap mainly by reducing training fit. Its validation F1 also falls. This does not show that reducing capacity will improve generalization. The two recipes also differ in augmentation and architecture, so this is not a controlled width-only comparison.

G2 has higher pooled validation accuracy in every gender category and every family-size bin. See `validation_slice_review.png` (visually inspected), `validation_slices.csv` and `paired_error_switches.csv`. Accuracy is the fraction correct; it is different from macro-F1.

Reweighting validation rows to the training class mix moves G2 F1 to 0.760872 / 0.730240. Matching the class/article mix gives 0.767080 / 0.732816, covering 98.95% / 99.19% of training mass. Matching class/article/family-size mix gives 0.770841 / 0.729293, covering 96.31% / 97.02%. These changes are small beside the roughly 0.25 gap. Missing strata, small groups and changed class mixtures limit this descriptive check. It cannot establish why the gap exists.

`analyse_saved.py` regenerates the saved-data tables. It reads the local downloaded evidence paths recorded in that script. `diagnostic_status.json` records the limits.

The saved bundles lack per-image clean training predictions. The remaining work needs the saved checkpoints to run in evaluation mode on a Colab GPU. `notebooks/task3_training/gender_saved_model_diagnostic.ipynb` and `src/fashion/train/task3_gender_diagnostic.py` are prepared locally for this. They save clean training/validation predictions, per-class scores, slice accuracy gaps, and batch/order stability results to a new timestamped Drive folder. They do not train or write the run registry. Publish these two files to the selected branch before running the notebook.

Local validation: notebook schema and all code cells compile; Ruff passes; 16 tests pass (15 notebook checks and one image-verification test covering changed and missing inputs). The GPU inference path has not run locally because PyTorch/GPU are unavailable here and local PyTorch installation was not requested.

The other task reviewed the partial results and recommends finishing GPU inference before choosing another model. It independently found that CompactBlurCNN fixes 301 G2 errors but introduces 507 new errors; its mean validation F1 is 0.0328 lower. It correctly noted that remaining differences cannot all be attributed to memorization. It requested image-content verification against `splits.csv`; this safeguard has now been added before any checkpoint inference. Stability checks cover fixed samples, not every image. No new candidate or acceptance threshold has been approved by these checks.
