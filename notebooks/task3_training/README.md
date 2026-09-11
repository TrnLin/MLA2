# Task 3 training and experiment notebooks

Read [the main report](../06_task3_part1_gender_usage.ipynb) for the results.
Run a notebook here only when you mean to repeat that experiment.
All 40 original companions are kept, including the EDA and saved-model diagnostics.

## Run them

- Colab runners: select the stated GPU, keep the existing ZIPs under
  `MyDrive/MLA2/data/`, and check `BRANCH` in the setup cell. Current code
  comes from GitHub; the saved data bundles do not need a new upload.
- Local runners: use the repository `.venv` kernel. Root discovery works
  from this folder or the repository root. See [the asset setup](../../reports/task3/ASSETS.md).
- Reuse each notebook’s named parent runs and its own output folder.
  Keep the saved folds, class map and experiment ID unchanged.
- Outputs shown here are the original run record. They are not a fresh
  execution of the current Git revision. Reading them needs no GPU.

The accepted final Gender model is the [single SAM25 refit](gender_sam25_refit.ipynb),
recorded in [Decision 0025](../../docs/decisions/0025-task3-gender-sam25-refit-final-model.md).
[SAM25 on five folds](gender_sam25_five_fold.ipynb) remains the development evidence.
The new [MixUp 0.20 five-fold runner](gender_mixup_five_fold.ipynb) trains all five
folds in Colab for a matched development comparison. See the
[run guide](../../docs/task3_gender_mixup_five_fold.md).
The original E1 Usage runner is [SmallCNN baselines](smallcnn_baseline_training.ipynb).

## Find an experiment

[moves.csv](moves.csv) records every full original filename and its retained successor.

| Former prefix | Retained notebook | Runtime |
|---|---|---|
| `04a` | [smallcnn_baseline_training.ipynb](smallcnn_baseline_training.ipynb) | Colab |
| `04b` | [smallcnn_child_experiments.ipynb](smallcnn_child_experiments.ipynb) | Colab |
| `04c` | [smallcnn_e3_experiments.ipynb](smallcnn_e3_experiments.ipynb) | Colab |
| `04d` | [tinyresnet18_pm_e4_experiments.ipynb](tinyresnet18_pm_e4_experiments.ipynb) | Colab |
| `04e` | [compactblurcnn_label_smoothing_e5_experiments.ipynb](compactblurcnn_label_smoothing_e5_experiments.ipynb) | Colab |
| `04f` | [gem_focal_e6_experiments.ipynb](gem_focal_e6_experiments.ipynb) | Colab |
| `04g` | [tinyconvnext_tinyhrnet_e7_experiments.ipynb](tinyconvnext_tinyhrnet_e7_experiments.ipynb) | Colab |
| `04h` | [early_stopping_translation_e8_experiments.ipynb](early_stopping_translation_e8_experiments.ipynb) | Colab |
| `04i` | [semantic_filter_exception_balance_e9_experiments.ipynb](semantic_filter_exception_balance_e9_experiments.ipynb) | Colab |
| `04j` | [audience_aux_e10_experiment.ipynb](audience_aux_e10_experiment.ipynb) | Colab |
| `04k` | [clean_slate_eda.ipynb](clean_slate_eda.ipynb) | Local |
| `04l` | [clean_slate_screen_1.ipynb](clean_slate_screen_1.ipynb) | Local |
| `04m` | [micro_swin_clean_slate_screen_2.ipynb](micro_swin_clean_slate_screen_2.ipynb) | Colab |
| `04n` | [gem_gender_v2_g1_foreground_mask.ipynb](gem_gender_v2_g1_foreground_mask.ipynb) | Colab |
| `04o` | [gem_gender_v2_g2_translation.ipynb](gem_gender_v2_g2_translation.ipynb) | Colab |
| `04p` | [gem_gender_v2_g3_component_weight.ipynb](gem_gender_v2_g3_component_weight.ipynb) | Colab |
| `04q` | [smallcnn_usage_v2_u1_component_weight.ipynb](smallcnn_usage_v2_u1_component_weight.ipynb) | Colab |
| `04r` | [gem_gender_v2_g2_confirmation.ipynb](gem_gender_v2_g2_confirmation.ipynb) | Colab |
| `04s` | [usage_v2_u2_full_rgb_hog_svm.ipynb](usage_v2_u2_full_rgb_hog_svm.ipynb) | Local |
| `04t` | [gender_gd1_mild_darkening.ipynb](gender_gd1_mild_darkening.ipynb) | Colab |
| `04u` | [gender_weight_decay_screen.ipynb](gender_weight_decay_screen.ipynb) | Colab |
| `04v` | [gender_saved_model_diagnostic.ipynb](gender_saved_model_diagnostic.ipynb) | Colab |
| `04w` | [gender_precision_check.ipynb](gender_precision_check.ipynb) | Colab |
| `04x` | [gender_narrow64_screen.ipynb](gender_narrow64_screen.ipynb) | Colab |
| `04y` | [gender_dropout_screen.ipynb](gender_dropout_screen.ipynb) | Colab |
| `04z` | [usage_two_stage_screen.ipynb](usage_two_stage_screen.ipynb) | Colab |
| `04aa` | [gender_dropout_darkening_screen.ipynb](gender_dropout_darkening_screen.ipynb) | Colab |
| `04ab` | [gender_stronger_dropout_screen.ipynb](gender_stronger_dropout_screen.ipynb) | Colab |
| `04ac` | [gender_grayscale_screen.ipynb](gender_grayscale_screen.ipynb) | Colab |
| `04ad` | [gender_name_truth_screen.ipynb](gender_name_truth_screen.ipynb) | Colab |
| `04ae` | [gender_group_weight_screen.ipynb](gender_group_weight_screen.ipynb) | Colab |
| `04af` | [gender_mixup_screen.ipynb](gender_mixup_screen.ipynb) | Colab |
| `04ag` | [usage_expanded_e8.ipynb](usage_expanded_e8.ipynb) | Colab |
| `04ah` | [gender_stronger_mixup_screen.ipynb](gender_stronger_mixup_screen.ipynb) | Colab |
| `04ai` | [gender_sam_screen.ipynb](gender_sam_screen.ipynb) | Colab |
| `04aj` | [gender_sam25_screen.ipynb](gender_sam25_screen.ipynb) | Colab |
| `04ak` | [gender_sam25_five_fold.ipynb](gender_sam25_five_fold.ipynb) | Colab |
| `04al` | [usage_expanded_v2_e8.ipynb](usage_expanded_v2_e8.ipynb) | Colab |
| `04am` | [usage_mixup_sam_screen.ipynb](usage_mixup_sam_screen.ipynb) | Colab |
| `04an` | [usage_replaced_v3_mixup_sam.ipynb](usage_replaced_v3_mixup_sam.ipynb) | Colab |
