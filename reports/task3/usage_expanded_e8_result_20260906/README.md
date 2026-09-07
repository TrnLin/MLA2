# Usage E8 with added images: result review

## Assignment goal and decision

The goal is good predictions on the teacher's unseen test set. The external
images supplement training; they do not change the target test population or
the model-selection criterion. Use the original teacher validation folds as
the available check of whether this change is likely to help that goal.

The expanded run lowers teacher-only macro-F1 from **0.4194 to 0.4086**, with
lower scores in four of five folds. Do not replace E8 with this run on the
present evidence. This is a validation result, not proof of the eventual test
score: the holdout and test labels remain unseen.

The higher combined-data score shows better recognition of added-source images.
It does not demonstrate improved performance on the teacher's test images and
must not be used to promote this model. The earlier recommendation based on a
combined-dataset prediction goal was an incorrect change of project scope.

## Teacher validation results

All five folds finished 30 epochs and saved their files in
`MyDrive/MLA2/task3_usage_expanded_e8/experiments/t3_usage_expanded_e8/usage/`.
The separate log worked for this run. The added images did not improve the score
on the original teacher validation images.

| Comparison on the same 32,772 teacher images | Macro-F1 |
|---|---:|
| Original E8 | 0.4194 |
| E8 recipe trained with 120 added images | 0.4086 |
| Change | -0.0108 |

The new model scored lower in four of five folds on teacher images only. The
pooled combined score is 0.5548 and must be compared with E8's combined score
of 0.4110, rather than its teacher-only score of 0.4194. Macro-F1 gives equal
weight to all nine classes, so a few new Home or Party examples can change it
substantially even when the added set is small.

| Teacher class | Images | Original E8 F1 | Added-images F1 |
|---|---:|---:|---:|
| Home | 1 | 0.000 | 0.000 |
| NA | 61 | 0.169 | 0.222 |
| Party | 12 | 0.077 | 0.000 |
| Smart Casual | 47 | 0.103 | 0.038 |
| Travel | 22 | 0.204 | 0.204 |

For teacher Party, none of the 12 validation images was classified correctly:
eight were called Casual and four Ethnic. For Smart Casual, only 2 of 47 were
correct. NA improved, with 15 of 61 correct.

The model did better on added-source Home (12/16 correct) and Party (52/97)
than on the matching teacher classes. Added Smart Casual (0/6) and Travel (0/1)
still failed. This is consistent with limited transfer between the source
datasets, but it does not prove why transfer failed. Home and Travel counts are
too small for strong conclusions. No NA examples were added, so the NA gain
cannot be attributed to adding that class directly.

Overfitting remains: mean clean teacher training F1 is 0.7766, while mean teacher
validation F1 is 0.4059. These are averages over folds, not the pooled scores in
the table above. Darkening also remains a weakness: a 15% brightness reduction
drops mean combined validation F1 by 0.3346, to 0.2129. Those corruption results
include added images and are not a comparison against E8's teacher-only score.

Retain the teacher-only result as the primary model-selection evidence. One
seed and tiny rare-class samples do not establish statistical significance.
This review does not change the chosen deployment model, retrain anything,
or inspect the holdout.

## Added-source diagnostic: not a model-selection result

Both models were also compared on the same mixed set of 32,892 validation
images. This is a secondary diagnostic with a changed source and class mix.

| Same validation images for both models | Original E8 macro-F1 | Added-images macro-F1 |
|---|---:|---:|
| Teacher plus added images: 32,892 rows | 0.4110 | 0.5548 |
| Teacher images only: 32,772 rows | 0.4194 | 0.4086 |

On the combined set, the new model improves macro-F1 by 0.1437 and wins all
five folds. On the 120 added validation images, E8 gets 0 correct and the new
model gets 64 correct. The new model therefore learned useful patterns for
those added sources, but the teacher-only comparison does not show a gain
toward the assignment's test-prediction goal.

Each frozen E8 fold checkpoint predicted only that fold's added validation
images. Its existing verified teacher predictions were reused. There was no
training, tuning, normalization refit, or holdout access. CPU reproduction
checks covered all 120 new-model added-image predictions and 81 E8 teacher
control images. Class decisions matched the saved GPU predictions in all
201 checks. Maximum probability difference was 0.001495; those numerical
differences are recorded in `combined_comparison.json`.

## Checks performed

- Downloaded the complete saved run with `rclone`, using read-only source access.
- Five unique completed registry rows; all five have 30 history rows and select epoch 30.
- All 40 recorded file hashes match: five checkpoints, five validation prediction
  files, and six diagnostic files per fold, including clean training predictions.
- Verified the frozen scratch-training recipe and the combined dataset hashes.
- Rechecked validation IDs, class labels, folds, product families, and probabilities
  against the saved split. No holdout or prediction rows enter this evaluation.
- Recomputed each fold and pooled score from the downloaded probabilities. The
  aggregate predictions exactly match the five fold prediction files.
- Verified the original E8 evidence and compared exactly the same teacher image IDs.

The earlier connector snapshots are retained alongside the review. The complete
raw files downloaded with `rclone` are under
`results/evidence/task3/usage_expanded_e8_20260906/`. The later hash and probability
checks supersede the metadata-only checks described in `drive_sources.json`.

## Reproduce

From the project root:

```bash
rclone copy gdrive:MLA2/task3_usage_expanded_e8/experiments/t3_usage_expanded_e8/usage results/evidence/task3/usage_expanded_e8_20260906 --exclude runs.csv.lock
./.venv/bin/python reports/task3/usage_expanded_e8_result_20260906/check_result.py
PYTHONPATH=/tmp/mla2-mixup-torch:src ./.venv/bin/python reports/task3/usage_expanded_e8_result_20260906/compare_combined.py
./.venv/bin/python reports/task3/usage_expanded_e8_result_20260906/plot_result.py
```

The report figure is `results/figures/task3/usage_expanded_e8_comparison_20260906.png`.
`combined_comparison.json`, `combined_fold_comparison.csv`, and
`e8_added_validation_predictions.csv` contain the secondary combined diagnostic.
`verified_summary.json`, `checks.json`, `fold_comparison.csv`, and
`robustness_summary.csv` contain the teacher-only review and file checks.

Sources: [saved run log](https://drive.google.com/file/d/1Cta0hwyh1WN5RrCKu0u-ar8gqroejsXe/view),
[teacher comparison](https://drive.google.com/file/d/1XjIl5KVUHBzMABQ5uIIkQ04S6nkGRugv/view),
and [class scores](https://drive.google.com/file/d/1kE0aZNatNWMe49qLKrPKOzU_pDfczBrY/view).
