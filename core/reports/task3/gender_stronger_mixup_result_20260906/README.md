# Stronger MixUp result

The alpha 0.4 run completed on folds 0 and 4. It passed 17 of 19 required
checks. It slightly reduced the average clean train–validation gap, but missed
the planned reduction and did not improve both folds. Keep alpha 0.2 as the
current candidate.

- Pooled validation macro-F1: 80.89% → 79.87%. The 74% floor passed.
- Mean clean training macro-F1: 93.81% → 92.06%.
- Mean clean train–validation gap: 12.91 → 12.20 percentage points.
- Gap reduction: 0.72 points; the required reduction was 2.00 points.
- Fold 0 gap: 12.44 → 12.49 points, a small worsening.
- Fold 4 gap: 13.38 → 11.90 points, an improvement of 1.48 points.
- Unisex recall: unchanged at 51.06% (361 of 707).
- Girls F1 fell 3.03 points; Boys F1 fell 1.88 points. Unisex F1 rose 0.30 points.
- Total validation errors increased by 41 across 13,110 images.

The two failures were `vs_mixup20.fold_0.gap_reduction` and
`vs_mixup20.mean_gap_reduction`. The validation floor did not cause the failure.
The remaining memory, model-size, confidence-quality and corruption guards passed.

This shows a small average reduction in the measured gap, not elimination of
overfitting. Training F1 fell 1.76 points while mean fold validation F1 fell
1.04 points. The pooled F1 decline was 1.02 points. Its saved paired-family
bootstrap interval was −2.05 to −0.03 points; repeated selection on these folds
is not corrected by that interval. No held-out test was used.

## Evidence checked

- Executed notebook: `notebooks/task3_training/gender_stronger_mixup_screen.ipynb`.
  It records commit `1233e4c99b5a318f9602073e0fb4de4d3e747dee`, two completed
  30-epoch runs, alpha 0.4 and the correct 74% floor.
- [Saved screen decision](https://drive.google.com/file/d/1hf3fkzfhLNmqo0Nh-D2Mw-31ZynpQWb2/view).
- [Saved direct-parent comparison](https://drive.google.com/file/d/10_Y-YnzgvJtSeLsupGfqMHblr_DZWFfh/view).
- Local copies: `decision.json`, `incremental.json`, `verified_summary.json`.

The two saved comparisons agree. Macro-F1 was independently recalculated from
both confusion matrices; fold gap arithmetic and all refinement decisions were
rechecked against the saved values. This was a result review, not a new training
run, checkpoint replay or full per-image prediction audit.
