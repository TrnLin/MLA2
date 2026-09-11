# 0020 — Task 3 Gender SAM25 development comparison

- Scope: comparison recipe
- Recipe date: 2026-09-06
- Documentation updated: 2026-09-11

SAM25 is a development comparison point for the selected MixUp recipe.
It uses a scratch 390,181-parameter GeM SmallCNN, dropout 0.30, MixUp alpha
0.20, SAM rho 0.05 and the recorded translation, darkening and grayscale
augmentation. Each fit stops at epoch 25 with cosine `T_max=30`.

The five fold models predict their excluded development rows. Each uses its
own training-fold normalization and the fixed corrected-label contract.
Never average the five models when computing development OOF scores.

The matched five-fold audit gives 79.73% corrected-label macro-F1 and a
10.15-point mean clean fit gap. MixUp 0.20 at epoch 30 has higher observed
class F1 and better probability quality, with a larger fit gap. Uncertainty
and corruption trade-offs remain part of the comparison.

All candidate comparisons and the final recipe decision are in
[Notebook 06, Sections 14–15](../../notebooks/06_task3_part1_gender_usage.ipynb).
The selected Gender model is the single MixUp refit at epoch 30.

- [Five-fold training contract](../task3_gender_sam25_five_fold.md)
- [Development review](../../reports/task3/gender_sam25_cv_result_20260906/README.md)
- [Fold model identities](../../reports/task3/gender_sam25_cv_result_20260906/model_manifest.json)
