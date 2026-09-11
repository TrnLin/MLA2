# 0025 — Task 3 Gender SAM25 refit training reference

- Scope: completed comparison artifact
- Training date: 2026-09-07
- Documentation updated: 2026-09-11

The SAM25 refit is a scratch training artifact from the Gender investigation.
It has 390,181 parameters and uses all 32,773 development images with the fixed
name-corrected labels. It trains for 25 epochs with cosine `T_max=30`, MixUp
alpha 0.20, SAM rho 0.05 and dropout 0.30. There is no validation selection or
early stopping in this fit.

Its original training receipt, checkpoint, normalization and history remain
in `results/evidence/task3/gender_sam25_refit_20260907/`. Its completed training
is included in the development inventory. The five-fold comparison scores
belong to separate fold models and cannot be assigned to this refit.

The selected Gender model is the **MixUp 0.20 refit at epoch 30**. The full
candidate comparison and choice are in
[Notebook 06](../../notebooks/06_task3_part1_gender_usage.ipynb). Only that
selected Gender refit is assessed in
[Notebook 07](../../notebooks/07_task3_part2_final_evaluation.ipynb).

- [SAM25 training contract](../task3_gender_sam25_refit.md)
- [Completed training notebook](../../notebooks/task3_training/gender_sam25_refit.ipynb)
- [Original training receipt](../../results/evidence/task3/gender_sam25_refit_20260907/model_manifest.json)
