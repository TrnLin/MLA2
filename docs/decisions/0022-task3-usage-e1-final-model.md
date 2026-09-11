# 0022 — Task 3 Usage E1 development reference

- Scope: baseline and training source contract
- Recipe date: 2026-09-07
- Documentation updated: 2026-09-11

E1 is the teacher-only scratch SmallCNN reference in the Usage investigation.
It uses channels 32/64/128/256, average pooling, unweighted cross-entropy,
AdamW learning rate 0.001, weight decay 0.0001, batch 128 and seed 2753.
Each fold runs 30 epochs with cosine decay to 0.00001 and keeps its final epoch.

On 32,772 original development rows, E1 gives 89.30% accuracy and 37.38%
nine-class macro-F1. E8 gives 88.85% and 41.94%. This illustrates the difference
between total image accuracy and equal-class performance. All other Usage
models and data-expansion trials remain in the development comparison.

The E1 manifest remains a byte-verified input to its refit training runner.
It is not the current model-selection record. The selected Usage model is the
single teacher-only E8 refit described in [0024](0024-task3-usage-e8-refit-final-model.md).

- [Frozen E1 training source](../../reports/task3/usage_final_e1_20260907/model_manifest.json)
- [Development analysis](../../notebooks/06_task3_part1_gender_usage.ipynb)
- [Selected E8 evaluation](../../notebooks/07_task3_part2_final_evaluation.ipynb)
