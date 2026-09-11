# Gender SAM25 training reference

This folder retains the SAM25 refit's training identity and registered run.
It supports the development inventory and comparison of completed training.
The underlying scratch checkpoint, configuration and training records are in
`results/evidence/task3/gender_sam25_refit_20260907/`.

The selected Gender model is the **MixUp 0.20 refit at epoch 30**. Its selection
is in [Notebook 06](../../../notebooks/06_task3_part1_gender_usage.ipynb), and its
assessment is in [Notebook 07](../../../notebooks/07_task3_part2_final_evaluation.ipynb).
Use `fashion.task3_gender_final.verify_gender_final` to verify that selected
MixUp checkpoint.
