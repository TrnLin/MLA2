# Selected Usage E8 refit — final evaluation

This folder contains only the single teacher-only Usage E8 refit's evaluation
used by [Notebook 07](../../../notebooks/07_task3_part2_final_evaluation.ipynb).
The checkpoint is epoch 30 of run
`t3_usage_e8_translation_teacher_all_development_refit_5553be0c138243e9`.

On 5,778 reserved images, it makes **5,125 correct predictions and 653 errors**:
**88.70% accuracy and 42.26% nine-class macro-F1**. Home has no reference examples
and contributes zero to the fixed nine-class macro-F1 definition.
The model finds five of 22 NA, Party, Smart Casual and Travel examples combined.
These results support reviewed suggestions, with weak rare-class reliability.

`refit_holdout_probabilities.csv` retains the original prediction bytes.
`holdout_predictions_and_labels.csv` contains the matching IDs, families,
original Usage labels and refit predictions. `holdout_per_class.csv` and
`evaluation.json` contain only this refit's results. `prediction_freeze.json`
retains the prediction time, checkpoint hash, prediction hash and runtime.
Its metadata scope update has its own date. `evaluation_provenance.json` pins
the retained files. No new inference was performed for the scope update.

The model trained from scratch on 32,772 eligible development images for
30 epochs. Its 391,209-parameter SmallCNN uses average pooling, effective-number
class weights and two-pixel training translations. Training took 528.77 seconds;
the saved CPU prediction pass took 11.45 seconds including image processing.
These timings are individual recorded runs, not serving benchmarks.

Run Notebook 07 to check file hashes, recompute metrics and rebuild its figures.
For a text-only replay from the project root:

```bash
PYTHONPATH=src ./.venv/bin/python reports/task3/usage_e8_refit_holdout_20260907/evaluate.py
```

This command reads the saved predictions. It does not train, predict again or
change a checkpoint.
