# Task 3 development evidence

The [main notebook](../../../notebooks/06_task3_part1_gender_usage.ipynb) has
20 numbered sections. It covers shared EDA, every recorded Gender and Usage
candidate, the development comparisons, recipe selection and full-development
refits. Reusable checks live in `src/fashion/`.

The selected recipes are Gender MixUp 0.20 with 30 epochs and teacher-only
Usage E8 with 30 epochs. The [final notebook](../../../notebooks/07_task3_part2_final_evaluation.ipynb)
assesses those two selected refits separately.

## Retained files

- `results/runs.csv` is the saved registry snapshot used by the stage tables.
- `results/classical_runs.csv` supplies the classical comparison records.
- `evidence_lock.json` pins development sources and each stage's run IDs.
- `analysis_assets.json` pins development audits, histories and image variants.
- `render.py` replays the current main notebook and creates `main_report.html`.

Run the preview from the project root:

```bash
./.venv/bin/python reports/task3/main_report_20260906/render.py
```

The notebook sums registered confusion counts before calculating macro-F1.
Original and corrected labels, two-fold screens and five-fold comparisons
remain separate. Refit loss curves describe completed training; they do not
provide a validation score.
