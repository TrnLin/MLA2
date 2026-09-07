# Accepted final Gender model

The owner accepted the **single full-development SAM25 refit, epoch 25**, on
7 September 2026 after reviewing the saved reserved-holdout comparison.
[Decision 0025](../../../docs/decisions/0025-task3-gender-sam25-refit-final-model.md)
supersedes the earlier five-model artifact. Original training and evaluation
receipts remain unchanged.

- `model_manifest.json` and its SHA-256 sidecar pin the final checkpoint, recipe,
  normalization, class order, source files and training evidence.
- `refit_runs.csv` preserves the completed training registry row. Its source is
  the archived Drive registry named in `registry_provenance`; the live registry
  was not rewritten by this acceptance.
- `holdout_sources.json` and its SHA-256 sidecar pin the separate saved holdout
  evidence used in `notebooks/04_task3_final_evaluation.ipynb`. Model verification in Notebook 4 does not load it.

Use [Notebook 4](../../../notebooks/04_task3_gender_usage.ipynb) for development,
training and inference details, and [Task 3 evaluation](../../../notebooks/04_task3_final_evaluation.ipynb)
for the holdout comparison and limits. The official teacher test set is prediction-only.

From the repository root, verify the accepted model without training or inference:

```bash
./.venv/bin/python -m fashion.task3_gender_final
```
