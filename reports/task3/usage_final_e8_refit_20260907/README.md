# Final Usage model: E8 refit

The accepted model is the single E8 + translation scratch refit, epoch 30,
trained on all 32,772 eligible teacher development images.

- [Frozen model manifest](model_manifest.json): checkpoint, configuration,
  normalization, class order, registered training evidence and source hashes.
- [Decision 0024](../../../docs/decisions/0024-task3-usage-e8-refit-final-model.md).
- [Notebook 4](../../../notebooks/04_task3_gender_usage.ipynb): development
  analysis, refit training and the final recipe.
- [Task 3 evaluation](../../../notebooks/04_task3_final_evaluation.ipynb): reserved-holdout
  comparisons, class errors, practical limits and final judgement.

Run ID: `t3_usage_e8_translation_teacher_all_development_refit_5553be0c138243e9`.

Checkpoint SHA-256:
`18da75a4ec935ab0d18c9ebdf9d1dabb6fa87ee484ead5f439de0192c4af2747`.

Verify from the repository root:

```bash
./.venv/bin/python -m fashion.task3_final
```

Inference uses this one model in evaluation mode, its saved full-development
normalization, then softmax and argmax. No translation, class-weight adjustment
or fold averaging is applied during prediction. The checkpoint and original
training-only manifest remain unchanged; this pack records later acceptance.

The teacher test set is prediction-only. The full submission must preserve
`id,gender,articleType,season,usage`. This pack contains no test scores.

The two refit rows in `refit_runs.csv` come from the completed training registries
downloaded with the run artifacts. They are snapshots of registered fits, not
new runs. The working registry is not replaced. Notebook 4 rebuilds its training
table from these rows and checks the saved summary.

Both notebooks passed Run All (40 and eight code cells). The update preserves
all existing Gender cells and outputs. The new tables, loss curves, class chart
and final recipe were inspected in their rendered HTML. All 65 relevant tests,
Python lint, formatting and diff checks passed.

Rebuild the local previews with:

```bash
./.venv/bin/python reports/task3/usage_final_e8_refit_20260907/render.py
```

This executes both notebooks to check them, saves only the new Usage outputs,
and exports `main_report.html` and `final_evaluation.html`. It does not train
models, run new inference, open raw holdout labels or score the teacher test set.
