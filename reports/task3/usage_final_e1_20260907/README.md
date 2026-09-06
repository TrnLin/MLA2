# Final Usage choice: E1

The final Usage artifact is the original teacher-only scratch SmallCNN,
using the equal probability average of its five saved fold checkpoints.
The owner accepted E1 for accuracy after reviewing the completed evaluations.
It is not the macro-F1 winner or an all-development single-model refit.

E1 gets **5,123/5,829 official test images right (87.89%)**. Nine-class
macro-F1 is **0.271542**. It misses every test NA, Party, Smart Casual and
Travel case. Home has no test examples. On the reserved holdout, it gets
5,192/5,778 right (89.86%), with macro-F1 0.363860.

## Files

- [Model manifest](model_manifest.json): the five checkpoint paths and hashes,
  full configurations, fold normalizations, classes and exact inference rule.
- [Decision 0022](../../../docs/decisions/0022-task3-usage-e1-final-model.md):
  the accepted trade-off and the Usage exception to the refit rule.
- [Evidence lock](evidence_lock.json): hashes for the saved final comparisons.
- [Main notebook](../../../notebooks/04_task3_gender_usage.ipynb): the complete
  report, including the preserved Gender analysis and finished Usage sections.
- [HTML preview](../main_report_20260906/main_report.html): the executed
  notebook with working local links. This is a generated review artifact.
- [Verification](verification.json): source hashes and completed checks.
- [Evaluated source mapping](source-archive-map.json): exact archived inference
  sources and separately pinned current scripts.

Sections 39–45 finish the Usage path: original experiments, 120/687-image
expansions, the matched two-fold MixUp/SAM and replacement trials, overfitting,
robustness, the E1 freeze, evaluation and practical limits. Five-fold scores
use 32,772 teacher development rows. The two-fold screens use 13,110 rows.
Mixed teacher/outside scores stay separate from teacher-only results.

## Reproduce the report

With the existing data and evidence present, run from the project root:

```bash
./.venv/bin/python reports/task3/main_report_20260906/render.py
```

This runs 127 saved-evidence analysis cells and exports HTML. It checks
development image hashes and decodes those images. It does not train, run
new holdout/test inference, look up evaluation labels or write a submission.
The main evidence pack and existing model files are required; no new ZIP
or checkpoint is created by this report work.

The original E1 inference recipe remains in
`reports/task3/usage_teacher_vs_expanded_test_20260906/inference_recipe.json`.
Existing Usage-only predictions remain in that directory's
`E1/usage_test_predictions.csv`. Do not replace them with predictions from
a new fit. The four-target submission still requires integration.

## Verification and review boundary

- All 127 code cells executed without errors; notebook schema and Python
  syntax checks passed. Ruff passed for the notebook and changed test file.
- All five E1 checkpoint hashes match the registry. Their normalizations
  match both saved evaluation recipes. Frozen inference source hashes match
  the evaluated source archive through the mapping below. The adapted current
  scripts have different hashes and are checked separately.
- All 32,772 saved E1 OOF rows match canonical IDs, folds, families and class
  indices. Their decisions reproduce the registered accuracy and macro-F1.
- The saved five-fold test probability average reproduces every E1 decision.
  CSV rounding changes probabilities by at most 2.04e-8; no inference was run.
- Per-class counts reproduce all four test summaries. Saved confusion counts
  reproduce E1 holdout and both expanded teacher-development results.
- 32 notebook, image-transform and baseline-contract tests passed. Notebook
  assertions now require the final E1 choice instead of an open Usage TODO.
- The four new figures and the rendered comparison, recipe and evaluation
  tables were visually inspected. Existing Gender analysis source is preserved,
  apart from wording that separates its holdout scope from Usage test evidence.

The main notebook and its tests were based on the current source checkout,
including its uncommitted Gender report changes. The source notebook hash is
recorded in `verification.json`. The source checkout, training notebooks,
training code, registry, data and checkpoints were not edited.

## Verify the frozen artifact after path moves

From the project root, with the private model/data assets from
[asset setup](../ASSETS.md) present, run:

```bash
./.venv/bin/python scripts/verify_task3_usage_manifest.py
```

This checks all 38 frozen manifest references: the recipe, source contracts,
five checkpoints, configurations, normalizations, metrics, OOF records and
saved predictions. It only reads and hashes files; it does not import or run
the archived scripts, train, or perform inference.

Two evaluated inference scripts changed when their live paths were moved.
Their exact evaluated bytes are tracked in `evaluated_sources/`.
`source-archive-map.json` maps each original manifest reference to its archive,
and pins each adapted current script separately. Other historical report
references use the `reports/task3_` → `reports/task3/` prefix move; other paths
stay relative to the checkout root. The frozen manifest itself is also checked.
Expected output is 38 frozen references and two separately verified current
scripts. No stale filename aliases or original source checkout are needed.
The current scripts are retained for their adapted paths, not claimed to be
byte-identical to the evaluated source or newly evaluated models.

The combined cleanup includes the saved source evidence packs, moved report
links, and decision 0022 in the decision index. See [asset setup](../ASSETS.md)
for local checkpoints and data. The five-page submission report is a separate
deliverable; this notebook retains the full analysis record.
