# Task 1 Results Explanation and Confusion Detail Design

**Date:** 2026-09-07
**Scope:** Task 1 `articleType` analysis, Notebook 02, evidence figures, and Task 1 HTML
**Change type:** Analysis and presentation only

## Goal

Make Notebook 02 follow the original experiment story, explain completed metrics directly after
each result block, and keep the full 124-class confusion matrix while adding smaller figures that
make its main errors readable.

The work reuses completed five-fold and OOF evidence. It does not change a model, metric, split,
checkpoint, prediction, or registry row.

## Approved decisions

- Preserve the existing notebook output blocks.
- Restore this order: scratch CNN, HOG, augmentation, class weighting, combined comparison,
  learning curves, failure analysis, and development decision.
- Keep Section 4 as the one place that defines the metrics.
- Put short result interpretation directly after each experiment output.
- Keep the full 124-class confusion matrix in the notebook.
- Add a Top-10 confusion chart, focused confusion matrix, and representative error images.
- Use compact figures in the main report and keep the full matrix as supporting evidence when
  page space is limited.

## Fixed constraints

- `data/processed/splits.csv` remains the only split.
- Only development OOF predictions feed the new analysis.
- Holdout and quarantine labels remain sealed.
- Existing checkpoints, histories, predictions, metrics, figures, and registry rows do not change.
- The fixed 124-class label order and metric formulas do not change.
- Reusable logic stays in `src/fashion/task1/`.
- No full, smoke, or final model run is part of this work.

## Current problems

The evidence is sound, but the current reading order differs from the intended experiment order.
HOG appears before the scratch CNN. Plain and augmented CNN results appear together. The learning
curve also discusses the weighted CNN before the weighting experiment is explained.

Section 4 defines the metrics, but later output blocks do not always state what one row represents,
which difference matters, what it means for this dataset, and whether the experiment idea passed.

The full confusion matrix proves all 124 classes were included. It is too dense to explain the
important mistakes at notebook or report width. The current text names confusion pairs without a
readable summary or source-image examples.

## Notebook narrative

### 1. Scratch CNN

Introduce the plain, unweighted, no-augmentation CNN as the neural control. Add a small read-only
display cell that reads the three saved CNN evidence tables and filters them to
`task1_cnn_no_aug_unweighted_v1`.

This cell does not train anything. It lets the scratch result appear before HOG while the old
combined CNN output remains untouched for the later augmentation section.

The observation explains fold rows, the five-fold summary, the pooled OOF row, and the role of the
plain CNN as the control.

### 2. HOG experiments

Place the existing HOG introduction, controller cell with its saved output, and observation after
the scratch CNN. Keep the old output attached to its cell.

Explain that KNN has stronger macro-F1 and Top-1, while SVM has stronger Top-5. KNN is therefore
the stronger single-label classical baseline. SVM is more useful for a short candidate list.

### 3. Data augmentation

Place the existing combined plain-versus-augmented controller cell and saved output here. Keep the
output unchanged. Introduce augmentation as one controlled change from the scratch CNN.

The observation records:

- plain CNN mean macro-F1: `0.5315`;
- augmented CNN mean macro-F1: `0.5218`;
- Top-1 accuracy is almost unchanged;
- augmentation has lower fold variation and validation loss;
- lower validation loss does not prove calibrated confidence;
- augmentation does not pass a macro-F1 improvement rule.

### 4. Class weighting

Keep the balanced-weight explanation, controller cell, saved output, and observation together.
This experiment keeps augmentation fixed and changes only the loss weighting.

Explain that weighting lowers macro-F1, weighted F1, Top-1, and Top-5. Reducing zero-F1 classes
from 21 to 20 does not offset that damage. Reject full inverse-frequency weighting for this CNN.

### 5. Combined comparison

Keep the current saved evidence display. Explain these three levels:

- a fold row measures one saved validation group;
- the five-fold summary gives the mean and variation across groups;
- pooled OOF combines one unseen prediction for every development item.

Fold standard deviation means variation across saved data groups. It must not be called repeat-run
or random-seed stability because only one seed was used per fold.

### 6. Learning curves

Move the learning-curve section after weighting and combined comparison. All three CNN conditions
will then be introduced before they appear in one plot.

Keep the warning that weighted and unweighted training losses use different objectives and should
not be compared directly.

### 7. Failure and confusion analysis

Show the existing weak-class evidence and full normalized 124-class confusion matrix first. Keep
the full matrix and old output unchanged.

Then add:

> The full matrix gives an overview of all 124 classes, but it is too dense to show individual
> mistakes clearly. The next figures provide a closer look at the most common confusion pairs and
> example images.

After this text, show the Top-10 table, Top-10 bar chart, focused matrix, and error-image grid.

### 8. Development decision

Keep the handoff after all result and failure evidence. Call it a **development decision**, not an
ultimate judgement. Notebook 06 owns the ultimate judgement after the protected holdout test.

## Result-writing contract

Section 4 defines each metric once. Later observation blocks interpret values without repeating
the definitions.

Each observation follows this order:

1. **What is shown:** say whether rows are folds, summaries, or pooled OOF results.
2. **What happened:** give only the values needed for comparison.
3. **What it means:** connect the values to class balance, probability loss, or overfitting.
4. **Decision:** accept, reject, or leave the experiment idea uncertain.

Use one compact paragraph plus one decision sentence per result block.

## Saved OOF loading

New plots must not depend on an in-memory smoke result. Add this read-only helper to
`src/fashion/task1/analysis.py`:

```python
def load_task1_oof_predictions(
    fold_metrics: pd.DataFrame,
    registry_rows: pd.DataFrame,
    *,
    candidate_id: str,
    expected_ids: Sequence[int],
    root: str | Path = ROOT,
) -> pd.DataFrame:
    ...
```

The helper must:

- require `run_id`, `fold`, and `candidate_id`;
- select exactly one row for folds 0 through 4;
- find one completed registry row for every run ID;
- resolve each registered `prediction_path` under the repository root;
- concatenate the five prediction files;
- call the existing OOF validation contract using development IDs;
- return a deterministic frame sorted by product ID;
- perform no write and never inspect protected labels.

Notebook 02 gets expected IDs from its existing development-only sample view and registry rows
from `RunRegistry().read()`.

## Detailed confusion table

Add this helper to `src/fashion/task1/analysis.py`:

```python
def build_task1_confusion_detail(
    predictions: pd.DataFrame,
    *,
    candidate_id: str,
    limit: int = 10,
) -> pd.DataFrame:
    ...
```

It requires `id`, `true_label`, and `predicted_label`. It returns:

| Column | Meaning |
|---|---|
| `rank` | Stable one-based display order |
| `candidate_id` | Exact candidate identity |
| `true_label` | Correct class |
| `predicted_label` | Wrong predicted class |
| `error_count` | Number of this directed error |
| `true_support` | All OOF rows with this true class |
| `error_rate` | `error_count / true_support` |
| `example_ids` | First three numeric IDs in stable order |

Only wrong predictions are grouped. Sort by descending count, then true label, then predicted
label. Reject non-positive limits. Empty input returns an empty frame with the fixed schema. Write
the table atomically to `results/evidence/task1/top_confusion_pairs.csv`.

## Figure design

All writers live in `src/fashion/task1/plotting.py`. They create parent folders, close figures,
and return the output path.

### Top-10 pair chart

```python
def write_task1_confusion_pair_figure(
    confusion_detail: pd.DataFrame,
    *,
    output: str | Path,
) -> Path:
    ...
```

- Use horizontal bars labelled `True -> Predicted`.
- Bar length is `error_count`, with the number printed on the bar.
- Put the largest pair at the top.
- State that direction runs from true to predicted label.
- Do not show internal candidate IDs on axes.

### Focused matrix

```python
def write_task1_focused_confusion_figure(
    predictions: pd.DataFrame,
    confusion_detail: pd.DataFrame,
    *,
    output: str | Path,
    max_classes: int = 8,
) -> Path:
    ...
```

- Select focus labels deterministically from the largest pairs, capped at eight.
- Rows are true focus labels.
- Columns are the same focus labels plus `Other`.
- `Other` collects predictions outside the focus set.
- Show both count and within-true-class percentage in each cell.
- Normalize using all OOF rows for each displayed true class.
- Explain `Other` and normalization in the caption.

### Error-image grid

```python
def write_task1_confusion_example_figure(
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    confusion_detail: pd.DataFrame,
    *,
    output: str | Path,
    pair_limit: int = 5,
    examples_per_pair: int = 2,
    root: str | Path = ROOT,
) -> Path:
    ...
```

- Use only IDs found in the selected OOF predictions.
- Require every ID to map to one unique development row in `splits.csv`.
- Resolve the existing `path` under the repository root.
- Apply EXIF orientation for display, but no training augmentation.
- Show product ID and `true -> predicted` above each image.
- Use up to five pairs and two images per pair.
- Fail clearly for missing, duplicate, protected, or unreadable images.

New outputs are:

- `results/evidence/task1/top_confusion_pairs.csv`;
- `results/figures/task1/top_confusion_pairs.png`;
- `results/figures/task1/focused_confusion_matrix.png`;
- `results/figures/task1/confusion_examples.png`.

The existing full confusion matrix path remains unchanged.

## Public API

Export these names from `fashion.task1`:

- `load_task1_oof_predictions`;
- `build_task1_confusion_detail`;
- `write_task1_confusion_pair_figure`;
- `write_task1_focused_confusion_figure`;
- `write_task1_confusion_example_figure`.

Existing public functions and imports remain unchanged.

## HTML report

Update `docs/task1-experiment-report.html` after notebook evidence is ready:

- follow the restored experiment order;
- use friendly model names;
- copy only the strongest result meanings;
- replace the repeated selected-model matrix with the Top-10 chart and example grid;
- keep one link to the full normalized matrix;
- use `development decision` instead of premature `ultimate judgement`;
- preserve links to raw tables and run IDs.

Do not display four full 124-class matrices side by side in the main report.

## Testing

### Analysis tests

Extend `tests/task1/test_analysis.py` for:

- exact five-fold registry loading and deterministic ID order;
- missing or duplicate folds;
- missing, duplicate, or incomplete registry rows;
- missing prediction files;
- incomplete or duplicate OOF IDs;
- exact confusion counts, support, rates, ranks, and example IDs;
- deterministic ties, empty input, and non-positive limits.

### Plotting tests

Extend `tests/task1/test_plotting.py` for:

- each new writer creating one non-empty PNG;
- fixed input schemas;
- focused class limits and `Other` predictions;
- development-only example images;
- clear failures for missing columns, protected IDs, and bad image paths;
- closed Matplotlib figures.

### Notebook tests

Update the notebook structure test for:

- the approved section order;
- metric definitions remaining in Section 4;
- an observation and decision after each result block;
- the full matrix remaining present;
- the closer-look transition and three new figures;
- unchanged smoke defaults and no embedded training loop.

Check output preservation by comparing the before/after output count and key saved full-run values.
Do not clear notebook outputs.

### HTML tests

Update `tests/task1/test_html_report.py` for:

- valid local links;
- references to the compact figures;
- removal of the repeated selected-model matrix;
- one remaining full-matrix link;
- `development decision` wording;
- displayed values matching saved CSV evidence.

## Verification

Before implementation, record the hash of `results/runs.csv` and a file/hash manifest for existing
Task 1 checkpoints, histories, and predictions. Confirm the hashes are unchanged afterward.

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_analysis.py tests/task1/test_plotting.py tests/task1/test_notebook_baseline.py tests/task1/test_html_report.py -q
./.venv/bin/python -m ruff check src/fashion/task1 tests/task1
```

Then check:

- the largest directed error is still `Casual Shoes -> Sports Shoes` with 350 errors;
- every example image belongs to a development OOF row;
- the focused matrix explains counts, percentages, and `Other`;
- new figures are readable at report width;
- old outputs and the full matrix remain present;
- the notebook follows the approved experiment order;
- no registry, model artifact, split, or protected artifact changed;
- no training or holdout access occurred.

## Error handling

- Missing full evidence shows a clear `not ready` message instead of starting training.
- Missing registry or prediction files name the missing run or path.
- OOF completeness failures stop all new evidence writes.
- Protected IDs stop the image grid before any figure is written.
- New evidence writes occur only after all inputs validate.
- Plotting failures do not modify existing evidence or figures.

## Non-goals

- No new model, loss, augmentation, sampler, or hyperparameter.
- No change to split, taxonomy, or metric formulas.
- No recalculation of existing training metrics.
- No model training or holdout evaluation.
- No deletion or clearing of notebook outputs.
- No replacement of the full confusion matrix.
- No final assignment-wide report.

## Success criteria

1. Notebook 02 follows the original experiment order.
2. Existing output blocks remain visible and unchanged.
3. Every result block says what is shown, what happened, what it means, and the decision.
4. Metric definitions remain in Section 4.
5. Learning curves appear only after all CNN conditions are introduced.
6. The full confusion matrix remains visible.
7. The three new views make its important errors readable.
8. New evidence is reproducible from registered development OOF predictions.
9. The HTML uses compact figures and correct development-decision wording.
10. Focused tests and lint pass without training or protected-data access.
