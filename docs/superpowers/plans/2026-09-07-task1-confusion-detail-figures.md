# Task 1 Results Explanation and Confusion Detail Plan

## Goal

Make Notebook 02 clearly explain its completed experiment results, then keep the full 124-class confusion matrix and add readable figures that explain the most important mistakes. Reuse saved metrics and out-of-fold predictions. Do not retrain models or open the holdout.

## Safety rules

- Read the selected candidate's saved five-fold prediction files through the registered run IDs.
- Do not create a new training run or append to `results/runs.csv`.
- Do not change checkpoints, model settings, splits, metrics, or the frozen holdout.
- Keep the full confusion matrix as complete evidence, but use smaller figures for explanation.
- Keep metric definitions in Section 4. Later observations interpret the actual results without repeating the full definitions.
- Keep the existing saved full-training outputs visible in the notebook.

## Workstream 1: Preserve the completed outputs

1. Leave the existing full-training outputs unchanged.
2. Do not clear, move, or regenerate those outputs as part of this work.
3. Add the new markdown discussion directly after the existing result output blocks.
4. Continue using the saved CSV evidence for new tables and figures.

## Workstream 2: Restore the original experiment flow

Reorder the Notebook 02 story to match the order in which the investigation was designed:

1. **Scratch CNN:** introduce the plain, unweighted CNN as the neural control and show its saved five-fold and OOF results.
2. **HOG experiments:** compare HOG k-NN and HOG linear SVM with the scratch CNN to test whether simpler shape features can compete.
3. **Data augmentation:** introduce mild augmentation as one controlled change from the plain CNN, then show the plain-versus-augmented comparison.
4. **Class weighting:** keep augmentation fixed, change only the loss weighting, and show the weighted result.
5. **Combined comparison:** show every final candidate under the same five-fold metric contract.
6. **Learning curves:** move the learning-curve section after class weighting so it does not discuss the weighted CNN before that experiment is introduced.
7. **Failure analysis:** show weak classes, the full confusion matrix, detailed confusion figures, and example images.
8. **Development decision:** record the chosen candidate and handoff to Notebook 06.

Make this change by moving markdown and saved-result display cells. Reuse the completed evidence. Do not rerun training.

## Workstream 3: Explain the metrics and results

1. Keep Section 4 as the single place that defines:
   - fixed-label macro-F1 and why it is primary;
   - weighted F1;
   - Top-1 accuracy;
   - Top-5 accuracy;
   - fold mean and sample standard deviation;
   - pooled OOF metrics;
   - validation loss.
2. After each saved result table, add a short observation block with four parts:
   - **What is shown:** what one row means and whether the table contains folds, summaries, or pooled OOF results.
   - **What happened:** name the best result and the important differences using the displayed values.
   - **What it means:** connect the values to common classes, rare classes, fold differences, overfitting, or probability quality.
   - **Decision:** say whether the tested idea is supported, rejected, or still uncertain.
3. Add this explanation after the classical results:
   - KNN has the better macro-F1 and Top-1 score;
   - SVM has better Top-5 performance;
   - explain that this makes KNN the stronger single-label baseline while SVM is better at short candidate lists.
4. Add this explanation after the plain-versus-augmented CNN results:
   - plain CNN has the best mean and pooled OOF macro-F1;
   - augmentation keeps Top-1 almost unchanged and lowers fold variation and validation loss;
   - lower validation loss does not by itself prove good calibration;
   - the result does not support a macro-F1 improvement claim.
5. Add this explanation after the weighted CNN results:
   - weighting lowers macro-F1, weighted F1, Top-1, and Top-5;
   - it does not solve the rare-class problem enough to justify the damage to other classes;
   - reject full inverse-frequency weighting for this model.
6. After the combined comparison, explain the difference between:
   - fold mean: the average of five validation results;
   - fold standard deviation: how much scores differ across the five saved data groups;
   - pooled OOF: one score from all development predictions combined.
7. Avoid saying fold standard deviation proves repeat-run stability. Only one seed was used per fold.
8. Keep observations short enough for the notebook story: one compact paragraph per result block, followed by a clear decision sentence.

## Workstream 4: Add confusion detail

1. Add analysis helpers in `src/fashion/task1/analysis.py` that build a selected-model Top-10 confusion-pair table with error counts, true-class support, and example IDs.
2. Add plotting helpers in `src/fashion/task1/plotting.py` for:
   - a horizontal Top-10 confusion-pair chart;
   - a focused confusion matrix for the main confused classes;
   - an image grid showing representative mistakes.
3. Export the new helpers from `src/fashion/task1/__init__.py`.
4. Add focused unit tests in `tests/task1/test_analysis.py` and `tests/task1/test_plotting.py`.
5. Update Section 11 of `notebooks/02_task1_article_type.ipynb` so it shows, in order:
   - the full 124-class confusion matrix;
   - a short note that the next figures give a closer look;
   - the Top-10 pair chart;
   - the focused matrix;
   - the example image grid.
6. Save the compact table and figures as:
   - `results/evidence/task1/top_confusion_pairs.csv`;
   - `results/figures/task1/top_confusion_pairs.png`;
   - `results/figures/task1/focused_confusion_matrix.png`;
   - `results/figures/task1/confusion_examples.png`.
7. Update `docs/task1-experiment-report.html`: remove the repeated selected-model matrix, use the compact chart and image grid in the main result story, and keep the full matrix as supporting evidence or appendix material.
8. Update `tests/task1/test_html_report.py` and the notebook structure test for the new files and text.

## Workstream 5: Keep the report aligned

1. Copy only the strongest result meanings into `docs/task1-experiment-report.html`; do not copy every notebook sentence.
2. Use friendly model names in report tables and figures instead of long internal candidate IDs.
3. Call the Task 1 choice a **development decision**, not an ultimate judgement, until the locked holdout evaluation is complete.
4. Keep raw values, run IDs, and complete tables linked as supporting evidence.
5. Use the Top-10 confusion chart and example grid in the main report. Keep the full matrix as appendix or linked evidence if page space is tight.

## Suggested notebook transition

> The full matrix gives an overview of all 124 classes, but it is too dense to show individual mistakes clearly. The next figures provide a closer look at the most common confusion pairs and example images.

## Checks

- The largest selected-model pair remains `Casual Shoes -> Sports Shoes` with 350 errors.
- All displayed examples come from saved development OOF predictions.
- The focused matrix states whether rows are counts or normalized values.
- The new figures remain readable at report width.
- Existing full confusion evidence remains available.
- Section 4 defines metrics once and later observations interpret rather than repeat them.
- Existing full-training outputs remain visible and unchanged.
- The notebook follows the original order: scratch CNN, HOG, augmentation, then class weighting.
- Learning curves appear only after all three CNN conditions have been introduced.
- Every result block states what is shown, what happened, what it means, and the decision.
- No new registry row, checkpoint, training run, or holdout access occurs.
- Run focused tests with:

  `./.venv/bin/python -m pytest tests/task1/test_analysis.py tests/task1/test_plotting.py tests/task1/test_html_report.py -q`
