# Task 3 final review — 11 September 2026

**Verdict: the training and saved scores are sound, but the current final selection cannot be certified as a fresh blind evaluation. The final handoff also has gaps.**

Scope: the current working tree, including the uncommitted notebook rewrite, Task 3 training and evaluation code, saved models, registries, decision records and notebook renders. This is a review. Existing code, notebooks, models and records were not changed. No new model training or holdout inference was run.

## Findings

### 1. [P1] State that the holdout was already opened

The [final notebook](../notebooks/07_task3_part2_final_evaluation.ipynb), Sections 1 and 5, says the development study fixes the model and that predictions are saved before reference labels are joined. These statements omit the key historical limit: the same holdout had already been examined before these final decisions.

- **Gender MixUp:** its [evaluation plan](../reports/task3/gender_mixup_refit_holdout_20260911/evaluation_plan.json) explicitly records `new_blind_evaluation: false` and “reserved rows already opened in prior project work.” The plan was saved at 05:34:40 UTC, predictions at 05:34:52 UTC, and scores at 05:34:53 UTC on September 11. The script does freeze predictions before joining labels in this pass. However, [the September 7 Gender decision](decisions/0025-task3-gender-sam25-refit-final-model.md) records an earlier acceptance after holdout review. The new MixUp development comparison and selection came later.
- **Usage E8:** its [prediction freeze](../reports/task3/usage_e8_refit_holdout_20260907/prediction_freeze.json) says this was not a new blind evaluation. Predictions were frozen at 07:59:39 UTC on September 7. Its [acceptance manifest](../reports/task3/usage_final_e8_refit_20260907/model_manifest.json) was recorded at 08:38:44 UTC and states “after evaluation review; not a new blind selection.” The earlier September 7 version of ADR 0024 also states this directly. The later development rationale does not change that sequence.

This establishes prior exposure, not proof that holdout results caused a specific model change. No direct holdout use in the reviewed training loops was found. The current MixUp comparison can be reproduced using development evidence alone. Neither fact proves that the whole selection process remained blind.

**Fix:** put this near the start of the final notebook: “These are fixed-model follow-up evaluations on previously opened holdout rows, not new blind evaluations. Predictions were frozen before label joining in each recorded pass, but final selection was not completed before the first holdout inspection.” Keep the original records and dates. Restore the missing scope disclosure checked by `tests/test_task3_usage_final.py:131`.

### 2. [P1] The packaged Gender model is not the evaluated model

The notebooks evaluate the 30-epoch MixUp refit with checkpoint hash `860f688162cccfcd903e8e4874b368697c0637b6e6a15baae3b4d4a3008f4ef9`.

The [Gender package](../results/task3/gender_model/package_manifest.json), its actual `final_epoch.pt`, [final verifier](../src/fashion/task3_gender_final.py), and accepted ADR 0025 still identify the 25-epoch SAM refit with hash `41a5f5ea027805e8edbbf667d080564776ca9873d9146e0dec15c0d4bec2b272`.

The final notebook warns against assuming the generic folder has the selected weights. That warning is accurate, but the handoff remains unfinished: following the package README runs a different model from the one credited with 78.74% F1.

**Fix:** publish a new, dated current-model handoff for the selected MixUp artifact and make the inference package and verifier agree with it. Preserve SAM's old acceptance records as history. Check the packaged checkpoint, normalization, class order and predictions against the selected artifact before submission.

### 3. [P2] Six new Gender runs are missing from the canonical registry

All five September 11 MixUp CV runs and the full-development refit have exactly one complete row in the saved `reports/task3/gender_mixup_selection_20260911/drive_runs.csv`. None is in the current [results/runs.csv](../results/runs.csv), which has 671 unique run IDs.

The new analysis uses saved registry snapshots, so its scores can still be verified. However, the project requires every run in the canonical registry, and Section 3 of the final notebook incorrectly says both selected fits are registered there. The missing refit is `t3_gender_name_truth_mixup_alpha020_refit_20260911T041436Z_3af0b94e`.

**Fix:** import the six verified rows through the registry workflow, preserving run IDs, hashes and original metadata. Check for conflicts and duplicates. Generate the final comparison inventory from the reconciled registry. This is a synchronization gap, not evidence that the runs were unrecorded during training.

### 4. [P2] Notebook replay requires Git history that a normal ZIP lacks

Both notebooks call `verify_usage_final(..., saved_training_source=True)`. [The helper](../src/fashion/task3_final.py), lines 95–109, unconditionally runs `git show` for the recorded training revision.

Replay succeeds in this checkout. A review copy with all assets available but no `.git` directory fails with `CalledProcessError` and “not a git repository,” while reading `5550d07ec91434a8b09d5d0da09200d9b5e3accb:src/fashion/config.py`. A shallow clone missing that revision has the same underlying dependency.

The assignment requires an executable source/model ZIP. A successful Run All in the development checkout does not establish that requirement.

**Fix:** include a hash-checked snapshot of the actual training source as an artifact, and verify those bytes during replay. Keep Git verification as an additional check where history exists. Test the finished package after extraction outside any Git repository.

### 5. [P2] Existing research needs an explicit link to the final results

External research is already present. The development notebook uses papers to motivate GeM, MixUp, class weighting and calibration, and discusses local adaptations and limits. The [research references](task3/research/references.md), Section 4, also include fashion-specific work by Parekh et al. and Seo et al. The Seo entry already explains why its setting is not directly comparable. This is useful existing work, not missing research.

The narrower gap is in the final-result discussion: the two notebooks do not explicitly connect the selected Task 3 results to the findings of those fashion-specific studies. The [assignment](COSC2753_2026B_Assignment%202.pdf), Section 3.3 and page 7, asks for comparison with other work using similar data or goals. A bibliography and method justification support the approach; the result comparison needs to say what agrees, what differs and what the different settings prevent us from concluding. No numerical leaderboard or identical dataset is required by this review.

The external Usage additions were development experiments; they are not a fresh external evaluation of the selected teacher-only E8 refit. The internal holdout is already opened. The exact final refits also lack a corruption benchmark. The notebook correctly admits that fold-model robustness cannot be assigned to refit weights.

**Fix:** reuse the existing research in a short final-result comparison, making the relevant task, data, input and training differences clear. No new training is needed to close this writing gap. Do not rank incompatible published scores as if they used the same experiment. Separately, if a fresh blind claim is required, freeze the current system and assess genuinely untouched product families under a fixed plan. Literature comparison does not make an already opened holdout blind; do not create a replacement test set by reshuffling already-used development rows.

## Comparison with standard ML practice

**Data separation — passes the reviewed implementation checks.** The split retains 32,773 development rows and 5,778 holdout rows. IDs, saved product families and exact image hashes do not overlap between development and protected partitions. Gender folds use the saved family allocation. Usage excludes the one blank target while preserving literal `NA`. The final Usage refit uses teacher data only.

**Preprocessing and training — passes.** RGB statistics are fitted on fold-training content pixels during CV and full-development pixels during refit. Padding is excluded. Augmentation and MixUp are training-only. The selected networks start from scratch. Both selected refits use a fixed 30-epoch schedule and the last checkpoint, with no validation or early stopping during refit. This follows the separation and preprocessing principles in the [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

**Development selection — reasonable, with limits.** The study compares several learner types and many controlled variants, preserves failed trials, separates two-fold screens from five-fold totals, and distinguishes original from corrected labels. MixUp's 80.46% corrected-label OOF F1 is not its original-label score: that score is 75.73%. The final holdout uses original labels. The 350 rule-based development label changes are explicit and do not alter holdout labels.

**Uncertainty — mostly well handled.** Family bootstrap respects related products. The notebook correctly reports that MixUp's 100,000-draw interval versus SAM25 crosses zero, that additional folds do not establish a clear win, and that one seed and different GPU types limit the comparison. Bootstrap over saved predictions does not include training variation or repeated-selection bias. Standard guidance distinguishes tuning evidence from independent performance estimation; nested CV can assess a selection procedure, but simply adding it after these decisions would not make the opened holdout fresh. See [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) and [selection bias](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

**Selection before first holdout inspection — does not pass.** The latest evaluation pass has a sensible prediction-freeze order. The historical sequence does not establish a final selection frozen before first exposure. File hashes prove identity; a training receipt's `holdout_evaluated: false` describes that training run, not the full project's history.

**Practical claims — appropriately limited.** The notebooks show weak rare classes, uncertain audience tags, raw confidence limits and the lack of measured staff-time savings. They recommend reviewed suggestions. A GUI, full four-target submission and operational acceptance are separate unfinished deliverables; this review does not certify them.

## What was verified

- Both main notebooks executed from fresh kernels: 184 development cells and 33 evaluation cells. Figure writes were redirected to temporary review folders. The development run passed a file-read guard against holdout/test result paths.
- All 45 companion training notebooks passed notebook-format validation; all 242 code cells compiled after standard IPython syntax transformation. None contains a saved error output. GPU training was not repeated.
- All five new MixUp fold artifacts and the refit passed saved-source, checkpoint, training-row, normalization, epoch, learning-rate and Drive-registry checks. The five-fold predictions cover each development ID exactly once and reproduce the selected OOF metrics.
- Both selected models' scored IDs and family groups match the canonical holdout. Independent scikit-learn calculations reproduce the saved accuracy and fixed-class F1: Gender 5,227/5,778 correct, 90.46% accuracy, 78.74% F1; Usage 5,125/5,778 correct, 88.70% accuracy, 42.26% F1.
- Rendered both notebooks in Chromium. No broken images or viewport-overflowing tables were found at 1440 pixels. Inspected development plots, final confusion matrices, error gallery, confidence plots and key rendered sections. Also inspected assignment page 7.
- Focused lint passed for the new report helpers, selected training runners and selected-report tests.
- Training-suite command: `PYTHONPATH=src ./.venv/bin/python -m pytest -q tests/train -k 'task3 or mixup or sam'`: **395 passed, 2 failed, 168 deselected**. Both failures require the new training notebooks to have empty outputs; those notebooks now contain completed-run evidence. These are stale expectations, not failed training mathematics.
- The focused artifact/notebook suite gave **110 passed, 7 failed**. Five failures concern Task 3: the missing blind-scope sentence, the old Gender verifier's registry-source hash, an outdated notebook inventory, and the two empty-output expectations above. Two additional failures concern Task 1 wording and a missing Task 2 model. Counts overlap with the training suite and must not be added.

The old Gender verifier's hash failure needs an explicit historical-source replay policy. Do not change an old lock merely to make a changed source file pass. The test expecting the blind-scope disclosure should be satisfied by honest wording, not deleted.

**Recommended next step:** fix the disclosure, current-model handoff, registry synchronization and ZIP replay first. Keep the present scores as follow-up evidence. New training does not solve these issues. A fresh, untouched evaluation set is needed only if the work must support a new blind-performance claim.
