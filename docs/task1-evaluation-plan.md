# Task 1 review and evaluation plan

Updated on 9 September 2026. The owner has chosen the plain CNN for final refit. The original review used `main` at `11a37317`; `main` has since advanced to `c1e19438`.
This plan includes run-record recovery and the plain-CNN selection update. No training, holdout inference, or submission generation is part of this change.

Task 1 has a sound development study. Its main gap is the final evaluation and usable model package. Completed: all 80 original Task 1 run records are restored on main, and Notebook 02 now selects the plain CNN. The selected OOF data, figures and HTML report were updated to match.

## Current status

| Area | Task 1: article type | Task 2: season | Task 3: gender and usage |
|---|---|---|---|
| Model comparisons | HOG k-NN, HOG SVM, and three CNN conditions; five folds each | Broad family, tuning, loss, multi-task and pretrained comparisons | Broad model, loss, data and training comparisons |
| Failure analysis | Learning curves, class F1, confusion pairs and example images | Also has robustness, confidence calibration, grouped uncertainty, attention maps and cost evidence | Detailed class, label, corruption and refit analyses |
| Final choice | Plain unweighted CNN selected on 9 September 2026 | I2 selected; scratch refit and handoff recorded | Single gender SAM25 refit and usage E8 refit accepted |
| Final model here | Fold checkpoints exist; no all-development refit/handoff found | Manifest exists, but `models/task2_season.pt` is missing locally | Both packaged checkpoints exist and match accepted checkpoint hashes |
| Internal holdout | No result found | Newer main now includes saved final-evaluation evidence | Completed, with an explicit warning that final acceptance followed holdout review |
| Official predictions | No article-type export found | Newer main includes a final prediction receipt | Separate 5,829-row gender and usage CSVs exist; their IDs/order match the prediction manifest. This review did not establish their model provenance |
| Shared run log | 80 original Task 1 records restored; all 590 other-task records preserved | 152 rows | Task 3 records present |

These are progress comparisons. Scores from different targets are not a fair ranking of the tasks.

Sources: `notebooks/02_task1_article_type.ipynb`, `docs/task2-season-execution-report.md`, `results/evidence/task2/final_handoff/manifest.json`, `notebooks/04_task3_final_evaluation.ipynb`, and ADRs 0024/0025.

## Repair results and remaining work

1. **Restore the verified Task 1 backup.** Source: `C:/Users/Khoa/Documents/MLA2-task1-runs-backup.csv`. It contains 80 unique original records: 69 experiment runs and 11 smoke checks. All 25 final comparison runs are covered. Their 65 referenced artifact files match their saved SHA-256 hashes, and all 25 fold scores match the saved tables. Across the complete backup, 183 referenced files match and seven older files are missing; there are no hash mismatches. Preserve those historical records and record the missing files. Completed on main: merged the 80 Task 1 rows with the existing 590 other-task rows. All existing fields and all original Task 1 fields are unchanged; the total is 670 rows. Audit: `results/evidence/task1/registry_recovery.json`. Portable source: `results/evidence/task1/registry_recovery_source.csv`. Do not fabricate or regenerate timestamps, settings or training records.
2. **The final runner is implemented and under verification.** Notebook 06 now replays saved evidence. Follow Tasks 2 and 3: run the refit and prediction stages through separate helpers, then let Notebook 06 verify and explain their saved results. Current Task 1 training is fold-based and selects an epoch using validation; it cannot use the holdout as its validation set for a final fit. A separate fixed-budget refit and image-only predictor are needed.
3. **Use the plain CNN and explain why.** Select `task1_cnn_no_aug_unweighted_v1`, with `task1_rgb_60x80_no_aug_v1` preprocessing and unweighted cross-entropy. It has the best mean and pooled macro-F1. Lower fold variation for augmentation is useful evidence, but not enough to override the declared primary metric. Date the change; preserve the earlier experiments as comparisons. Refresh the selected OOF file, confusion table, figures and report so they all refer to the plain CNN.
4. **The shared holdout has already been used by other tasks.** Task 3's ADRs explicitly record acceptance after evaluation. Newer main also contains Task 2 final-evaluation evidence. Do not describe the whole project holdout as untouched. Record which targets/results were accessed and whether they influenced Task 1. If Task 1 choices were influenced by holdout outcomes, disclose that limit; do not create a replacement split.

## What Task 1 already proves

| Model | Five-fold mean macro-F1 | Fold standard deviation |
|---|---:|---:|
| Plain CNN, selected | 0.5315 | 0.0331 |
| Augmented CNN | 0.5218 | 0.0158 |
| HOG k-NN | 0.5023 | 0.0140 |
| HOG SVM | 0.4889 | 0.0165 |
| Augmented CNN with inverse-frequency weights | 0.4564 | 0.0181 |

Macro-F1 gives each of the 124 labels equal weight. The selected CNN has pooled out-of-fold macro-F1 0.5569, Top-1 accuracy 84.22%, and Top-5 accuracy 97.80%. These are development results, not holdout results. Pooling all predictions is a different calculation from averaging five fold scores.

The selected prediction file contains exactly 32,773 unique development IDs. Its recorded folds match the current canonical folds. The selected plain-CNN summary reports pooled macro-F1 0.5569261693. Twenty-one classes have zero F1. The five selected fold checkpoints are present. The restored ledger links these scores back to the original run and artifact hashes.

Existing failure evidence is useful: Casual Shoes versus Sports Shoes, Tshirts versus Tops, and Flats versus Heels. For the plain CNN, Flats to Heels affects 39.6% of true Flats (141/356), and Sports Shoes to Casual Shoes affects 16.9% (286/1,691). Explain both the number of errors and the share of the class affected.

Sources: `results/evidence/task1/comparison.csv`, `classical_comparison.csv`, `oof_metrics.csv`, `selected_oof_predictions.csv`, and `top_confusion_pairs.csv`.

## Updated execution plan

1. **Completed: restore the ledger and select the plain CNN.** Merge the verified backup, confirm unique IDs and unchanged other-task records, update Notebook 02 to select the plain CNN, and rebuild the selected-model figures from its saved predictions. Keep the canonical split and all saved folds.
2. **Write the evaluation contract.** Freeze candidate, class order, preprocessing, seed, training budget, checkpoint rule, metrics, slices and stress tests before seeing Task 1 holdout results. Make a dated JSON manifest. Complete the shared handoff record and document Task 3's earlier access.
3. **Refit once on development only.** Train the selected plain CNN from random weights on the 32,773 eligible development rows. Fit normalization on development only. Register this as a new run. Proposed budget: retain the existing 20-epoch recipe and save epoch 20, with no validation-based epoch selection. This recipe is now recorded in `configs/task1/final_evaluation.json` before training. Use no random training augmentation and ordinary unweighted cross-entropy. Save the exact scheduler, normalization, class map, configuration and checkpoint hashes.
4. **Predict the internal holdout, then score it.** Use the same locked model for all 5,778 holdout images. Save all 124 probabilities and IDs before joining labels by ID. Use the protected loader with explicit unlock only after the freeze. No augmentation, refitting, model switching or threshold fitting here.
5. **Write the final judgement.** Report the measurements below, compare the observed result with development expectations, and state where human review is needed. A weak result is a result to explain, not a reason to tune on holdout.
6. **Predict the assignment test set.** Run that same checkpoint and saved preprocessing on the 5,829 official images. Fill only `articleType`; merge the other owners' outputs by ID. Preserve exactly `id,gender,articleType,season,usage`, template order, allowed labels and no blanks. The teacher supplied no test labels, so this is prediction, not an accuracy evaluation. Keep `styles_prediction.csv` as the source template and write the completed file separately.
7. **Package and demo.** Include the model, small image-only prediction function, dependencies and exact run instructions. Test loading and predicting in a fresh process. Show an ordinary success and a meaningful failure in the GUI.

## Notebook 06: use Tasks 2 and 3 as small workflow references

Use `notebooks/06_task2_season_evaluation.ipynb` as the main workflow reference and `notebooks/04_task3_final_evaluation.ipynb` as the reference for clear explanations. Keep Notebook 06 as the shared final-evaluation notebook: add Task 1's evidence and a short group status table, with links to the completed Task 2 and Task 3 analyses. Do not copy their whole notebooks or rerun their evaluations.

Both reference notebooks replay saved results. Follow that pattern: a normal **Run All reads and verifies artifacts**, then shows tables, figures and explanations. It must not train, open raw holdout labels, run model inference or overwrite official predictions. If a required artifact is missing or its hash differs, stop with a clear message naming the missing stage.

| Existing Notebook 06 sections | What to show for Task 1 | Small reference to follow |
|---|---|---|
| 1-4: contract, frozen inputs and CV | Plain CNN identity, 124-label map, five-fold comparison, reason for selection and known limits | Task 2: contract and model identity; Task 3: clear distinction between development and final-refit scores |
| 5-7: refit, preprocessing and checkpoint | Saved full-development run, fixed epoch rule, training-only normalization, checkpoint/config hashes | Both tasks: one scratch refit, saved preprocessing and a verifiable model package |
| 8: holdout access | Dated receipt showing predictions were fixed before labels were joined; document earlier access by other tasks | Task 2: separate prediction and scoring receipts; Task 3: honest access history |
| 9: independent evaluation | Holdout coverage, fixed 124-class macro-F1, Top-1/Top-5, class scores and uncertainty | Task 2: scorecard, class analysis and product-family bootstrap |
| 10: failures and practical limits | Confusion pairs, rare-class counts, fixed photo stress tests, confidence diagnostics and measured cost | Task 2: robustness and cost; Task 3: explain small class counts and what the evidence cannot prove |
| 11: final judgement | What worked, what failed, where human review is needed, and a measurable extension | Task 3: connect each conclusion to an observed result and its limit |
| 12-13: predictions and submission | Task 1's 5,829-row export, ID/label checks, combined-file status and source links | Task 2: prediction receipt; Task 3: final model and system handoff |

For each main result, use the same short pattern: **question -> saved table/figure -> result -> meaning -> limit**. Repeat only a compact development comparison; keep the full experiment story in Notebook 02. Label development CV, pooled OOF and final holdout scores clearly so readers do not mistake them for the same measurement.

### Separate execution stages and files

These Task 1 entry points are now implemented. Real training and scoring are still pending.

| Stage | Proposed entry point | Saved output |
|---|---|---|
| Refit | `scripts/refit_task1_article_type.py` | New row in `results/runs.csv`; `models/task1_article_type.pt` and `models/task1_article_type.manifest.json` |
| Predict holdout without labels | `scripts/build_task1_final_evaluation.py predict-holdout` | `results/evidence/task1/final_evaluation/holdout_predictions.csv` with IDs and 124 probabilities; `prediction_receipt.json` |
| Score frozen holdout predictions | `scripts/build_task1_final_evaluation.py score --evaluation-unlocked` | `unlock_receipt.json`, metrics, class/error tables and `evaluation_manifest.json` in the same evidence folder |
| Predict assignment images | `scripts/build_task1_final_evaluation.py predict-test` | `results/article_type_test_predictions.csv` with `id,articleType`; `test_prediction_receipt.json` |
| Verify and replay | `scripts/build_task1_final_evaluation.py audit`, then Notebook 06 | Verified tables and figures; cited figures under `results/figures/task1/final_evaluation/` |

Put reusable training, inference and evaluation logic in `src/fashion/task1/`; keep these scripts thin and the notebook narrative. Save the evaluation choices in `configs/task1/final_evaluation.json` before holdout inference. Receipts must connect the exact checkpoint, preprocessing, class map, split, image IDs, prediction files and timestamps. Keep full precision in saved files; round only displayed tables.

Task 2's `scripts/build_task2_final_evaluation.py` and `src/fashion/task2/final_evaluation_runner.py` show the prediction/score/audit pattern. Its prediction stage also creates official test predictions before scoring. For Task 1, keep the requested order: **holdout prediction -> holdout evaluation -> official prediction**, using the same locked model throughout. Completed stages should be audited and replayed; a failed or partial write should be inspected before any retry, with no silent overwrite.

### What to share and what to keep specific to Task 1

- Reuse common file hashing, atomic writes, the canonical data loaders and generic group-bootstrap helpers where compatible. Keep Task 1's image transform and fixed 124-class metric contract. Task 2's Season runner has hard-coded labels, model identity and temperature; it cannot be called directly for article type.
- Match Task 2's small stress-test set for easier group reporting: clean images, JPEG quality 85 with subsampling 2, brightness factors 0.85 and 1.15, and Gaussian blur radius 1. Apply these to the image before Task 1's deterministic preprocessing. Freeze the choices now; use the results to describe limits, not select another model. This replaces the earlier draft's JPEG quality 50 proposal.
- Use raw confidence for the initial Task 1 report. Any risk-coverage plot is diagnostic; no review threshold is picked from holdout. Calibration is optional work that must be fitted and assessed using development evidence first. Do not reuse Task 2's temperature or Task 3's label corrections.
- Keep HOG, augmentation and weighting comparisons in development evidence. If adding Task 2's simple majority baseline to the holdout scorecard, declare it first, fit its label using development only, record the run and freeze its predictions before scoring. Do not rerun all five candidates on holdout to choose a winner.
- Reference Task 3's honest discussion of evaluation timing, but retain Task 1's pre-evaluation plain-CNN choice. No model switching after Task 1 holdout scoring. Other tasks' class counts and scores are not evidence of Task 1 performance.

Completion checks: all model and prediction hashes agree; exactly 5,778 holdout IDs and 5,829 official IDs are covered; quarantine is excluded; class order is unchanged; original Task 1 metrics agree with the saved probability tables; repeated Notebook 06 Run All leaves model files, prediction files, receipts and `results/runs.csv` unchanged. Each displayed conclusion must point to its saved evidence. The group table should link existing Task 2/3 results rather than claim they were newly evaluated by Notebook 06.

## Measurements to declare before holdout

| Question | Evidence |
|---|---|
| Does it handle all labels? | Fixed 124-label macro-F1 as primary; per-class precision, recall, F1 and support. Keep absent-label zeros explicit |
| How often is its first answer right? | Top-1 accuracy and weighted F1 |
| Is a short suggestion list useful? | Top-5 accuracy; this does not replace the single-label submission |
| Which groups fail? | Class-size bands defined from development counts: 1-20, 21-100, and over 100 examples; grayscale versus colour; shoe/top confusion pairs |
| How uncertain is the estimate? | 95% intervals by resampling whole product-family groups, preserving related images together. Small/absent classes remain weak evidence |
| Does photo quality matter? | Match Task 2's fixed probes: JPEG quality 85 (subsampling 2), brightness multiplied by 0.85 and 1.15, and Gaussian blur radius 1 pixel. Compare each with clean inputs; select nothing from these results |
| Can the app use it? | Model size, batch-one CPU latency after warm-up, median and 95th percentile over repeated runs; state machine, thread count and whether loading is timed |

Use development predictions first for any confidence calibration or review threshold. Never pick a threshold from holdout. Compact OOF files contain labels only; confidence work needs the original probability files. Save cited figures under `results/figures/task1/`.

## Prepare evidence for Assessment 3

The untracked Assessment 3 PDF awards 20 points each for literature review, critical analysis, extension, demonstration, and presentation/Q&A. Its HD review asks for at least two additional closely related papers across the project and literature relevant to every task. This does not mean two new training runs.

For Task 1, prepare these four small pieces:

- **Comparison and explanation:** one five-model table; why CNN features helped; why augmentation lowered fold variation but not macro-F1; why strong inverse-frequency weights hurt. Do not call a failed improvement a success.
- **Evidence of limits:** one readable confusion chart and example images; rare-class results with counts; clean versus degraded inputs; one honest final judgement. Existing development errors are already useful for this.
- **Literature comparison:** compare problem, labels, image conditions, annotations and training assumptions, then explain which ideas fit this project. Do not compare published scores as if they used our 124 labels and folds.
- **A measurable extension:** propose top-five suggestions with a human-review option for uncertain article types. Fit the confidence rule on development only. On a later untouched evaluation set, measure wrong automatic labels, fraction handled automatically, Top-5 coverage, rare-class recall and review time. Compare with automatic Top-1 assignment. A separate, labelled customer-photo set would test the catalogue-to-phone-photo gap; blur tests alone do not prove that.

Two useful paper starting points, verified from the publisher's pages:

1. [Liu et al., DeepFashion, CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html): clothing recognition across store and consumer photos, with rich attributes and landmarks. Use it to discuss photo changes and why richer annotations could help. Its data and annotations differ from ours.
2. [Cui et al., Class-Balanced Loss, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html): reweights using the effective number of samples. Compare this with Task 1's tested inverse-frequency weights; they are different methods. It is a justified future experiment, not an improvement already proved here.

Read the full papers before presenting detailed claims. The presentation is under ten minutes for the whole group, not ten minutes per task. The PDF header says a two-minute demo while its body allows two to four; planning a two-minute group demo fits both. All tasks must be shown, and all students must appear with faces in the recording. Slides/video must be submitted before presentation day as specified by the class.

Requirement sources: `rubrics/RUBRIC.md`; `docs/COSC2753_2026B_Assignment 2.pdf` (pages 3, 6-7); `Assessment Task 3_ Presentation & Technical Interview (40%)_.pdf` (pages 2-3, 5-7).

Next after this repair: implement and lock the all-development plain-CNN refit, then holdout evaluation, then official predictions. Broad new model searches are lower priority than closing these gaps.


## Implementation handoff — 9 September 2026

The tools and Notebook 06 are implemented. Tests use synthetic images; no real final refit, Task 1 holdout scoring or official export has been run by this change.

Run these commands from the repository root, in order. Let each finish before starting the next:

```powershell
.venv/Scripts/python.exe scripts/refit_task1_article_type.py --mode run
.venv/Scripts/python.exe scripts/build_task1_final_evaluation.py predict-holdout
.venv/Scripts/python.exe scripts/build_task1_final_evaluation.py score --evaluation-unlocked
.venv/Scripts/python.exe scripts/build_task1_final_evaluation.py audit
```

Open `notebooks/06_final_evaluation.ipynb` and Run All to review the saved results. Then export the assignment predictions:

```powershell
.venv/Scripts/python.exe scripts/build_task1_final_evaluation.py predict-test
.venv/Scripts/python.exe scripts/build_task1_final_evaluation.py audit
```

The refit uses CUDA if available, otherwise CPU. The evaluation runner uses CPU so the saved cost measurement is explicit. Keep the model package, history, canonical inputs, registry row and source files together. Run All reads files and checks scores; it does not run the model or change artifacts.

A completed or failed stage cannot silently run again. Inspect its attempt record and any partial files before a deliberate recovery. Do not delete receipts to tune on the holdout. Official export contains Task 1 only; the group merge and app demonstration remain separate work.

Verification: 15 focused refit/inference/evaluation checks passed after the final fixes. The wider Task 1 + notebook + documentation run had 295 passed, 1 skipped and 2 failures: missing local `models/task2_season.pt` and `results/evidence/task3/`. These missing other-task artifacts were not changed. Code lint and `git diff --check` passed.


## Notebook separation — 9 September 2026

At the owner's request, the implemented Task 1 artifact-replay notebook was copied unchanged to `notebooks/02_task1_final_eval.ipynb`. The shared `notebooks/06_final_evaluation.ipynb` was restored to its original committed scaffold. Use `02_task1_final_eval.ipynb` for the Task 1 replay steps above. Its contents were preserved exactly, including the existing headings. The proposed training/holdout/test execution notebook is still separate pending work.


## Final-run notebook and epoch review

`notebooks/02_task1_final_run.ipynb` now provides the execution cells: local or Colab/Drive setup, existing ZIP reuse, final refit, blind holdout predictions, explicit scoring and official test export. It has not been run on real data. `02_task1_final_eval.ipynb` stays artifact replay; Notebook 06 stays the original group scaffold.

The five plain-CNN best development epochs are 20,19,18,19,20 (median19). Keeping epoch20 preserves the tested OneCycle schedule. Mean epoch20 macro-F1 is approximately0.5302 versus0.5315 for fold-selected checkpoints; this small difference supports retaining the fixed20 budget, not a claim of an optimal budget. No Task1 holdout evidence was used.

Colab needs these local changes published to GitHub first. No commit or push has been made. Its dedicated Drive checkout saves outputs persistently. Merge the new Task1 registry row back by ID instead of overwriting teammates' newer records.
