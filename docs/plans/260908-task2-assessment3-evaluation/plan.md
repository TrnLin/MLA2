---
title: "Task 2 independent evaluation and Assessment 3 preparation"
description: "Complete Kai's Season evaluation notebook from the frozen I2 bundle and turn the evidence into a defensible Assessment 3 presentation and interview pack."
status: pending
priority: P1
effort: 2-4 days
branch: evaluation/task2-assessment3-prep
tags: [feature, docs, critical, task2, assessment3]
created: 2026-09-08
---

# Task 2 independent evaluation and Assessment 3 preparation

## 1. Outcome

Produce a locked, replayable Task 2 evaluation notebook for Kai and an evidence pack that can support Assessment 3. The notebook must evaluate the frozen Season I2 bundle on the internal holdout after the group freeze. It must not retrain, retune, switch the winner, or change metrics after holdout labels are exposed.

This plan is scoped to Task 2. The shared `06_final_evaluation.ipynb` remains the group-level aggregator. Kai's owner notebook is `notebooks/06_task2_season_evaluation.ipynb` and is later summarised in the shared Notebook 06.

## 2. Current facts

- Current base is `main` after the merged integration and Task 3 pull.
- Work branch is `evaluation/task2-assessment3-prep`.
- Frozen Task 2 winner is I2: `g4-i2-article-type-lambda-0-3-c1`.
- I2 is a scratch SmallCNN with `weights=None`, image-only inference, and ArticleType used only as a masked training auxiliary target.
- Frozen development primary metric is pooled five-fold OOF Season macro-F1: `0.7526869559580971`.
- Frozen temperature is `1.3650015953177774`; no review threshold is justified yet.
- The frozen model bundle is `models/task2_season.pt` and its manifest is `models/task2_season.manifest.json`.
- The component handoff still says `group_freeze_verified=false`, `holdout_opened=false`, and `evaluation_claim_allowed=false`. No holdout scoring may run until the group records the freeze.

## 3. Baseline decision: B0 or B1

Use **B0 as the primary holdout baseline**.

| Comparison | Purpose | Holdout role |
|---|---|---|
| B0 training-fold majority | Tests whether I2 beats a label-prior rule with no image understanding | Primary sanity comparator; deterministic and available from the frozen class prior |
| B1 HOG + HSV LinearSVC | Tests a strong classical image representation | Secondary comparator only if a B1 holdout prediction receipt is frozen before labels are opened |
| C2 scratch ResNet18 | Development reference used to justify I2 selection | Do not create a new post-freeze holdout run; show its development evidence only unless a receipt already exists |
| I2 frozen SmallCNN | Final selected model | Main holdout result |

The development model-selection story remains `I2 versus C2`, with B0 and B1 showing progressively stronger baselines. The independent holdout judgement should first answer “does the final model beat a simple prior?” with B0. B1 may strengthen the comparison, but it must not be trained or selected after holdout access.

## 4. Notebook architecture

`notebooks/06_task2_season_evaluation.ipynb` will use English prose and one code cell per leaf subsection followed by one interpretation cell.

1. Evaluation scope and holdout boundary.
2. Frozen I2 bundle and hash audit.
3. Prediction receipt before labels.
4. One-shot unlock and label coverage.
5. Clean holdout scorecard: I2 versus B0, and B1 only when pre-frozen.
6. Per-class metrics and confusion matrix.
7. OOF-to-holdout generalisation.
8. Family-blocked bootstrap uncertainty.
9. Calibration and review behaviour.
10. Predeclared slices: Spring, ArticleType conflict, year, file size, family size, grayscale/RGB.
11. Robustness: clean, JPEG quality 85, brightness 0.85/1.15, blur radius 1.
12. Deterministic error examples and Grad-CAM review.
13. Runtime, memory, coverage, and practical viability.
14. Literature comparison and ultimate judgement.
15. Artifact audit and handoff to shared Notebook 06.

Default mode is `artifact_replay`. It reads verified aggregate evidence and cannot silently open protected labels. A separate controlled scoring step creates the immutable evidence bundle after the group unlock receipt exists.

## 5. Execution phases

### Phase 0 - Branch and repository audit

- Keep `main` untouched.
- Work only on `evaluation/task2-assessment3-prep`.
- Verify the Task 2 handoff, manifest, model hash, split hash, label-map hash, implementation hash, and registry binding.
- Confirm no Task 2 candidate may be changed after the freeze timestamp.

### Phase 1 - Predeclare the evaluation

- Freeze the metric contract, baseline policy, slice definitions, perturbations, bootstrap seed, and report wording before labels are opened.
- Record the group freeze receipt and acknowledge that Task 3 already has holdout evidence; do not claim that the entire group has never accessed holdout.
- Keep Task 2's independent claim precise: the I2 decision predates Task 2 Season holdout scoring.

### Phase 2 - Produce label-free holdout predictions

- Load the canonical split through the protected, label-redacted API.
- Select all 5,778 holdout IDs and no quarantine IDs.
- Predict clean and predeclared perturbation images with the frozen I2 bundle.
- Verify unique ID coverage, finite four-class probabilities, sum-to-one probabilities, image-read failures, model hash, and manifest hash.
- Write an immutable prediction receipt. Do not write or inspect holdout labels in this phase.

### Phase 3 - Controlled one-shot scoring

- Only the designated evaluation owner opens protected labels once.
- Score valid Season labels and report blank/invalid labels separately.
- Compute primary and secondary metrics, per-class metrics, count and row-normalised confusion matrices, and coverage.
- Run 10,000 product-family bootstrap draws with seed 2753 for I2 and I2-minus-B0.
- Keep all outputs immutable and hash them in an evaluation manifest.
- Never retrain, refit temperature, choose a threshold, or switch model after scoring.

### Phase 4 - Analysis and judgement

- Compare development OOF and holdout results without hiding the generalisation gap.
- Analyse Spring and the largest confusion routes.
- Test declared slices and robustness conditions.
- Report calibration, risk-coverage as a diagnostic, latency, model size, memory, and prediction failures.
- Use deterministic error selection and clearly label Grad-CAM as a non-causal diagnostic.
- State whether the model is appropriate for catalogue decision support, not just whether its accuracy is high.

### Phase 5 - Assessment 3 evidence pack

Assessment 3 is directly connected to Assignment 2. The Task 2 notebook supplies the evidence, but it does not by itself satisfy the whole 40% assessment.

Presentation timing:

| Time | Content | Task 2 evidence |
|---|---|---|
| 0:00-2:00 | Problem, EDA, literature | Season ambiguity, class imbalance, shortcut risks, related papers |
| 2:00-5:00 | Approach and comparisons | B0 -> B1 -> CNN families -> C2 -> I2 incremental selection |
| 5:00-8:00 | Results and critical analysis | OOF, holdout, class errors, calibration, robustness, cost, limitations |
| 8:00-10:00 | Extension and future work | low-light/shift-aware review system with a measurable evaluation protocol |

The presentation must include a logical literature argument, not an annotated list. For Task 2, use:

- Seo et al. (PLOS One 2025): same broad Fashion Product Images source but different target, resolution, pretrained and multimodal setup; use as a boundary comparison, not a score benchmark.
- Kolisnik et al. (Expert Systems with Applications 2021): same dataset family and hierarchical prediction; useful for visual/contextual fashion classification, not a direct Season benchmark.
- Ferreira et al. (KDD 2018): structured multi-task fashion classification on a different Farfetch dataset; motivates auxiliary related targets and explains domain mismatch.
- Multimodal Sequential Fashion Attribute Prediction (Information 2019): includes Season as an attribute on different Rakuten data; motivates why Season is contextual and why direct score comparison is invalid.
- Guo et al. (ICML 2017) and Ovadia et al. (NeurIPS 2019): justify calibration and the warning that confidence can fail under dataset shift.
- Koh et al. (ICML 2021, WILDS): justify independent shift-aware evaluation and slice reporting.
- Field and Welsh (JRSS-B 2007): justify clustered/family bootstrap instead of treating related product images as independent.

At least two additional peer-reviewed papers beyond the supplied project references must be explained in relation to the project. Each paper needs: problem, method, useful idea, limitation, and exact connection to Task 2.

### Proposed extension for the 20-point Extension criterion

Problem formulation: build a human-in-the-loop Season tagging service that detects low-light or out-of-distribution catalog images, sends uncertain cases to a reviewer, and logs corrections for future monitoring. The current evidence motivates this extension because brightness 0.85 severely damages I2 performance and Spring recall.

Proposed method:

1. Add an image-quality/OOD gate using brightness, blur, and confidence features.
2. Route high-risk cases to a human reviewer instead of forcing a Season label.
3. Collect reviewed labels in a versioned queue; do not silently retrain.
4. Periodically compare clean, shifted, and reviewer-corrected slices.

Success criteria:

- Macro-F1 and Spring recall on a future time-based or source-based test set.
- Coverage-risk curve at a predeclared review budget.
- Calibration under brightness/blur shift.
- Reviewer override rate and average review time.
- No increase in false confidence on low-quality images.

## 6. Commit sequence

1. `docs(plan): preserve task2 evaluation and assessment3 roadmap`
2. `test(task2): define evaluation preflight and baseline boundary`
3. `feat(task2): add frozen evaluation preflight`
4. `test(task2): define label-free prediction receipt`
5. `feat(task2): add holdout prediction receipt`
6. `test(task2): define one-shot scoring contract`
7. `feat(task2): add immutable holdout scoring evidence`
8. `test(notebook): define Kai task2 evaluation structure`
9. `docs(notebook): add Kai task2 evaluation notebook`
10. `docs(assessment3): add task2 literature and extension notes`
11. `test(task2): audit evaluation artifacts and notebook replay`
12. `docs(task2): hand off holdout judgement to shared Notebook 06`

Do not squash these commits. Do not commit protected raw labels or unreviewed prediction dumps unless the group explicitly approves the submission package.

## 7. Definition of done

- The owner notebook is replayable and contains no unresolved TODOs for Task 2.
- Holdout predictions cover exactly the eligible holdout IDs before labels are exposed.
- I2 is compared with B0 on holdout; B1 is included only when its receipt was frozen before unlock.
- Primary, secondary, per-class, confusion, calibration, bootstrap, slice, robustness, failure, cost, and coverage evidence exists.
- Every claim points to an artifact, run ID, or hash.
- The notebook never calls a new split or `train_test_split`.
- No model, temperature, threshold, or metric changes after holdout scoring.
- The final judgement includes evidence, limitations, intended use, and human-review boundary.
- Assessment 3 has a four-part timed story, at least two additional peer-reviewed papers, an extension formulation, a success metric plan, a short demo script, and interview questions with model answers.
- Shared Notebook 06 receives only the compact Task 2 summary and links to the owner evidence.

## 8. Key references

- [Assessment 3 PDF](C:/Users/Khoai/Downloads/Assessment%20Task%203_%20Presentation%20%26%20Technical%20Interview%20%2840%25%29_.pdf)
- [Assignment 2 rubric](../../../rubrics/RUBRIC.md)
- [Task 2 execution report](../../task2-season-execution-report.md)
- [Shared final-evaluation scaffold](../../../notebooks/06_final_evaluation.ipynb)
- [PLOS One ResNet-BERT fashion classification](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0324621)
- [Condition-CNN fashion hierarchy](https://www.sciencedirect.com/science/article/pii/S0957417421006291)
- [Structured output fashion classification](https://arxiv.org/abs/1806.09445)
- [Multimodal sequential fashion attribute prediction](https://www.mdpi.com/2078-2489/10/10/308)
- [Calibration under dataset shift](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)
- [WILDS distribution-shift benchmark](https://proceedings.mlr.press/v139/koh21a.html)
