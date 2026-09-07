# Task 3 research plan: gender and usage classification

Status: research and planning only. No model result is claimed here.

Task 3 predicts two catalogue labels from each teacher image:

- `gender`: 5 classes;
- `usage`: 9 classes.

The development process is evidence-led. It starts from the established EDA, defines one primary
learnable baseline, diagnoses that baseline completely, and only then permits one child experiment
with one main changed factor. The same loop repeats separately for `gender` and `usage`.

## Executive recommendation

1. Keep `gender` and `usage` as separate outputs. Do not make a combined 45-way label.
2. Keep majority and stratified predictors as lower bounds and integrity checks, not as the primary
   learnable baseline.
3. Keep HOG-plus-colour logistic regression as the classical comparison reference.
4. Use one exact small scratch CNN design as the primary learnable baseline for both targets,
   trained as two separate models.
5. After every complete candidate, diagnose training curves, pooled out-of-fold metrics, per-class
   behaviour, fixed failure examples, cost, and a small fixed robustness suite.
6. Before creating a child configuration, write the observed weakness, evidence paths, hypothesis,
   one changed factor, expected result, and rejection condition.
7. Run one child for one target at a time. Accept it as the new parent, reject it and return to the
   old parent, or stop when the limitation is mainly in the data or labels.
8. Treat architecture, resolution, augmentation, imbalance, optimiser, and sharing methods as a
   conditional option library. None belongs in a fixed model screen.
9. Test a shared two-head model only when separate-model cost is a measured problem and a matched
   comparison is possible.
10. Use pretrained systems only as clearly marked comparison benchmarks after the eligible
    development chain is fixed. They are never eligible for the submitted model, official
    predictions, or application.

The assignment rewards comparison breadth, justified decisions, and honest failure analysis more
than one extra weakly motivated run. Sequential work still compares multiple algorithms and
techniques; it makes the reason for each comparison visible. See the
[rubric](../../../rubrics/RUBRIC.md).

## Required experiment loop

```text
established EDA
  -> lower bounds and classical reference
  -> one primary learnable baseline
  -> complete diagnostic bundle
  -> written parent-child hypothesis
  -> one child with one main changed factor
  -> accept, reject, or stop
  -> repeat only from the accepted parent
```

No child model configuration may be created until the parent's decision record exists. No result
may be used to justify a change that was written only after the child had already run.

## Roles of the initial methods

| Role | Method | What it decides | May become a child parent? |
|---|---|---|---|
| Lower bound | Fold-training majority predictor | Exposes the accuracy illusion | No |
| Sanity lower bound | Fold-training stratified predictor | Checks class order, masks, metrics, and seeds | No |
| Integrity check | Bias-only, shuffled-label, and tiny-batch tests | Detects broken training or leakage | No |
| Shortcut diagnostic | Ground-truth `articleType` majority lookup | Shows catalogue association risk; it is not a deployable input | No |
| Classical reference | HOG plus HSV colour and logistic regression | Measures hand-built visual features | Comparison reference |
| Primary learnable baseline | Exact small scratch CNN, trained separately per target | Starts the parent-child development chain | Yes |

## Evidence language

Each document separates three kinds of statements:

- **Repository fact:** evidence already present in this project.
- **Sourced fact:** evidence from a standard, paper, or official technical guide.
- **Recommendation:** a proposed Task 3 decision based on those facts.

Recommendations are not results. A hypothesis becomes an experiment only after its parent evidence
and decision record are saved. A model choice becomes a result only after the controlled run is
registered and its evidence bundle exists.

## Reading order

| Order | File | Purpose |
|---:|---|---|
| 1 | [Problem and lifecycle](01_problem_and_lifecycle.md) | Defines the task, users, label meaning, ethics, life cycle, and top-level success gates. |
| 2 | [Data and validation design](02_data_and_validation_design.md) | Fixes the data boundary, masks, leakage rules, preprocessing fit scope, and five-fold protocol. |
| 3 | [Model choice](03_model_choice.md) | Defines method roles, the exact primary baseline, and the conditional child option library. |
| 4 | [Experiment plan](04_experiment_plan.md) | Defines the parent-diagnose-hypothesise-child-decision loop and its gates. |
| 5 | [Evaluation framework](05_evaluation_framework.md) | Defines metrics, repeated diagnostic bundles, comparisons, selection gates, and holdout rules. |
| 6 | [Error, robustness, and ethics](06_error_robustness_and_ethics.md) | Defines per-candidate and finalist failure, robustness, review, and ethical work. |
| 7 | [Final selection and deployment](07_final_selection_and_deployment.md) | Defines method freeze, final refit, checkpoint lock, holdout unlock, predictions, and monitoring. |
| 8 | [Reproducibility and artifacts](08_reproducibility_and_artifacts.md) | Defines registry, hypothesis, prediction, decision, environment, and handoff artifacts. |
| 9 | [References](references.md) | Consolidates primary sources and explains what each source supports. |

## Phase map

| Phase | Main output | Exit condition |
|---|---|---|
| Frame | Intended use, forbidden use, owners, risks | Problem and safety contract fixed |
| Understand data | Counts, masks, folds, family boundary, shortcut and transform risks | EDA-derived baseline choices recorded |
| Build evidence system | Registry, configuration, OOF, metric, error, robustness, and cost contracts | Tests pass before any training |
| Establish anchors | Lower-bound and classical-reference evidence | Pipeline and comparison anchors valid |
| Run primary baseline | Separate small-CNN OOF runs for both targets | Complete registered baseline evidence exists |
| Diagnose one target | Curves, metrics, classes, failures, cost, and robustness | One dominant weakness or a stop reason is written |
| Test one hypothesis | One child with one main changed factor | Child evidence is complete |
| Decide | Accept, reject, or stop | Decision record points to evidence and parent |
| Confirm finalists | Five folds and three fixed seeds with the method frozen | Training-randomness evidence complete |
| Freeze and independently evaluate | All-development refit and one holdout opening | Frozen independent report exists |
| Hand off | Official CSV, application checks, model card, monitoring | Handoff checklist passes |

## Non-negotiable project rules

- Use [`data/processed/splits.csv`](../../../data/processed/splits.csv) as the only split.
- Keep holdout and quarantine target values sealed until the final evaluation notebook.
- Use teacher images only for Tasks 1–3.
- Fit models, normalisation, class weights, sampling, and learned calibration without holdout data.
- Preserve the separate official columns `gender` and `usage`.
- Preserve the target masks. A blank label is missing; literal `NA` is not missing.
- Keep `Home` in the output. Never use its single row to choose a child model.
- Train submission-eligible models from random initialisation.
- Register every training execution through `fashion.train.registry` in `results/runs.csv`.
- Refit the frozen method on all development rows before the one-time holdout unlock.
- Do not tune, switch models, change metrics, or change thresholds after unlock.

These rules come from the [assignment specification](<../../COSC2753_2026B_Assignment 2.pdf>),
[accepted decisions](../../decisions/README.md), and the locked
[final-evaluation notebook](../../../notebooks/06_final_evaluation.ipynb).

## Current readiness and implementation order

**Repository facts.** Data preparation and the protected split boundary are ready. The Task 3 and
final-evaluation notebooks are still scaffolds. The project currently has no `results/runs.csv`, no
`src/fashion/train/` package, and no pinned deep-learning or classical-ML dependencies in
[`pyproject.toml`](../../../pyproject.toml).

Implement in this order:

1. Pin the selected learning dependencies and environment.
2. Implement the run registry, configuration hashing, eligibility assertions, and parent-child
   experiment contract.
3. Implement fold-safe labels, masks, preprocessing, OOF predictions, metrics, and artifact
   validation.
4. Implement lower-bound predictors and the exact classical reference.
5. Implement only the exact small-CNN baseline and its shared training engine.
6. Implement the repeated curve, class, failure, cost, and core-robustness diagnostic bundle.
7. Run and diagnose the baseline.
8. Implement only the one child selected by that evidence. Do not pre-implement a model zoo.
9. Keep Notebook 04 narrative: it calls shared code, reads artifacts, and records decisions.

The first evidence-producing training run should be the tiny-batch integrity check, not a large
CNN. The first full five-fold neural evidence should be the primary small-CNN baseline.

## Source index

External claims are cited near the relevant text. The consolidated list is in
[references.md](references.md).
