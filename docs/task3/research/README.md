# Task 3 research plan: gender and usage classification

Status: research and planning only. No model result is claimed here.

Task 3 predicts two catalogue labels from each teacher image:

- `gender`: 5 classes
- `usage`: 9 classes

The recommended path is to build strong separate scratch-trained models first, then test one
matched shared-backbone model with two heads. Use all five saved family-safe folds. Select models
from pooled out-of-fold evidence, freeze the full method, refit on all development rows, and open the
independent holdout once.

## Executive recommendation

1. Keep `gender` and `usage` as separate outputs. Do not make a combined 45-way label.
2. Start with majority, sanity, HOG-plus-colour, and small-CNN baselines.
3. Compare scratch MobileNetV3-Small with a scratch ResNet-18 using a low-resolution stem.
4. Compare `(height=80, width=60)` with `(height=128, width=96)`.
5. Start with unweighted cross-entropy. Compare one capped class-balanced loss after the clean
   baseline exists.
6. Treat the truly blank usage row with a loss mask. Treat literal `usage="NA"` as a real class.
7. Keep `usage="Home"` in the output. Its one development example is not enough to claim
   generalisation.
8. Compare two separate backbones with one shared backbone and two heads under matched conditions.
9. Accept sharing only when neither task is harmed beyond the predeclared margin and practical cost
   falls enough.
10. Use pretrained systems only as clearly marked comparison benchmarks. They are not eligible for
    the submitted model, official predictions, or application.

The assignment rewards comparison breadth, justified decisions, and honest failure analysis more
than one extra tuning run. See the [rubric](../../../rubrics/RUBRIC.md).

## Evidence language

Each document separates three kinds of statements:

- **Repository fact:** evidence already present in this project.
- **Sourced fact:** evidence from a standard, paper, or official technical guide.
- **Recommendation:** a proposed Task 3 decision based on those facts.

Recommendations are not recorded as results. A model choice becomes a result only after the
controlled experiments are run and registered.

## Reading order

| Order | File | Purpose |
|---:|---|---|
| 1 | [Problem and lifecycle](01_problem_and_lifecycle.md) | Defines the task, users, label meaning, ethics, life cycle, and top-level success gates. |
| 2 | [Data and validation design](02_data_and_validation_design.md) | Fixes the data boundary, masks, leakage rules, preprocessing fit scope, and five-fold protocol. |
| 3 | [Model choice](03_model_choice.md) | Gives the detailed model decision tree and reasoning for every major model and loss family. |
| 4 | [Experiment plan](04_experiment_plan.md) | Turns the model questions into controlled, staged runs with budgets and stopping rules. |
| 5 | [Evaluation framework](05_evaluation_framework.md) | Defines metrics, aggregation, uncertainty, calibration, efficiency, selection gates, and holdout rules. |
| 6 | [Error, robustness, and ethics](06_error_robustness_and_ethics.md) | Predeclares failure slices, image perturbations, review rules, uncertainty use, and ethical limits. |
| 7 | [Final selection and deployment](07_final_selection_and_deployment.md) | Defines the method freeze, final refit, checkpoint lock, holdout unlock, predictions, and monitoring. |
| 8 | [Reproducibility and artifacts](08_reproducibility_and_artifacts.md) | Lists the run, prediction, evidence, environment, and handoff artifacts that must exist. |
| 9 | [References](references.md) | Consolidates primary sources and explains what each source supports. |

## Phase map

| Phase | Main output | Primary file |
|---|---|---|
| Frame | Intended use, forbidden use, owners, risks | [01](01_problem_and_lifecycle.md) |
| Understand data | Counts, masks, folds, family boundary, shortcut risks | [02](02_data_and_validation_design.md) |
| Fix measurement | Frozen metrics, comparisons, gates | [05](05_evaluation_framework.md) |
| Establish baselines | Dummy, classical, small CNN | [03](03_model_choice.md), [04](04_experiment_plan.md) |
| Develop | Architecture, resolution, augmentation, imbalance, sharing | [03](03_model_choice.md), [04](04_experiment_plan.md) |
| Diagnose | Errors, calibration, uncertainty, robustness, cost | [05](05_evaluation_framework.md), [06](06_error_robustness_and_ethics.md) |
| Freeze | Selected run IDs, fixed method, hashes | [07](07_final_selection_and_deployment.md) |
| Independently evaluate | All-development refit and one holdout opening | [07](07_final_selection_and_deployment.md) |
| Hand off | Official CSV, application checks, model card, monitoring | [07](07_final_selection_and_deployment.md), [08](08_reproducibility_and_artifacts.md) |

## Non-negotiable project rules

- Use [`data/processed/splits.csv`](../../../data/processed/splits.csv) as the only split.
- Keep holdout and quarantine target values sealed until the final evaluation notebook.
- Use teacher images only for Tasks 1–3.
- Fit models, normalisation, class weights, sampling, and learned calibration without holdout data.
- Preserve the separate official columns `gender` and `usage`.
- Preserve the target masks. A blank label is missing; literal `NA` is not missing.
- Train submission-eligible models from random initialisation.
- Register every training execution in the required run registry once it exists.
- Refit the frozen method on all development rows before the one-time holdout unlock.
- Do not tune, switch models, change metrics, or change thresholds after unlock.

These rules come from the [assignment specification](<../../COSC2753_2026B_Assignment 2.pdf>),
[accepted decisions](../../decisions/README.md), and the locked
[final-evaluation notebook](../../../notebooks/06_final_evaluation.ipynb).

## Current readiness

**Repository facts.** Data preparation and the protected split boundary are ready. The Task 3 and
final-evaluation notebooks are still scaffolds. The project currently has no `results/runs.csv`, no
`src/fashion/train/` package, and no pinned deep-learning or classical-ML dependencies in
[`pyproject.toml`](../../../pyproject.toml).

**Recommendation.** Before the first real training run, create the registry and shared evaluation
contract, then pin the chosen framework and its environment. The first evidence-producing run
should be the dummy and leakage-sanity suite, not a large CNN.

## Source index

External claims are cited near the relevant text. The consolidated list is in
[references.md](references.md).
