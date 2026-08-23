# Notebooks

Use notebooks to tell the investigation story and show evidence. Reusable logic stays in
`src/fashion/`.

## Reading order and status

| # | Notebook | Owner | Status | Purpose |
|---|---|---|---|---|
| 00 | `00_problem_definition.ipynb` | shared | complete | users, task contracts, boundaries, success criteria, and risks |
| 01 | `01_data_preparation.ipynb` | shared data owner | complete and executed | raw audit through the shared model-ready data contract |
| 02 | `02_task1_article_type.ipynb` | Task 1 teammate | planning scaffold | article-type experiments and provisional judgement |
| 03 | `03_task2_season.ipynb` | Task 2 teammate | planning scaffold | season experiments, ambiguity, and shortcut analysis |
| 04 | `04_task3_gender_usage.ipynb` | Task 3 teammate | planning scaffold | separate gender and usage decisions plus shared-model comparison |
| 05 | `05_task4_visual_search.ipynb` | Task 4 teammate | planning scaffold | retrieval experiments, coverage, cost, and provisional judgement |
| 06 | `06_final_evaluation.ipynb` | group evaluation owner | locked scaffold | one independent holdout evaluation and ultimate judgement |

The five scaffold notebooks contain no completed training code, result, exact metric choice, or model
claim. Their `TODO(owner)` fields show what the assigned teammate must decide and justify.

## Shared rules for every task

- `data/processed/splits.csv` is the only split and must be read through protected loaders.
- Holdout and quarantine targets stay sealed until Notebook 06.
- Submitted models are trained from scratch. Pretrained systems are separate benchmarks only.
- Both image variants share one product ID and never count as independent evidence.
- Task-specific preprocessing and exact metrics are chosen after baseline evidence and frozen before
  fair comparison.
- Every training run appends to `results/runs.csv` through the shared registry.
- Figures and tables used by the report trace to run IDs instead of hand-written numbers.

## Notebook 01 contract

`01_data_preparation.ipynb` is the one official shared data-preparation workflow. It is intentionally
self-contained: a fresh-kernel **Run All** validates or rebuilds the shared data contract, performs
focused train-only analysis, and saves report evidence and figures. Its job includes raw SHA-256
hashing, image and metadata reconciliation, duplicate and family control, the sole split,
protected-label checks, shared transforms, normalization, and the Task 4 query and gallery boundary.

Code stays visible for auditability but is collapsed by default. Every code cell has an immediately
preceding guide that is removed from the code-free HTML report. Cached validation is the default. Use
`FASHION_DATA_PREPARATION_MODE=full` for the slow forensic rebuild after raw inputs change.

## Task and final-evaluation boundary

Notebooks 02–05 reuse shared preparation and add only task-specific hypotheses, preprocessing
comparisons, metric decisions, registered experiments, validation error analysis, efficiency evidence,
and provisional judgements. Notebook 06 receives frozen run IDs and opens protected evaluation exactly
once. It reports the result honestly and never tunes after the unlock.
