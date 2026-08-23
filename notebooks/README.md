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

The five scaffold notebooks contain Markdown only. They contain no completed training code,
result, exact metric choice, model choice, transform choice, or fake claim. Their
`TODO(owner)` fields show what the assigned teammate must decide and justify.

## Shared rules for every task

- `data/processed/splits.csv` is the only split and must be read through protected loaders.
- `development` contains five precomputed folds. The task owner records one fixed fold or all five
  before running experiments.
- Holdout and quarantine targets stay sealed until Notebook 06. Holdout is opened once after all
  choices are frozen.
- Submitted models are trained from scratch. Pretrained systems are separate benchmarks only.
- Tasks 1–3 use teacher images. Task 4 owns any decision about another image collection.
- Task-specific preprocessing and exact metrics are chosen and frozen inside the task notebook.
- Learned preprocessing values are fitted on the training folds of each round, not all development.
- Every training run appends to `results/runs.csv` through the shared registry.
- Figures and tables used by the report trace to run IDs instead of hand-written numbers.

## Notebook 01 contract

`01_data_preparation.ipynb` is the one official shared data-preparation workflow. It is intentionally
self-contained: a fresh-kernel **Run All** validates or rebuilds the shared data contract, performs
development-only analysis, and saves report evidence and figures. Its job includes raw SHA-256
hashing before decode, exact ID reconciliation, duplicate/family/quarantine control, the sole split,
five CV folds, protected-label checks, class-support evidence, NMI, image-quality descriptions, and
transform-risk examples.

Code stays visible for auditability. Every code cell displays its result immediately and is followed
by a short finding or definition-only note. Cached validation is the default. Set
`FASHION_PREPARATION_MODE=full` for the slow child-process rebuild after teacher inputs change.

Notebook 01 does not fit model statistics or select image size, crop, padding, loss, sampler, model,
metric, or retrieval protocol. `development_image_profile.json` is descriptive and has
`allowed_for_model_fit: false`.

## Task and final-evaluation boundary

Notebooks 02–05 reuse shared preparation and add only task-specific hypotheses, preprocessing
comparisons, metric decisions, registered experiments, validation error analysis, efficiency evidence,
and provisional judgements. Notebook 05 also owns query size, query/gallery, relevance, Top-K, index,
and ranking-evaluation decisions. Notebook 06 receives frozen run IDs and opens protected evaluation
exactly once. It reports the result honestly and never tunes after the unlock.
