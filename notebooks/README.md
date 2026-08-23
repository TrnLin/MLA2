# Notebooks

Notebooks tell the investigation story. Reusable code lives in `src/fashion/`.

## Reading order

| # | Notebook | Status | Purpose |
|---|---|---|---|
| 00 | `00_problem_definition.ipynb` | complete | users, task boundaries, risks, and success dimensions |
| 01 | `01_data_preparation.ipynb` | complete and executed | teacher audit, sole split, five folds, and development-only evidence |
| 02 | `02_task1_article_type.ipynb` | planning scaffold | article-type comparisons and judgement |
| 03 | `03_task2_season.ipynb` | planning scaffold | season comparisons and judgement |
| 04 | `04_task3_gender_usage.ipynb` | planning scaffold | separate gender and usage outputs |
| 05 | `05_task4_visual_search.ipynb` | planning scaffold | Top-K search choices and comparisons |
| 06 | `06_final_evaluation.ipynb` | locked scaffold | one holdout evaluation and ultimate judgement |

Notebooks 02–06 contain Markdown only. They make no model, metric, loss, sampler,
transform, or Task 4 protocol choice. Each `TODO(owner)` belongs to the task owner.

## Shared rules

- Load only `data/processed/splits.csv` through the shared APIs.
- Choose one fixed `cv_fold` or all five folds before experiments.
- Fit learned preprocessing only on the training folds of each round.
- Keep holdout and quarantine targets sealed until Notebook 06.
- Train submitted models from scratch.
- Write every run to `results/runs.csv` through `fashion.train.registry`.
- Tasks 1–3 use `data/raw/teacher` images.
- Task 4 owns query size, image size, optional external images, query/gallery rules,
  relevance, K, the index, and ranking evaluation.

## Notebook 01

Notebook 01 is the official shared preparation workflow. Cached mode is the default.
Use full mode only after teacher inputs change:

```bash
FASHION_DATA_PREPARATION_MODE=full ./.venv/bin/python -m jupyter lab
```

It hashes raw bytes before decode, reconciles exact ID sets, controls duplicate and
family leakage, validates five folds, describes development labels and images, and
writes report evidence. Every code result is followed by a short finding.

Notebook 01 does not open holdout targets. It does not read external images. It does
not select a transform, model, metric, or retrieval protocol.
