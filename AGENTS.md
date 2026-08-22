# MLA2 — COSC2753 Assignment 2 (Fashion Intelligence)

Group ML project over ~38.6k 60×80 fashion product images: four classification targets
(`articleType`, `season`, `gender`, `usage`) plus a Top-K visual search system.

## What the grade rewards

`rubrics/RUBRIC.md` scores Approach (50), Ultimate Judgement (30), Report (20).
**No criterion scores accuracy.** Marks come from comparison breadth, justified choices,
and honest analysis of failures. Given a choice between one more training run and better
analysis of the runs already done, choose the analysis.

## Hard constraints

- **Train models from scratch.** Pretrained weights belong in the comparison benchmarks
  only — a submitted model carrying `pretrained=True` violates the spec.
- **`data/processed/splits.csv` is the only split.** Every notebook, the search index, and
  the app read it. A `train_test_split` call anywhere else invalidates cross-model
  comparison and leaks evaluation images into the Task 4 index.
- **Every training run appends a row to `results/runs.csv`** through
  `fashion.train.registry`. The report's comparison tables are generated from that file.
- **`data/raw/teacher/test/styles_prediction.csv` format is fixed**:
  `id,gender,articleType,season,usage`.
  This is why Task 3 predicts gender and usage as separate targets.

## Conventions

- `notebooks/00_eda.ipynb` is the official EDA exception: its analysis and plot code stays
  visible in the notebook so one Run All tells the whole story. Reusable dataset, split,
  image-variant, training, and evaluation contracts stay in `src/fashion/`.
- Later notebooks stay narrative and import reusable logic from `src/fashion/`.
- Write a figure to `results/figures/` when the report will cite it.
- Use `./.venv/bin/python`.

## How to talk to me

Talk to me like I'm 5. Small words, short sentences, short paragraphs. If a big word is
needed, explain it right after. Only return what's actually necessary.

Just tell me what you did, did it work, what do I do now.

If I have to decide something: 2 options max, the context I need to pick fast, and which
one you'd go with.

Keep paths and commands exact.

## Where to look

- `rubrics/RUBRIC.md` — marking bands and HD checklists. Read when scoping work or
  deciding what to cut.
- `docs/COSC2753_2026B_Assignment 2.pdf` — the spec. Read for deliverables, submission
  format, and naming conventions.
