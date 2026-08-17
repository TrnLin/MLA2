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
- **`data/test/styles_prediction.csv` format is fixed**: `id,gender,articleType,season,usage`.
  This is why Task 3 predicts gender and usage as separate targets.

## Dataset gotchas

Measured from the data, not assumed:

- `usage` contains the literal string `"NA"` (71 rows), distinct from blank (1 row).
  `season` has 20 blanks.
- `articleType` spans 125 classes, from Tshirts (6,781) down to classes with one example.
  Singleton classes cannot be learned or fairly evaluated — handling them is a decision
  the report must defend.
- `usage` is severely skewed: Casual 29,641, Home 1.
- `styles_train.csv` has two trailing empty columns in its header; parsers must tolerate it.
- Images are 60×80. Upscaling much past 96px adds no information, only compute.
- Verify every CSV row has a matching image file before trusting any count.

## Conventions

- Notebooks stay narrative — import from `src/fashion/`, keep logic out of cells.
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
- `docs/superpowers/specs/` — design decisions already made, with their rationale. Read
  before proposing changes to structure or workflow.
