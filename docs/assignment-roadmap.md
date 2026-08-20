# MLA2 Assignment Roadmap

**Status:** Not started
**Due:** 12 September 2026 at 23:59

This file gives the work order only. It does not choose methods. Record choices
and their evidence in `docs/decisions/`.

## Sources of truth

- Assignment requirements: `docs/COSC2753_2026B_Assignment 2.pdf`
- Marking priorities: `rubrics/RUBRIC.md`
- Accepted choices: `docs/decisions/`
- Repository setup: `README.md`

## Progress

- [ ] Phase 0 — Inspect the data and record data decisions
- [ ] Phase 1 — Build trusted training data and one shared split
- [ ] Phase 2 — Build shared training and evaluation tools
- [ ] Phase 3 — Compare classification approaches
- [ ] Phase 4 — Compare visual search approaches
- [ ] Phase 5 — Evaluate final choices and analyse failures
- [ ] Phase 6 — Generate and check official predictions
- [ ] Phase 7 — Build the app, report, and submission package

## Phase goals

### Phase 0 — Understand the data

Check metadata, labels, images, missing values, duplicates, class balance, and
possible leakage. Save useful evidence and record decisions before cleaning data.

### Phase 1 — Prepare trusted data

Build rebuildable manifests and one shared split under `data/processed/`. Add
checks that protect the official test set and keep related images together.

### Phase 2 — Build shared experiment tools

Create one path for loading data, training, evaluation, checkpoints, inference,
and run records. Pass a small smoke run before large experiments.

### Phase 3 — Compare classifiers

Compare genuinely different approaches for `articleType`, `season`, `gender`,
and `usage`. Record timing, model size, per-class results, and failure examples.

### Phase 4 — Compare visual search

Build and compare Top-K retrieval approaches. Evaluate both relevance and speed.

### Phase 5 — Judge final choices

Freeze choices before final evaluation. Compare runs, study failures, and explain
trade-offs honestly. Use published work only with clear notes about differences
in data and evaluation.

### Phase 6 — Create predictions

Generate the required prediction CSV and check its columns, order, IDs, labels,
and missing values.

### Phase 7 — Package the work

Build the web app, write the report, and test the whole submission in a clean
environment.

## Permanent rules

- Train submitted models from scratch.
- Use one shared split for every task.
- Record every training run.
- Keep raw data unchanged.
- Keep reusable logic out of notebooks.
- Prefer useful comparisons and honest failure analysis over one extra run.
