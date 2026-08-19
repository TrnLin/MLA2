# MLA2 Assignment Roadmap

**Status:** Active  
**Due:** 12 September 2026 at 23:59  
**Internal model deadline:** 5 September 2026

This is the shared progress tracker. It stays high-level. Each phase gets a separate,
detailed plan only when that phase starts.

## Sources of truth

- Assignment requirements: `docs/COSC2753_2026B_Assignment 2.pdf`
- Marking priorities: `rubrics/RUBRIC.md`
- Dataset findings: `docs/eda-problem-review.md`
- Accepted choices: `docs/decisions/`
- Repository setup: `README.md`

## Progress

- [x] Phase 0 — EDA and dataset decisions
- [ ] **Phase 1 — Trusted training data (active)**
- [ ] Phase 2 — Shared training and evaluation system
  - [ ] Minimum evaluation framework ready
  - [ ] Shared smoke run passed
  - [ ] Classifier input-size pilot complete
- [ ] Phase 3 — Parallel classification investigations
- [ ] Phase 4 — Visual search
- [ ] Phase 5 — Final evaluation and retraining
- [ ] Phase 6 — Official predictions
- [ ] Phase 7 — Web application, report, and submission

## Schedule

- **19–23 Aug:** clean data, build the manifest, and create the shared split
- **24–25 Aug:** build the shared training and evaluation framework, pass a smoke run, and
  complete the classifier input-size pilot
- **26 Aug–1 Sep:** run Tasks 1–4 in parallel and collect report evidence
- **2 Sep:** freeze final model choices
- **3 Sep:** run the untouched catalogue-holdout and outside-data evaluations
- **3–4 Sep:** retrain selected classifiers on all allowed internal training products
- **5 Sep:** generate and validate the official prediction CSV
- **6–11 Sep:** finish the web application, report, packaging, and clean-environment test
- **12 Sep:** submit

## Phase outcomes

### Phase 1 — Trusted training data

Implement accepted cleanup decisions using targeted review. Produce the reviewed correction
and masking table, `data/processed/train_manifest.csv`, and the only allowed split,
`data/processed/splits.csv`.

Before any cleanup or label access, load the official test IDs from
`data/raw/teacher/test/styles_prediction.csv` and exclude those IDs from the original
metadata and image population. The original dataset's labels for those IDs stay quarantined.

The development population uses an 80/20 training/validation split. Official-training IDs
46,919–51,999 remain an untouched catalogue-shift holdout. Exact duplicate groups and both
image views of each product stay in one partition. Commit the shared split.

**Complete when:** all counts reconcile, processed files rebuild exactly, test labels remain
quarantined, and split-safety checks pass.

### Phase 2 — Shared experiment system

Before real model training starts, build one common path for loading paired image views,
training, evaluation, checkpoints, inference, and run registration. The minimum evaluation
framework records comparable metrics, per-class support, sharp- and blurry-view results,
saved predictions, timing, and model size. Training may randomly use either the original
sharp view or the blurry teacher view of one product; the two views never become two product
rows.

Every run appends to the committed `results/runs.csv`. The four task owners separate only
after one train-save-load-evaluate smoke run passes. A controlled article-type pilot chooses
one classifier input size; blurry-view macro-F1 is the primary selection result.

**Complete when:** the smoke run is reproducible and uses the shared split.

### Phases 3 and 4 — Parallel investigations

Run the four assignment lanes in parallel:

1. article type;
2. season;
3. separate gender and usage classifiers; and
4. Top-K visual search.

Submitted classifiers are trained from scratch. Pretrained models are comparison benchmarks
only. Save metrics, figures, timing, and failure examples during each run even though report
prose is written later.

**Complete when:** each target and visual search has reproducible comparison evidence and a
frozen final candidate.

### Phase 5 — Final evaluation and retraining

Freeze choices before viewing the catalogue holdout. Evaluate it once and report unsupported
holdout classes honestly. Use a small, separately labelled Fashionpedia set as an outside
robustness check. Select permissively licensed images, crop one clear garment, use two
reviewers, and mask unresolved target disagreements.

After evaluation, retrain the selected classifier designs on all 38,612 allowed internal
training products using settings fixed during development.

Report note: after this retrain, no internal data is left to test the final model. All
reported metrics come from the pre-retrain development models. The report must state this
trade-off explicitly.

**Complete when:** every final choice has independent evidence and a deployment reason.

### Phase 6 — Official predictions

Predict from the supplied blurry `data/raw/teacher/test/images_test/` images only. Do not use
official test data for model selection.

Generate exactly `id,gender,articleType,season,usage`, with 5,829 unique ordered IDs, valid
labels, no blanks, and no extra columns.

**Complete when:** the checked prediction CSV is reproducible on 5 September.

### Phase 7 — Application and submission

Use the frozen models in a simple web interface with predictions, honest confidence
information, and Top-K similar products. Finish the five-page report, package all required
files, and rehearse the submission in a clean environment.

The existing public link for the original image dataset and its preparation steps must be
recorded so the marker can reproduce training.

**Complete when:** the application and package run from the README and all three Canvas
uploads are ready.

## Non-negotiable rules

- Train submitted models from scratch.
- Use `data/processed/splits.csv` as the only split.
- Record every run through `fashion.train.registry` in `results/runs.csv`.
- Exclude official test IDs from the original dataset before exposing any target columns.
- Never use official test labels or official test images for model selection.
- Keep raw data unchanged and reusable logic out of notebooks.
- Run Python with `./.venv/bin/python`.
- Prefer useful comparisons and honest failure analysis over one extra training run.
