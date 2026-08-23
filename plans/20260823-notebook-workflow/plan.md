---
title: "Notebook workflow restructuring"
description: "Build a protected development/holdout workflow with five reusable CV folds and teacher-only shared preparation."
status: in-progress
priority: P1
effort: 10h
branch: data-processing
tags: [docs, notebooks, data, reproducibility]
created: 2026-08-23
---

# Notebook workflow restructuring

## Goal

Preserve the sealed holdout and quarantine IDs. Merge the old train and validation rows into one
development set, add five group-safe folds, and rebuild the notebooks so another teammate can use the
prepared data without guessing or leaking protected outcomes.

## Sources of truth

- `docs/COSC2753_2026B_Assignment 2.pdf`
- `rubrics/RUBRIC.md`
- `AGENTS.md` and `docs/decisions/`
- `notebooks/01_data_preparation.ipynb`
- `C:/Users/Khoai/RMIT/Machine_Learning/Machine Learning at RMIT/ASM2/notebooks/01_Data_preparation.ipynb`
- `C:/Users/Khoai/RMIT/Machine_Learning/Machine Learning tổng/`

## Phases

| # | Phase | Status | Commit boundary | Link |
|---|---|---|---|---|
| 1 | Correct problem definition | Pending | `docs(notebooks): correct problem definition scope` | [phase 01](./phase-01-problem-definition.md) |
| 2 | Rebuild shared data preparation | In progress | data contracts, teacher-only workflow, then notebook | [phase 02](./phase-02-data-preparation.md) |
| 3 | Refresh future task handoffs | Pending | `docs(notebooks): refresh task workflow scaffolds` | [phase 03](./phase-03-task-scaffolds.md) |
| 4 | Verify and reconcile project contracts | Pending | tests, handoff, and final plan sync | [phase 04](./phase-04-verification.md) |

## Notebook reading order

1. `00_problem_definition.ipynb`
2. `01_data_preparation.ipynb`
3. `02_task1_article_type.ipynb`
4. `03_task2_season.ipynb`
5. `04_task3_gender_usage.ipynb`
6. `05_task4_visual_search.ipynb`
7. `06_final_evaluation.ipynb`

## Permanent rules

- Final submitted models train from scratch.
- `data/processed/splits.csv` remains the only split.
- Holdout and quarantine targets stay sealed during development.
- Raw files remain unchanged.
- Reusable logic stays in `src/fashion/`; notebooks explain, call, and audit it.
- The canonical partitions are `development`, `holdout`, and `quarantine`.
- Every development row has one group-safe `cv_fold` from 0 to 4.
- Descriptive EDA may use development; learned preprocessing must fit on fold-training rows only.
- Shared preparation reads teacher data only. External images are an optional Task 4 decision.
- No shared resize, crop, padding, augmentation, or normalization policy is selected here.
- Every future training run writes through the shared run registry.
- Task notebooks may add only evidence-justified preprocessing differences.

## Dependencies

- Phase 1 can be completed before the data migration.
- Phase 2 owns split migration, protected loaders, teacher-only preparation, artifacts, and Notebook 01.
- Phase 3 starts after the new `development` and `cv_fold` contract is stable.
- Phase 4 runs after all files and generated evidence are current.
