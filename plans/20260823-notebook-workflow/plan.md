---
title: "Notebook workflow restructuring"
description: "Split the current EDA into a clear problem definition, an auditable data-preparation workflow, and task handoff notebooks."
status: in-progress
priority: P1
effort: 10h
branch: data-processing
tags: [docs, notebooks, data, reproducibility]
created: 2026-08-23
---

# Notebook workflow restructuring

## Goal

Keep the trusted preparation code and evidence. Rebuild the notebook story so a new reader sees the
problem, raw hashing, image/tag reconciliation, duplicate controls, split, train-only analysis, and
shared preprocessing in the correct order.

## Sources of truth

- `docs/COSC2753_2026B_Assignment 2.pdf`
- `rubrics/RUBRIC.md`
- `AGENTS.md` and `docs/decisions/`
- `notebooks/00_eda.ipynb`
- `C:/Users/Khoai/RMIT/Machine_Learning/Machine Learning at RMIT/ASM2/notebooks/01_Data_preparation.ipynb`
- `C:/Users/Khoai/RMIT/Machine_Learning/Machine Learning tổng/`

## Phases

| # | Phase | Status | Commit boundary | Link |
|---|---|---|---|---|
| 1 | Complete problem definition | In progress | `docs(notebooks): add problem definition workflow` | [phase 01](./phase-01-problem-definition.md) |
| 2 | Restructure data preparation | Pending | `refactor(notebooks): clarify data preparation workflow` | [phase 02](./phase-02-data-preparation.md) |
| 3 | Add future task handoffs | Pending | `docs(notebooks): add task workflow scaffolds` | [phase 03](./phase-03-task-scaffolds.md) |
| 4 | Verify and reconcile project contracts | Pending | Amend only the phase that owns any required correction | [phase 04](./phase-04-verification.md) |

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
- Two image variants remain one product-level unit.
- Every future training run writes through the shared run registry.
- Task notebooks may add only evidence-justified preprocessing differences.

## Dependencies

- Phase 1 can be completed independently.
- Phase 2 owns the rename of `00_eda.ipynb` and all direct path/test/documentation updates.
- Phase 3 starts after Phase 2 fixes the notebook contract and reading order.
- Phase 4 runs after all notebook files exist.
