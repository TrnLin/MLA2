# 0003 — Protected EDA and train-only statistics

- Status: Accepted
- Date: 2026-08-20

## Context

Exploring holdout targets before the final judgement weakens independent evaluation.
Using validation, holdout, or prediction images for fitted normalization also leaks
information into model preparation.

## Decision

Use only the `train` partition for target, image, and normalization evidence. Use
`train` and `val` only for development diagnostics, with classes selected from
training and percentages divided by each partition's valid-label count. Never
aggregate holdout or quarantine target outcomes in EDA. Never use prediction image
properties in modelling EDA.

## Why

This keeps the final holdout estimate honest and makes unequal partition sizes safe
to compare. It also gives every future model the same fitted preprocessing inputs.

## Consequences

Some rare training classes can be absent from validation and must be reported as a
coverage limit. Protected labels become available only during the locked final
evaluation stage, by joining the evaluator's local raw teacher CSV. They are blank
in the persisted split and absent from the lean prepared pack.

## Evidence

- `rubrics/RUBRIC.md` requirement for justified and independent evaluation.
- Scope and provenance checks inside `notebooks/00_eda.ipynb`.
- Training-ID digest in the main `data/processed/paired_normalization.json` and the
  comparison baseline `data/processed/normalization_original_only.json`.
