# Phase 03 - Add future task notebook handoffs

## Ownership

- Create markdown-first scaffolds for notebooks `02` through `06`.
- Update `notebooks/README.md` with reading order, ownership, and notebook status.
- Do not implement training, tuning, retrieval models, holdout evaluation, or official predictions.

## Shared scaffold pattern

- Status and owner handoff.
- Task contract and hypothesis placeholders for the teammate to complete.
- Shared teacher-data inputs and development/`cv_fold` contract reused from Notebook 01.
- Owner choice between one predeclared fold and all five folds; never choose the best fold afterward.
- Fold-local learned preprocessing rule.
- Recommended task-specific preprocessing or sampling options, with a blank decision and rationale.
- Primary/supporting metric selection checklist, with no final metric preselected.
- Baseline and comparison matrix placeholders.
- Training registry and checkpoint contract.
- Validation metrics and error analysis.
- Efficiency, robustness, and limitations.
- Provisional judgement and next actions.

## Task-specific focus

- Task 1: prompts for long-tail taxonomy, imbalance options, and rare-class errors.
- Task 2: prompts for weak visual signal, article-type shortcut risk, calibration, and ambiguity.
- Task 3: prompts for separate versus shared representations, label masks, loss weights, and negative transfer.
- Task 4: owns arbitrary query sizes, resize/crop/padding comparisons, optional external images,
  query/gallery design, embeddings, relevance, cutoff, ranking evidence, and latency.
- Final evaluation: freeze run IDs, refit on all development, lock the checkpoint, unlock holdout once,
  make an independent judgement, then prepare prediction and app handoff.

## Todo

- [x] Replace stale train/validation and paired-image contracts.
- [x] Add CV-mode, fold-local preprocessing, metric-freeze, and decision-log TODOs.
- [x] Keep all five notebooks Markdown-only with no completed model claims.
- [x] Validate notebook JSON and heading order.
- [x] Commit the scaffold refresh separately.

## Success criteria

- A teammate can own one notebook without guessing its scope, inputs, outputs, metrics, or stop rule.
- No future notebook duplicates shared preparation code or accesses protected targets early.
