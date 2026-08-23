# Phase 03 - Add future task notebook handoffs

## Ownership

- Create markdown-first scaffolds for notebooks `02` through `06`.
- Update `notebooks/README.md` with reading order, ownership, and notebook status.
- Do not implement training, tuning, retrieval models, holdout evaluation, or official predictions.

## Shared scaffold pattern

- Status and owner handoff.
- Task contract and hypothesis placeholders for the teammate to complete.
- Shared data/preprocessing inputs reused from Notebook 01.
- Recommended task-specific preprocessing or sampling options, with a blank decision and rationale.
- Primary/supporting metric selection checklist, with no final metric preselected.
- Baseline and comparison matrix placeholders.
- Training registry and checkpoint contract.
- Validation metrics and error analysis.
- Efficiency, robustness, and limitations.
- Provisional judgement and next actions.

## Task-specific focus

- Task 1: prompts for long-tail taxonomy, supported classes, imbalance options, and rare-class errors.
- Task 2: prompts for weak visual signal, article-type shortcut risk, calibration, and ambiguity.
- Task 3: prompts for separate versus shared representations, label masks, loss weights, and negative transfer.
- Task 4: prompts for query/gallery isolation, embeddings, variant fusion, relevance coverage, and latency.
- Final evaluation: freeze run IDs, unlock holdout once, independent comparison, deployment judgement,
  prediction and app handoff.

## Todo

- [x] Create all five scaffold notebooks with clear headers and TODOs.
- [x] State that scaffolds contain no completed model claims.
- [x] Validate notebook JSON and heading order.
- [x] Commit each scaffold in a separate reviewable commit.

## Success criteria

- A teammate can own one notebook without guessing its scope, inputs, outputs, metrics, or stop rule.
- No future notebook duplicates shared preparation code or accesses protected targets early.
