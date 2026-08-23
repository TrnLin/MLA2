# Phase 03 - Add future task notebook handoffs

## Ownership

- Create markdown-first scaffolds for notebooks `02` through `06`.
- Update `notebooks/README.md` with reading order, ownership, and notebook status.
- Do not implement training, tuning, retrieval models, holdout evaluation, or official predictions.

## Shared scaffold pattern

- Status and owner handoff.
- Task contract and hypothesis.
- Shared data/preprocessing inputs reused from Notebook 01.
- Task-specific preprocessing or sampling only when evidence justifies it.
- Baseline and comparison matrix.
- Training registry and checkpoint contract.
- Validation metrics and error analysis.
- Efficiency, robustness, and limitations.
- Provisional judgement and next actions.

## Task-specific focus

- Task 1: long-tail taxonomy, supported classes, weighted/focal loss comparison, rare-class errors.
- Task 2: weak visual season signal, article-type shortcut risk, calibration, ambiguous errors.
- Task 3: separate versus shared representation, missing-label masks, loss weights, negative transfer.
- Task 4: query/gallery isolation, product embeddings, variant fusion, relevance coverage, search latency.
- Final evaluation: freeze run IDs, unlock holdout once, independent comparison, deployment judgement,
  prediction and app handoff.

## Todo

- [ ] Create all five scaffold notebooks with clear headers and TODOs.
- [ ] State that scaffolds contain no completed model claims.
- [ ] Validate notebook JSON and heading order.
- [ ] Commit the completed phase separately.

## Success criteria

- A teammate can own one notebook without guessing its scope, inputs, outputs, metrics, or stop rule.
- No future notebook duplicates shared preparation code or accesses protected targets early.

