# Phase 01 - Complete the problem definition notebook

## Ownership

- Create `notebooks/00_problem_definition.ipynb`.
- Add structural tests for its required sections if needed.
- Do not move or edit the current EDA notebook in this phase.

## Required sections

- Executive summary.
- Real-world context, users, and decisions supported.
- Prediction unit, model input, and four assignment tasks.
- Data roles and protected evaluation boundary.
- Success criteria and metrics per target.
- Constraints, non-goals, assumptions, and risks.
- End-to-end system flow and assignment deliverables.
- Readiness gate for data preparation.

## Content rules

- Explain Task 3 as separate `gender` and `usage` outputs.
- Explain product IDs versus image-variant inputs.
- State that the specification sets no arbitrary accuracy threshold.
- Use macro-F1 and per-class recall for classification development.
- Use nDCG@5, Recall@5, coverage, and latency for visual search.
- Include model size, latency, robustness, and failure analysis in the final judgement criteria.
- Keep the notebook narrative and runnable from a fresh kernel without prepared data.

## Todo

- [ ] Create the notebook with complete headers and subheaders.
- [ ] Validate notebook JSON and heading hierarchy.
- [ ] Check every statement against the assignment, rubric, and accepted decisions.
- [ ] Commit the completed phase separately.

## Success criteria

- A reader can state the input, output, unit, user, metric, constraint, and final-evaluation rule
  without opening another notebook.
- No data preparation, EDA, or model training is duplicated here.

