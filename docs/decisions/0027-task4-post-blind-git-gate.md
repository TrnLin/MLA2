# 0027 — Task 4 post-blind git gate

- Status: Accepted
- Date: 2026-09-09

## Context

The first real Task 12 scoring attempt stopped safely before creating
`unlock_attempt.json` or reading the raw labels. The scorer rejected the current
commit because the prediction receipt names source commit
`bf64d2b8e350f29d658eb1496eaec40a3048f836`, while Task 11 had correctly
committed the six blind receipt-bound files as evidence commit
`42ed6c371e04ab5ffeb180ceca0d3321629072a3`.

The exact-commit check contradicted the documented Task 11 to Task 12 workflow:
Task 11 had to commit the blind evidence before Task 12 could score it.

## Decision

Before any unlock marker or raw-label read, the scorer requires a clean tree, a
valid receipt commit that is an ancestor of `HEAD`, and a narrow allowlist for
every path changed since that receipt commit. Clean means no tracked change and
no nonignored untracked file; ignored data, model, environment, and cache files
remain allowed. The allowlist contains only the exact hashed blind artifacts
and prediction receipt, this scorer and its state auditor, their focused test,
and this decision.

After renewed user approval, the controller creates the fixed ref
`refs/tags/task4-holdout-scoring-approved-v1` at the exact reviewed scoring
commit. The initial gate requires that ref to exist and dereference exactly to
the current `HEAD`, pinning the full reviewed tree without a self-referential
commit hash. This tag is a procedural approval marker. It is not signed and is
not a cryptographic signature.

Production scoring must run with isolated Python so `PYTHON*` environment
variables, user site packages, and current-directory Python shadows cannot
alter imports. The approved command is:

```text
./.venv/bin/python -I scripts/build_task4_final_evaluation.py score --evaluation-unlocked
```

The receipt hashes and byte counts remain authoritative. Any model, config,
split, encoder, metric, protocol, analysis, bootstrap, blind-prediction, CLI, or
other path change still fails closed before label access. The current git state,
blind source commit, and sorted post-blind path list are copied into the unlock
attempt, unlock receipt, and evaluation manifest. The fixed scoring approval
ref and its dereferenced commit are copied there too. The auditor requires all
three records to agree.

This changes no evaluation term, ranking, model, metric, condition, bootstrap,
or analysis. The failed command did not unlock labels and is not an evaluation
run.

## Consequences

A new explicit user approval is required before another Task 12 scoring attempt.
After `unlock_attempt.json` is created, no source change or retry is allowed,
even if scoring later fails.
