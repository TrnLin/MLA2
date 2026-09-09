# 0007 — Supported deployment taxonomy

- Status: Superseded by `0010-official-output-and-supported-metrics.md`
- Date: 2026-08-21

## Context

This decision made primary metrics comparable, but incorrectly removed rare
image-backed labels from the model and submission output space. ADR 0010 keeps
the fair metric slice while restoring every trainable official label.

A class with fewer than three independent training product families is too weak
for the primary macro-F1 comparison. This rule must be fitted after allocation
from training data only; validation and protected holdout labels cannot choose it.

## Decision

The original decision excluded weak classes from outputs as well as metrics. ADR
0010 supersedes that part: keep train-observed official outputs, but mask classes
with fewer than three training families from the primary metric only. Do not merge
semantically different products just to increase support.

Fit label maps from the training partition after splitting. An unexpected label
not in the train map receives index `-1`, is masked, and is reported.

## Evidence

- The old 107/17 article-type counts were influenced by future partitions and are
  retired.
- The train-fitted replacement keeps 124 official article-type outputs, with 100
  in the primary slice and 24 reported as limited-support official outputs.
- `usage` keeps 9 official outputs and 8 primary classes. Literal `NA` remains a
  valid official class. `season` keeps 4 classes and `gender` keeps 5.
- Training-family sensitivity for threshold 2/3/4/5 keeps 105/100/98/94 article
  types. No protected holdout class-coverage audit is produced.

## Consequences

Primary macro-F1 uses the frozen train-fitted supported slice. Limited-support
official outputs remain in training and the fixed submission schema, and are
reported separately rather than silently relabelled.
