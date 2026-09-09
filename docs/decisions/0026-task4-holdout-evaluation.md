# 0026 — Task 4 holdout evaluation

- Status: Accepted
- Date: 2026-09-08

## Context

Decision 0019 defines Protocol A and Protocol B for development fold `1` only.
It says that, after every choice is frozen, the selected method is refit on all
development data and Notebook 06 opens the labelled holdout once. It does not
define the holdout queries, galleries, directions, comparators, stress
conditions, uncertainty method, or evidence boundary.

Decision 0023's Consequences section makes the refit stricter: Notebook 06 must
refit R5 on all development folds as a registered `final_refit` run before the
one holdout evaluation. The missing holdout protocol and the refit requirement
must both be resolved in writing before the holdout is unlocked.

## Decision

### Holdout retrieval protocols

This decision extends decision 0019's two development-fold protocols to the
holdout. Protocol A is the primary broad-similarity evaluation. All 5,778
holdout products are queries against a teacher-image gallery containing all
32,773 development products. Relevance grades stay unchanged from decision
0019: grade `2` for the same `articleType` and `baseColour`, grade `1` for the
same `articleType` with a different colour, and grade `0` otherwise. The
primary score is mean per-query linear nDCG@10. Results are also reported at K
values `5`, `10`, and `20`, with broad Precision@K, strict Precision@K, tie
rate, and an `articleType` class macro as supporting evidence. The query mean,
not the class macro, is primary. All 61 quarantine rows are excluded from every
query and gallery.

Protocol B is supporting family-recovery evidence. Its queries and raw gallery
are both the holdout set. Before ranking, each query excludes its own `id` and
candidates sharing its `sha256` or `duplicate_group`. Relevant candidates share
`product_family_group`. Recall@10 is primary, with Hit Rate@10, Precision@10,
tie rate, and coverage also reported. Protocol B does not enter a combined
winner score. The holdout contains 4,110 product families; 705 contain more than
one row, and the holdout families have zero overlap with development families.

A query is excluded from a metric average only when that metric is undefined:
Protocol A has zero ideal DCG, or Protocol B has no eligible family positive
after exclusions. Undefined queries are excluded and counted; they are never
assigned zero.

### Query source directions

Two query directions are evaluated against the same all-development teacher
gallery: holdout teacher queries and holdout V1 queries. All 5,778 holdout IDs
have a V1 image.

### Comparators

The five pre-declared rows are the submitted scratch R5 autoencoder, a seeded
random-ranking floor, the untrained `spatial-hsv-edge-4x4-v2` probe, HOG plus
HSV-edge fusion, and B1 pretrained ResNet18. B1 is a benchmark ceiling only and
is never a candidate.

This decision narrowly supersedes decision 0023's Decision clause saying that
B1 is “permanently ineligible for final refit, holdout judgement, and
submission” only for holdout scoring. B1 may be scored on the holdout as a
labelled benchmark-only ceiling row. B1 remains permanently ineligible for
final refit, submission, and any selection decision.

### Query conditions

Seven conditions are applied to the query image only and only in the teacher
direction: `clean`, `jpeg_quality_85`, `gaussian_blur_radius_1`,
`brightness_0_85`, `brightness_1_15`, `wide_canvas`, and `tall_canvas`. The
gallery is never degraded.

### Uncertainty

Uncertainty uses a family-blocked paired bootstrap over the 4,110 holdout
families, with 10,000 replicates and seed `2753`. One shared draw is used across
all methods. Report a 95% percentile interval for R5 Protocol A nDCG@10 and a
paired interval for R5 minus each comparator. No p-value is reported.

### No final refit

This decision supersedes decision 0023's Consequences clause requiring Notebook
06 to refit R5 on all development folds and create a registered `final_refit`
run. The submitted model is the existing fold-1-trained R5 already exported to
`models/task4_r5`.

There are three reasons:

1. The rubric scores no accuracy metric, so additional training data buys no
   marks.
2. An all-development refit has no held-out fold, so `train_epochs` has no
   milestone score and `select_best_checkpoint` has nothing to rank. Nine
   independent validators (`TrainingSessionConfig.__post_init__`,
   `learned_data._validate_validation_fold`,
   `CrossSourcePairDataset.__init__`, `validate_training_loader`,
   `cache.fit_cached_fold_rgb_statistics`,
   `Task4RunRegistry._validate_row`'s fold bound, the milestone scorer, and five
   fold assertions in `gallery_artifact.py`) would need loosening for one
   unrepeatable run.
3. Refitting RGB statistics on all development changes the input distribution,
   so the refit model would not be the artifact described by the existing
   five-fold stability evidence.

The exact recorded cost is that 6,556 fold-1 development products, 20.0% of the
32,773 development rows, were used for epoch selection and are never used for
training. This is a permanent limitation.

### One holdout unlock

Notebook `notebooks/task-4/10_task4_part3_final_evaluation.ipynb` is replay-only. The
single unlock happens through
`scripts/build_task4_final_evaluation.py score --evaluation-unlocked` and never
again. This decision supersedes decision 0023's Holdout boundary statement that
“The holdout stays sealed until Notebook 06” only as to the unlock mechanism:
the audited script breaks the seal, and Notebook 06 replays the sealed result.

## Why

One frozen contract makes the final score independent by construction: the
model, data views, metrics, comparators, conditions, and uncertainty method are
declared before labels are opened, and the result cannot flow back into any
choice. The rubric scores judgement, not accuracy, so a clear independent test
and honest limits matter more than an unsupported final refit. The rubric also
scores comparison breadth, and a pretrained ceiling row makes the scratch
model's holdout number interpretable only when B1 is scored on the same holdout.

## Consequences

The holdout result cannot change the model, gallery, preprocessing, metric,
condition, threshold, or any other choice. The 6,556-product fold-1 training gap
is a permanent stated limitation.

`results/evidence/task4/final/` remains development-only.
`results/evidence/task4/final_evaluation/` contains post-unlock evidence, kept
separate so the boundary is visible.

## Evidence

- `docs/superpowers/specs/2026-09-08-task4-holdout-final-evaluation-design.md`
- `docs/superpowers/plans/2026-09-08-task4-holdout-final-evaluation.md`
- `configs/task4/holdout_final_evaluation.json`
- `scripts/build_task4_final_evaluation.py`
- `src/fashion/task4_evaluation/blind.py`
- `src/fashion/task4_evaluation/score.py`
- `src/fashion/task4_evaluation/audit.py`
- `tests/test_documentation.py`
- `tests/train/test_ranking_metrics.py`
- `tests/task4_evaluation/test_views.py`
- `tests/task4_evaluation/test_conditions.py`
- `tests/task4_evaluation/test_spec.py`
- `tests/task4_evaluation/test_blind.py`
- `tests/task4_evaluation/test_score.py`
- `tests/task4_evaluation/test_artifacts.py`
- `tests/task4_evaluation/test_notebook.py`
