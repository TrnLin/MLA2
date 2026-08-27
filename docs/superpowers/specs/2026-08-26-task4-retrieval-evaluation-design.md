# Task 4 Retrieval Evaluation Design

- Status: Approved in conversation
- Date: 2026-08-26

## Goal

Freeze one model-independent evaluation contract before Task 4 preprocessing or model
experiments begin. The contract must compare broad fashion similarity, same-family
recovery, runtime cost, and failures without exposing the sealed holdout.

## Constraints

- `data/processed/splits.csv` is the only split.
- Development data is used for model selection. Holdout remains sealed until Notebook 06.
- Quarantine rows and official teacher-test products never enter development queries,
  galleries, fitting, or model selection.
- V1 is an image variant keyed by product ID, not independent data.
- Retrieval is scored per product ID. Multiple image variants must collapse to one ranked
  product before Top-K scoring.
- Metric code accepts labelled frames. It must not unlock protected labels itself.
- Every candidate uses the same frozen fold and evaluation rules.

## Fold policy

The repository seed is `2753`. The recorded draw
`random.Random(2753).choice(range(5))` selected fold `1`, which is now frozen rather
than redrawn at runtime.

For ordinary candidate comparison:

- Train on development folds `0`, `2`, `3`, and `4`.
- Use development fold `1` as the unseen validation fold.
- Never rotate the validation fold between epochs. A model remembers data from earlier
  epochs, so rotation would contaminate later validation scores.

After candidate comparison, take the two configurations with the highest Protocol A
mean nDCG@10 and retrain them from scratch with each of the other folds held out in turn.
Report the unweighted mean and standard deviation across all five fresh runs. These
stability runs do not change the predeclared candidate comparison rules.

## Protocol A: broad similarity

Protocol A is the primary model-selection test.

- Queries: every eligible row in validation fold `1`.
- Gallery: every eligible row in the other four development folds.
- Relevance grade `2`: same `articleType` and same `baseColour`.
- Relevance grade `1`: same `articleType` and different `baseColour`.
- Relevance grade `0`: different `articleType`.

This is a metadata proxy, not human similarity ground truth. It mainly measures whether
the model groups clothing type and colour sensibly.

The split must make query and gallery disjoint by `id`, `sha256`, `duplicate_group`, and
`product_family_group`. Any crossing is a contract failure and stops evaluation. The
fold allocation is the real leakage defence; runtime assertions guard against future
split or protocol changes.

### Protocol A metrics

The primary score is mean per-query nDCG@10. DCG uses linear gains equal to the relevance
grade, so a grade-2 match is worth twice a grade-1 match:

`DCG@K = sum(grade_at_rank / log2(rank + 1))`

`nDCG@K = DCG@K / ideal_DCG@K`

The ideal ranking uses all eligible positives available in that query's gallery and is
truncated at K. Report results at K values `5`, `10`, and `20`, with nDCG@10 declared as
the winner-selection score.

Also report:

- Precision@K for grade `>= 1` matches.
- Precision@K for strict grade `2` matches.
- A macro result formed by averaging query scores within each present `articleType`,
  then averaging those class means.

The per-query mean chooses the winner. The class macro is supporting evidence and does
not silently replace the primary result.

## Protocol B: same-family recovery

Protocol B is supporting evidence and does not enter a custom combined winner score.

- Queries: eligible rows in unseen validation fold `1`.
- Per-query gallery: other rows from the same unseen fold.
- Relevant products: the same `product_family_group`.
- Remove the query's own ID before ranking.
- Remove candidates sharing the query's `sha256` or `duplicate_group` so exact duplicates
  cannot make the test trivial.
- Exclude queries with no remaining family positive and report their count and coverage.

Because the model was not trained on fold `1`, this tests family recovery on unseen
products. Product families never cross folds, so this question cannot be measured by
Protocol A.

Report Recall@10 as the main family score, plus Hit Rate@10 and Precision@10.

## Undefined queries and coverage

A query is excluded from a metric average only when that metric has no possible positive:

- Protocol A: ideal DCG is zero.
- Protocol B: no eligible same-family candidate remains after exclusions.

Do not assign these queries a score of zero. Every result must report total queries,
scored queries, excluded queries, zero-grade-2 queries, and queries with fewer than K
strict positives where relevant.

## Ranking contract

- Apply all per-query exclusions before taking Top-K.
- Collapse image variants to one result per product ID before taking Top-K.
- Sort by ascending distance, then ascending numeric product ID.
- Require unique product IDs in each ranked list.
- Report the tie rate because simple baselines may produce many equal distances.

## Operational and failure evidence

For each evaluated method, record:

- End-to-end and search-only latency at batch size one, including p50 and p95.
- Hardware, warm-up policy, and number of timed queries.
- Index bytes, embedding dimension, parameter count, and checkpoint bytes.
- Whether p95 end-to-end latency is below one second and the index is below 1 GB.

Predeclared failure slices are rare `articleType` classes, grayscale images, unusual
image geometry, and rare `articleType`-`baseColour` pairs. Overall results must not hide
these slices.

## Final evaluation

After all choices are frozen:

1. Refit the selected method on all development data.
2. In Notebook 06 only, unlock the labelled holdout through the approved data API.
3. Use holdout products as Protocol A queries and development products as the gallery.
4. Evaluate once and make no model or protocol changes after seeing the result.

Protocol B is not available for final holdout because product families do not cross the
development-holdout boundary. Published-work comparison remains necessary and must state
metric and dataset differences rather than claim false like-for-like results.

## Software boundaries

`src/fashion/retrieval/protocol.py` will own pure, model-independent view construction,
exclusion checks, relevance grading, coverage, and ranking metrics. It will receive data
frames and ranked product IDs as inputs and will not read protected labels or train
models.

`tests/test_retrieval_protocol.py` will define the contract with small synthetic frames,
including leakage failures, exact-duplicate removal, metric calculations, undefined-query
coverage, ID collapse, and deterministic ties.

The Task 4 notebook will explain the frozen rules and show evidence produced by the
module. Existing scaffold tests that require owner decisions to remain open must be
updated in the same change because those decisions are now closed.

