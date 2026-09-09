# 0019 — Task 4 retrieval evaluation

- Status: Accepted
- Date: 2026-08-26

## Context

Task 4 needs one model-independent evaluation contract before preprocessing and
model experiments begin. Every candidate must face the same unseen products and
ranking rules without opening the sealed holdout. Decision 0009 established the
self-match principle, but its fixed query counts and validation-split details are
now stale.

## Decision

The candidate-comparison validation fold is fixed at fold `1`. This is the recorded
result of `random.Random(2753).choice(range(5))`; it is not redrawn at runtime.
Ordinary runs train on development folds `0`, `2`, `3`, and `4`.

Protocol A is the primary broad-similarity test. Every eligible fold-`1` product is
a query, and eligible products in the other four development folds form the
gallery. Relevance is grade `2` for the same `articleType` and `baseColour`, grade
`1` for the same `articleType` with a different colour, and grade `0` otherwise.
The winner-selection score is mean per-query linear nDCG@10. Linear means that a
grade-2 result contributes twice the gain of a grade-1 result. Results are also
reported at K values `5`, `10`, and `20`, with broad Precision@K, strict
Precision@K, and an `articleType` class macro as supporting evidence. The query
mean, not the class macro, selects the winner.

Protocol B is supporting same-family recovery evidence. It uses fold `1` as both
the query set and raw gallery because these products were unseen during training.
The query's own ID and candidates sharing its `sha256` or `duplicate_group` are
removed before ranking. Relevant products share `product_family_group`. Report
Recall@10 as the main family score, plus Hit Rate@10 and Precision@10. Protocol B
does not enter a combined winner score.

A query is excluded from a metric average only when that metric is undefined:
Protocol A has zero ideal DCG, or Protocol B has no eligible family positive after
exclusions. Such queries are not assigned zero. Report total, scored, excluded,
zero-grade-2, and fewer-than-K strict-positive query counts where relevant.

All exclusions happen before Top-K. Image variants collapse to one result per
product ID. Rankings sort by ascending distance, then ascending numeric product ID,
and report their tie rate.

The two configurations with the highest Protocol A mean nDCG@10 are rerun from
scratch with each development fold held out in turn. Report the unweighted
five-run mean and standard deviation. These stability runs do not change the
candidate-comparison contract.

After every choice is frozen, the selected method is refit on all development
data. Notebook 06 alone may unlock the labelled holdout, evaluate it once, and
must not change the model or protocol afterward.

This decision supersedes decision 0009's stale fixed query counts and retired
validation-split details. It preserves 0009's rule against retrieval self-match
leakage.

## Why

One frozen fold makes the main comparison affordable and repeatable. Five fresh
runs for only the top two candidates add stability evidence without multiplying
the full experiment matrix. Protocol A measures broad type-and-colour grouping,
while Protocol B separately checks recovery of unseen product families.

## Consequences

Protocol A query and gallery products must be disjoint by `id`, `sha256`,
`duplicate_group`, and `product_family_group`; any crossing stops evaluation.
Product families cannot cross development folds, so Protocol B must stay within
the unseen fold. Coverage and tie behaviour are reported instead of hidden.

Metadata relevance is only a proxy for human visual similarity and requires
qualitative failure review. V1 is an ID-keyed image variant of the same products,
not independent data, so V1 comparisons cannot be described as independent
validation.

## Evidence

- `docs/superpowers/specs/2026-08-26-task4-retrieval-evaluation-design.md`
- `src/fashion/retrieval/protocol.py`
- `tests/test_retrieval_protocol.py`
- `results/evidence/task4/retrieval_protocol_coverage.csv`
