# 0009 — Task 4 retrieval isolation

- Status: Accepted
- Date: 2026-08-21

## Context

Using evaluation queries inside the search index creates trivial self-matches and
leaks product-family variants.

## Decision

For scored development evaluation, use validation rows as queries and train rows
as the only gallery. Remove matches sharing query ID, SHA-256, or product-family
group before Top-K. Labels may score retrieved items but never enter the embedding
model as inputs.

Use validation queries from the supported metric slice. Use graded proxy
relevance: 2 for the same official article type and base colour, 1 for the same
official article type with a different colour, and 0 otherwise. Report nDCG@5
over all three grades. Recall@5 treats only grade 2 as relevant. Its denominator
is every distinct grade-2 product in the allowed train gallery; its numerator is
the distinct grade-2 products in the first five product IDs. Queries with zero
grade-2 positives are excluded from the macro mean and reported as coverage.

Both image variants may be encoded. Their scores collapse to their shared product
ID before Top-K, using the better variant score. The same product can occupy only
one slot and contributes only one scored item. Under the corrected 100-class,
train-fitted supported slice, 5,768 of 5,781 validation products are eligible.
The 13 excluded products are outside the supported article-type slice only; final
evaluation reports them separately as a coverage/failure slice rather than
silently omitting them. Current supported-query coverage is 110
zero-grade-2 queries and 412 queries with fewer than five grade-2 positives. The
old 116/422 figures came from the prohibited all-partition-fitted taxonomy.

A separate final application gallery may include all catalogue data permitted by
the assignment only after final evaluation is complete. It is never used to claim
development performance.

## Evidence

The protocol builder asserts train-only gallery membership, complete low/high
pairs, product-level deduplication, and zero query/gallery intersection by ID,
SHA-256, and product-family group. Tests cover zero, one, and fewer-than-K
positives plus disagreeing variant scores. They also reject opening the final
gallery before evaluation is marked complete.

## Consequences

The metadata score is only a proxy for human visual similarity. The report must
state that limitation and compare it with reviewed qualitative results.
