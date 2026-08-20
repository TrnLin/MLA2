# 0004 — Use Review-Driven Cleaning and Target-Specific Masking

**Status:** Accepted  
**Date:** 2026-08-19

## Context

The usable training population contains missing target values, literal `usage == "NA"`,
scope questions, heuristic label-review candidates, and a small number of clear metadata
errors. Automatically deleting rows or rewriting labels would discard useful images and
could replace one uncertain label with another.

## Decision

Keep raw files immutable. Record every accepted exclusion, correction, and target mask in
the versioned processed review table defined by decision 0003.

Apply these rules:

1. keep readable fashion and beauty products; exclude only visually confirmed non-fashion
   or unusable products, including the blank image at ID `44998`;
2. treat 20 blank season labels, one blank usage label, and 71 literal `"NA"` usage values
   as target-specific missing labels;
3. keep affected products for other valid targets and visual search;
4. correct clear errors, including ID 45824 to `Footwear → Shoes → Flats` and ID 38223 to
   `Sunglasses`;
5. review gender candidates in related product groups and mask gender when evidence stays
   uncertain; and
6. review uncommon hierarchy mappings without forcing defensible overlaps into one path.

Do not infer corrections from product-name rules alone.

## Consequences

- Different targets can use different row counts without deleting products globally.
- The cleaned article-type vocabulary falls from 124 to 123 after correcting the sole
  `Ipad` row to the existing `Flats` class.
- Cleanup remains auditable and reversible because raw data never changes.
- Manual review is required before the processed review table is built.

## Evidence

See [`../eda-problem-review.md`](../eda-problem-review.md) and the generated evidence under
`results/figures/eda/`.
