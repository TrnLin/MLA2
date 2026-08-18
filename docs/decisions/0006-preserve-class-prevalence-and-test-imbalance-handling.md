# 0006 — Preserve Class Prevalence and Test Imbalance Handling

**Status:** Accepted  
**Date:** 2026-08-19

## Context

The targets are naturally imbalanced. Tshirts are 17.6% of article type, while Casual is
76.8% of labelled usage. Thirty-two article-type classes have fewer than ten products, and
the Home usage class has one. Removing majority rows or generating fake minority products
would change the catalogue distribution without creating genuine product diversity.

## Decision

Preserve the natural class prevalence in validation and catalogue-holdout partitions. Keep
all valid training products and do not rebalance the dataset by majority deletion or
generative images.

Treat imbalance as a controlled model comparison:

1. compare ordinary training with one imbalance-aware method under the same split, seed,
   model, and training budget;
2. for usage, compare cross-entropy with focal loss and include an always-Casual 76.8%
   accuracy sanity baseline;
3. do not heavily oversample tiny classes, which would mostly teach memorisation;
4. report macro-F1, per-class recall, support, confusion, calibration, and support-band
   results rather than accuracy alone; and
5. mark classes without enough validation support as unsupported.

## Consequences

- Evaluation remains representative of the supplied catalogue.
- Imbalance mitigation must prove its value rather than being assumed to help.
- Rare-class limitations are reported with evidence and are not used as a general excuse
  for weak results.
- No method is claimed to make singleton classes reliably learnable.

## Evidence

See [`../eda-problem-review.md`](../eda-problem-review.md),
`results/figures/eda/target-skew.csv`, and
`results/figures/eda/article-type-support.csv`.
