# 0004 — Product-family and near-duplicate split policy

- Status: Superseded by `0011-auditable-family-review-boundary.md`
- Date: 2026-08-21

## Context

The old evidence below preserved useful candidate identities, but its manual calls
did not record a reviewer, date, tool, blindness, or second review. It must not be
quoted as completed human review. ADR 0011 replaces the operational decision.

Exact hashes alone miss recompressed images, alternate views, and colour variants.
The dHash audit also produces many white-background collisions. Exact normalized
product names cross partitions often, but review shows that this text is noisy.

## Decision

Build families before the sole split in this order:

1. Generate every dHash-distance-0-to-2 candidate over labelled and official
   prediction images. Score full-canvas and foreground-cropped pixels.
2. Accept only pairs that pass the frozen hash, MSE, MAE, and foreground-ratio
   rule. Every non-exact cross-role acceptance also needs a recorded image review.
3. Quarantine labelled rows in confirmed cross-role visual components and exact
   hashes with conflicting valid labels.
4. On remaining rows, union exact hashes, accepted internal visual pairs, and an
   exact case-folded, whitespace-collapsed product-name key.
5. Allocate each whole family to one partition in `data/processed/splits.csv`.

The name key is a conservative block only. It never proves identity and never
changes a label.

## Why

The pixel rule removes most white-background hash collisions. The broad name
block makes evaluation harder when it merges distinct designs, but prevents a
shared name family from leaking across partitions. This is safer than using the
name as an identity label.

## Consequences

The rebuilt split is approved for model comparison. Any rule, review, or family
change must rebuild the sole split, label maps, normalization, and EDA evidence.
The rule favours precision, so subtle ungrouped variants may remain.

## Evidence

- 113,593 dHash candidates: 14,849 at distance 0, 28,410 at 1, and
  70,334 at 2.
- 120 automatic-rule pairs reviewed: 119 same/variant and 1 different;
  observed precision 99.17%, Wilson 95% lower bound 95.43%. This estimates
  precision, not recall.
- All 10 non-exact cross-role rule matches reviewed: 6 accepted and 4 rejected.
- 120 normalized-name pairs reviewed: 28 same/variant, 84 different, 8 uncertain.
  This confirms that the name key is a blocking signal, not identity proof.
- Before the family policy, 2,549 non-empty normalized-name groups with more than
  one SHA crossed the superseded split: 10,959 rows and 10,788 distinct SHA values.
  The old split digest and query scope are frozen in the review evidence.
- Sole split: train 26,992; validation 5,782; holdout 5,781; quarantine 57.
- Active family, normalized-name, and exact-SHA crossings: 0, 0, and 0.
- Seed 2753 and deterministic IDs/hashes are recorded in generated evidence.
