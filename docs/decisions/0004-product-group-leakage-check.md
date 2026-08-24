# 0004 - Product-family and near-duplicate split policy

- Status: Superseded by `0011-auditable-family-review-boundary.md`
- Date: 2026-08-22

## Context

Exact hashes miss recompressed images, alternate views, and colour variants. dHash also creates many
white-background collisions. Exact normalized product names are useful blocking signals but do not
prove that rows show one product.

## Superseded decision

Build families before the sole split:

1. Generate dHash-distance-0-to-2 candidates across labelled and official prediction images.
2. Score full-canvas and foreground-cropped pixels with the frozen automatic rule.
3. Quarantine every cross-role automatic visual component and each conflicting exact-hash group.
4. On remaining rows, union exact hashes, accepted internal visual pairs, and the normalized name.
5. Allocate each whole family through `data/processed/splits.csv`.

The name key is a conservative split block only. It never changes a label. ADR 0011 removes all
human inputs from this pipeline and records the current family-unit cost.
