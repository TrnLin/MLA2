# 0008 — Aspect-preserving image contract

- Status: Superseded by 0015
- Date: 2026-08-21

## Context

The old 128×128 transform turns a normal 60×80 image into 96×128 content, so
4,096 of 16,384 pixels are padding. Padding also biased the old statistics.

## Decision

Use 96×128 tensors, written as `(height, width) = (128, 96)` in code. Preserve
aspect ratio for unusual sizes. Fit RGB mean and standard deviation on real
transformed content pixels from train only. After standardization, set any padding
to zero, the channel mean.

## Evidence

- Common 60×80 images use 0% padding instead of 25%.
- Each float32 RGB tensor uses 144 KiB instead of 192 KiB: 25% fewer pixels.
- On 512 deterministic training images over five runs, the chosen transform was
  1.06 times as fast on the reviewed machine.
- Train-only content statistics cover 26,992 images. Dataset-wide padding share
  is 0.0058% because only 17 source images have unusual dimensions.

## Consequences

Training, evaluation, prediction, retrieval, and the app must share this contract.
No registered training run exists yet, so there is no honest validation macro-F1
comparison. The choice is based on exact geometry/cost and the development-only
speed check; revisit it through registered runs if model evidence disagrees.
