# 0002 — Treat High- and Low-Resolution Files as Paired Views

**Status:** Accepted  
**Date:** 2026-08-17

## Context

The original and teacher image with the same ID show the same product. The original file is
sharp and high-resolution; the assignment train and test files are mostly 60×80 thumbnails.
Training only on sharp images may create a resolution mismatch at test time.

The two files are not independent products. Counting them as separate rows, or placing them
in different partitions, would duplicate a product and leak validation information.

## Decision

Represent each official train ID once in the training manifest, with:

- one label set;
- a path to the original high-resolution image; and
- a path to the teacher low-resolution image.

Assign the ID to a split before loading either image. Both views must always remain in the
same partition.

For the paired-view experiment:

1. randomly choose the high- or low-resolution view during training;
2. resize both to the model's fixed input size;
3. optionally apply mild blur and JPEG compression;
4. evaluate sharp and low-resolution validation views separately; and
5. use low-resolution macro-F1 as the primary model-selection result.

## Consequences

- The model can learn robustness to sharp and blurry inputs.
- Products are not double-counted.
- Validation scores cannot be inflated by putting two versions of one product on opposite
  sides of the split.
- Low-resolution-only training remains the required baseline for comparison.

## Evidence

The same-ID audit compared 512 pairs. Median dHash distance was 0, every pair was within
distance 6, and resized-image median PSNR was about 43.5 dB. See
[`../dataset-quality-comparison.md`](../dataset-quality-comparison.md).
