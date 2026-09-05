# 0021 — Task 4 high-resolution input

- Status: Accepted
- Date: 2026-08-27
- Supersedes: size choice in 0020

## Context

Decision 0020 selected `96×128` because it ranked first under the fixed,
untrained HSV-and-edge probe. That probe was useful for a cheap comparison, but
it cannot test whether a learned model benefits from fine texture, trim, logos,
or other small V1 details.

The probe evidence remains important and is not rewritten:

- `96×128`: `0.49247873`
- `60×80`: `0.49198675`
- `240×320`: `0.48987763`

Thus `240×320` ranked third on the equal teacher/V1 same-source probe score.
Its teacher→teacher score was strongest, but its V1→V1 and cross-source scores
were weaker. The holdout remained sealed while reconsidering this choice.

## Decision

Use RGB tensors with `(height, width) = (320, 240)`, written as `240×320` in
width-by-height displays.

Keep every other part of decision 0020's image contract:

- apply EXIF orientation;
- composite transparency onto white and convert to RGB;
- resize with LANCZOS while preserving aspect ratio;
- centre the result on white padding and retain a content mask;
- cache development pixels losslessly as `uint8`;
- scale to `[0, 1]` at model input;
- fit source-specific mean and standard deviation on the current training folds
  only, then reuse those saved values for validation and queries.

This is an explicit policy override, not a claim that `240×320` won the probe.
The project prioritizes retaining V1 detail for the learned-model study. The
fixed probe winner, rank, and score remain recorded beside the deployed
contract.

ADR 0019's retrieval metrics, fold, and winner rules do not change.

## Consequences

Each float32 RGB input is `900 KiB`, 6.25 times the `144 KiB` used by
`96×128`. Each selected-size development cache is about 7 GiB per source.
Training and query encoding will therefore be slower and use more memory.

Teacher images begin at `60×80`, so upsampling them cannot create detail. The
extra resolution mainly preserves information available in V1 and arbitrary
high-resolution user queries. Later evidence must report whether learned
models use that detail enough to justify the cost.

The known large-background failure remains. Letterboxing is deterministic, but
it does not locate a small garment inside a large canvas.

The holdout remains sealed until Notebook 06. Any further size change requires
another decision record and fresh development evidence.

## Artifacts

- `results/evidence/task4/preprocessing_comparison.csv`
- `results/evidence/task4/preprocessing_size_selection.csv`
- `results/evidence/task4/preprocessing_stability.csv`
- `results/evidence/task4/preprocessing_robustness.csv`
- `results/evidence/task4/preprocessing_contract.json`
- `results/evidence/task4/preprocessing_normalization_fold1.json`
- `results/figures/task4/preprocessing_comparison.png`
