# 0020 — Task 4 image preprocessing

- Status: Accepted; size choice superseded by 0021
- Date: 2026-08-27

Decision 0021 keeps this transform contract but changes the frozen size from
`96×128` to `240×320`. This record remains the evidence-backed probe decision.

## Context

Task 4 needs one input contract before the non-deep baseline and learned models
can be compared. Teacher images are `60×80`; V1 is mostly much larger. User
queries may have different modes and shapes. A choice based only on geometry
would not show whether extra pixels help retrieval.

## Decision

Use RGB tensors with `(height, width) = (128, 96)`, written as `96×128` in
width-by-height displays.

For every teacher, V1, or user image:

- apply EXIF orientation;
- composite transparency onto white and convert to RGB;
- resize with LANCZOS while preserving aspect ratio;
- centre the result on white padding;
- retain a content mask;
- cache development pixels losslessly as `uint8`;
- scale to `[0, 1]` at model input.

For learned models, fit RGB mean and standard deviation on the current round's
training folds and source only. Apply those same values to validation, query,
and gallery images; never refit an incoming query. Standardized padding is
zero, the fitted channel mean.

The comparison used a fixed, untrained `4×4` spatial HSV-and-edge probe with
equal normalized feature blocks and exact cosine ranking. It tested `60×80`,
`96×128`, and `240×320` under ADR 0019. The size score was the equal mean of
teacher→teacher and V1→V1 Protocol A query-mean nDCG@10. Teacher→V1,
V1→teacher, and Protocol B remained supporting evidence.

This equal-source mean is an explicit addition to ADR 0019, which defines the
within-direction nDCG@10 score but has no source axis. The winner is still
chosen on fixed fold `1`. The later five-fold check is supporting stability
evidence and does not reopen or change the fixed-fold size decision.

The later learned-model study still decides whether teacher or V1 supplies the
deployed index. The preprocessing probe does not decide that source.

## Evidence

On fixed fold `1`:

- `96×128`: `0.49247873`
- `60×80`: `0.49198675`
- `240×320`: `0.48987763`

The top-two five-fold check kept the same order:

- `96×128`: mean `0.49287316`, standard deviation `0.00279159`
- `60×80`: mean `0.49244586`, standard deviation `0.00265309`

The gain is small and is smaller than fold-to-fold variation. `96×128` is
nevertheless selected because the frozen rule chooses the highest primary
score. It uses 2.56 times as many pixels as `60×80`, while `240×320` uses 16
times as many and reduced the paired source mean.

Cross-source retrieval became worse as size increased. At `96×128`,
teacher→V1 scored `0.485494` and V1→teacher scored `0.483382`. This warns that
source appearance is a real deployment issue rather than proof that more V1
detail always helps.

Wide and tall white-canvas queries also failed badly. Mean nDCG@10 fell from
about `0.49` to `0.11`–`0.17`, with very low Top-10 overlap. Letterboxing
therefore defines deterministic arbitrary-size handling but does not solve
small-object or large-background queries. That failure must remain visible for
later model robustness and app guidance.

## Consequences

Historically, this decision required future Task 4 candidates to receive the
same `96×128` geometry and colour contract. Decision 0021 supersedes that size:
the active contract is now `240×320`, while the transform and leakage rules
above remain unchanged. The selected-size teacher and V1 development caches
contain 32,773 IDs each and no holdout pixels. Fold-1 normalization contains
26,217 training products per source.

The holdout remains sealed until Notebook 06. A future change to size, padding,
colour handling, or normalization requires a new decision record and fresh
development evidence; it cannot be chosen after seeing holdout results.

## Artifacts

- `results/evidence/task4/preprocessing_comparison.csv`
- `results/evidence/task4/preprocessing_size_selection.csv`
- `results/evidence/task4/preprocessing_stability.csv`
- `results/evidence/task4/preprocessing_robustness.csv`
- `results/evidence/task4/preprocessing_contract.json`
- `results/evidence/task4/preprocessing_normalization_fold1.json`
- `results/figures/task4/preprocessing_comparison.png`
