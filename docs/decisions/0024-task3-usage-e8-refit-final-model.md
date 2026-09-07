# 0024 — Freeze the Task 3 Usage E8 refit

- Status: Accepted
- Date: 2026-09-07
- Supersedes: [0022](0022-task3-usage-e1-final-model.md), the final Usage artifact
- Amends: [0014](0014-development-holdout-cv-boundary.md), the recorded acceptance timing

## Context

The owner selected the completed E8 refit as the final Usage model. E8 uses
class weighting and small translations. Its original five-fold development
comparison led the nine teacher-only experiments on macro-F1, which gives each
class equal weight. E1 led on accuracy. The expanded-data and later MixUp/SAM
trials did not establish a better teacher-only macro-F1 alternative.

## Decision

Freeze the single scratch model from run
`t3_usage_e8_translation_teacher_all_development_refit_5553be0c138243e9`.
The authoritative handoff is
`reports/task3/usage_final_e8_refit_20260907/model_manifest.json`.
Its checkpoint SHA-256 is
`18da75a4ec935ab0d18c9ebdf9d1dabb6fa87ee484ead5f439de0192c4af2747`.

The model was freshly trained on all 32,772 eligible teacher development rows
from `data/processed/splits.csv`. It uses one 391,209-parameter SmallCNN,
30 fixed epochs, seed 2753, batch 128, AdamW at 0.001, weight decay 0.0001,
and cosine decay to 0.00001. Epoch 30 is final, not a best epoch chosen later.
It uses effective-number weighted cross-entropy (beta 0.999, cap 5) and
translation by up to two pixels with probability 0.5 during training.
Class weights and normalization were fitted on full development only.

For inference, apply saved RGB preprocessing at height 80 and width 60 and
the refit's own saved normalization. Use evaluation mode, then softmax and
argmax. Do not augment images, multiply probabilities by training class weights,
or average the earlier fold checkpoints. Preserve the class order:
`Casual, Ethnic, Formal, Home, NA, Party, Smart Casual, Sports, Travel`.

Notebook 4 holds development analysis, refit training details and the frozen
recipe. Reserved-holdout scores, errors and final evaluation analysis belong
in `04_task3_final_evaluation.ipynb`. The teacher test set is prediction-only in the submitted work;
the teacher did not supply its labels. Do not include private external-reference
test scores in either notebook or the final report.

## Why

This accepts equal-class performance as the main Usage objective, with accuracy
as a visible trade-off. Original E8 reached 41.94% development macro-F1 versus
E1's 37.38%, while accuracy was 88.85% versus 89.30%. These are OOF results from
the earlier fold models. They are not validation scores for the full-development
refit, which has no validation split or early stopping.

The completed refit uses one model and all eligible development rows. The
reserved-holdout comparison and its small-class limitations are recorded in
[Task 3 evaluation](../../notebooks/04_task3_final_evaluation.ipynb). Acceptance followed
evaluation review; it must not be presented as a newly blind selection or a
first opening of the holdout. This record does not backdate the choice.

## Consequences

E1 and the earlier E8 fold models remain historical comparisons. The old E1
manifest and the refit's original training-only manifest keep their exact bytes.
This later acceptance manifest records the chosen model separately. Gender's
current decision and all other task boundaries remain unchanged.

Rare Usage labels, overlapping occasion meanings, one training seed and photo
sensitivity still limit practical use. The result supports catalogue suggestions
with human review, not reliable automatic assignment of every occasion.
One forward pass replaces the old five-model inference recipe; no application
speedup is claimed without a matched latency benchmark.

The eventual official output must retain `id,gender,articleType,season,usage`.
A Usage-only CSV is not the full submission. This freeze does not deploy the
application or generate a new submission.

## Evidence

- [Main development notebook](../../notebooks/04_task3_gender_usage.ipynb)
- [Reserved-holdout analysis](../../notebooks/04_task3_final_evaluation.ipynb)
- [Accepted model manifest](../../reports/task3/usage_final_e8_refit_20260907/model_manifest.json)
- [Completed E8 training notebook](../../notebooks/task3_training/usage_e8_refit.ipynb)
- [Original E8 training contract](../task3_usage_e8_refit.md)
- Assignment: train from scratch and preserve the official prediction format.
