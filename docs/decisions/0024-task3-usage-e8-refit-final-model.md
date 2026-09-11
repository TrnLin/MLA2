# 0024 — Freeze the Task 3 Usage E8 refit

- Status: Accepted
- Date: 2026-09-07
- Development rationale clarified: 2026-09-11; artifact and original acceptance date unchanged
- Supersedes: [0022](0022-task3-usage-e1-final-model.md), the final Usage artifact
- Amends: [0014](0014-development-holdout-cv-boundary.md), the recorded acceptance timing

## Context

The owner retains the completed E8 refit as the final Usage model. The rationale
below uses development evidence to explain the accepted trade-offs: class coverage,
shift tolerance and probability quality against accuracy, lighting sensitivity and
uncertainty. It does not turn the earlier failed improvement gates into passes.
Reserved-holdout analysis belongs in the final-evaluation notebook. The owner's
private examination is not used as selection evidence in the notebook's argument.
Original dated records remain preserved. This clarification does not make a new
blind-evaluation claim.

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

The development notebook holds development analysis, refit training details and the frozen
recipe. Reserved-holdout scores, errors and final evaluation analysis belong
in `07_task3_part2_final_evaluation.ipynb`. The teacher test set is prediction-only in the submitted work;
the teacher did not supply its labels. Do not include private external-reference
test scores in either notebook or the final report.

## Why

The intended use is catalogue suggestions checked by a person. All nine labels
matter, so overall accuracy alone is insufficient. On the same five development
folds, E8 reached 41.94% macro-F1 versus E1's 37.38%, while accuracy fell from
89.30% to 88.85%. E8 detected one Party, six Smart Casual and five Travel items
where E1 detected none. We accept the observed accuracy cost for added class
coverage, while acknowledging that most rare-label predictions remain wrong.

E2 is the immediate class-weighted predecessor. Adding translation in E8 reduced
the mean fold macro-F1 loss under small shifts from 8.69 to 2.69 points. Log loss
fell from 0.3428 to 0.3292, and 15-bin calibration error from 3.31% to 0.79%.
Errors made with confidence of at least 99% fell from 278 to 64. Position tolerance
and better probability quality support retaining E8 alongside its observed class
coverage; they do not establish accurate confidence for every rare class.

These benefits have a measured cost. Darkening caused a 16.19-point mean fold
macro-F1 loss for E8 versus 12.89 points for E2. NA F1 and Sports detections also
fell. The paired whole-family bootstrap interval for clean macro-F1 against E2
crosses zero; against E3, the 95% interval is -2.39 to +3.18 points. A clear clean
advantage over those alternatives is not established. These intervals concern
saved fold predictions, not training-seed variation or the effects of repeated
model selection. The earlier improvement gates remain failed. Retaining E8 is
an explicit judgement across benefits and costs, not a claim that those gates
selected it. E2 remains the simpler, stronger dark-image alternative and E1 the
original accuracy alternative.

The full path is E1 baseline → E2 weighting → separate E3–E9 branches,
then S1/S2 and U1/U2/U3 screens, probability diagnostics and E8 data expansions.
E3 (41.61% F1) remains close, with better accuracy and dark-image behaviour than
E8. E5 loses only 9.51 F1 points under darkening, while E6 has slightly lower
NLL (0.3286 versus 0.3292). E4/E7 architecture changes reduce F1; E9 loses too
much common-class performance. On matched folds 0 and 4, E2 scores 40.73%:
S1/S2 score 31.93%/39.73%, U1 41.38% with worse probability scores, U2 36.29%
after calibration (40.59% raw margins), and U3 B10 40.34% with worse NLL/Brier.
The U3 review records verified artifacts but incomplete fold-4 registry proof.
These screens are not ranked against five-fold totals.

The saved equal E2+E3+E8 average is a promising diagnostic: five-fold F1 42.31%,
accuracy 89.91% and NLL 0.3032 exceed E8. Its interval versus E2 is positive,
but it lacks matched corruption and final-refit evidence, misses all Party and
requires three models. It was not advanced. Choosing E8 accepts that observed
clean-score cost in return for its tested shift behaviour and one-model package;
it does not mean the average failed numerically. All six saved averages and the
E2/E3/E8 prior adjustments are displayed in the development notebook. The prior
adjustments worsen NLL; the E8 adjustment also lowers F1. No adjustment is retained.

The added-data trials and later matched regularisation screens did not establish
a stronger teacher-data alternative for the intended class-coverage objective.
Keeping the compact teacher-only recipe avoids adding those source and training
changes without demonstrated benefit.

Refitting implements the chosen recipe using all 32,772 eligible development
rows, including scarce labels, with fresh scratch weights and the fixed 30-epoch
schedule. Normalization and class weights use development data only. Training
completion confirms the recipe was executed; it does not establish superiority
over fold models or supply a new validation score. The scores, bootstrap
intervals and robustness measurements above belong to the development fold
models and must not be attributed to the full-development checkpoint.

The reserved-holdout assessment remains in
[Task 3 evaluation](../../notebooks/07_task3_part2_final_evaluation.ipynb).
The original manifest preserves its recorded date; this rationale does not backdate
the choice or infer that private examination determined it.

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

- [Main development notebook](../../notebooks/06_task3_part1_gender_usage.ipynb)
- [Reserved-holdout analysis](../../notebooks/07_task3_part2_final_evaluation.ipynb)
- [Accepted model manifest](../../reports/task3/usage_final_e8_refit_20260907/model_manifest.json)
- [Completed E8 training notebook](../../notebooks/task3_training/usage_e8_refit.ipynb)
- [Original E8 training contract](../task3_usage_e8_refit.md)
- Assignment: train from scratch and preserve the official prediction format.
