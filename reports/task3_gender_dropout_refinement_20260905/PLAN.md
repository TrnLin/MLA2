# Gender refinement: dropout plus mild darkening

Status: implemented as `notebooks/04aa_task3_gender_dropout_darkening_screen.ipynb`
with reusable logic in `src/fashion/train/task3_gender_dropout_darkening.py`.
No new training has been performed. The model is not accepted.

## Why this is worth testing

G-Drop30 improved measured validation F1 on both screen folds and reduced the
mean clean training–validation gap from 0.24838546 to 0.21241153. Pooled validation
macro-F1 rose from 0.74572609 to 0.75425536. The paired family interval for that
gain was [-0.00670895, 0.02405347], so improvement is not established beyond these
reused screen folds. The 0.03597392 gap reduction remains below the required 0.050.

The current remaining corruption failures are the differences in induced F1
changes versus matched E6: darkening -0.08364930 and grayscale -0.04310049,
against a lower bound of -0.020. These are not raw corrupted-image F1 scores.

Earlier G-D1 used mild darkening without dropout. Its mean darkening-induced
change was -0.05585252 versus G2's -0.21476839: a substantial measured improvement
in lighting tolerance. However, its mean clean gap was 0.25254134, slightly worse
than G2. Those are historical evaluation results, not a new matched IEEE
comparison. They motivate a test; their gains cannot simply be added to dropout's.

The question is whether the measured dropout benefit can coexist with the
lighting tolerance learned through mild darkening. This tests complementary
interventions. It does not assume that darkening will close the remaining gap or
repair grayscale sensitivity.

## One added factor relative to G-Drop30

Keep the full-width G2 architecture [32, 64, 128, 256], fixed GeM p=3 and
post-GeM classifier dropout p=0.30. Keep 390,181 trainable parameters.

Add the existing `translation_2px_p05_mild_darkening_p025` augmentation policy:

- Keep the current integer translation, independently sampled on each axis from
  -2 to +2 pixels with probability 0.50 and the existing white fill.
- After translation, darken an image with probability 0.25, drawing its brightness
  multiplier uniformly from 0.90 to 1.00.
- Use the same independent, seeded darkening random stream as G-D1 so adding
  darkening does not change the translation random stream.
- Apply the same transform law to every gender class. Preserve full RGB inputs.

This reuses the established image transform. The existing pixel preview was
visually reviewed: mild darkening preserves visible item structure while changing
both item and background brightness. That preview does not establish that every
image retains all useful information.

Keep plain cross-entropy, AdamW weight decay 0.0001, the G2 learning-rate schedule,
batch 128, 30 epochs, seed 2753, fold-training normalization and final-epoch
checkpoint. Train from scratch on canonical folds 0 and 4 only. Dropout remains
disabled during clean train, validation and corruption evaluation.

## Comparators and lineage

The direct experimental comparator is G-Drop30, using these two completed runs:

- Fold 0: `t3_gender_dropout_030_gender_smallcnngem3_f0_s2753_2ab5e633206b_20260905T111550Zce42ad`
- Fold 4: `t3_gender_dropout_030_gender_smallcnngem3_f4_s2753_2ab5e633206b_20260905T112541Z8145b5`

Use their verified configurations, registry rows, source hashes and saved
checkpoints as evidence. Do not invent parent runs for folds 1–3. Implementation
must support this two-fold research comparator explicitly rather than silently
presenting the combination as a single change from G2.

G2 remains the fixed reference for the validation, class, confidence and gap
requirements. E6 remains the fixed reference for corruption requirements.
G-Drop30 and G-D1 remain failed research candidates; using them to motivate this
combination does not accept either model or rewrite its historical decision.

Preserve the training runtime/precision used by 04y. Evaluate all comparison
checkpoints with the same explicit IEEE FP32 policy, input preparation and batch
128. Keep these evaluation snapshots separate from historical registered scores.
No model is initialized from a comparison checkpoint.

## Fixed overall screen checks

Carry forward the existing G-Drop30 rules without relaxing or retuning them:

- Pooled validation macro-F1 difference versus matched IEEE G2 >= -0.030.
- Paired whole-family bootstrap 95% lower bound >= -0.030, using 10,000 draws
  within canonical folds and seed 2753. Each fold's validation difference must
  also be >= -0.030.
- Mean clean training–validation macro-F1 gap reduction versus G2 >= 0.050,
  with a strictly smaller gap on both folds. Use final-checkpoint clean training
  predictions, not online training scores with dropout/augmentation active.
- No pooled class loses more than 0.020 F1 versus G2.
- NLL increase <= 0.020 and ECE-15 increase <= 0.010 versus G2.
- Translation-induced F1 change improves by >= 0.030 versus E6; every other
  standard corruption's induced change worsens by no more than 0.020 versus E6.
- Exactly 390,181 parameters and peak allocated GPU memory strictly below
  3,000,000,000 bytes. Report speed without a speed cap. Use normal GPU execution.

All checks must pass. Preserve split, image-hash, checkpoint, registry and
precision provenance checks. Each actual training run must append its registry
row. Keep the held-out test sealed. Do not automatically expand to other folds.

## What the result must explain

Alongside the fixed decision, report paired differences versus G-Drop30 for
clean validation, clean training, gap, each class and every corruption. Include
raw clean/corrupted F1 as well as induced changes, so a lower clean baseline
cannot make a robustness comparison look better without being visible.

Expected benefit: better lighting tolerance while retaining some dropout benefit.
Risks: lost clean validation quality, a remaining grayscale failure, unchanged
training fit, or an adverse interaction with dropout. No numerical gain is
predicted. A lighting improvement alone is a partial result, not a full pass.

Do not raise dropout, add grayscale augmentation, change widths or change the
stopping rule in this same trial. If a specific failure remains, use the new
evidence to predeclare the following change rather than running a parameter grid
against repeatedly reused folds.

## Evidence scope

G-Drop30: saved 04y notebook plus Drive `screen_decision.json` and
`ieee_oof_predictions.csv` under
`MyDrive/MLA2/task3/experiments/t3_gender_dropout_030/gender`.
All 13,110 validation rows were checked against the canonical IDs, labels and
families; pooled/fold F1, pooled NLL and ECE were recomputed from probabilities.
The full new checkpoint hashes, corruption predictions and bootstrap were not
independently recomputed in that review.

G-D1: `reports/task3_gender_gd1_result_20260905/fold_comparison.csv`, its saved
per-fold robustness files and prior verified decision. Image preview:
`reports/task3_data_intervention_20260905/augmentation_preview.png`.

The two-fold results guide development; they do not remove model-selection bias
or provide independent final evaluation.
