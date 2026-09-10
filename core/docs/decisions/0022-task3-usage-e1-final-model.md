# 0022 — Freeze the Task 3 Usage E1 model

- Status: Superseded by [0024](0024-task3-usage-e8-refit-final-model.md); retained as the historical E1 decision
- Date: 2026-09-07
- Amends: [0014](0014-development-holdout-cv-boundary.md), for the final Usage artifact only

## Context

The Usage investigation compared scratch image models, class weighting,
regularisation, classical features, a transformer screen and rare-class data
expansion. E1 led the nine original five-fold experiments on accuracy; E8 led
on macro-F1, which gives each class equal weight. The owner has chosen E1 for
overall accuracy after reviewing the completed holdout and test results.
This later preference does not turn earlier failed macro-F1 gates into passes.

## Decision

- Select experiment `t3_primary_baseline_smallcnn`, target `usage`, teacher only.
- Freeze exactly the five original `final_epoch.pt` checkpoints in
  `reports/task3/usage_final_e1_20260907/model_manifest.json`. Their run IDs,
  SHA-256 hashes, configurations and fold normalizations define the artifact.
- Keep the scratch 391,209-parameter SmallCNN: channels 32/64/128/256,
  global average pooling, no classifier dropout or augmentation, unweighted
  cross-entropy, AdamW, learning rate 0.001, weight decay 0.0001, batch 128,
  seed 2753, 30 epochs and cosine decay to 0.00001. Use the saved final epoch;
  do not choose another epoch or retrain the model.
- Inputs are teacher RGB images at height 80, width 60. Apply the saved EXIF,
  aspect-preserving resize, letterbox and fold-specific normalization contract.
  No high-resolution training images or external additions belong to E1.
- For a new image, run each fold model in evaluation mode, take its softmax
  probabilities, average the five vectors equally, then take the largest value.
  Class order is `Casual, Ethnic, Formal, Home, NA, Party, Smart Casual, Sports,
  Travel`. No threshold, calibration or prior adjustment is selected.
- For development OOF scores, use only the model that held out that row.
  Averaging five models on development rows would include training exposure.
- For Usage only, this evaluated five-model ensemble replaces the
  all-development single-model refit prescribed by 0014. No refit is selected.
  The Gender exception in 0020 and other tasks' rules remain in force.
- Keep the evaluated artifact fixed. Holdout and recovered test labels have
  already been inspected; they cannot serve as new blind selection data.

## Why

On 32,772 original development rows, E1 scores 89.30% accuracy and 0.373756
macro-F1. Original E8 scores 88.85% and 0.419393. E8 with 687 additions reaches
89.51% development accuracy, so E1 does not win that expanded comparison.

On the same 5,829 official test images, the four saved five-fold probability
averages give E1 5,123 correct (87.89%), original E8 4,989 (85.59%), E8 with
120 additions 5,000 (85.78%), and E8 with 687 additions 4,991 (85.62%).
E1 is the accepted accuracy choice among those four evaluated artifacts.
Original E8 has the better test macro-F1: 0.308491 versus E1's 0.271542.
The decision does not claim E1 wins every metric or every possible blend.

The later two-fold v2 MixUp/SAM and v3 replacement trials did not improve
teacher macro-F1 on their matched 13,110 rows. They are separate screens,
not five-fold alternatives with the same evaluation population.

## Consequences

E1 gets 5,192/5,778 holdout images correct: 89.86% accuracy and 0.363860
nine-class macro-F1. On the official test, it misses all 245 NA, 16 Party,
12 Smart Casual and one Travel examples. Home has no test examples. Casual
dominates the test, and E1's accuracy advantage mainly comes from that class.
The fixed nine-class macro-F1 includes zero for absent Home; an eight-present-
class diagnostic must stay separately named.

The teacher supplied no official test labels. The saved evaluation used
matching high-resolution metadata as a reference by ID, for scoring only.
Predictions were frozen before their scoring pass, but reference labels had
already been seen in earlier comparisons. Final acceptance was made after
evaluation review. Do not describe this decision as newly blind or claim
that product-family independence of the test has been established.

Overfitting and sensitivity to darkening and small shifts remain. The model
is a catalogue-tag suggestion with known rare-class failures. It is not
validated for reliable automatic assignment of every occasion. Five forward
passes cost more than one; no application latency benchmark is claimed.
The existing E1 `id,usage` predictions do not replace the complete required
`id,gender,articleType,season,usage` submission. The teacher template is unchanged.

## Evidence

- [Frozen model manifest](../../reports/task3/usage_final_e1_20260907/model_manifest.json)
- [Original validation ranking and test review](../../reports/task3/usage_teacher_vs_expanded_test_20260906/README.md)
- [Original saved inference recipe](../../reports/task3/usage_teacher_vs_expanded_test_20260906/inference_recipe.json)
- [Reserved holdout review](../../reports/task3/usage_holdout_20260906/README.md)
- [Completed 687-image test comparison](../../reports/task3/usage_expanded_v2_e8_test_20260907/README.md)
- [Two-fold v2 trial](../../reports/task3/usage_mixup_sam_result_20260906/README.md)
- [Two-fold v3 replacement trial](../../reports/task3/usage_replaced_v3_mixup_sam_result_20260907/README.md)
- [Main report notebook](../../notebooks/06_task3_part1_gender_usage.ipynb)
