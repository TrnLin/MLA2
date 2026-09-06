# SAM with MixUp 0.2

Use `notebooks/04ai_task3_gender_sam_screen.ipynb` on a fresh Colab L4.
Push the notebook and source changes first, then Run All. It uses the existing
Drive paths for the completed 04af alpha 0.2 parents, earlier G2/E6 and refinement
comparisons, registry and 04w precision evidence. The failed 04ah runs are not needed.

This changes only the learning step: first-order, non-adaptive L2 SAM with
rho 0.05 around AdamW. Keep MixUp 0.2, name-truth labels, 390,181 parameters,
dropout 0.30, grayscale 0.10, translation and darkening, batch 128, seed 2753,
learning rate 0.001, weight decay 0.0001 and cosine decay to 0.00001.
Train folds 0 and 4 from scratch for 30 epochs; select the final epoch.

SAM computes two gradients on the same augmented, mixed batch. It temporarily
moves weights by `0.05 * gradient / (global_L2_norm + 1e-12)`, then computes the
second gradient. It restores exact original weights before one AdamW update.
All trainable parameters, including BatchNorm affine parameters, enter the norm.
The first gradient excludes AdamW's decoupled weight decay.

Both passes reuse the same dropout mask. Torch RNG advances once. Both use
training batch statistics; running means, variances and counters retain only
the first pass. MixUp runs once, so every original row appears once per epoch.
Non-finite losses or gradients fail the registered run. Errors restore original
parameters and normalization buffers; failed optimizer state is not resumed.

The [SAM paper](https://arxiv.org/abs/2010.01412) and
[official implementation](https://github.com/google-research/sam) motivate the
two-gradient update. This fixed radius is an adaptation for this AdamW recipe,
not evidence of an optimal setting or a promised gain.

Keep all 19 checks from 04ah:

- The 14 non-F1 G2/E6 checks retain the gap, calibration, corruption, size and
  memory requirements. Peak allocated GPU memory must stay below 3 GB.
- Mean clean gap must shrink by at least 2 percentage points versus 04af,
  from about 12.91 to 10.91 points or less; both folds must improve.
- Pooled validation macro-F1 must stay at or above 74%.
- Unisex recall must not fall from its matched 04af value.

Relative pooled, fold and class F1 changes and their bootstrap interval remain
diagnostics. Exact matched parent scores drive decisions. These development
folds have been used repeatedly; the result is not an independent blind test.

Save clean training and validation class scores at epochs 10, 15, 20, 25 and 30.
These checks do not select a checkpoint. Final comparisons still use matched
IEEE FP32 evaluation. Online mixed-input F1 stays blank. Both mixed loss traces
are recorded separately; the second loss is measured at the perturbed weights.

Output:
`MyDrive/MLA2/task3/experiments/t3_gender_name_truth_mixup_alpha020_sam005/gender`.

Read `screen_decision.json` and `incremental_comparison.json` first.
The latter's inherited `dropout_*` fields refer to the alpha 0.2 parents.
Each run saves `sam_training.json`, `mixup_training.json`,
`clean_epoch_diagnostics.json`, history, predictions, metrics and a final
checkpoint. Reuse verifies the receipts and their hashes as well as the
existing registry and checkpoint checks. The source audit binds implementation
files, policies, labels, parent artifacts and the parent's original code commit.

Expect roughly twice the training compute, plus periodic clean diagnostics.
Actual GPU time and memory are measured, not estimated for acceptance.
`train_seconds` includes per-epoch validation and the extra clean checks.

Validation covers a numerical two-gradient reference, one AdamW state update,
dropout replay, one BatchNorm update, failure restoration, one MixUp draw,
code/policy/parent tampering, and a complete 30-epoch synthetic-image CPU run
through the real model, registry, metrics, receipts and checkpoint path.
Hardware and source boundaries are replaced only in that synthetic test.
This checks implementation; real L4 performance and the 3 GB limit remain
unmeasured until the Colab trial. Stop and review after folds 0 and 4.
