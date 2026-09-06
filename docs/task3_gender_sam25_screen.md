# Fixed epoch-25 SAM screen

Use `notebooks/04aj_task3_gender_sam25_screen.ipynb` on a fresh Colab L4.
Push this code first, then Run All.

Train from scratch for exactly **25 epochs**, using the same SAM rho 0.05 and
MixUp 0.2 recipe. Keep the cosine scheduler at **T_max = 30**. This preserves
the first 25 learning-rate values of 04ai; changing T_max to 25 would change
the whole training path. Save epoch 25 as the final checkpoint. Do not warm-start
from epoch 30 or select a checkpoint during the run.

04ai's epoch-25 diagnostics had a 9.94-point clean gap, 79.86% pooled validation
F1 and 361/707 Unisex items found correctly. The final epoch-30 model found
five fewer. Epoch-25 weights were not saved and lacked the full evaluation.
This trial was chosen after reviewing validation results and is not a fresh
independent test.

Keep the same canonical folds 0 and 4, corrected labels, 390,181-parameter model,
augmentations, seed 2753, batch 128, AdamW controls and SAM update math.
Only the fixed stopping budget changes from the SAM experiment.

Keep all 19 checks against the same 04af and G2/E6 comparisons:
mean gap falls by at least two points versus 04af, both fold gaps shrink,
pooled validation F1 stays at least 74%, Unisex recall does not fall,
and all 14 remaining G2/E6 guards pass. Run full matched IEEE evaluation
and corruption checks on the saved epoch-25 model.

Each fit is registered before training. Configuration records `epochs=25`
and `cosine_t_max=30`. History, checkpoint and receipts must show 25 completed
epochs. Metrics record `early_stopped=false` because 25 is the planned budget.
The new verifier checks every learning rate against the original cosine path.
The old SAM and MixUp screens retain their 30-epoch defaults.

Output:
`MyDrive/MLA2/task3/experiments/t3_gender_name_truth_mixup_alpha020_sam005_epoch25/gender`.

The notebook uses the same Drive inputs as 04ai. It needs the completed 04af
parents and their earlier comparison/precision evidence, not the 04ai weights.
The 04ai result is the documented reason for choosing epoch 25. Keep its failed
decision unchanged. Stop after the two folds and review `screen_decision.json`
before any further training.
