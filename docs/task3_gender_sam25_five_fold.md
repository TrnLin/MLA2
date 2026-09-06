# Five-fold fixed SAM25 training

Run `notebooks/task3_training/gender_sam25_five_fold.ipynb` on a fresh Colab L4.
Push the code first. Run All trains five new scratch models in canonical fold
order 0, 1, 2, 3, 4. Expect about 40 minutes of training plus Drive checks and
final evaluation, based on the completed two-fold timing.

Keep the passing 04aj recipe: 25 epochs with cosine T_max=30, MixUp 0.2,
SAM rho 0.05, the same 390,181-parameter GeM model, augmentations, corrected
gender labels, seed and optimizer settings. The first run trains all five
folds; restarting the notebook verifies and reuses completed CV folds.
Interrupted fits remain recorded and restart from scratch. There is no
mid-epoch checkpoint resume or best-epoch selection.

The output is a five-model set, not one model trained on all development data.
Each model trains on four folds and predicts the remaining fold. Pooled OOF
predictions must cover every eligible development image exactly once.
For later inference on new images, average the five probability vectors after
applying each model's own normalization. Never use that average for OOF scores.

The preflight checks the exact reviewed 04aj decision, both source models,
their registry entries, receipts and IEEE evaluations. Its original source
audit is compared against code commit
`5f0789506239944f2e836348b00bdc5e34d5658d`; new CV routing does not invalidate
the old evidence. The new audit binds current training code and the same data.

G2 provides one verified same-fold comparison reference per new registry row.
Those parent fields describe comparison lineage, not warm starts or the direct
recipe predecessor. The recipe source remains 04aj and is recorded separately.
No G2, 04aj or other parent weights are loaded for training. No extra parent
training is needed. No five-fold comparison against the two-fold 04af result
is claimed.

Output directory:
`MyDrive/MLA2/task3/experiments/t3_gender_name_truth_mixup_alpha020_sam005_epoch25_cv/gender`.

- Five registered run directories with final epoch-25 weights, normalization,
  configuration, history, metrics and SAM/MixUp receipts.
- `model_manifest.json`: all five models, portable relative file paths,
  class order and SHA-256 hashes.
- `aggregate/`: IEEE OOF predictions, pooled metrics, per-class scores,
  confusion matrix, errors, corruption scores and per-fold fit gaps.
- `aggregate/fold_summary.png`, also copied to
  `results/figures/task3/gender_sam25_cv.png` in the Colab repository.
- Corrected-label and original teacher-label diagnostics.
- `cv_summary.json`: all folds, screen folds 0/4 and additional folds 1/2/3.

Training and evaluation must stay below 3 GB allocated GPU memory. A damaged
completed fold or changed audit stops reuse. Duplicate completed runs require
an explicit choice; the runner never picks a model by its validation score.
Keep one active notebook writer per experiment.

`complete_for_review` means all five models and checks finished. It does not
declare a winner. The old 19-check screen remains unchanged. All five folds
are development data; folds 1/2/3 have been used by earlier experiments too.
This notebook does not evaluate the final holdout or teacher test, tune the
recipe, train a full-data refit, or measure five-model ensemble performance.
