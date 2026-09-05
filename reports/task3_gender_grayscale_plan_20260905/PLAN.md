# Gender occasional-grayscale trial

Status: training code and Colab notebook prepared. No new model has been trained.

## Question and fixed change

Does occasional grayscale training reduce color sensitivity while retaining
clean-image class scores and reducing the clean training–validation gap?

Start from **04aa: dropout 0.30 plus mild darkening**. Add grayscale with
probability **0.10 per training image presentation**, uniformly across classes.
This is a single predeclared rate, not an optimized value or a sweep. About 90%
of presentations retain their color; the actual count is random. Color remains
useful, so this first trial keeps it on most inputs. Higher rates may help more
or may discard too much useful information; this run does not decide that.

The saved 0.30 model correctly classified 125 of 216 Girls items in color and
60 in grayscale on folds 0 and 4. The error inspection found sensitivity to
color and ambiguity between child/adult catalog categories. The 0.45 dropout
trial reduced clean validation F1 without meeting the existing gap or grayscale
rules. Its checkpoints are not parents of this new trial. Catalog name/label
conflicts remain separate audit candidates; this experiment makes no label,
sample-selection, split or held-out-test changes.

## Exact training recipe

1. Translate ±2 px per axis with probability 0.50, using the existing stream.
2. Darken with probability 0.25 and a uniform brightness factor in [0.90, 1.00],
   using the existing independent persistent-worker stream.
3. Convert RGB → L → RGB with probability 0.10, using a third independent
   persistent-worker stream seeded with `torch.initial_seed() ^ 0x47524159`.
4. Apply the unchanged image transformation and clean fold-training RGB
   normalization. Grayscale inputs still have three channels; normalization
   uses the original per-channel statistics.

The new grayscale step does not consume translation, darkening, sampler or
global PyTorch random draws. It runs after darkening. A training image can
receive both transformations. There is no fixed grayscale subset or per-epoch
reset of the extra random stream.

Keep full RGB images, widths `[32, 64, 128, 256]`, GeM p=3, dropout **0.30**,
390,181 parameters, plain cross-entropy, AdamW weight decay 0.0001, learning rate
0.001 with the existing cosine schedule to 0.00001, batch 128, 30 epochs,
seed 2753, two persistent workers and the final-epoch checkpoint. Both models
start from random weights. Every actual fit uses the registry-aware trainer.

Clean training-score evaluation and ordinary validation use original-color
images, with no training augmentation and with dropout disabled. The existing
robustness evaluation still applies its explicit, fixed corruptions.

## Source checks and comparisons

The direct parents are the completed 04aa runs:

- Fold 0: `t3_gender_dropout_030_mild_darkening_gender_smallcnngem3_f0_s2753_cfbed3f0fed4_20260905T123721Z5d546f`
- Fold 4: `t3_gender_dropout_030_mild_darkening_gender_smallcnngem3_f4_s2753_cfbed3f0fed4_20260905T124728Z7dfb12`

Verify both parents, their original source audit, the earlier dropout/G2/E6
lineage and 04w precision evidence before fitting. Record a new source audit.
Reuse a completed child only when its configuration, lineage and artifacts
match. Re-evaluate G2, E6, the two direct parents and the new checkpoints under
the same IEEE FP32 policy and batch size. Keep the 04w/04aa L4 training runtime.

## Unchanged acceptance rules

All 19 existing checks must pass. No threshold is relaxed for this trial:

- Pooled and each-fold validation macro-F1 difference versus G2 ≥ −0.030.
- Paired whole-family bootstrap 95% lower bound ≥ −0.030, using 10,000 draws
  within canonical folds and seed 2753.
- Mean clean training–validation gap reduction versus G2 ≥ 0.050, with a
  strictly smaller gap on both folds.
- Every pooled class F1 difference versus G2 ≥ −0.020; NLL increase ≤ 0.020
  and ECE-15 increase ≤ 0.010.
- Translation-induced F1 change improves by ≥ 0.030 versus E6; every other
  standard corruption's induced-change difference versus E6 ≥ −0.020.
- Exactly 390,181 parameters and peak allocated GPU memory strictly below
  3,000,000,000 bytes. No CPU offload or speed cap. A memory failure prevents
  starting the next fold.

Also report clean train/validation, class, gap and raw/induced corruption
differences against the direct 0.30-plus-darkening parents. These are descriptive,
not new acceptance gates. Specifically inspect Boys/Girls clean scores together
with grayscale scores. A smaller corruption drop can result from a worse clean
baseline, and a lower training score alone is not success.

## Run and output

Notebook: `notebooks/04ac_task3_gender_grayscale_screen.ipynb`.
Entry point: `fashion.train.task3_gender_grayscale.run_gender_grayscale_screen`.
Push the code and notebook, use a fresh Colab L4 runtime, then Run All.

Output: `MyDrive/MLA2/task3/experiments/t3_gender_dropout_030_mild_darkening_grayscale_010/gender`.

Save `screen_decision.json`, `clean_gap_comparison.csv`, `ieee_oof_predictions.csv`,
`incremental_comparison.json`, `dropout_corruption_comparison.csv`, `source_audit.json`
and the matched `comparison_ieee_v2/` snapshots. In the incremental artifacts,
`dropout_*` means the direct 0.30-plus-darkening parents without training grayscale.

Stop after folds 0 and 4 for review. Do not automatically change the grayscale
rate, expand folds, refit or open the held-out test. Heat-map diagnostics are
separate from this training change and are not implemented by this notebook.
