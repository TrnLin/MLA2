# Gender stronger-dropout screen

Status: implemented for a future Colab run. No new training has been performed.

The completed 04aa screen improved dark-image handling but still failed the mean
clean gap and grayscale rules. Its mean clean training F1 is 0.965007, pooled
validation F1 0.742180, and mean clean gap 0.223359. Girls, Boys and Unisex retain
the largest gaps. Adding darkening barely reduced training fit versus dropout
alone. Earlier dropout reduced the gap more than the width-only or stronger
weight-decay trials, so stronger dropout is the next single-factor hypothesis.

## Frozen change

Raise post-GeM classifier dropout from **0.30 to 0.45**. This hides each pooled
feature with 45% probability during training instead of 30%. The value is one
predeclared increase, not a tuned optimum or a parameter sweep. It may lower
training fit, but may also hurt minority-class validation or leave grayscale weak.

Keep the completed 04aa mild-darkening policy exactly: integer translation ±2 px
per axis with probability 0.50, then darkening with probability 0.25 and a uniform
brightness factor of 0.90–1.00. Keep its separate seeded darkening random stream.
Keep full RGB inputs, widths `[32, 64, 128, 256]`, fixed GeM p=3, 390,181 parameters,
plain cross-entropy, AdamW weight decay 0.0001, learning rate 0.001 with the same
cosine schedule to 0.00001, batch 128, 30 epochs, seed 2753, fold-training RGB
normalization and final-epoch checkpoint. Train from scratch on canonical folds
0 and 4 only. No checkpoint initializes the new model. Dropout and training
augmentation are disabled for clean training, validation and corruption evaluation.

The direct parents are the two completed 04aa runs:

- Fold 0: `t3_gender_dropout_030_mild_darkening_gender_smallcnngem3_f0_s2753_cfbed3f0fed4_20260905T123721Z5d546f`
- Fold 4: `t3_gender_dropout_030_mild_darkening_gender_smallcnngem3_f4_s2753_cfbed3f0fed4_20260905T124728Z7dfb12`

Verify their registry, configuration, checkpoints and original source audit,
including the original dropout-to-G2-to-E6 lineage. Their failed decision stays
unchanged; research-parent status does not mean model acceptance.

## Unchanged decision

Use the same L4 runtime and training precision as 04w/04aa. Evaluate G2, E6, the
two direct parents and the new checkpoints with the same IEEE FP32 policy and
batch 128. All 19 existing checks must pass:

- Pooled and each-fold validation macro-F1 difference versus G2 >= −0.030.
- Paired whole-family bootstrap 95% lower bound >= −0.030; 10,000 draws within
  canonical folds, seed 2753.
- Mean clean train–validation gap reduction versus G2 >= 0.050, and both fold
  gaps strictly smaller. Use clean final-checkpoint training scores.
- Every pooled class F1 difference versus G2 >= −0.020; NLL increase <= 0.020,
  ECE-15 increase <= 0.010.
- Translation-induced F1 change improves by >= 0.030 versus E6. Every other
  standard corruption's induced-change difference versus E6 >= −0.020.
- Exactly 390,181 parameters; peak allocated GPU memory strictly below
  3,000,000,000 bytes, with no CPU offload or speed cap.

Save additional clean train/validation, gap, class and raw/induced corruption
differences versus the direct **0.30 + darkening** parents. These describe the
effect of raising dropout and add no acceptance gates. A smaller training score
alone is not success. Do not open the held-out test or automatically expand folds.

## Run and outputs

Notebook: `notebooks/04ab_task3_gender_stronger_dropout_screen.ipynb`.
Entry point: `fashion.train.task3_gender_stronger_dropout.run_gender_stronger_dropout_screen`.
Output: `MyDrive/MLA2/task3/experiments/t3_gender_dropout_045_mild_darkening/gender`.

Each actual fit uses the existing registry-aware trainer. Complete matching runs
are reused only after source-audit, lineage, configuration and artifact checks.
Save `screen_decision.json`, `clean_gap_comparison.csv`, `ieee_oof_predictions.csv`,
`incremental_comparison.json`, `dropout_corruption_comparison.csv`, `source_audit.json`
and the matched `comparison_ieee_v2/` snapshots. In the incremental files,
`dropout_*` columns refer to the direct 0.30-plus-darkening parents, identified by
name and exact run IDs in the screen decision.
