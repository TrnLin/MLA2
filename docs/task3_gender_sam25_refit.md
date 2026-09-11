# Gender SAM25 comparison refit

Open `notebooks/task3_training/gender_sam25_refit.ipynb` in a fresh Colab L4
runtime after the new code is available on GitHub, then Run All.
The launcher fetches `task-3-gender-usage-classification` from
`https://github.com/TrnLin/MLA2.git`. It reuses
`MyDrive/MLA2/data/task3-data.zip`; no new data ZIP is needed.
Only missing training images are extracted. Git supplies the canonical split,
label map, frozen source configuration and current code.

This trains one fresh SmallCNN with GeM p=3 on all **32,773 development images**.
It keeps the name-corrected Gender labels, 390,181 parameters, dropout 0.30,
MixUp alpha 0.2, SAM rho 0.05, seed 2753, batch 128, AdamW learning rate 0.001,
weight decay 0.0001, and the translation/darkening/grayscale recipe.
Train exactly **25 epochs** with cosine **T_max=30**, minimum rate 0.00001.
There is no best-epoch choice. Input is RGB at 60×80 pixels; normalization is
fitted afresh on all development content pixels, excluding padding.

The code checks the exact saved SAM25 configuration, split and corrected-label
hashes, all development image hashes, class order and protected family boundary.
Existing SAM and MixUp mechanics are reused, including two gradient passes on
the same mixed batch and one AdamW update. The saved training precision settings
are applied and restored; runtime versions are recorded. The reference runtime
was PyTorch 2.11.0+cu128 on Colab L4. Peak allocated GPU memory stays below 3 GB.

The shared registry receives a row before fitting starts, with no validation
fold and zero validation products. This is a refit row, not an OOF comparison.
Training failures remain recorded. Both Drive and local registries are updated
through `fashion.train.registry`. Keep one active writer for this experiment.

Output directory:
`MyDrive/MLA2/task3/experiments/t3_gender_name_truth_mixup_alpha020_sam005_epoch25_refit/gender`.

- A unique run directory holds `final_epoch.pt`, `config.json`,
  `normalization.json`, `training_rows.csv`, `history.csv`, `metrics.json`,
  and SAM/MixUp receipts.
- `model_manifest.json` gives the model files, hashes and class order:
  Boys, Girls, Men, Unisex, Women.
- Repeat Run All checks and reuses a completed refit. It stops if its files,
  code, data or configuration changed. An interrupted fit restarts with fresh
  weights in another run directory; it does not resume partial weights.

Use the single checkpoint with its saved normalization for later prediction.
No holdout/test evaluation or submission file is produced here. Mixed training
loss is a training diagnostic; it cannot establish validation performance.
The five-fold scores belong to the earlier fold models, not this new checkpoint.

The completed training artifact is described in
[ADR 0025](decisions/0025-task3-gender-sam25-refit-final-model.md).
Its original training receipt remains unchanged. SAM25 contributes to the
development comparison; the selected Gender model is the MixUp 0.20 refit at
epoch 30. See [the main notebook](../notebooks/06_task3_part1_gender_usage.ipynb)
for the full candidate comparison and recipe choice.

Equivalent command, from a checkout with the teacher images already available:

```bash
./.venv/bin/python -m fashion.train.task3_gender_sam25_refit \
  --output-root results/task3
```
