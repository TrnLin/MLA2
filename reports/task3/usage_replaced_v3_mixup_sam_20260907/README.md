# Usage v3: repeat MixUp + SAM with the reviewed replacements

Open [task3_training/usage_replaced_v3_mixup_sam.ipynb](../../../notebooks/task3_training/usage_replaced_v3_mixup_sam.ipynb)
in VS Code. Select a fresh Colab GPU kernel and choose **Run All**.

This repeats the completed v2 MixUp + SAM recipe on v3. It trains **folds 0 and 4 only**,
with fresh SmallCNN weights, 30 epochs and the final-epoch checkpoint. MixUp stays at
alpha 0.2 and SAM at rho 0.05 over AdamW. Batch size 128, seed 2753, translation,
cosine scheduling and the effective-number class-weight rule stay fixed. Class weights
and normalization are fitted again using each training fold only.

## Files in Drive

Keep these existing files and results:

- `MyDrive/MLA2/data/task3-data.zip` — teacher images.
- `MyDrive/MLA2/data/usage_mixup_sam_training.zip` — existing v2 outside images and splits.
  `teacher_plus_rare_usage_v2_training.zip` or `teacher_plus_rare_usage_v2.zip` also works.
- `MyDrive/MLA2/task3_usage_mixup_sam/` — the completed fold-0/fold-4 reference results.

Put the new **`teacher_plus_rare_usage_v3_delta.zip`** in `MyDrive/MLA2/data/` once.
The local file is `reports/task3/rare_replacement_20260906/teacher_plus_rare_usage_v3_delta.zip`
(about 10 MB). It contains new images and the v3 manifests, so this one data update is
needed. It does not contain a new training-code bundle. Its SHA-256 is
`e625d81fc993ecb607377478eea445b77eea56fc245029502bfafe2e0e2ba4c6`.

Code comes from `https://github.com/TrnLin/MLA2.git`, branch
`task-3-gender-usage-classification`. The notebook records the exact fetched commit.
It extracts only allowed data paths from ZIPs, never their old Python files. Later
code-only updates need no new data upload. All images are read from local Colab disk.

## Frozen dataset

Use `data/processed/teacher_plus_rare_usage_v3_20260906/splits.csv` explicitly.
There are still 39,299 rows: 38,612 teacher rows plus 687 outside images. V3 retains
557 outside rows and replaces 130: Smart Casual 62, Party 44, Travel 17 and Home 7.
It preserves all teacher rows/folds, every retained outside row/fold and class totals.
The nine Usage classes include literal `NA`; no outside NA or non-Usage targets are added.

The approved split hash is
`19d514fbd56f48459b3ec207f49bb3e59f1c23bc901804fa0c32f13eb3066ecc`.
The source-geometry hash is
`da01f7e831ac415d9a1ed398be43be35dd00afa8265da630b2a1baa9dd441f2b`.
The preflight verifies hashes, the replacement counts, retained rows, target masks,
source borders and all five family boundaries. Before fitting it decodes and verifies
every eligible development image. Protected image pixels are not read by that preflight.

The new photos add missing watches, wallets, ties, sandals, clutches, perfumes and camera
bags, plus closer shirt/dress/cover views. Usage meaning, source bias and photo/audience
differences remain. Teacher development has only 47 Smart, 22 Travel, 12 Party and one
Home example. This dataset change is not evidence of a model improvement.

## Results and comparison

New results go to:

`MyDrive/MLA2/task3_usage_replaced_v3_mixup_sam/experiments/t3_usage_replaced_v3_mixup_sam/usage/`

Each fold records its checkpoint, predictions, clean training diagnostics, per-source
scores, MixUp and SAM receipts, history, normalization and robustness results. The
dedicated `results/runs.csv` and local `v3_registry/results/runs.csv` use the shared
`fashion.train.registry` writer. Existing experiment logs are not overwritten.

`aggregate_folds_0_4/teacher_comparison.json` compares **v3 MixUp + SAM with v2 MixUp + SAM**
using exactly the same teacher validation IDs. It includes per-fold F1, clean teacher
training gaps, all nine per-class F1/recall values and separate outside-source diagnostics.
The retained-v2 and replacement groups have different members, so those source scores
must not be presented as a matched external comparison.

A completed v3 fold is reused only after all recorded hashes, settings, row counts,
prediction IDs and training receipts pass verification. Interrupted folds restart fresh.
No old checkpoint is loaded for initialization. The notebook does not run the other
three folds, select another epoch, promote a model or open holdout/test labels.

## Local command

Once the data and baseline results are available locally, the same runner is:

```bash
./.venv/bin/python -m fashion.train.task3_usage_replaced_v3 \
  --output-root results/evidence/task3/usage_replaced_v3_mixup_sam_20260907 \
  --baseline-directory results/evidence/task3/usage_mixup_sam_20260906/experiments/t3_usage_expanded_v2_mixup_sam/usage \
  --baseline-registry results/evidence/task3/usage_mixup_sam_20260906/experiments/t3_usage_expanded_v2_mixup_sam/usage/results/runs.csv
```

Production fitting requires PyTorch and CUDA. No production training was started while
preparing this code. See [preflight.json](preflight.json) for the real-data/reference check
and [verification.md](verification.md) for the code checks.
