# Final Usage E8 + translation refit

Open `notebooks/task3_training/usage_e8_refit.ipynb` in VS Code.
Select your Colab GPU kernel. The notebook fetches code from
`https://github.com/TrnLin/MLA2.git`, branch `task-3-gender-usage-classification`.
For later code changes, commit and push before starting the GitHub fetch.

Reuse `MyDrive/MLA2/data/task3-data.zip`. Steps 1–4 mount Drive, fetch code,
extract only missing teacher images and check inputs. Step 5 starts training.
Step 6 checks the saved files. No new data ZIP or old checkpoint download is needed.

## Fixed recipe

This is the original teacher-only E8, `t3_usage_translation_2px_smallcnn`.
The saved five-fold configs, `usage_translation_2px_spec` and the original
inference recipe agree. The small, tracked `docs/task3_usage_e8_refit_source.json`
contains exact config snapshots and their original hashes, run IDs and source paths.
It also pins the split, class map, image transform and architecture code.

Train one fresh scratch SmallCNN on **32,772 eligible teacher Usage development
rows** from `data/processed/splits.csv`. Keep all saved folds and product families.
The one development row without an eligible Usage label is excluded. Holdout,
quarantine, test and external images stay out of training and fitted preprocessing.

- 391,209 parameters; channels 32/64/128/256; global average pooling; no dropout.
- Translation only: 50% chance; each axis is an integer from -2 to 2 pixels.
  Use the existing bilinear, white-fill transform before the normal image resize.
- Effective-number weighted cross-entropy: beta **0.999**, cap **5.0**.
  Recompute counts from all admitted development rows. Normalize present-class
  weights to mean one, then cap; do not normalize again after the cap.
- AdamW: learning rate **0.001**, weight decay **0.0001**, batch **128**, seed **2753**.
- Exactly **30 epochs**. Cosine T_max **30**, minimum rate **0.00001**. Save epoch 30.
  No validation, early stopping, MixUp, SAM or checkpoint initialization.
- Teacher RGB: height **80**, width **60**. Keep EXIF orientation, aspect-preserving
  LANCZOS resize and centred white letterbox. Fit RGB mean/std on all admitted
  development content pixels, before random translation. Exclude padding from
  normalization statistics; standardized padding is zero.

Class order: Casual, Ethnic, Formal, Home, NA, Party, Smart Casual, Sports, Travel.
Counts: 25,151; 2,183; 1,949; 1; 61; 12; 47; 3,346; 22. Literal `NA` is a class.
Fold weights are checked as evidence, then fresh weights are computed for this
larger training population. Fold weights are not reused for the final fit.

`base_config` retains the original baseline architecture/optimizer settings.
As in the saved E8 runs, `child_experiment` supplies the real translation and loss.
`training_augmentation`, `effective_loss_name` and `class_weight_contract` make
those effective settings explicit in the refit config.

## Output and reruns

Drive output:
`MyDrive/MLA2/task3/experiments/t3_usage_e8_translation_teacher_all_development_refit/usage/`.

It saves `final_epoch.pt`, `config.json`, `class_map.json`, `class_weights.json`,
`normalization.json`, `training_rows.csv`, `history.csv`, `metrics.json`,
`environment.json` and `model_manifest.json`. The manifest pins file hashes and
records training completion separately from evaluation, which remains false.
The config records source and implementation hashes. Training rows keep IDs,
folds, families, labels, image paths and image hashes.

The shared `fashion.train.registry` records the full run lifecycle in the Drive
main log and mirrors it to the checkout log during training. Other task rows stay
intact. A completed rerun verifies the artifacts and main log and returns without
writing files or training again. It does not restore or synchronize missing mirrors.
A failed, interrupted or incomplete run stops for inspection; no silent restart.
The experiment namespace is separate from E1 and all old E8 fold artifacts.

With teacher images and a working GPU environment already in the local checkout:

```bash
PYTHONPATH=src ./.venv/bin/python -m fashion.train.task3_usage_e8_refit --preflight
PYTHONPATH=src ./.venv/bin/python -m fashion.train.task3_usage_e8_refit \
  --output-root results/task3
```

The first command writes nothing and needs no GPU. The second starts training.
Its default log is `results/runs.csv`. Local outputs use `results/task3/` in
place of `MyDrive/MLA2/task3/`.

## Later judgement

Verify the completed manifest and checkpoint hash. Construct `Task3BaselineCNN`
from the saved `base_config`; strictly load `model_state_dict`. Use the saved
normalization and nine-class map. Predict with one model in evaluation mode,
without random translation; take softmax then argmax. Class weights affect
training only and must not multiply prediction probabilities.

Write evaluation to a separate folder linked to this checkpoint hash. Report
its own accuracy, nine-class macro-F1, per-class failures and measured inference
cost. Old E8 development macro-F1 **0.419393** belongs to the five fold models,
not this new refit. Evaluate the fixed checkpoint without selecting another
recipe or epoch from its final scores.

This preparation does not train on real data, evaluate a model, replace E1, or
write submission predictions. A later submission still needs the full fixed
`id,gender,articleType,season,usage` format.

## Preparation checks

Read-only preflight passed on all **32,772** local teacher images, including the
canonical training identity and each saved E8 fold's counts and class weights.
No real-data normalization, training or evaluation was run.

```bash
PYTHONPATH=src:/tmp/mla2-mixup-torch ./.venv/bin/python -m pytest \
  tests/train/test_task3_usage_e8_refit.py -q
PYTHONPATH=src:/tmp/mla2-mixup-torch ./.venv/bin/python -m pytest \
  tests/train/test_registry.py tests/train/test_task3_child_experiments.py -q
```

Results: **11 passed in 13.92 seconds**, then **91 passed in 3.59 seconds**.
The E8 tests include a real full SmallCNN fit for 30 epochs on 65 synthetic images,
with unequal class weights and translation enabled, followed by strict checkpoint
reload, schedule and normalization checks, immutable rerun checks and failure handling.
All logs and generated model files in those tests are temporary.

Ruff lint and format checks passed. The notebook schema validates and all six code
cells compile. Its static HTML preview was rendered and visually checked at
`/tmp/usage-e8-preview/usage_e8_refit.html`, with screenshot
`/tmp/usage-e8-preview/usage_e8_refit.png`. No Colab browser testing was done.
Tests used genuine PyTorch **2.11.0+cpu** from `/tmp/mla2-mixup-torch` through
`./.venv/bin/python`. No shared environment changes or Drive downloads were needed.
GPU execution remains to be run by the owner.
