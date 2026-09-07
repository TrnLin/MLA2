# Final Usage E1 refit

The owner's 7 September 2026 request adds a single full-development refit.
This changes the intended final artifact from the ensemble choice in ADR 0022.
The old decision, evaluated checkpoints, manifest and scores remain historical evidence.
The new model is not trained or evaluated by adding this code. Its later scores must
be reported separately. Do not attach the ensemble's accuracy to the refit.

## Fixed method

Train one fresh scratch SmallCNN on **32,772 eligible Usage development rows** from
`data/processed/splits.csv`. The 32,773-row development partition has one row without
an eligible Usage label. Keep the saved folds and families; use their full union.
No external additions, holdout, quarantine or test images enter this fit.

Keep E1's 391,209 parameters, channels 32/64/128/256 and global average pooling.
No classifier dropout, image augmentation, class weights or pretrained weights.
Use ordinary cross-entropy, AdamW, learning rate 0.001, weight decay 0.0001,
batch 128, seed 2753 and exactly 30 epochs. Cosine T_max is 30, with minimum
rate 0.00001. Save epoch 30; never select an epoch using holdout or test.

Teacher RGB input stays height 80, width 60. Keep EXIF orientation, aspect-preserving
LANCZOS resize and centred white letterbox. Fit fresh RGB mean/std on all admitted
development content pixels, excluding padding. Standardized padding is zero.

Class order: Casual, Ethnic, Formal, Home, NA, Party, Smart Casual, Sports, Travel.
Counts: 25,151; 2,183; 1,949; 1; 61; 12; 47; 3,346; 22 respectively.
The literal `NA` is a class, not a missing value.

## Run steps

The Colab launcher fetches these four files from `task-3-gender-usage-classification` on
`https://github.com/TrnLin/MLA2.git`:

- `src/fashion/train/task3_usage_e1_refit.py`
- `notebooks/task3_training/usage_e1_refit.ipynb`
- `docs/task3_usage_e1_refit.md`
- `tests/train/test_task3_usage_e1_refit.py`

For later code updates, commit the changed files and run
`git push origin task-3-gender-usage-classification` before starting Colab.

Open `notebooks/task3_training/usage_e1_refit.ipynb` in a fresh Colab GPU runtime.
Run cells 1–4 to fetch code, reuse `MyDrive/MLA2/data/task3-data.zip`, and check inputs.
Run step 5 to train. Step 6 checks the saved history. No new data ZIP is needed.
The launcher extracts only missing admitted teacher images; Git supplies the split,
class map and frozen source manifest. Do not run two runtimes for the same output.

With local teacher images and a working PyTorch GPU environment, run from the repository:

```bash
PYTHONPATH=src ./.venv/bin/python -m fashion.train.task3_usage_e1_refit --preflight
PYTHONPATH=src ./.venv/bin/python -m fashion.train.task3_usage_e1_refit \
  --output-root results/task3
```

The first command is read-only. The second trains. The local registry defaults to
`results/runs.csv`. Colab uses `MyDrive/MLA2/task3/results/runs.csv` as its main log
and the checkout log as a mirror, preserving other tasks' rows.

## Saved output and reruns

Local output is
`results/task3/experiments/t3_usage_e1_teacher_all_development_refit/usage/`.
Colab uses the same `experiments/.../usage/` suffix under `MyDrive/MLA2/task3/`.

It saves the final checkpoint and SHA-256, config, class map, normalization,
training rows with IDs/families/folds/image hashes, history, training-only metrics,
environment, implementation hashes and `model_manifest.json`.
The source manifest, all five embedded recipes, canonical split, label map,
model/transform code and each input image are checked before training.
The canonical split hash also pins every protected boundary and excluded row.

The manifest starts as running, then records training completion or failure.
Evaluation stays false. A repeat run verifies every completed file and registry
identity, including requested mirrors. It reuses the completed model without fitting.
A lock and a stable experiment/run identity protect against concurrent or duplicate fits.
An incomplete output or existing registry entry stops for inspection, even if the
manifest is missing. Failed runs do not silently restart. Preserve them and agree
on an explicit recovery plan before another physical fit. Do not delete evidence
or change the recipe just to bypass the stop.

## Later judgement

Verify the completed manifest and checkpoint hash, reconstruct `Task3BaselineCNN`
from the saved config, and load `model_state_dict` strictly. Use its saved
normalization and class map. Call `eval()`, take softmax, then argmax. Use one
model; do not load fold weights or average the old ensemble.

Write evaluation into a new evidence folder linked to this checkpoint hash.
Report accuracy, fixed nine-class macro-F1, class support and failures, plus
measured inference cost. Do not evaluate training rows as independent evidence.
Do not use later scores to choose another epoch, normalization or recipe.
The old holdout and recovered test labels were already viewed during selection.
Any later scoring is a fixed-model reassessment, **not newly blind evaluation**.
The recovered test labels are reference metadata, not teacher-provided labels.

This code does not run evaluation or produce submission predictions. A separate
evaluation pass is still required after training. The final submission must retain
`id,gender,articleType,season,usage`; Usage-only predictions are insufficient.

## Preparation checks

The read-only preflight passed against all 32,772 local teacher images.
Seven Usage tests passed, including a real 30-epoch scratch fit of the full model
on 45 synthetic images, checkpoint reload, the cosine schedule, padding exclusion,
registry mirrors, changed-file rejection and stopping after interruption.
Tests used temporary logs and outputs. No real-data fit or evaluation was run.

Verification used `./.venv/bin/python` with genuine PyTorch `2.11.0+cpu` from
`/tmp/mla2-mixup-torch`, including its native extension. The shared environment
was not changed. This verifies CPU training mechanics; Colab GPU/Drive execution
has not been tested here. Ruff lint and format checks passed.

```bash
PYTHONPATH=src:/tmp/mla2-mixup-torch ./.venv/bin/python -m pytest \
  tests/train/test_task3_usage_e1_refit.py -q
```

The unexecuted notebook was exported to HTML and visually inspected at
`/tmp/usage-e1-preview/usage_e1_refit.html`; its screenshot is
`/tmp/usage-e1-preview/usage_e1_refit.png`.
