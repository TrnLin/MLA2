# Usage E8 on the combined dataset

Open `notebooks/04ag_task3_usage_expanded_e8.ipynb` in Google Colab.

1. Put `teacher_plus_rare_usage_training.zip` in `MyDrive/MLA2/data/`.
2. Keep the existing `task3-data.zip` in that same folder.
3. Choose a GPU runtime. Click **Run All**.

The ZIP includes the code, combined split, 120 added images and five saved E8 evidence
bundles. It does not need a commit or push. Colab unpacks it into its own folder.
The generated ZIP, image data and model weights stay local; upload the provided ZIP
from this report folder to Drive before running the notebook from GitHub.

## What runs

One nine-class Usage model. Each of the five folds starts from scratch and trains for
30 epochs. The recipe is E8: SmallCNN, 60×80 RGB, batch 128, AdamW, weighted
cross-entropy and small image shifts. Seed 2753, FP32, final epoch checkpoint.

E8's saved five-fold pooled teacher macro-F1 is **0.4193932**. This was the highest
completed five-fold score among the compared single GPU Usage models. Reusing this
recipe does not promise a higher score with the new data.

Only `data/processed/teacher_plus_rare_usage_20260906/splits.csv` selects the new
training and validation rows. The original teacher rows and folds remain unchanged.
The existing source manifest supplies only the saved image-border geometry; its
old source labels and train/validation columns do not select this training task.
Class weights and image mean/std are fitted on combined training rows only.

## Saved output

In Drive, `MyDrive/MLA2/task3/experiments/t3_usage_expanded_e8/usage/` contains:

- Each fold: checkpoint, config, normalization, epoch history, clean training
  predictions, validation predictions, source-specific scores and corruption checks.
- `aggregate/teacher_comparison.json`: E8 and expanded scores on the same teacher IDs.
- `aggregate/teacher_per_class.csv`: teacher class scores and changes.
- Standard aggregate scores, confusion matrix and failure index.

Every real run appends to `MyDrive/MLA2/task3/results/runs.csv`, mirrored into the
local Colab `results/runs.csv`. A new execution gets a new ID. Resume reuses only
complete runs with the exact recipe, dataset and intact saved files.

Compare **teacher-only** validation scores with E8. The combined score has a changed
class mix. Added-source scores use small samples: Smart Casual has two added families,
Travel has one added image, and NA has no added images. Holdout and prediction images
remain sealed. The new run does not automatically replace the chosen model.

## Rebuild after editing the code

From the project root:

```bash
./.venv/bin/python reports/task3_usage_expanded_e8_20260906/build_bundle.py
```

Upload the rebuilt ZIP before starting a fresh Colab session. The bundle receipt
records its hash and verifies every file after compression.
