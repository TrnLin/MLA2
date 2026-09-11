# Gender MixUp 0.20 on all five folds

Open `notebooks/task3_training/gender_mixup_five_fold.ipynb` in Colab and select an L4 GPU.
The new notebook, runner and `configs/task3/gender_mixup_alpha020_source.json` must first
be on GitHub. Set `BRANCH` to the branch containing them, then use **Run all**.
The configured branch is `task3-mixup-five-fold-training`.

The setup follows `gender_sam25_five_fold.ipynb` and `gender_sam25_refit.ipynb`:
mount Drive, fetch code from GitHub, and reuse `MyDrive/MLA2/data/task3-data.zip`.
Only missing development images are extracted. The runtime must match
`requirements/colab-task3-runtime.txt`; the runner checks it before training.
Do not install the local project's newer dependency set over Colab's CUDA stack.

Each saved fold 0–4 gets a fresh model trained on the other four folds.
Use the original 390,181-parameter GeM p=3 CNN, dropout 0.30, MixUp alpha 0.20,
translation, mild darkening and grayscale. AdamW uses learning rate 0.001,
minimum rate 0.00001, weight decay 0.0001, batch 128 and seed 2753.
There is no SAM. Train 30 epochs with cosine T_max=30 and always save epoch 30.
Validation scores do not select a checkpoint. Fit RGB statistics separately on
each fold's training images. Keep the corrected Gender labels and canonical folds.

The source JSON is a byte-for-byte copy of the original MixUp fold-0 screen config
from `reports/task3/gender_mixup_result_20260906/saved/`.
Its SHA-256 is `5a8214917e34a489f8301614cb95fa6058e751c894e801ab77b35686ff0b7043`.
It supplies the recipe only; old weights and fold-specific row counts are not reused.
Fresh row contracts, implementation hashes and runtime details are recorded per fold.

Allow about 45 minutes for training on an L4, plus image checks and final scoring.
This is an estimate from the two old MixUp runs, not a new timing measurement.

Results are saved under:
`MyDrive/MLA2/task3/experiments/t3_gender_name_truth_mixup_alpha020_cv/gender/`.
Every run also enters `MyDrive/MLA2/task3/results/runs.csv`, mirrored to local
`results/runs.csv`. Each fold has a checkpoint, normalization, config, row lists,
30-epoch history, MixUp receipt, metrics and clean/corrupted predictions.
Final scoring uses the same IEEE FP32 policy as SAM25 CV.

`cv_summary.json` links the five model folders and their manifests.
`oof_predictions.csv` has one held-out prediction per development image.
`fold_summary.csv` gives each fold's F1, calibration and clean training gap.
`robustness.csv` and the `oof_*.csv` files preserve the five corruption checks.
Use the saved predictions for a paired family bootstrap against SAM25 after training.
Do not average all five models to score development images: four trained on each row.

A repeat Run all verifies file hashes and registry records before reusing completed folds.
Failed or interrupted fits restart from scratch in new folders. Keep one active Colab
writer per experiment. A stale running registry row after a hard disconnect remains
as the record of that attempt. Changed code/data/settings require a separate output root.

These are development results after earlier model selection, not a new blind test.
No holdout/test images are scored, no final refit is run, and the accepted final model
is not changed. Review the completed comparison before choosing a final model.
