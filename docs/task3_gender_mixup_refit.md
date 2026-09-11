# Gender MixUp 0.20 refit in Colab

Open `notebooks/task3_training/gender_mixup_refit.ipynb` in Colab.
Select an **L4 GPU**, then **Run all**. The new notebook and
`src/fashion/train/task3_gender_mixup_refit.py` must first be pushed to
`task3-mixup-five-fold-training`, the branch selected in the notebook.
Reuse `MyDrive/MLA2/data/task3-data.zip`. No new dataset ZIP is needed.

This trains one fresh model on all **32,773** Gender development rows,
including every saved fold. It uses the same pinned source config and label checks
as the five-fold MixUp runner. It does not need completed CV checkpoints or load
their weights. No holdout/test image enters normalization or training.

Keep MixUp alpha **0.20**, GeM p=3, dropout **0.30**, the same translation,
darkening and grayscale transforms, batch **128** and seed **2753**.
Use AdamW with learning rate **0.001**, minimum **0.00001** and weight decay
**0.0001**. There is no SAM. Train **30 epochs**, cosine **T_max=30**, then save
epoch **30**. Fit RGB statistics on the development images only.
There is no validation split, early stopping, or best-checkpoint selection.
The runtime check matches the five-fold Colab stack in
`requirements/colab-task3-runtime.txt`.

Outputs go to:
`MyDrive/MLA2/task3/experiments/t3_gender_name_truth_mixup_alpha020_refit/gender/`.

- `model_manifest.json` links the one checkpoint, config, normalization and hashes.
- Each attempt has its own folder with `final_epoch.pt`, `config.json`,
  `normalization.json`, `history.csv`, `training_rows.csv`, `mixup_training.json`
  and `metrics.json`.
- Every attempt enters `MyDrive/MLA2/task3/results/runs.csv` before training,
  mirrored to local `results/runs.csv`. Failed attempts remain recorded.

Repeat Run all to verify and reuse a completed refit. Interrupted training starts
again from scratch in a new folder. Changed code, data or settings require a
different output root. Keep one active Colab session for this refit experiment.

Training loss measures mixed inputs, so it is not a validation score.
Use the five-fold results to compare MixUp and SAM25. This refit is saved as a
candidate and does not replace the accepted model, run holdout/test evaluation,
or create a submission file. For later inference, load this checkpoint with its
own normalization, apply softmax, and take the largest class probability.
