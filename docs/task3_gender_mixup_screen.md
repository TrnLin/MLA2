# Gender MixUp screen

Run `notebooks/task3_training/gender_mixup_screen.ipynb` on a fresh Colab L4
after pushing the notebook and source files. Use **Run All**. The notebook checks
completed 04ad parents and their earlier evidence on Drive before any fit.

This trial tests MixUp alpha **0.2** from unweighted **04ad**. It keeps the same
name-truth labels, dropout 0.30, grayscale probability 0.10, mild darkening,
architecture, seed 2753, batch 128, 30 epochs, and final-epoch checkpoint.
Only folds **0 and 4** run. Each model starts from random weights.

For every batch, draw one `lambda ~ Beta(0.2, 0.2)` and a random permutation
of that batch. Blend the normalized, augmented images by lambda and train with
`lambda * CE(logits, y) + (1-lambda) * CE(logits, partner_y)`.
Self-pairs and same-label pairs are allowed. No class or sample weights apply.
A dedicated NumPy PCG64 stream, seeded with `2753 XOR 0x4D495855`, persists
across epochs without drawing from augmentation or dropout random streams.

Every training row appears once per epoch. The trainer rejects rows outside
the current training fold, wrong labels, duplicate rows, and incomplete epochs.
Neither validation nor held-out images can be mixing partners.

Online training loss uses mixed images; online training F1 is blank because
ordinary hard-label F1 is not meaningful for blended labels. The final gap
uses **clean, unmixed** training and validation scores in evaluation mode,
with ordinary unweighted cross-entropy. Corruption evaluations also stay unmixed.

The unchanged 19 gates compare matched G2/E6 evaluations on identical labels.
They require at least **0.050** mean clean-gap reduction, both folds' gaps
to shrink, and the existing validation, class, confidence, corruption and
resource guards. A separate comparison shows the direct change versus 04ad,
with a paired whole-family bootstrap interval. Original teacher-label scores
remain a diagnostic. A pass does not select a final submission model.

Results go to
`MyDrive/MLA2/task3/experiments/t3_gender_name_truth_mixup_alpha020/gender`.
Each run saves `mixup_training.json` with its policy, training-row hash and
per-epoch coverage, batch count, lambda sum, pair counts and mixing-plan hash.
The metrics bind this receipt by SHA-256; reuse checks it and the history scope.
The mixing-plan hash is a fingerprint, not a stored list from which to replay pairs.
Source audits bind the training contracts and MixUp implementation hashes.
All fits enter `results/runs.csv` through the shared registry.

This separate trial explicitly allows MixUp despite the earlier frozen
no-MixUp plan. Previous experiments retain their original rules and results.
MixUp may hide small product cues. Review validation and rare-class scores
along with the clean gap; lower mixed training F1 is not a success measure.
This is a development experiment chosen after seeing earlier results.
Stop after the two folds and review the saved comparison.
