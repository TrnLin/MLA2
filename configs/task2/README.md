# Task 2 experiment declarations

Each JSON file is an immutable scientific question. Run it through the matching
`fashion.task2` loader; do not edit a config after a physical run exists. A correction
gets a new experiment ID and a new file.

`g0_pipeline_smoke.json` is a pipeline gate, not comparison evidence. It must pass
before any baseline or model-family screen starts.

`b0_majority.json` and `b1_hog_hsv_svm.json` freeze the two five-fold comparison
anchors. B1 uses unweighted LinearSVC decision scores; its softmax-transformed values
must not be described as calibrated probabilities.

The three `g1_*.json` files form one matched scratch-family screen. Only
`experiment_id` and `model_family` may differ. Every family receives P0/A0, seed 2753,
five folds, AdamW, effective batch 128, and exactly eight epochs of opportunity.

`g1_c2_resnet18.json` is the measured P0 reference for the G2 input-size question.
`g2_p1_c2_resnet18.json` changes only the experiment identity, stage, and image size
from P0 `(80, 60)` to P1 `(128, 96)`. It keeps C2, A0, seed, folds, optimiser, batch
sizes, loss, and eight-epoch budget fixed. Do not rerun P0 under a new experiment ID.
