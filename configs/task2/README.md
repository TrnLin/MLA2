# Task 2 experiment declarations

Each JSON file is an immutable scientific question. Run it through the matching
`fashion.task2` loader; do not edit a config after a physical run exists. A correction
gets a new experiment ID and a new file.

`g0_pipeline_smoke.json` is a pipeline gate, not comparison evidence. It must pass
before any baseline or model-family screen starts.

`b0_majority.json` and `b1_hog_hsv_svm.json` freeze the two five-fold comparison
anchors. B1 uses unweighted LinearSVC decision scores; its softmax-transformed values
must not be described as calibrated probabilities.
