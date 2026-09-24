# Task 2 post-submission RF study

This is a **separate experiment after submission**, not a replacement for the
submitted Season model. The original `models/task2_season.pt`, its evaluation,
and the official teacher-test predictions are unchanged.

## What was fitted

Before trying LeakyReLU as a remedy, the activation audit inspected the share
of **whole channels** that never activated across each complete validation fold.
It found zero dead and zero near-dead channels across 4,800 channel-fold
observations. The roughly 56.6% zero-valued activations are ordinary sparsity,
not 56.6% dead neurons. The incomplete LeakyReLU attempts were retained in the
registry, but the dead-neuron hypothesis did not justify a full continuation;
see [`summary.json`](../results/evidence/task2/post_submission/summary.json).

The five-fold development comparison selected the fixed **I2 embedding + Random
Forest (RF)** recipe: OOF macro-F1 was 0.76096, versus 0.75269 for I2 alone.
HistGradientBoosting and the fixed 50/50 ensembles did not exceed the RF head.
The selection evidence is
[`boosting_summary.json`](../results/evidence/task2/post_submission/boosting_summary.json).

The already-verified I2 CNN had itself been refitted on all 32,753 valid
development Season rows. Its weights were **frozen** here. The new run extracted
one 256-dimensional embedding per development image, then fitted a 300-tree RF
with Gini splits, `max_features=sqrt`, `min_samples_leaf=2`,
`class_weight=balanced_subsample`, and seed 2753. No holdout row or label entered
this fit. The RF's out-of-bag accuracy in
[`fit_history.json`](../results/evidence/task2/post_submission/full_refit/fit_history.json)
is a **training diagnostic**, not an independent evaluation score.

The new bundle consists of the original I2 `.pt` plus
`models/task2_postsubmit_i2_rf.joblib`. The latter is a local ignored weight;
its tracked manifest is
[`task2_postsubmit_i2_rf.manifest.json`](../models/task2_postsubmit_i2_rf.manifest.json).
Both weight files are required to replay inference. The manifest verifies their
hashes, canonical split and class map, config, implementation, training ID set,
and registry run ID. These local hashes do not prove the source is trusted:
**do not load an untrusted `.joblib`**, because Joblib uses pickle. The run
registry's `parameter_count` records total RF tree nodes here; it is not
comparable to a neural network's trainable parameter count.

## Retrospective internal-holdout evaluation

The RF predicted the same 5,778 holdout IDs under the same five fixed image
conditions as the original I2. Predictions and a hash receipt were saved before
joining protected labels. On clean images:

| Measure | Submitted I2 | I2 + RF |
|---|---:|---:|
| Macro-F1 | 0.75338 | **0.76567** |
| Accuracy | 0.76428 | **0.77310** |
| Balanced accuracy | 0.71971 | **0.73126** |
| Spring F1 | 0.75773 | **0.77285** |
| Negative log likelihood (lower is better) | **0.60454** | 0.63392 |
| Recorded CPU clean throughput (images/s) | **462.44** | 126.45 |
| Combined weight bytes | **4,856,199** | 51,168,654 |

The paired macro-F1 difference is **+0.01228**. Resampling whole product
families 10,000 times gives a middle-95% interval of **[+0.00625, +0.01864]**.
That interval describes sampling variation within this dataset, not transfer to
new retailers or cameras. Per-class metrics, accuracy intervals, and confusion
counts are in the
[`full_refit` evidence folder](../results/evidence/task2/post_submission/full_refit/).

The I2 probabilities use its frozen temperature scaling; RF probabilities are
uncalibrated. The log-loss row compares these deployed outputs, not models
given the same calibration treatment. The throughput readings also come from
separate CPU runs, so they are indicative rather than a controlled speed
benchmark. The RF's recorded rate is about 3.6 times lower.

The added head does **not** fix the main robustness failure: at brightness 0.85,
RF macro-F1 falls to 0.37927 and Spring recall to 0.00429 (I2: 0.37122 and
0.00429). Its probability log loss is also worse, so higher clean macro-F1 is
not an all-round win.
See the [scorecard](../results/figures/task2/post_submission/full_refit/holdout_scorecard.png),
[paired interval](../results/figures/task2/post_submission/full_refit/holdout_bootstrap.png),
and [fixed-condition plot](../results/figures/task2/post_submission/full_refit/holdout_robustness.png).

**Claim boundary:** The internal holdout had already been opened for the
submitted I2 before this post-submission study. This comparison is therefore
retrospective and exploratory; it is not a fresh independent selection test,
not a new assignment submission score, and not authority to replace the
submitted model. No RF official teacher-test export was made.

## Replay and handoff

From the repository root, with the original I2 `.pt`, new RF `.joblib`, raw
images, and tracked artifacts available:

```powershell
.\.venv\Scripts\python.exe scripts/run_task2_post_submission_rf_refit.py --step all --mode run_or_load
```

`run_or_load` verifies existing artifacts instead of overwriting them. The
refit and evaluation can also be run separately with `--step refit` and
`--step evaluate`. The RF weight is intentionally ignored by Git; transfer it
separately together with the original I2 `.pt`. Keep the manifest and evidence
in Git so the receiver can check the transferred bytes. The new run is recorded
in `results/runs.csv` under
`postsubmit-i2-embedding-rf-full-development-fall-s2753-3f82275bfac9`.

## Later controlled RF-head grid

We then asked whether small changes to the **RF head only** could improve the
earlier development result. The I2 encoder, its five fold-specific checkpoints,
training IDs, 256-dimensional embeddings, 300 trees, class weighting, and
random seeds stayed fixed. The original RF (`min_samples_leaf=2`,
`max_features=sqrt`) was rerun as a parity control: all five folds reproduced
the pinned earlier predictions and probabilities. Four variants changed only
the minimum examples per leaf (1 or 4) and/or the fraction of features tested
at a tree split (`0.25`, i.e. 64 of 256 features, versus `sqrt`, i.e. 16).

Before training, we required an improvement of at least **+0.003 pooled OOF
macro-F1**, no Spring F1 loss beyond 0.005, and a positive lower bound for a
paired 95% product-family bootstrap interval. All candidates used the same
32,753 canonical development IDs. The internal holdout was not used to choose
the grid or inspect its candidates.

| RF head | Pooled OOF macro-F1 | Change vs original | Spring F1 | Mean fold fit time |
|---|---:|---:|---:|---:|
| Original (`leaf=2`, `sqrt`) | **0.760959** | — | 0.769643 | 26.0 s |
| `leaf=1`, `sqrt` | 0.759695 | −0.001264 | 0.770130 | 27.3 s |
| `leaf=4`, `sqrt` | 0.760833 | −0.000125 | 0.772528 | 24.4 s |
| `leaf=2`, `0.25` | 0.760277 | −0.000681 | 0.772968 | 105.5 s |
| `leaf=4`, `0.25` | 0.760196 | −0.000763 | 0.772648 | 95.8 s |

None passed the gain gate. Every paired 95% interval included zero; for the
closest variant (`leaf=4`, `sqrt`) it was **[−0.002402, +0.002153]**. Some
variants raised Spring F1 slightly but lowered overall macro-F1. Trying 64
features per split took about four times longer to fit on these fold runs. The
cost numbers measure the RF head, **not** image preprocessing plus CNN plus RF
end-to-end. The exploratory grid therefore **retains the already fitted RF**;
no new full-development refit, holdout evaluation, or official prediction was
performed. The submitted I2 model is unchanged.

The [zero-centred paired chart](../results/figures/task2/post_submission/rf_grid/paired_delta.png)
shows the actual differences and intervals without exaggerating them on a
truncated absolute-score axis. Exact metrics, per-class scores, fold scores,
costs, run IDs, and hashes are in the
[grid evidence](../results/evidence/task2/post_submission/rf_grid/summary.json).
Re-run it with:

```powershell
.\.venv\Scripts\python.exe -m fashion.task2.post_submission_rf_grid
```

The figure can be rebuilt from verified evidence with
`scripts/plot_task2_post_submission_rf_grid.py`. The bootstrap interval
describes resampling uncertainty for these product families; because the same
five folds were also used to compare variants, it is not independent proof of
an improvement after model selection.
