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

The audited G2-P decision retains P0. `g2_a1_c2_resnet18.json` therefore uses the
existing P0/A0 run as its reference and changes only experiment identity, stage, and
augmentation from A0 to A1. A1 adds mild colour jitter to A0 at `(80, 60)`. Do not
rerun A0 under a new experiment ID.

The audited G2-A decision retains P0/A0. Compact tuning reuses the G1 C1 and C2 runs as
T0 (`3e-4`, `1e-4`). The four `g2_t*.json` declarations add only T1 (`1e-3`, `1e-4`)
and T2 (`3e-4`, `1e-3`) for those two finalists. They keep folds, seed, transforms,
loss, batch sizes, and the eight-epoch budget fixed. Do not rerun either T0 reference.

The audited G2-T decision selects C1-T1 and retains C2-T0. The two `g3_*.json`
declarations preserve those family-specific optimiser pairs and every shared P0/A0 data,
loss, fold, seed, batch, AMP, clipping, and warm-up choice. They change only experiment
identity, stage, maximum epochs from 8 to 30, and patience from 8 to 5. This is the
matched full-budget finalist comparison; do not expand the architecture or tuning grid.

`g4_i1_effective_number_c1.json` is the controlled minority-class intervention. It
copies G3 C1-T1 exactly and changes only experiment identity, stage, and loss. For each
fold, class counts come only from that fold's training rows. The loss uses Cui et al.'s
effective-number weights with `beta=0.9999`, normalized to mean one. Retain I1 only if
Spring F1 improves by at least 0.010 while pooled macro-F1 falls by no more than 0.002.
Compare OOF metrics rather than weighted-loss values because the loss scales differ.

The two `g4_i2_article_type_lambda_*.json` files test one separate question: whether a
training-only ArticleType head improves the shared C1 representation. They copy G3
C1-T1's split, P0/A0 data, optimiser, seed, and full budget. Only experiment identity,
stage, loss declaration, and auxiliary weight differ. `lambda=0.1` tests mild transfer;
`lambda=0.3` tests a stronger signal. Missing ArticleType labels are masked instead of
dropping valid Season rows. Selection and early stopping remain based on Season
macro-F1. Inference accepts images only; true ArticleType is never an inference input.

`g4_p0s_resnet18_standard_scratch.json` and
`g4_pstar_resnet18_standard_pretrained.json` form one matched benchmark pair. They copy
G3 C2-T0's five folds, P0/A0 transforms, optimiser, seed, and full budget. The pair
differs only in experiment identity and whether the standard-stem ResNet18 starts from
random or ImageNet weights. Both rows are benchmark-only and never final-eligible. P*
is not tuned and cannot become the Task 2 winner. P* minus P0S estimates the effect of
initialisation under this project's fixed 80x60 pipeline; it is not an ImageNet-recipe
benchmark and must not be compared causally with the different I2 architecture.

`g5_c2_t0_resnet18_seed_2026.json` and
`g5_i2_article_type_lambda_0_3_c1_seed_2026.json` are the frozen eligible stability
pair. They copy the retained G3 C2-T0 comparator and selected I2 lambda 0.3 candidate.
Each changes only experiment identity, stage, and seed from 2753 to 2026. Both remain
scratch, image-only at inference, and final-eligible. The stability gate asks whether
I2 remains above C2 at both seeds; it does not reopen architecture, transform,
optimiser, loss, or auxiliary-lambda selection. P* is excluded.

`g6_shortcut_error_slices.json` freezes the first post-modelling analysis gate before
its results are calculated. It reads the four eligible C2/I2 OOF packs across seeds
2753 and 2026. ArticleType majorities and file-size quartiles are fit on four training
folds only, then applied to the matching validation fold. Acquisition year, canonical
product-family size, and image mode are joined only after prediction and are never model
inputs. The contract also fixes the Spring error outputs, high-confidence threshold,
low-support warning, and class order. Slice results can weaken a claim, but cannot add a
new architecture or freeze the ultimate winner by themselves.

`g6_robustness_cost.json` pairs the primary-seed C2 and I2 fold checkpoints on
all 32,753 development products under JPEG-85, two fixed brightness shifts, and
radius-1 Gaussian blur. It also freezes single-image CPU and available-CUDA
latency and memory measurement rules. These probes explain deployment risk;
they cannot reopen the G5 model choice or access holdout data.
