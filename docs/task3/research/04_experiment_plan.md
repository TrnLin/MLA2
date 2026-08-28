# Task 3 evidence-led experiment plan

[Previous: model choice](03_model_choice.md) · [Research index](README.md) ·
[Next: evaluation framework](05_evaluation_framework.md) · [References](references.md)

## 1. Purpose

This plan turns Task 3 into a sequence of evidence-backed decisions:

```text
EDA evidence
  -> one primary learnable baseline
  -> complete diagnosis
  -> one written hypothesis
  -> one child with one main changed factor
  -> accept, reject, or stop
```

The loop runs separately for `gender` and `usage`. A later model is not selected because it is
popular, modern, or present in a prewritten matrix. It exists only when the accepted parent exposes
a weakness or practical trade-off that the child can test.

No result enters the report unless its run, predictions, configuration, diagnostic bundle, and
decision record are traceable through the required registry.

## 2. Common controls

Unless the one written hypothesis names a field as the changed factor, keep it fixed:

| Control | Fixed rule |
|---|---|
| Data | Teacher images referenced by [`splits.csv`](../../../data/processed/splits.csv) |
| Development scope | All five saved family-safe folds for scored evidence |
| Validation | Natural fold distribution; no resampling |
| Labels | Stable five-class gender and nine-class usage maps |
| Masks | Target-specific `has_<target>_label` fields |
| Initialisation | Random for every eligible neural model |
| Screening seed | Seed 2753 |
| Finalist seeds | Three predeclared seeds |
| OOF output | IDs, families, folds, logits, probabilities, labels, masks, and run IDs |
| Normalisation | Fitted on each outer training complement |
| Class weights or sampler | Fitted on each outer training complement |
| Holdout | Never accessed during this plan |
| Metrics | Frozen definitions from [05](05_evaluation_framework.md) |
| Cost hardware | Same named device and timing protocol |

Changing an architecture counts as one main factor even though its internal layers differ. Do not
change architecture and resolution, loss, augmentation, optimiser, or budget in the same child.

## 3. Method roles

### 3.1 Lower bounds

- Fold-training majority predictor: exposes misleading accuracy under imbalance.
- Fold-training stratified predictor: checks class order, masks, metrics, and random seeds.
- Bias-only no-image model: checks that the training loop can reproduce class priors.

These are lower bounds. They are not the primary learnable baseline and do not become parents.

### 3.2 Integrity checks

- Tiny-batch overfit with augmentation disabled.
- Shuffled-label validation.
- OOF row and family assertions.

These detect broken code or leakage. They are not algorithm comparisons.

### 3.3 Shortcut diagnostic

The development-only `articleType` majority lookup remains a descriptive shortcut warning. It is
not a deployable baseline because it uses another ground-truth output.

### 3.4 Classical comparison reference

HOG-plus-HSV multinomial logistic regression provides classical algorithm breadth and measures
whether hand-built shape and colour features are enough.

### 3.5 Primary learnable baseline

One exact small scratch CNN design, trained as separate gender and usage models, starts the
parent-child development chain. It is the first neural model allowed to produce full five-fold OOF
evidence.

## 4. Evidence levels

### Level E0: engineering check

- A few batches or one partial fold.
- Used only to catch crashes, bad shapes, masks, non-finite loss, or impossible memory use.
- Cannot reject a scientifically plausible model based on score.
- If parameters are updated, record the run as an engineering run.

### Level E1: development decision evidence

- All five folds.
- One seed.
- Fixed and recorded training budget.
- Full OOF predictions and the repeated diagnostic bundle.
- Suitable for one parent-child decision.

### Level E2: finalist confirmation

- All five folds.
- Three fixed seeds.
- Frozen architecture, input, loss, augmentation, optimiser, schedule, and epoch rule.
- Suitable for final development selection.

### Level E3: independent evidence

- Frozen all-development refit.
- One holdout opening.
- Performed only under [07](07_final_selection_and_deployment.md).

## 5. Phase 0: registry, contracts, and integrity

The evidence system must exist before a real training run.

| ID | Question | Method | Required output | Exit rule |
|---|---|---|---|---|
| S0.1 | Do classes and masks match the repository contract? | Count each training complement and validation fold | Fold support assertions | All counts match saved evidence |
| S0.2 | Is OOF assembly exact? | Deterministic predictions keyed by fold | One-row-per-ID assertions | Expected target row counts pass |
| S0.3 | Is the registry safe? | Start, complete, and fail synthetic run records | Registry and config-hash assertions | One traceable execution row per run |
| S0.4 | Is eligibility enforced? | Construct scratch and comparison-only configurations | Scratch/pretrained assertions | Ineligible systems cannot enter winner selection |
| S0.5 | Can the training loop learn? | Tiny-batch overfit for each target and shared-mask path | Loss and train-score curves | Near-memorisation without augmentation |
| S0.6 | Is there obvious leakage? | Shuffled training labels | OOF predictions | Validation stays near chance/prior behaviour |

Stop the experiment programme if an integrity check fails. Fix the contract and rerun the same
check. Do not choose another model.

## 6. Phase 1: lower bounds and exact classical reference

### 6.1 Lower-bound OOF evidence

For every fold:

1. derive class priors from its training complement;
2. predict its natural validation rows;
3. preserve the fixed target order and masks;
4. save OOF predictions and metrics.

### 6.2 Exact classical reference

Use the same deterministic `(80,60)` input geometry as the primary CNN:

| Field | Fixed value |
|---|---|
| Shape feature | Grayscale HOG |
| HOG orientations | 9 |
| Pixels per cell | `(8,8)` |
| Cells per block | `(2,2)` |
| HOG block norm | `L2-Hys` |
| Colour feature | HSV histogram |
| HSV bins | 16 per channel |
| Feature scaling | `StandardScaler`, fitted inside the fold-training complement |
| Classifier | Multinomial logistic regression |
| Penalty and strength | L2, `C=1.0` |
| Solver and iterations | `lbfgs`, `max_iter=2000` |
| Class weighting | None |

Train separate gender and usage classifiers. If a fold-training complement lacks a fixed class,
restore its output column with zero positive probability. In particular, fold 4 cannot learn
`Home`.

The classical result remains a valid eligible comparison. It is not the parent of a prewritten model
zoo. If it beats the neural baseline, report that honestly.

## 7. Phase 2: exact primary learnable baseline

### 7.1 EDA-derived input and task choices

| Choice | Baseline setting | Evidence-led reason |
|---|---|---|
| Task form | Two separate target models with one common design | Official outputs are separate; one usage label is missing; only 26 of 45 target pairs occur |
| Input size | `height=80, width=60` | Almost every source image is already 60×80; upsampling creates no new detail |
| Orientation and colour | EXIF-normalise and convert to RGB | Grayscale and ordinary RGB images must share one tensor contract |
| Geometry | Preserve 3:4 aspect; letterbox only unusual images | EDA shows that stretching changes shape and cropping can remove the product |
| Resize | LANCZOS for the 12 unusual-size images | Deterministic handling without deleting valid images |
| Padding | White before scaling; neutral zero after standardisation | Catalogue backgrounds are usually bright; fitted statistics must exclude padding |
| Normalisation | Fold-training RGB mean/std over content pixels only | Keeps fitted preprocessing inside the training boundary |
| Augmentation | None | Exposes the raw train-validation gap and transform sensitivity |
| Loss | Ordinary unweighted cross-entropy | Measures imbalance failure before adding an imbalance method |
| Output heads | Fixed 5 and 9 logits | Preserves all labels, including `NA` and `Home` |

### 7.2 Exact architecture

```text
input: 3 x 80 x 60

Conv 3x3, 3->32, padding 1, no bias
BatchNorm, ReLU, MaxPool 2x2

Conv 3x3, 32->64, padding 1, no bias
BatchNorm, ReLU, MaxPool 2x2

Conv 3x3, 64->128, padding 1, no bias
BatchNorm, ReLU, MaxPool 2x2

Conv 3x3, 128->256, padding 1, no bias
BatchNorm, ReLU

Adaptive global average pooling
Linear 256->5 for gender or Linear 256->9 for usage
```

The model has about 0.39 million trainable parameters. Three pooling operations leave roughly a
`10×7` feature map before global pooling. There is no dropout in the baseline.

### 7.3 Exact starting recipe

```text
initialisation: Kaiming random initialisation
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
batch_size: 128
epochs: 30
scheduler: cosine decay to 0.00001
mixed_precision: false
early_stopping: false
OOF checkpoint: final epoch
screening seed: 2753
```

The batch and epoch values are frozen engineering starting rules, not model results. E0 must confirm
that batch 128 fits the named device. If the baseline curves have not settled by epoch 30, the next
experiment changes only the budget or schedule and reruns the same model before any architecture
change.

### 7.4 Mask behaviour

- Gender uses all 32,773 development labels.
- Usage loss and usage metrics exclude only the truly blank usage row.
- Literal `usage="NA"` remains a normal class.
- Every usage fold keeps all nine output logits.
- Fold 4 records `Home` as a zero-positive-support limitation.

## 8. Phase 3: required diagnosis after every E1 candidate

No next configuration may be written until all seven items exist:

| Diagnostic | Minimum artifact | Decision supported |
|---|---|---|
| Training behaviour | Per-fold train/validation loss and macro-F1 curves | Optimisation, underfit, or overfit |
| OOF performance | Pooled primary and secondary metrics plus fold table | Parent quality and stability |
| Class behaviour | Support, predicted count, precision, recall, F1, and confusion matrices | Collapse, flooding, or useful minority change |
| Failure examples | Fixed correct, incorrect, high-confidence-error, grayscale, unusual-size, and rare-class index | Mechanism behind the aggregate result |
| Cost | Parameters, bytes, seconds/epoch, total time, peak memory, and named-device latency | Practical trade-off |
| Core robustness | JPEG 75, brightness ×0.85/×1.15, 3% translation, and grayscale | Sensitivity that can justify one later transform change |
| Comparison | Paired difference against the immediate parent and both initial references | Accept, reject, or practical tie |

The full calibration, corruption, Grad-CAM, and manual-review suites remain mandatory for finalists.
They do not replace this smaller repeated diagnostic gate.

## 9. Phase 4: hypothesis and decision contract

Before a child run, save:

```text
hypothesis_id
target
parent_experiment_id
parent_run_ids
observed_weakness
trigger_observation_ids
evidence_paths
hypothesis
single_changed_factor
fixed_controls
expected_result
rejection_condition
created_before_child_run=true
```

After the child evidence bundle is complete, save:

```text
decision_id
hypothesis_id
parent_run_ids
child_run_ids
paired_primary_difference
class_changes
failure_changes
robustness_changes
cost_changes
decision=accept|reject|stop
decision_reason
accepted_parent_for_next_cycle
limitations
```

Rules:

- **Accept:** the child becomes the only parent for the next cycle.
- **Reject:** return to the old parent; do not stack the failed change into another child.
- **Stop:** the evidence is sufficient, the limitation is mainly data/label based, or another run
  would not answer a distinct question.

## 10. Gender child gates

Choose at most one row after diagnosing the accepted gender parent.

| Observed parent evidence | Hypothesis | Only changed factor | Reject when |
|---|---|---|---|
| Training is still improving at epoch 30 | The fixed budget ended too early | Epoch/schedule budget | The extra budget adds no stable OOF gain |
| Training and validation are both weak, close, and settled | More optimisation-friendly capacity may help | Architecture to low-resolution-stem ResNet-18 | Underfit remains or overfit/cost dominates |
| Training is strong but OOF is weak | Invariance is missing | Add the fixed light augmentation policy | OOF, classes, or calibration worsen |
| `Boys`, `Girls`, or `Unisex` collapses while common classes work | Loss contribution is too majority-led | Capped class-balanced cross-entropy | Precision floods or common-class damage is severe |
| Failure review shows lost fine detail without overfit | Larger internal feature maps may help | Input to `(128,96)` | Gain is unstable or cost is not justified |
| Quality passes but named-device cost fails | A compact backbone may preserve quality more cheaply | Architecture to scratch MobileNetV3-Small | Quality/robustness loss exceeds the practical margin |
| Core colour or brightness robustness fails | The parent relies too strongly on colour conditions | Matching mild colour augmentation only | Clean or minority behaviour worsens |

Do not run ResNet-18 and MobileNetV3-Small together. They answer different observed problems.

## 11. Usage child gates

Choose at most one row after diagnosing the accepted usage parent.

| Observed parent evidence | Hypothesis | Only changed factor | Reject or stop when |
|---|---|---|---|
| Training is still improving at epoch 30 | The fixed budget ended too early | Epoch/schedule budget | Extra training does not help |
| `Casual` dominates and supported minorities collapse while common features fit | Loss contribution is too majority-led | Capped class-balanced cross-entropy | Rare predictions flood or common performance collapses |
| Common classes are also weak and train/validation remain close | Feature capacity is insufficient | Architecture to low-resolution-stem ResNet-18 | Underfit remains or ambiguity dominates |
| Training is strong but OOF is weak | Invariance is missing | Fixed light augmentation | OOF, class, or calibration evidence worsens |
| Failures show lost product detail without overfit | Larger feature maps may preserve cues | Input to `(128,96)` | Cost rises without stable useful gain |
| Quality passes but two-model cost is too high | A compact backbone may reduce system cost | Architecture to scratch MobileNetV3-Small | Usage quality or rare behaviour exceeds the harm margin |
| Errors are mainly weak business labels or classes with 1–22 examples | More model complexity cannot create missing visual evidence | No child; stop | Record the data limitation |

`Home` never triggers a child. Always inspect both the all-nine usage macro-F1 and the companion
without `Home`.

## 12. Conditional method definitions

### 12.1 Light augmentation

When the accepted parent's gap or robustness evidence triggers it, add only:

- horizontal flip with probability 0.5;
- rotation up to 5 degrees;
- translation up to 5%;
- small scale jitter without destructive cropping;
- brightness and contrast change up to 10%.

Keep validation deterministic. Do not add Mixup, CutMix, random erasing, or a second loss change in
the same child.

### 12.2 Capped class-balanced cross-entropy

When class collapse triggers it:

```text
beta = 0.999
raw_weight[c] = (1 - beta) / (1 - beta ** training_count[c])
normalise nonzero raw weights to mean 1
cap each resolved weight at 5
normalise the batch loss by the sum of applied sample weights
```

Compute counts inside the fold-training complement. A zero-count class receives no invented positive
weight. Do not combine weighting and sampling in this child.

### 12.3 Architecture children

- Low-resolution-stem ResNet-18 is available only for measured underfit or optimisation limits.
- Scratch MobileNetV3-Small is available only for a measured cost problem.
- ResNet-34 or EfficientNet-B0 remains a later option only if an accepted smaller parent still shows
  underfit and the next hypothesis names capacity as the sole factor.
- Large ResNets, ConvNeXt, and scratch transformers remain out of scope without new evidence.

### 12.4 Resolution child

`(128,96)` is available only when fixed failure examples suggest lost detail and the parent does
not show material overfit. Do not schedule it automatically.

## 13. Separate versus shared gate

A shared two-head model is not a mandatory phase.

It becomes eligible only when:

1. separate gender and usage parents have complete evidence;
2. their combined checkpoint size or measured latency fails a named application limit;
3. a common backbone and matched recipe can be defined;
4. the no-harm margin is frozen before the run.

Then compare:

| System | Purpose |
|---|---|
| Matched gender-only parent | Gender reference |
| Matched usage-only parent | Usage reference |
| One shared backbone with two heads | Cost-reduction child |

The only main changed factor is separate versus shared task structure. Start with equal normalised
head losses. Reject shared if either target fails the paired no-harm gate, a supported class
collapses, or the cost saving is not material.

Only after a near-pass with valuable cost savings may one later hypothesis test one conflict method,
such as uncertainty weighting or PCGrad.

## 14. Pretrained comparison-only lane

After the eligible development chain and finalist set are fixed, one ImageNet-pretrained ResNet-18
may measure the transfer-learning gap.

Required registry flags:

```text
scratch=false
submission_eligible=false
official_prediction_eligible=false
application_eligible=false
```

It uses the same fold and target outputs but the preprocessing required by its exact weights. It
cannot become a parent, change the eligible winner, or produce official predictions.

## 15. Finalist confirmation

Confirm only the accepted final parent for each target, plus a shared candidate if its cost gate
fired and it passed development screening.

For each finalist:

- all five folds;
- three fixed seeds;
- frozen input, augmentation, loss, optimiser, schedule, and epoch rule;
- full OOF logits;
- full calibration, error, robustness, explanation, and cost artifacts.

Do not add a hyperparameter after seeing a bad seed.

## 16. Adaptive run budget

There is no fixed minimum such as 110–115 fold jobs.

| Work | Approximate scope |
|---|---:|
| Lower bounds | No neural training |
| Classical reference, both targets | 10 classical fold fits |
| Primary CNN baseline, both targets | 10 neural fold jobs |
| One target-specific child | 5 neural fold jobs |
| One finalist confirmation | 15 neural fold jobs |
| Shared screening | 5 jobs, only if triggered |
| Shared confirmation | 15 jobs, only if screening passes |
| Pretrained benchmark | 5–10 jobs after eligible selection |

Compute-saving rules:

- use E0 only to catch engineering failures;
- reuse an identical registered reference;
- run one target child at a time;
- stop after a rejected child unless a different observed weakness still justifies a new hypothesis;
- prefer diagnosis and report evidence over an extra weakly motivated run.

## 17. Run-order loop

1. Implement registry, contracts, OOF, metrics, and diagnostic artifacts.
2. Pass S0 integrity checks.
3. Generate majority and stratified lower-bound OOF evidence.
4. Run the exact classical references.
5. Run the exact small-CNN baseline for gender and usage.
6. Complete both baseline diagnostic bundles.
7. Select one target and write one hypothesis from its dominant observed weakness.
8. Run one child with one main changed factor.
9. Complete its diagnostic bundle.
10. Accept, reject, or stop.
11. Repeat only from an accepted parent.
12. Confirm finalists with three seeds.
13. Run a shared comparison only if its measured cost gate fired.
14. Run one pretrained comparison only after eligible selection is fixed.
15. Apply the final selection rules from [05](05_evaluation_framework.md).
16. Enter the one-way final process in [07](07_final_selection_and_deployment.md).

## 18. Implementation order

Implement shared code under `src/fashion/` in this order:

1. `train/registry.py`: execution rows, status, hashes, eligibility, parent and hypothesis IDs.
2. Configuration and target contracts: canonical serialisation, split/label digests, masks, and
   fixed output orders.
3. Fold-safe preprocessing and prediction schemas.
4. OOF aggregation, metrics, per-class reports, and artifact assertions.
5. Lower-bound and classical-reference runners.
6. The exact small CNN and common training engine.
7. Curve, failure, core-robustness, and cost diagnostics.
8. The Notebook 04 artifact reader and decision ledger.
9. Only the single child selected by completed parent evidence.

Do not implement a ResNet, MobileNet, shared system, sampler, focal loss, or mixing augmentation
until an approved hypothesis selects that specific change.

## 19. Required record for every experiment decision

For each experiment, record:

- question and hypothesis ID;
- target and parent run IDs;
- trigger observations and evidence paths;
- changed factor and fixed controls;
- folds and seeds;
- result, OOF, curve, failure, robustness, and cost artifact paths;
- conclusion and accept/reject/stop decision;
- accepted parent for the next cycle;
- limitations and eligibility.

Notebook 04 narrates these records and imports reusable evidence. It must not contain a second
private training implementation.
