# Task 3 model choice

[Previous: data and validation](02_data_and_validation_design.md) · [Research index](README.md) ·
[Next: experiment plan](04_experiment_plan.md) · [References](references.md)

## 1. Decision to make

Task 3 needs a credible method under these constraints:

- 32,773 development products;
- small 60×80 teacher images;
- five family-safe folds;
- five gender classes;
- nine usage classes;
- severe imbalance;
- one truly missing usage label;
- one `Home` development example;
- scratch training for every submitted system;
- separate official output columns;
- a rubric that rewards breadth, judgement, and honest failure analysis.

The research question is not “which famous model wins a fixed screen?” It is:

> What is the smallest evidence-backed change that addresses the accepted parent's observed
> weakness or practical trade-off for this target?

## 2. Selection principles

### 2.1 EDA chooses the primary baseline boundary

The established EDA fixes the first learnable model's data, task, geometry, and diagnostic choices:

- native-like `(80,60)` input because almost all images are already 60×80;
- aspect preservation because stretch changes shape and crop can remove content;
- RGB conversion because 294 grayscale images remain valid;
- separate gender and usage models because the official outputs and masks are separate;
- unweighted loss and no augmentation so the first curves expose imbalance and overfit rather than
  hiding them;
- all five saved folds because family-safe OOF evidence is the development comparison unit.

### 2.2 Diagnose before adding complexity

Every candidate must finish the diagnostic bundle from
[04_experiment_plan.md](04_experiment_plan.md#8-phase-3-required-diagnosis-after-every-e1-candidate)
before another configuration is created.

A child must have:

- one accepted parent;
- one written observed weakness;
- one hypothesis;
- one main changed factor;
- fixed controls;
- an expected result and rejection condition written before the child runs.

### 2.3 Hold the comparison boundary fixed

Within one parent-child comparison, keep fixed:

- development folds;
- label maps and masks;
- input size unless resolution is the changed factor;
- augmentation unless augmentation is the changed factor;
- loss unless imbalance is the changed factor;
- optimiser and training exposure unless optimisation is the changed factor;
- seed;
- OOF aggregation and evaluation code;
- cost-measurement hardware.

### 2.4 Separate eligibility from scientific interest

A pretrained model may measure the value of transfer learning. It remains ineligible for the
submitted model, official predictions, and application. Eligibility is a registry field.

### 2.5 Select each output honestly

Gender and usage have separate parent chains. A gender gain cannot hide a usage loss. Do not use a
hidden average of their primary metrics.

## 3. Model-role hierarchy

| Role | Method | Purpose | Parent status |
|---|---|---|---|
| Lower bound | Training-fold majority predictor | Exposes misleading accuracy | Never a parent |
| Sanity lower bound | Training-fold stratified predictor | Checks metric and seed wiring | Never a parent |
| Integrity check | Bias-only, shuffled labels, tiny-batch overfit | Detects broken code or leakage | Never a parent |
| Shortcut warning | Ground-truth `articleType` lookup | Measures metadata association only | Forbidden model input |
| Classical reference | Exact HOG-plus-HSV logistic regression | Measures hand-built visual evidence | Comparison reference |
| Primary learnable baseline | Exact small scratch CNN | Starts each target chain | First parent |
| Conditional child | One method selected from parent evidence | Tests one named weakness or trade-off | Parent only if accepted |
| Pretrained benchmark | ImageNet-pretrained ResNet-18 | Measures transfer gap | Never an eligible parent |

## 4. Lower bounds and integrity checks

### 4.1 Majority predictor

The development majority shares are about 54.17% for `Men` and 76.75% for `Casual`. Their
descriptive full-development macro-F1 values are only about 0.1405 and 0.0965.

The real OOF baseline derives its majority from each training complement. It decides whether an image
method learns more than class priors.

### 4.2 Stratified random predictor

Sample labels from the training-side prior using the fixed seed. This checks encoding, class order,
masks, metrics, and repeatability. It is not a practical candidate.

### 4.3 Bias-only model

A trainable bias vector with no image features should converge toward training priors. It checks the
training loop and, later, the shared-mask implementation.

### 4.4 Label-shuffle test

Shuffle training targets while leaving validation labels unchanged. Validation should fall near
chance or prior behaviour. A strong result indicates leakage, ID misalignment, accidental metadata
input, cache contamination, or a metric bug.

### 4.5 Tiny-batch overfit test

Train on 16–64 examples with augmentation disabled. The small CNN should nearly memorise them.
Failure means the tensor, label map, mask, loss, optimiser, or model mode is broken. Do not run a
full model while this test fails.

### 4.6 Article-type lookup is not a baseline

Article-type majority lookup agrees descriptively with about 90.0% of usage labels and 78.5% of
gender labels. It uses another correct answer field and cannot be supplied to the core image model.
Keep it only as a shortcut diagnostic and later error slice.

## 5. Exact classical comparison reference

### 5.1 Features

Use:

- grayscale HOG with 9 orientations, `(8,8)` pixels per cell, `(2,2)` cells per block, and
  `L2-Hys` block normalisation;
- one 16-bin histogram for each HSV channel.

HOG is an established local-gradient and shape descriptor.
[Dalal and Triggs](https://doi.org/10.1109/CVPR.2005.177).

Fit feature scaling inside each training complement only.

### 5.2 Classifier

Use multinomial logistic regression with:

```text
penalty=L2
C=1.0
solver=lbfgs
max_iter=2000
class_weight=None
```

Reasons:

- it provides direct multiclass probabilities;
- its regularisation has a clear meaning;
- it is fast enough for all folds;
- it uses only project data;
- it provides an algorithmically different reference to the CNN.

For fold 4 usage, restore the fixed `Home` output column with zero learned positive probability.

### 5.3 Interpretation

If the classical reference is near the small CNN, shape and colour may be sufficient or the CNN may
be under-trained. If it loses badly, learned local combinations and context matter. Either outcome
is useful comparison evidence.

## 6. Exact primary learnable baseline

Use the same design for both targets but train two separate models.

### 6.1 Input contract

```text
height=80
width=60
EXIF transpose=true
RGB conversion=true
preserve aspect ratio=true
letterbox only unusual geometry=true
resize interpolation=LANCZOS
padding colour=white before scaling
normalisation=fold-training RGB mean/std over content pixels
standardised padding=0
random augmentation=none
```

### 6.2 Architecture

```text
3x80x60 input

Conv3x3 3->32, bias=false, padding=1
BatchNorm, ReLU, MaxPool2x2

Conv3x3 32->64, bias=false, padding=1
BatchNorm, ReLU, MaxPool2x2

Conv3x3 64->128, bias=false, padding=1
BatchNorm, ReLU, MaxPool2x2

Conv3x3 128->256, bias=false, padding=1
BatchNorm, ReLU

Adaptive global average pooling
Linear 256->5 or Linear 256->9
```

The design has about 0.39 million parameters. Three pooling operations preserve about a `10×7`
map before global pooling. Global pooling avoids a large flattened dense head. The baseline has no
dropout.

### 6.3 Training recipe

```text
initialisation=Kaiming random
optimizer=AdamW
learning_rate=0.001
weight_decay=0.0001
batch_size=128
epochs=30
scheduler=cosine decay to 0.00001
mixed_precision=false
early_stopping=false
OOF_checkpoint=final epoch
seed=2753
```

The batch and epoch settings are bounded engineering defaults. Their adequacy is tested by the
curves. If the model is still learning at epoch 30, extend only the budget before changing another
factor.

### 6.4 Why ordinary cross-entropy

Ordinary unweighted cross-entropy is a clean reference. It lets the first per-class report show
whether the optimiser ignores minority classes. Imbalance methods become eligible only after that
failure is observed.

### 6.5 Why no augmentation

The first run must show the raw train-validation gap and corruption sensitivity. EDA shows that
large crops and geometry changes can remove or distort the tiny product, so no stochastic transform
is accepted before model evidence exists.

## 7. Parent diagnosis

Every E1 parent must answer:

1. Did loss remain finite and did the tiny-batch test pass?
2. Did training settle within the fixed budget?
3. Are training and validation both weak, or is there a large gap?
4. Does pooled OOF macro-F1 beat lower bounds and how does it compare with the classical reference?
5. Which supported classes collapse or flood?
6. What do fixed failure examples show?
7. Which core corruption causes the largest drop?
8. Does parameter, checkpoint, training, or measured inference cost fail a named limit?

The next hypothesis uses the strongest observed weakness. If several weaknesses exist, use this
priority:

1. integrity or incomplete optimisation;
2. general underfit or overfit;
3. supported-class collapse;
4. robustness or lost-detail evidence;
5. deployment cost;
6. optional sharing.

## 8. Conditional architecture children

### 8.1 Low-resolution-stem ResNet-18

Residual connections can make deeper networks easier to optimise.
[ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

Use this child only when the accepted parent's train and validation evidence both remain weak and
close after optimisation has settled.

Use:

```text
3x3 convolution, stride 1
normalisation
activation
no early max-pool
standard residual stages
global average pooling
target head
```

Keep input, augmentation, loss, optimiser, seed, folds, and budget fixed. The main changed factor is
the architecture. Reject it when underfit remains, overfit grows, or cost is not justified.

### 8.2 Scratch MobileNetV3-Small

MobileNetV3 is a compact, hardware-aware family.
[MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf).

Use it only when an accepted parent has acceptable quality but fails a named latency, storage, or
memory limit. Keep every recipe factor fixed and measure actual hardware cost. Nominal FLOPs do not
prove latency.

Do not run MobileNet beside ResNet as an automatic screen.

### 8.3 Later capacity option

ResNet-34 or EfficientNet-B0 may be considered only when an accepted smaller parent still shows
underfit and the next written hypothesis names capacity as the one factor. Do not run both.

Do not prioritise ResNet-50/101, ConvNeXt, large EfficientNets, ViT, or Swin without new evidence.

## 9. Loss design and masking

### 9.1 Separate models

```text
loss_target = sum(mask_i * cross_entropy_i) / sum(mask_i)
```

Gender has all masks true. Usage has one false mask.

### 9.2 Shared two-head model

If sharing is later triggered:

```text
loss = lambda_gender
       * sum(gender_mask_i * gender_ce_i) / sum(gender_mask_i)
       + lambda_usage
       * sum(usage_mask_i * usage_ce_i) / sum(usage_mask_i)
```

Start with `lambda_gender=lambda_usage=1`. ID `28319` can update gender and the shared backbone
but contributes zero usage loss. Literal `NA` contributes normal usage loss.

### 9.3 Fixed output space

Every usage head outputs nine logits in every fold. Fold 4 gives the `Home` logit no positive
training example. Report it as untrainable regardless of its prediction.

## 10. Conditional imbalance methods

### 10.1 Trigger

An imbalance child is permitted only when:

- ordinary CE has completed;
- common classes show useful learning;
- one or more supported minority classes collapse or are rarely predicted;
- failure review does not show that ambiguity or missing evidence is the main cause.

### 10.2 First imbalance child

Use capped class-balanced effective-number weights:

```text
beta=0.999
raw_weight[c]=(1-beta)/(1-beta**training_count[c])
normalise nonzero weights to mean 1
cap each resolved weight at 5
normalise loss by the sum of applied sample weights
```

Compute counts inside the outer training complement. A zero-count class receives no invented
positive weight.

[Class-balanced loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html).

Judge it using macro-F1, per-class precision and recall, predicted counts, accuracy cost,
calibration, fold stability, and failure examples. Reject it when rare predictions flood or common
performance suffers severely.

### 10.3 Later imbalance options

Weighted sampling or focal loss may be considered only after the first imbalance child is rejected
and a new diagnosis shows why loss scale was not enough.

Do not:

- combine weighting and sampling in the first child;
- use uncapped inverse-frequency weights;
- repeatedly oversample `Home`;
- claim that loss engineering creates missing evidence.

## 11. Conditional resolution and transform children

### 11.1 Resolution

`(128,96)` is available only when:

- fixed errors show small or lost visual detail;
- the accepted parent is not materially overfit;
- the extra cost can be measured.

Keep architecture, loss, augmentation, optimiser, seed, folds, and budget fixed. Do not default to
224×224.

### 11.2 Light augmentation

Use only when the accepted parent shows overfit or a matching robustness weakness:

- horizontal flip probability 0.5;
- rotation up to 5 degrees;
- translation up to 5%;
- small scale jitter without destructive cropping;
- brightness and contrast changes up to 10%.

Do not add Mixup, CutMix, random erasing, or a second loss change in the same child.

### 11.3 Strong augmentation and regularisation

Mixup, CutMix, random erasing, dropout, label smoothing, or a weight-decay change each requires a
new parent diagnosis and a separate hypothesis. Mask-safe tests are required before mixing methods.

## 12. Separate and shared systems

### 12.1 Separate models are the correctness starting point

- Each target gets full backbone capacity.
- Each target may accept a different child.
- Task gradients cannot conflict.
- Usage masking cannot alter gender training.
- Failure analysis is simpler.

Costs are two image passes, two checkpoints, and duplicate features.

### 12.2 Shared model trigger

Both outputs come from the same image, so sharing may reduce cost. It can also create negative
transfer because the tasks have different imbalance and visual difficulty.

[Which Tasks Should Be Learned Together?](https://proceedings.mlr.press/v119/standley20a.html).

A shared comparison is permitted only when:

- separate parents have complete evidence;
- combined cost fails a named application limit;
- one common backbone and matched recipe can be defined;
- the no-harm margin is frozen first.

Compare matched gender-only, usage-only, and shared systems. Change only the task-sharing structure.

```text
delta_gender = gender_macro_F1_shared - gender_macro_F1_separate
delta_usage  = usage_macro_F1_shared  - usage_macro_F1_separate
```

Require the lower 95% paired family-bootstrap bound to exceed `-0.01` for both outputs, no
supported-class collapse, and material combined cost reduction.

Only a near-pass with valuable savings can trigger one later conflict-method hypothesis, such as
uncertainty weighting or PCGrad.

## 13. Why a combined 45-way class is rejected

- Nineteen possible pairs are unobserved.
- Rare classes become rarer joint classes.
- One missing usage label removes valid gender supervision.
- Unseen combinations are difficult to express.
- Official outputs still need separate columns.
- Per-output error, calibration, and loss weighting become unclear.

A shared two-head model can test common features without these problems.

## 14. Optimisation and budget

The baseline uses AdamW and one fixed cosine schedule. Optimiser, learning rate, schedule, or epoch
budget changes are children only when the accepted parent's curves justify them.

Examples:

- still improving at epoch 30: change only the budget;
- unstable or divergent loss: change only learning rate after integrity checks;
- credible but poor ResNet optimisation: one SGD-momentum child may be justified.

Do not run a broad optimiser grid across architecture, resolution, augmentation, and loss.

## 15. Pretrained comparison-only system

After eligible selection is fixed, one ImageNet-pretrained ResNet-18 may measure the transfer gap.
Use one clearly defined fine-tuning protocol.

Record:

```text
scratch=false
pretrained_source=ImageNet-1K
submission_eligible=false
official_prediction_eligible=false
application_eligible=false
```

It cannot become an eligible parent or change the eligible winner.

Hidden external learning also includes pretrained feature extractors, foundation embeddings,
external self-supervised checkpoints, pretrained segmentation, and external pseudo-labels.

## 16. Cost and deployment evidence

For every E1 parent and child, record at least:

- parameter count;
- checkpoint bytes;
- seconds per epoch and total training time;
- peak accelerator memory;
- named-device batch-1 p50/p95 latency.

For finalists, add MACs/FLOPs, CPU and GPU timing where relevant, throughput, load time, and peak
inference memory.

Compare shared cost with the sum of both separate systems. A simpler model may win when performance
is practically tied and it is cheaper, more stable, better calibrated, or more robust.

## 17. Per-target decision emphasis

### Gender

Watch for:

- `Boys` and `Girls` collapsing to adult categories;
- `Unisex` collapsing to `Men` or `Women`;
- article-type, person, colour, or background shortcuts;
- a performance-cost trade-off that could justify a compact architecture.

### Usage

Watch for:

- `Casual` majority collapse;
- rare-class flooding after weighting;
- `Formal` versus `Smart Casual`;
- `Casual` versus `Travel` or `Party`;
- business labels that are not visible from pixels;
- the one-row `Home` class, which never selects a child.

## 18. Acceptance logic

Advance a child only when:

- split, mask, scratch, and registry assertions pass;
- its complete diagnostic bundle exists;
- it answers the written hypothesis;
- paired evidence against its parent shows a useful or practical gain;
- supported classes do not collapse without an accepted trade-off;
- gains do not depend on `Home`;
- added complexity and cost are justified.

Reject or stop when:

- the run violates data or eligibility rules;
- the gain disappears in paired comparison;
- minority gains come from uncontrolled false positives;
- the method needs extreme rare-example duplication;
- a larger model adds overfit rather than useful capacity;
- a shared model harms either target beyond the margin;
- failures are mainly label ambiguity or missing evidence;
- another run would reduce time needed for diagnosis and final judgement.

The exact loop and artifact requirements are in
[04_experiment_plan.md](04_experiment_plan.md). Metrics and statistical rules are in
[05_evaluation_framework.md](05_evaluation_framework.md).
