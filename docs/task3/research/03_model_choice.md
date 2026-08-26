# Task 3 model choice

[Previous: data and validation](02_data_and_validation_design.md) · [Research index](README.md) ·
[Next: experiment plan](04_experiment_plan.md) · [References](references.md)

## 1. Decision to make

Task 3 does not need the model with the highest isolated score at any cost. It needs the most
credible method under these constraints:

- 32,773 development products;
- small 60×80 teacher images;
- five family-safe folds;
- five gender classes;
- nine usage classes;
- severe imbalance;
- one truly missing usage label;
- one `Home` development example;
- scratch training for the submitted system;
- separate official output columns;
- a report that rewards comparison, judgement, and honest failure analysis.

**Recommendation.** Treat model selection as a staged question:

1. Does the pipeline learn more than class priors?
2. Do learned features beat classical image features?
3. Which small scratch CNN gives the best stable evidence for each target?
4. Does extra resolution help?
5. Does imbalance treatment help without creating false positives?
6. Can one shared backbone match separate models while reducing cost?
7. Is more capacity justified by underfitting evidence?
8. How large is the transfer-learning gap, using an ineligible benchmark?

## 2. Selection principles

### 2.1 Start simple and add complexity for a measured reason

Every added method must answer a named question. Examples:

- HOG asks whether hand-built shape and colour are sufficient.
- A small CNN asks whether learned local features help.
- ResNet-18 asks whether a stronger scratch optimiser and larger capacity help.
- MobileNetV3 asks whether a compact mobile family gives a better cost tradeoff.
- A shared model asks whether common features reduce system cost without negative transfer.
- Class-balanced loss asks whether minority recall improves without broad degradation.

Do not add a model only because it is modern or famous.

### 2.2 Hold the comparison boundary fixed

Within a controlled comparison, keep fixed:

- development folds;
- label maps and masks;
- input resolution unless resolution is the tested variable;
- augmentation unless augmentation is the tested variable;
- optimiser and training exposure where possible;
- seed during one-seed screening;
- evaluation code;
- OOF aggregation;
- cost-measurement hardware.

### 2.3 Separate eligibility from scientific interest

A pretrained model may be scientifically useful because it measures the value of transfer learning.
It remains ineligible for submission. Eligibility must be a registry field, not a note remembered at
the end.

### 2.4 Select each output honestly

Gender and usage can have different best separate models. Do not force both to use one architecture
unless the shared-system evidence justifies that constraint.

Do not hide a usage loss behind a gender gain by averaging the two primary scores.

## 3. Model-choice decision tree

```text
Start
│
├─ Does the method load pretrained weights or learned external features?
│  ├─ Yes → comparison-only lane
│  │         Record scratch=false and submission_eligible=false
│  └─ No  → submission-eligible lane
│
├─ Does the method preserve separate gender and usage outputs and masks?
│  ├─ No → reject
│  └─ Yes
│
├─ Establish lower bounds
│  ├─ majority predictor
│  ├─ stratified random predictor
│  ├─ shuffled-label and tiny-batch sanity tests
│  └─ HOG + colour + linear classifier
│
├─ Train a small scratch CNN
│  ├─ cannot overfit a tiny batch → debug; do not run the matrix
│  ├─ train and validation both weak → likely underfit; add capacity
│  └─ train strong, validation weak → regularise before adding capacity
│
├─ Compare compact scratch families
│  ├─ MobileNetV3-Small
│  └─ low-resolution-stem ResNet-18
│
├─ Compare input resolution and light augmentation
│
├─ Compare unweighted and one capped imbalance treatment
│
├─ Choose best separate candidate per target
│
├─ Build matched shared backbone with two heads
│  ├─ both tasks non-inferior and system cheaper → shared finalist
│  ├─ one task harmed → separate models win
│  └─ small conflict with large cost benefit → try one conflict method
│
├─ Do learning curves prove remaining underfit?
│  ├─ Yes → one larger family: ResNet-34 or EfficientNet-B0
│  └─ No  → stop adding capacity
│
├─ Run one pretrained ResNet-18 comparison benchmark
│
└─ Confirm eligible finalists with all five folds and three seeds
```

## 4. Non-neural baselines

### 4.1 Majority predictor

**Repository facts.** The development majority shares are about 54.17% for `Men` and 76.75% for
`Casual`. If a predictor always chooses the majority class, the descriptive full-development
macro-F1 values are only about:

- gender: 0.1405;
- usage: 0.0965.

These calculations show why accuracy alone is misleading. The actual fold baseline must derive the
majority from each fold’s training complement.

**Decision supported.** Does a candidate learn meaningful class separation beyond prior frequency?

**Keep.** This baseline is mandatory, even though it is simple.

### 4.2 Stratified random predictor

Sample a label according to the training-side class distribution.

**Decision supported.** Are label encoding, class order, masks, metrics, and random-seed handling
working as expected?

**Keep as sanity evidence.** It is not a practical candidate.

### 4.3 No-image prior model

A trainable bias vector with no image features should converge toward training priors.

**Decision supported.** Does the training loop reproduce the expected prior-only behaviour? This is
especially useful for checking the shared loss and mask implementation.

### 4.4 Label-shuffle test

Shuffle targets within the training side while leaving validation labels unchanged.

Expected result: validation falls near chance/prior behaviour.

If it remains strong, inspect:

- family leakage;
- ID/label alignment;
- accidental metadata input;
- validation images in training caches;
- a metric calculated against the wrong target.

### 4.5 Tiny-batch overfit test

Train on 16–64 examples with augmentation disabled. A CNN should almost memorise this tiny set.

Failure means the image tensor, label map, mask, loss, optimiser, or model mode is broken. Do not
interpret a large-model run until this test passes.

## 5. Classical image-feature baseline

### 5.1 Recommended feature design

Use a concatenation of:

- HOG features from a grayscale view;
- a compact HSV or RGB colour histogram;
- optional coarse downsampled RGB pixels;
- optional simple foreground/background occupancy features.

Fit feature standardisation on each training complement only.

HOG is an established descriptor for local gradients and shape.
[Dalal and Triggs, HOG](https://doi.org/10.1109/CVPR.2005.177).

### 5.2 Recommended classifier

Start with multinomial logistic regression.

Reasons:

- direct multiclass probabilities;
- easier probability analysis than an uncalibrated SVM;
- interpretable regularisation strength;
- fast enough for all five folds;
- scratch-compatible because it learns only from project data.

A linear SVM is an optional second classical comparison. Its margins need a separate calibration
procedure before probability metrics can be compared fairly.

### 5.3 What it can reveal

If HOG-plus-colour is close to a CNN:

- the tiny images may contain mostly shape and colour information;
- the CNN may be under-trained;
- a smaller deployment model may be enough.

If it loses badly:

- learned context and local feature combinations matter;
- usage likely needs more than silhouette and colour;
- the classical result still provides useful breadth and a lower bound.

### 5.4 Expected limitations

- HOG is weak at semantic occasion labels.
- Colour can amplify stereotypes and background shortcuts.
- A linear decision surface may not separate visually similar usage classes.
- Hand-built features cannot easily learn which item in a multi-product image is the labelled one.

## 6. Small custom CNN

### 6.1 Purpose

The small CNN is a capacity and engineering anchor. It should be easy to understand and fast to run.

Suggested shape:

- three or four convolution stages;
- `3×3` kernels;
- gradually increasing channels, for example 32–64–128–256;
- normalisation and ReLU or SiLU;
- controlled downsampling;
- global average pooling;
- small dropout in the classifier head;
- one target head for separate models or two heads for the shared experiment.

### 6.2 Why global average pooling

- Fewer parameters than a large flattened dense layer.
- Accepts the two proposed input resolutions more easily.
- Reduces overfitting risk.
- Gives a clear final feature vector for separate or shared heads.

### 6.3 Decisions supported

- Does end-to-end feature learning beat HOG?
- Is a named architecture necessary?
- Is the data pipeline healthy before expensive models?
- Does light augmentation reduce the train–validation gap?

### 6.4 Failure interpretation

| Observation | Likely meaning | Next action |
|---|---|---|
| Low training and validation performance | Underfit or optimiser problem | Check learning rate, then add capacity |
| High training, weak validation | Overfit | Add light augmentation, weight decay, or dropout |
| Strong gender, collapsed usage | Usage imbalance/ambiguity | Inspect loss, per-class recall, and shortcuts |
| Highly unstable folds | Family or rare-class dependence | Keep fold detail; do not jump to a larger model immediately |

## 7. ResNet family

Residual learning introduced shortcut connections that make deeper networks easier to optimise.
[ResNet paper](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

### 7.1 Recommended ResNet-18 adaptation

Use a low-resolution stem:

```text
3×3 convolution, stride 1
normalisation
activation
no early max-pool
standard residual stages
global average pooling
target head(s)
```

Do not use the standard ImageNet `7×7`, stride-2 convolution followed by stride-2 max-pooling as the
only ResNet configuration. On a 60×80 image, it reduces spatial detail to roughly 15×20 before the
main residual stages.

### 7.2 Why ResNet-18 is a strong main candidate

- Moderate capacity for 32k images.
- Residual connections help scratch optimisation.
- Easy to adapt to small inputs.
- Easy to build matched separate and shared variants.
- Common enough that its design and cost are easy to explain.
- A useful architecture match for the pretrained comparison lane.

### 7.3 Risks

- About 11 million parameters in a standard form may still overfit.
- Batch normalisation can be unstable with very small batches.
- A standard library implementation may silently retain the unsuitable ImageNet stem.
- A pretrained constructor may accidentally download weights unless eligibility is asserted.

### 7.4 When to test ResNet-34

Only when:

- ResNet-18 training macro-F1 remains clearly below a useful level;
- training and validation curves remain close, suggesting underfit rather than overfit;
- optimisation checks pass;
- the extra cost fits the experiment budget.

Do not test ResNet-50 by default. More capacity is not evidence of a better fit for this small-image
scratch problem.

## 8. MobileNetV3 family

MobileNetV3 combines mobile-oriented blocks with hardware-aware architecture search.
[MobileNetV3 paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf).

### 8.1 Why MobileNetV3-Small belongs in the main matrix

- Small parameter count.
- Low nominal operation count.
- Attractive for the application.
- Gives a meaningful efficiency contrast with ResNet-18.
- A shared MobileNet could make both outputs with one compact image pass.

### 8.2 Risks

- Depthwise operations are not equally fast on every CPU or runtime.
- Scratch optimisation may be less forgiving than ResNet-18.
- Aggressive internal downsampling may still lose tiny-image detail.
- Nominal FLOPs may not predict measured latency.

### 8.3 Adaptation requirements

- Check the first stride and total downsampling at both proposed resolutions.
- Initialise all learned weights randomly.
- Use a task-sized classifier head.
- Measure actual CPU and GPU latency.
- Do not assume the library’s ImageNet preprocessing is correct for the scratch lane.

## 9. EfficientNet and other capacity candidates

EfficientNet proposes compound scaling of depth, width, and resolution.
[EfficientNet paper](https://proceedings.mlr.press/v97/tan19a.html).

### EfficientNet-B0

Keep as a conditional third family because:

- it is compact;
- it provides a different scaling design;
- it may offer a useful cost/quality point.

Do not put it in the first minimum matrix unless compute permits. ResNet-18 and MobileNetV3-Small
already answer the most important capacity-versus-efficiency question.

### Large CNNs and Vision Transformers

Do not prioritise:

- ResNet-50/101;
- ConvNeXt;
- large EfficientNets;
- ViT or Swin trained from scratch;
- very deep custom networks.

Reasons:

- no project evidence yet says the smaller networks underfit;
- 60×80 images offer limited spatial detail;
- scratch transformers usually need careful recipes and more data;
- their cost reduces the number of controlled folds, seeds, and error analyses the team can afford;
- the rubric rewards justified comparison, not model size.

They become reasonable only after a smaller-family learning-curve diagnosis proves a capacity limit.

## 10. Loss design and label masking

### 10.1 Separate models

For each target, train only rows where `mask_target[i]` is true:

```text
loss_target = sum_i(mask_target[i] * cross_entropy_target[i])
              / sum_i(mask_target[i])
```

Gender has all masks true. Usage has one false mask.

### 10.2 Shared two-head model

Use:

```text
loss = lambda_gender
       * sum_i(gender_mask[i] * gender_cross_entropy[i]) / sum_i(gender_mask[i])
       + lambda_usage
       * sum_i(usage_mask[i] * usage_cross_entropy[i]) / sum_i(usage_mask[i])
```

Start with `lambda_gender = lambda_usage = 1` after each head’s loss is normalised by valid rows.

This means:

- ID `28319` can update gender and the shared backbone;
- it cannot update usage loss;
- literal `NA` updates usage normally;
- the loss scale does not shrink merely because one target has a missing row.

### 10.3 Fixed output space

Every CNN usage head should output nine logits in every fold, including fold 4 where `Home` has zero
positive training examples.

The fold-4 `Home` logit receives only indirect negative softmax pressure. It has no positive evidence.
The result must be labelled untrainable, regardless of the prediction.

A classical classifier may internally omit an absent training class. Its output adapter must restore
the fixed nine-class order and assign no learned positive probability to that absent class.

### 10.4 Cross-entropy baseline

Start with ordinary multiclass cross-entropy because it:

- has a clear interpretation;
- provides a clean reference;
- works with fixed class heads;
- supports class weights later;
- avoids adding an imbalance method before measuring the problem.

## 11. Imbalance methods

### 11.1 Why accuracy is insufficient

Always predicting `Casual` gives roughly 76.75% descriptive development accuracy while failing every
other usage class. The majority result therefore looks useful under accuracy and useless under
macro-F1.

Imbalance treatment must be judged by:

- macro-F1;
- per-class precision and recall;
- predicted class counts;
- accuracy cost;
- calibration;
- fold and seed stability.

### 11.2 Unweighted cross-entropy

**Role:** reference method.

**Advantages:** stable, simple, usually good for common classes, no noisy rare-class amplification.

**Risk:** the optimiser can largely ignore `Party`, `Travel`, `Smart Casual`, and `NA`.

Keep it in every architecture screen.

### 11.3 Inverse-frequency class weights

Raw inverse frequency gives very large rare-class weights. With one `Home` example, the ratio to
`Casual` is 25,151:1. Using that directly can create unstable gradients and many false positives.

**Recommendation.** Do not use uncapped inverse-frequency weighting.

### 11.4 Class-balanced effective-number weights

The class-balanced method uses:

```text
class_weight[c] = (1 - beta) / (1 - beta ** class_count[c])
```

where `class_count[c]` is the training-side class count. It reduces the idea that every repeated example adds
fully independent information. [Class-balanced loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html).

Task 3 protocol:

1. Compute `class_count[c]` on the outer training complement.
2. Compute weights using a predeclared beta.
3. Normalise nonzero-class weights to mean one.
4. Cap the maximum weight using a predeclared cap.
5. Record the resulting vector.
6. Keep natural validation distribution.
7. For a zero-count class such as fold-4 `Home`, record no positive training support; never divide by
   zero or create a large synthetic weight.

The exact beta and cap belong in the bounded experiment plan. Do not search many values.

### 11.5 Weighted sampling

Weighted sampling changes how often examples appear. It can improve minority exposure, but:

- it repeatedly shows the same rare images;
- it increases memorisation risk;
- it changes the natural gender–usage pair distribution in a shared model;
- one `Home` image can be repeated many times without adding information;
- probability calibration can worsen because the training prior changes.

**Recommendation.** Test capped weighted sampling only if class-balanced loss is insufficient. Do not
combine both in the first comparison. Never apply the sampler to validation.

### 11.6 Focal loss

Focal loss reduces the contribution of easy examples. It was introduced for dense detection with a
large easy-negative imbalance. [Focal loss](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf).

Potential benefit:

- focuses learning on hard minority examples.

Risks here:

- hard examples may be noisy or label-ambiguous;
- rare classes can contain the least reliable evidence;
- gamma adds another tuning choice;
- it does not solve zero support for `Home`.

**Recommendation.** Keep focal loss outside the minimum matrix. Add one fixed-gamma comparison only
when ordinary and class-balanced cross-entropy both show easy-class domination.

### 11.7 Label smoothing

Small label smoothing can regularise an overconfident model, but it can:

- reduce the already weak signal for a rare class;
- complicate calibration interpretation;
- hide teacher-label uncertainty without modelling it explicitly.

Do not use it in the baseline. Consider a small fixed value only for a clearly overfit finalist, and
compare NLL/Brier as well as macro-F1.

### 11.8 Rare-class collection and hierarchy

In a real product, the best response to a one-example class would be more reviewed data or a changed
taxonomy. This assignment fixes the dataset and labels, so the model experiment cannot solve that
problem.

Do not claim that loss engineering replaces missing evidence.

## 12. Separate models

### 12.1 Why separate models are the correctness baseline

- Each target receives all backbone capacity.
- Each can choose a different winning architecture.
- Each can use a different imbalance method.
- Task gradients cannot conflict.
- Checkpoint selection and failure analysis are simpler.
- Usage masking has no effect on gender training.

### 12.2 Costs

- Two image passes.
- Sum of two checkpoints.
- Higher combined memory.
- More application integration work.
- Duplicate visual features.

### 12.3 Per-target likely needs

Gender may rely strongly on article type, silhouette, styling, and catalogue conventions. It has five
classes and moderate imbalance.

Usage has nine classes, extreme imbalance, and weakly visual business concepts. It may need:

- stronger features;
- more careful class treatment;
- more review and abstention;
- different calibration;
- a different winning backbone.

**Recommendation.** Do not assume one separate architecture must win both outputs.

## 13. Shared backbone with two heads

### 13.1 Why test sharing

Both outputs come from the same image. Shared early features can represent:

- shape;
- texture;
- colour;
- product region;
- article-type cues;
- person/background context.

One backbone may reduce inference cost substantially.

Fashion attribute research also motivates testing related attributes jointly, but it does not prove
that these two labels cooperate in this dataset.
[Fine-grained fashion attribute extraction](https://openaccess.thecvf.com/content/CVPR2021W/CVFAD/html/Parekh_Fine-Grained_Visual_Attribute_Extraction_From_Fashion_Wear_CVPRW_2021_paper.html).

### 13.2 Why sharing may fail

Research shows that multitask objectives can compete and make a shared model worse than separate
models. [Which Tasks Should Be Learned Together?](https://proceedings.mlr.press/v119/standley20a.html).

Task 3 conflict sources include:

- usage has a much more extreme long tail;
- common usage gradients may dominate shared features;
- gender can be easier and converge sooner;
- task-specific best augmentations may differ;
- article-type shortcuts may help both headline metrics while hurting rare cases;
- `Home` has no positive evidence in one fold;
- the one missing usage row produces only a gender gradient.

### 13.3 Fair M1 comparison

Build:

1. Gender-only model with backbone B.
2. Usage-only model with backbone B.
3. Shared model with one B and two heads.

Keep fixed:

- random seed;
- input size;
- augmentation;
- optimiser family;
- sample exposure;
- backbone definition;
- loss type per head;
- fold and metric code.

Compare shared performance against the matching separate target, and compare shared system cost
against the sum of the two separate systems.

### 13.4 Negative-transfer quantities

```text
delta_gender = gender_macro_F1_shared - gender_macro_F1_separate
delta_usage  = usage_macro_F1_shared  - usage_macro_F1_separate
```

The recommended no-harm rule is:

```text
lower 95% paired family-bootstrap bound > -0.01
```

for both outputs.

The one-percentage-point margin is a project recommendation, not a universal standard. Freeze it
before seeing M1 results.

### 13.5 Loss weighting

Start with equal weights after per-head valid-row normalisation.

If one task dominates, inspect:

- head loss magnitude;
- backbone gradient norm from each head;
- rate of negative gradient cosine;
- learning curves;
- per-class gains and losses.

Do not immediately tune many lambda values. That can create a hidden composite objective.

### 13.6 Conditional conflict methods

If M1 nearly passes and shared cost is valuable, test one method:

- uncertainty-based learned task weights; or
- PCGrad, which projects conflicting task gradients.

[Uncertainty weighting](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf),
[PCGrad](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

Reject further shared tuning when:

- one task remains outside the no-harm margin;
- gains come only from a rare one-example outcome;
- the shared model is not materially cheaper;
- complexity makes the conclusion hard to defend.

### 13.7 Partial sharing

A backbone that branches after early stages may reduce conflict while retaining some efficiency.
This is an optional fallback, not a minimum experiment. It adds another architecture choice and
should be attempted only when:

- full sharing shows clear early-feature benefit;
- late gradients conflict;
- deployment cost makes two full backbones unacceptable.

## 14. Why a combined 45-way class is rejected

The combined target has several structural problems:

- 19 possible pairs are unobserved;
- rare classes become rarer joint classes;
- one missing usage label removes valid gender supervision;
- unseen combinations are impossible without special decoding;
- official output conversion is artificial;
- a correct gender and wrong usage cannot be diagnosed cleanly;
- per-output calibration and abstention become difficult;
- separate loss weighting cannot address different imbalance levels.

A shared two-head model obtains the intended shared-feature benefit without these costs.

## 15. Input resolution and architecture interaction

### 15.1 Native-like `(80,60)`

Advantages:

- minimal invented pixels;
- low compute;
- fastest experiments and app inference;
- matches most source geometry.

Risks:

- small details disappear after network downsampling;
- rare style cues may be only a few pixels wide.

### 15.2 Upsampled `(128,96)`

Advantages:

- larger feature maps through early stages;
- may make optimisation and small patterns easier;
- preserves the 3:4 source aspect ratio.

Risks:

- no new information is created;
- interpolation can smooth detail;
- higher compute and memory;
- an apparent gain may come from a better architecture scale rather than true added evidence.

### 15.3 Recommendation

Compare the two sizes only on the leading small CNN family. Keep interpolation, augmentation, loss,
seed, folds, and training exposure fixed.

Do not default to 224×224. It increases work substantially and comes from common pretrained-model
recipes, not from this data.

## 16. Augmentation and regularisation

### 16.1 No-augmentation reference

Required to show the raw train–validation gap and to verify that augmentations help rather than
merely change the task.

### 16.2 Light augmentation — recommended main setting

- Horizontal flip.
- Rotation up to about 5°.
- Translation up to about 5%.
- Small scale jitter without destructive cropping.
- Mild brightness and contrast change.
- Mild saturation/hue change where colour meaning is preserved.

Why light:

- catalogue placement can shift;
- images are tiny;
- large crops can remove the product;
- strong colour changes can destroy real product cues;
- vertical flips are not realistic.

### 16.3 Mixup

Mixup linearly combines images and labels and can improve regularisation.
[Mixup paper](https://openreview.net/pdf?id=r1Ddp1-Rb).

Mask problem:

- if one source has missing usage, a mixed usage target needs a defined valid contribution;
- separate target masks must be mixed independently;
- probability calibration may change.

Use only after mask-safe tests exist. A simple safe policy is to apply mixup to rows with both Task 3
labels valid and train the one missing-usage row without mixup.

### 16.4 CutMix

CutMix replaces an image region and mixes labels by region area.
[CutMix paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf).

Risk at 60×80: one patch may remove most of the labelled product or insert a second product, which
already occurs naturally in some images.

Keep it optional and compare it only after light augmentation.

### 16.5 Random erasing

Potentially useful for occlusion robustness, but risky because the product can occupy few pixels.
Use a small erased area only, if at all.

### 16.6 Weight decay

Use as the main parameter regulariser. Search a small logarithmic range, not a broad continuous one.
Keep the value recorded per architecture because MobileNet and ResNet may respond differently.

### 16.7 Dropout

Use a small amount in the classifier head when training evidence shows overfitting. Heavy dropout in
the backbone is not the first choice.

### 16.8 Normalisation layer

Batch normalisation is reasonable when the effective batch is large enough. If hardware forces very
small batches, compare GroupNorm or use gradient accumulation while recognising that accumulation
does not change BatchNorm’s per-step statistics.

## 17. Optimiser, schedule, and training budget

### 17.1 Optimiser

Use one common optimiser during architecture screening so optimiser choice does not become a hidden
model advantage.

Recommended practical order:

1. AdamW for the first controlled screen because it needs less architecture-specific tuning.
2. One SGD-with-momentum comparison on the leading ResNet only if training curves suggest it could
   matter.

Do not run a wide optimiser search for every architecture.

### 17.2 Learning rate

Use a very small bounded set per family, selected before full confirmation. For example, two or three
log-spaced values. Record warmup and effective batch size.

### 17.3 Scheduler

A fixed cosine decay with short warmup is a defensible common schedule. The exact choice matters
less than using the same declared schedule in matched comparisons and recording it completely.

### 17.4 Epoch selection

Screening curves may identify a sensible budget. Finalist confirmation should use a frozen epoch or
schedule rather than selecting a different best validation checkpoint for every scored fold.

For a shared model, avoid a hidden average metric for checkpoint choice. Use a fixed schedule or a
predeclared Pareto rule that requires neither task to degrade.

### 17.5 Mixed precision

Mixed precision may reduce training cost. Record whether it is used. Verify that loss scaling is
stable for rare weighted classes and that full-precision evaluation gives equivalent predictions
within tolerance.

## 18. Pretrained comparison-only systems

### 18.1 Recommended benchmark

Use one ImageNet-pretrained ResNet-18 with the same five-fold target outputs.

Possible benchmark forms:

- frozen backbone plus trained heads; or
- full fine-tuning.

Full fine-tuning is the more realistic transfer benchmark, but one clearly defined form is enough.

### 18.2 Required separation

Record:

```text
scratch=false
pretrained_source=ImageNet-1K
submission_eligible=false
official_prediction_eligible=false
application_eligible=false
```

TorchVision represents random initialisation with `weights=None` and pretrained initialisation with a
weight enum. [TorchVision model documentation](https://docs.pytorch.org/vision/master/models.html).

### 18.3 What the benchmark decides

- How much external visual knowledge helps.
- Whether the scratch limitation is a major performance constraint.
- Whether poor rare-class results mainly come from labels/data rather than feature learning.

It does not decide the submitted winner.

### 18.4 Avoid hidden pretraining

The following also count as learned external information:

- pretrained feature extractors used before a classical head;
- self-supervised external checkpoints;
- foundation-model embeddings;
- pretrained segmentation used to crop the product;
- external pseudo-labels.

Keep all such systems outside the eligible lane unless the specification is explicitly changed.

## 19. Compute and deployment tradeoffs

### 19.1 Measurements

For every finalist, record:

- parameter count;
- checkpoint bytes;
- MACs or FLOPs at the exact resolution;
- training seconds per epoch and total;
- peak accelerator memory;
- CPU and GPU batch-1 p50/p95 latency;
- batch throughput;
- model load time;
- peak inference RAM.

### 19.2 Separate versus shared cost

Compare:

```text
separate cost = gender backbone + usage backbone
shared cost   = one backbone + two small heads
```

Do not compare shared cost with only one separate target. The application needs both outputs.

### 19.3 Practical selection

A model within one primary-metric percentage point of the leader can win when it is:

- materially faster;
- much smaller;
- more stable across seeds;
- better calibrated;
- more robust;
- simpler to explain and maintain.

The [evaluation framework](05_evaluation_framework.md) turns this into explicit winner rules.

## 20. Per-target recommendation

### Gender

Recommended order:

1. Majority baseline.
2. HOG-plus-colour logistic regression.
3. Small CNN.
4. MobileNetV3-Small and low-resolution ResNet-18.
5. Light augmentation.
6. Only then test class balancing if `Girls` or `Boys` recall remains poor.

Expected issue: article-type and styling shortcuts may produce high common-class performance while
hurting `Unisex`, `Girls`, and `Boys`.

### Usage

Recommended order:

1. Majority baseline, with macro-F1 emphasised.
2. HOG-plus-colour logistic regression.
3. Small CNN.
4. MobileNetV3-Small and low-resolution ResNet-18.
5. Light augmentation.
6. Capped class-balanced cross-entropy.
7. Optional sampler or focal loss only if the first imbalance method fails.

Expected issue: weak visual semantics and very rare classes. No model should be called strong on
`Home`; `Party`, `Travel`, `Smart Casual`, and `NA` need cautious class-level reporting.

### Shared system

Build it only after the best separate architecture and training recipe are known. Use the same
backbone and matched recipe. Equal normalised head losses are the first configuration.

## 21. Candidate decision table

| Candidate | Include? | Main reason | Main risk | Eligible? |
|---|---|---|---|---|
| Majority predictor | Yes | Exposes accuracy illusion | No image learning | Baseline only |
| Stratified random | Yes | Pipeline sanity | Not useful operationally | Baseline only |
| HOG+colour logistic | Yes | Classical scratch breadth | Weak semantics | Yes |
| Small CNN | Yes | Capacity and pipeline anchor | May underfit | Yes |
| MobileNetV3-Small | Yes | Efficiency candidate | Hardware-dependent latency | Yes, random weights |
| Low-res ResNet-18 | Yes | Main scratch optimisation candidate | Larger checkpoint | Yes, random weights |
| ResNet-34 | Conditional | Capacity check | Extra cost/overfit | Yes, random weights |
| EfficientNet-B0 | Conditional | Alternative compact scaling | More matrix breadth/cost | Yes, random weights |
| One shared backbone, two heads | Yes after separate baseline | Efficiency and related features | Negative transfer | Yes, random weights |
| Partial-sharing model | Conditional | Conflict/efficiency compromise | Added complexity | Yes, random weights |
| Combined 45-way class | No | None beyond shared features | Sparse, mask and output problems | Reject |
| Large ResNet/ConvNeXt/ViT | Not initially | Capacity | Poor evidence-to-cost fit | Possible but not recommended |
| Pretrained ResNet-18 | Yes, separate lane | Measures transfer gap | Violates submitted-model rule | No |
| Foundation-model features | No minimum need | External comparison | Hidden pretraining and scope change | No |

## 22. Model-choice acceptance logic

Advance a candidate when:

- its data and eligibility assertions pass;
- pooled OOF primary performance beats the required lower bound;
- gains are not caused only by `Home`;
- no supported class collapses without explanation;
- fold and seed variation are acceptable;
- training curves show a credible fit;
- its added complexity answers a named question.

Reject or stop investing when:

- it violates scratch or split rules;
- it uses ground-truth metadata unavailable to the image-only model;
- its gain disappears in paired family-bootstrap comparison;
- it improves accuracy by worsening minority failure;
- it needs extreme rare-example duplication;
- it is slower and larger without a stable primary gain;
- a shared model harms either task beyond the no-harm margin;
- a larger model shows more overfit rather than more useful capacity;
- the experiment would reduce time needed for error analysis and final judgement.

The exact experiment IDs and stopping budget are in
[04_experiment_plan.md](04_experiment_plan.md). Metrics and statistical rules are in
[05_evaluation_framework.md](05_evaluation_framework.md).
