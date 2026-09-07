# Task 3 data and validation design

[Previous: problem and lifecycle](01_problem_and_lifecycle.md) · [Research index](README.md) ·
[Next: model choice](03_model_choice.md) · [References](references.md)

## 1. Purpose

This document fixes the Task 3 data boundary before model work begins. It defines:

- which rows and images may be used;
- what each label and mask means;
- how the five saved folds are used;
- which statistics must be fitted inside a training fold;
- how pooled out-of-fold predictions are formed;
- how leakage is detected and prevented;
- which limitations must remain visible in later reports.

No Task 3 experiment may create another random split.

## 2. Canonical data boundary

**Repository fact.** [`data/processed/splits.csv`](../../../data/processed/splits.csv) is the only
split. Its canonical partitions are:

| Partition | Products | Share of image-backed rows | Role |
|---|---:|---:|---|
| Development | 32,773 | 84.878% | Model development, five-fold comparison, final refit |
| Holdout | 5,778 | 14.964% | One independent evaluation after method freeze |
| Quarantine | 61 | 0.158% | Excluded because of data-integrity concerns |

The partition evidence is saved in
[`partition_summary.csv`](../../../results/evidence/data_preparation/partition_summary.csv).

The development fold assignment has this SHA-256 digest:

```text
bad7bc4ae65fbbfd815567f4ccfa308d6e57dc650bc15c0b8e798867a335f2fd
```

The partition ID-set digests are:

```text
development 2639d731f89942fd9598f9552bb092dd634830c3e6a63d0b569cfc2aa4362d01
holdout     48a9e2e389657744d2771a3ab5ca34e85e015b21b18ef4d0ed29c2e648cf8fc0
quarantine  5a672e34e485c8ddad0cef37c2dc37d99f8774f8874279a5d0ed69aff7022f43
```

**Recommendation.** Store these digests in every Task 3 run record. Refuse to compare runs when the
split digest differs.

## 3. Image scope

**Repository fact.** Tasks 1–3 use teacher images only. The optional external Fashion Product Images
collection is owned by Task 4 and must not enter Task 3. See
[decision 0015](../../decisions/0015-teacher-only-shared-image-preparation.md).

Development image properties include:

- 32,773 image-backed development products;
- usual dimensions of width 60 and height 80;
- 294 grayscale images, retained rather than deleted;
- 12 images with unusual dimensions;
- mostly bright catalogue backgrounds in the audited sample;
- product photos that may include a person, several items, or a small labelled item.

Visual evidence is in the
[data-preparation figures](../../../results/figures/data_preparation/), including target
distributions, transform risks, rare-class support, shortcut heatmaps, and contact sheets.

**Recommendation.** Keep all valid teacher images. Convert them deterministically to RGB. Use
aspect-preserving resize or letterbox for unusual geometry. Do not delete grayscale or unusual-size
images just because they are uncommon; make them evaluation slices.

## 4. Target facts

### 4.1 Gender

All 32,773 development products have a valid gender label.

| Label | Products | Development share |
|---|---:|---:|
| Men | 17,753 | 54.170% |
| Women | 12,027 | 36.698% |
| Unisex | 1,766 | 5.389% |
| Boys | 684 | 2.087% |
| Girls | 543 | 1.657% |

The largest-to-smallest ratio is about 32.69:1. All five classes appear in every validation fold and
in every training complement.

### 4.2 Usage

32,772 development products have a valid usage label. Product ID `28319` has a valid gender label
but a truly blank usage label.

| Label | Products | Development share |
|---|---:|---:|
| Casual | 25,151 | 76.745% |
| Sports | 3,346 | 10.210% |
| Ethnic | 2,183 | 6.661% |
| Formal | 1,949 | 5.947% |
| NA | 61 | 0.186% |
| Smart Casual | 47 | 0.143% |
| Travel | 22 | 0.067% |
| Party | 12 | 0.037% |
| Home | 1 | 0.003% |

The largest-to-smallest ratio is 25,151:1.

`Home` occurs in one product family in fold 4. The fold-4 training complement therefore contains
eight usage classes while fold-4 validation contains all nine. The other validation folds contain
eight usage classes because their validation side does not contain `Home`.

The canonical evidence is in
[`development_class_summary.csv`](../../../data/processed/development_class_summary.csv) and
[`cv_fold_summary.json`](../../../data/processed/cv_fold_summary.json).

### 4.3 Joint gender–usage support

Only 26 of the possible 45 combinations are observed. The valid-label counts are:

| Gender | Casual | Ethnic | Formal | Home | NA | Party | Smart Casual | Sports | Travel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Boys | 660 | 8 | 0 | 0 | 0 | 0 | 0 | 16 | 0 |
| Girls | 536 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Men | 13,319 | 78 | 1,857 | 0 | 19 | 0 | 38 | 2,440 | 1 |
| Unisex | 1,490 | 0 | 1 | 1 | 15 | 0 | 0 | 241 | 18 |
| Women | 9,146 | 2,090 | 91 | 0 | 27 | 12 | 9 | 649 | 3 |

This sparsity is evidence against a combined gender–usage class. It is also a reason to slice errors
by the pair: a model may learn common business combinations instead of visual evidence.

## 5. Label and mask contract

The canonical target fields are:

```text
gender
usage
has_gender_label
has_usage_label
```

Rules:

1. A true `has_<target>_label` value means the row may contribute to that target’s loss and metric.
2. A truly blank target is missing.
3. Literal `usage="NA"` is a valid class and has `has_usage_label=True`.
4. The label maps are built from all nonblank image-backed development labels.
5. A rare label is never deleted, renamed, merged, or masked because of low support.
6. A validation label absent from that fold’s training complement remains in validation and is
   reported as an untrainable-class limitation.
7. Holdout and quarantine targets remain blank in persisted shared artifacts until explicit final
   evaluation unlock.

These rules come from [decision 0016](../../decisions/0016-development-label-scope.md) and the
dataset contract in [`dataset.py`](../../../src/fashion/data/dataset.py).

### Masked training behaviour

For separate models:

- Gender uses all 32,773 development rows.
- Usage excludes the one blank-usage row from usage loss.

For a shared two-head model:

- The blank-usage row still trains the gender head and shared backbone through gender loss.
- It contributes zero usage loss.
- `NA` contributes normal usage loss.

The precise loss equation is defined in [model choice](03_model_choice.md#10-loss-design-and-label-masking).

## 6. Family-safe validation

### Why the family is the allocation unit

Related products may share exact pixels, near-duplicate pixels, or a normalised product name. If
family members enter both training and validation, a model can appear to generalise while mostly
remembering a product family.

**Repository fact.** The split assigns `product_family_group` atomically. Exact hashes, accepted
visual groups, and normalised-name groups have zero fold crossings. See
[decision 0014](../../decisions/0014-development-holdout-cv-boundary.md).

Some conservative family groups contain several labels:

- 2 development families contain more than one gender label;
- 238 development families contain more than one valid usage label.

This means a family is a dependence and leakage unit, not a claim that all its labels are identical.

### Fold sizes

| Validation fold | Training products | Validation products | Training families | Validation families | Task 3 coverage warning |
|---:|---:|---:|---:|---:|---|
| 0 | 26,220 | 6,553 | 18,341 | 4,564 | No gender/usage train-absence warning |
| 1 | 26,217 | 6,556 | 18,312 | 4,593 | No gender/usage train-absence warning |
| 2 | 26,220 | 6,553 | 18,328 | 4,577 | No gender/usage train-absence warning |
| 3 | 26,219 | 6,554 | 18,304 | 4,601 | No gender/usage train-absence warning |
| 4 | 26,216 | 6,557 | 18,335 | 4,570 | `Home` is validation-only |

### Recommended five-fold protocol

For outer fold `k`:

1. Select development rows with `cv_fold != k` as the training complement.
2. Select development rows with `cv_fold == k` as validation.
3. Fit normalisation, weights, samplers, model parameters, and any learned preprocessing on the
   training complement only.
4. Use a deterministic validation transform.
5. Predict every validation row once.
6. Save IDs, families, masks, true labels, logits, probabilities, predicted labels, and run IDs.
7. Repeat for folds 0–4.
8. Concatenate the five validation outputs into one pooled OOF table.
9. Assert that every valid development ID appears exactly once for its target.
10. Calculate primary metrics from that pooled table.

All models compared in Task 3 must use this same five-fold scope. A single fold can be used for a
short engineering smoke test, but not as final comparative evidence.

## 7. What may use all development labels

Descriptive data analysis may use all development rows. Examples include:

- label counts;
- fold support tables;
- pair counts;
- descriptive association measures;
- image-quality summaries;
- fixed class-order construction.

These are not learned model parameters.

The following must be fitted only on the training complement of each outer fold:

- RGB mean and standard deviation;
- PCA or feature scaling;
- HOG/colour feature standardisation;
- class priors used by a learned baseline;
- class weights;
- sampling probabilities;
- learned augmentation policies;
- model parameters;
- early-stopping decisions used by a scored model;
- probability calibration used for a scored fold.

The full-development values may be fitted only after method freeze, for the final development refit.

## 8. Image preprocessing boundary

### Shared deterministic preparation

The existing image utility can:

- apply EXIF orientation;
- convert to RGB;
- perform aspect-preserving letterbox resize;
- return a content mask;
- map pixels to `[0,1]`;
- apply supplied mean and standard deviation;
- make standardised padding neutral.

See [`images.py`](../../../src/fashion/data/images.py).

### EDA-derived primary baseline transform

The helper's default `(128,96)` size is not the Task 3 baseline. The primary baseline records one
fixed transform:

1. EXIF-normalise.
2. Convert to RGB.
3. Use `(height=80, width=60)`.
4. Preserve the 3:4 width-to-height ratio.
5. Leave ordinary 60×80 images at native geometry.
6. Resize only unusual geometry with LANCZOS and letterbox rather than stretch or crop.
7. Use white padding before scaling.
8. Fit RGB statistics on training content pixels, excluding padding.
9. Apply the same saved statistics to validation and make standardised padding neutral.
10. Use no random baseline augmentation and keep validation deterministic.

This follows the EDA: almost all images are already 60×80, 294 grayscale images remain valid, only
12 images have unusual dimensions, and the transform-risk figure shows that stretch changes shape
while crop can remove content.

`(128,96)`, augmentation, another normalisation rule, or another interpolation method is a
conditional child factor. It is tested only when an accepted parent's curves, fixed errors, or core
robustness evidence justify that one change.

Avoid square stretching. Do not use ImageNet mean and standard deviation for a scratch model. The
pretrained benchmark uses the preprocessing required by its exact weights and stays in the
comparison-only lane.

## 9. Augmentation fit boundary

Augmentation is a model-development choice, not shared data preparation.

Rules:

- Apply random augmentation to training images only.
- Never randomly augment validation or holdout metrics.
- Fix the random seed path for repeatability.
- Record the exact operations and ranges.
- Do not use official prediction images to choose augmentation.
- Do not inspect holdout failures and then add a matching augmentation.

A deterministic robustness copy may be created for evaluation, but it is not part of training unless
the experiment plan explicitly defines a separate augmentation comparison.

## 10. Validation checkpoint and tuning boundary

Selecting a checkpoint on the same validation fold later used for the reported score adds optimism.

**Recommendation.** Use two evidence levels:

1. **Parent-child development:** use the fixed final epoch for E1 OOF evidence. Curves diagnose
   whether the next single-factor child should change the budget.
2. **Confirmation:** freeze the epoch or schedule, then rerun finalist folds and seeds without
   choosing a separate best checkpoint from each scored validation fold.

Nested CV is not required by the accepted project decision. The written sequential chain, three-seed
finalist confirmation, and sealed holdout are the practical controls. Do not perform an open-ended
search over the same five folds.

## 11. Leakage threats and controls

| Threat | Misleading result | Required control |
|---|---|---|
| New random split | Incomparable models and possible family overlap | Reject any `train_test_split` or new fold file |
| Family member on both sides | Memorisation appears as generalisation | Split by saved `product_family_group` only |
| Full-development RGB statistics in CV | Validation influences input scaling | Fit statistics on each training complement |
| Full-development class weights | Validation label distribution influences training | Compute weights from training complement only |
| Resampled validation | Artificial class balance changes the test question | Keep natural validation rows and weights |
| Holdout class audit before freeze | Model and metric choices adapt to test data | Keep protected targets sealed |
| External high-resolution images | Violates Task 3 data role and changes comparison | Teacher image paths only |
| Ground-truth article type as input | Shortcut uses another answer field | Use it only as an error-analysis slice |
| Product name as input | Changes image-only Task 3 into multimodal classification | Do not feed text to the core Task 3 models |
| ImageNet weights in eligible model | Violates scratch requirement | Record and assert `weights=None` or equivalent |
| `NA` parsed as missing | Removes a real class | Read target strings without default NA conversion |
| Blank usage treated as `NA` | Creates a false label | Respect `has_usage_label` |
| Best-fold reporting | Hides instability | Pool all five OOF folds and show each fold |
| Holdout retry | Independent evidence is lost | One explicit unlock after checkpoint lock |

## 12. Data-derived shortcut risks

The development evidence reports normalised mutual information (NMI):

| Target pair | NMI |
|---|---:|
| articleType–usage | 0.2538 |
| articleType–gender | 0.1942 |
| gender–usage | 0.1042 |
| articleType–season | 0.1742 |
| season–usage | 0.0383 |
| season–gender | 0.0241 |

The article-type majority lookup agrees descriptively with about 90.0% of usage labels and 78.5% of
gender labels. This is not a deployable baseline because it uses ground-truth article type. It shows
that an image model can appear strong by first recognising product type and then copying a dominant
catalogue association.

**Recommendation.** Every finalist should report metrics by article type and by gender–usage pair,
then use fixed visual explanations and manual review to check whether it focuses on the labelled
product rather than a person or background.

## 13. Rare-class protocol

For `Home`:

- Keep the ninth output logit.
- Do not create synthetic independent evidence from the one image.
- Do not merge it into another label.
- Do not mask its validation row.
- Report fold 4 as an untrainable-class limitation.
- Do not interpret a correct OOF guess as learned generalisation.
- Make `Home` review-only in the application.

For `Party`, `Travel`, `Smart Casual`, and `NA`:

- Keep all natural training and validation rows.
- Report product and family support.
- Compare class weighting with unweighted loss.
- Avoid unrestricted oversampling.
- Show class precision, recall, F1, predicted count, and confidence.
- Label uncertainty clearly when support is small.

The [evaluation framework](05_evaluation_framework.md) explains how `Home` is kept in the primary
metric while preventing one lucky example from controlling the final story.

## 14. Validation output contract

Each scored OOF row should contain at least:

```text
id
product_family_group
cv_fold
target
target_valid
true_label
predicted_label
logit_<class> or probability_<class> for every fixed class
confidence
top2_margin
entropy
run_id
model_id
seed
```

For a shared model, save both target outputs against one shared checkpoint ID.

Assertions:

- exactly 32,773 valid gender OOF rows per fold-complete run;
- exactly 32,772 valid usage OOF rows per fold-complete run;
- no duplicate target–ID pair;
- no label outside the fixed development map;
- probabilities are finite and sum to one within tolerance;
- masked labels never enter target metrics;
- `NA` appears as a normal class value;
- fold 4 records the zero-training-support `Home` condition.

## 15. Holdout boundary

Before method freeze, Task 3 must not inspect:

- holdout target values;
- holdout class counts;
- holdout class coverage;
- holdout confusion patterns;
- holdout calibration;
- holdout failures.

After freeze, the final evaluation notebook may explicitly unlock labels, apply the locked model and
preprocessing, and calculate the frozen evaluation once. It must not change the model because of the
result.

The complete procedure is in
[07_final_selection_and_deployment.md](07_final_selection_and_deployment.md).
