# Task 3 error analysis, robustness, and ethics

[Previous: evaluation framework](05_evaluation_framework.md) · [Research index](README.md) ·
[Next: final selection and deployment](07_final_selection_and_deployment.md) ·
[References](references.md)

## 1. Purpose

Headline metrics answer “how often?” They do not answer:

- which products fail;
- why they fail;
- whether the failure is a model problem or a label limitation;
- whether confidence is trustworthy;
- whether small image changes break the prediction;
- whether the model relies on a person, background, colour, or product-type shortcut;
- whether using the output could harm a catalogue user or reinforce stereotypes.

This document predeclares the error, robustness, uncertainty, ethics, and human-review work needed to
answer those questions.

## 2. Error-analysis principles

1. Use saved OOF predictions so reviewed development images were not in their predicting model’s
   training side.
2. Fix slices and sample-selection rules before looking for a flattering story.
3. Show correct and incorrect examples.
4. Include high-confidence errors, not only uncertain mistakes.
5. Keep product IDs and run IDs for every reviewed image.
6. Separate model error, label ambiguity, data quality, and unsupported-class limits.
7. Do not infer demographic traits that the dataset does not contain.
8. Treat visual explanations as diagnostics, not proof.

## 3. Predeclared quantitative slices

### 3.1 Target slices

| Slice | Why it matters | Required output |
|---|---|---|
| Each gender class | Exposes majority and child/unisex collapse | Support, precision, recall, F1, confidence |
| Each usage class | Exposes extreme long-tail failure | Support, precision, recall, F1, predicted count |
| Gender–usage pair | Detects reliance on common business combinations | Accuracy/F1 where support permits, confusion notes |
| Valid versus missing usage | Confirms mask behaviour | Count and shared-model treatment |

### 3.2 Fold and family slices

| Slice | Why it matters |
|---|---|
| Validation fold | Reveals partition instability |
| Family size 1 versus greater than 1 | Checks whether related-product groups differ |
| Conservative mixed-label family | Finds conflicts in broad family groups |
| `Home` fold-4 row | Records zero positive training support |

### 3.3 Image-quality slices

- RGB versus grayscale.
- Usual 60×80 versus unusual dimensions.
- Brightness quartiles.
- Contrast quartiles.
- Near-white background quartiles.
- File-size quartiles.
- Very small visible product.
- Product partly cropped.
- Visible person.
- Several products or accessories.
- Text/logo-heavy image.

The first five can be generated from existing preparation evidence. Person, multi-product, and small
product status may require a fixed manual tag set.

### 3.4 Catalogue-context slices

- Article type.
- Master category and subcategory, for analysis only.
- High article-type-majority agreement versus low agreement.
- Common versus rare gender–usage pair.
- Product-name family grouping basis, without feeding the name to the model.

Ground-truth article type is not a Task 3 model input. It is an analysis key used to find shortcuts.

### 3.5 Confidence slices

- Highest-confidence correct predictions.
- Highest-confidence errors.
- Lowest-confidence correct predictions.
- Smallest top-1/top-2 margin.
- Highest entropy.
- Separate/shared disagreement.
- Seed disagreement, if finalist seeds are available.

High-confidence errors are the most important for human review because they show when confidence can
mislead.

## 4. Slice reporting rules

For every slice, show:

- product count;
- family count;
- class composition;
- primary/secondary metric where defined;
- difference from the full OOF population;
- confidence and calibration summary;
- uncertainty or a clear “descriptive only” note.

Do not rank slices with tiny support as if their estimates were stable.

Suggested support language:

- 100 or more families: normal quantitative comparison, still with uncertainty.
- 30–99 families: cautious quantitative comparison.
- 2–29 families: descriptive and highly uncertain.
- 1 family: case study only.

These are reporting rules, not rules for dropping labels.

## 5. Failure taxonomy

Every manually reviewed error should receive one or more tags.

| Failure tag | Meaning | Example response |
|---|---|---|
| `visually_ambiguous` | More than one supplied label looks plausible | Keep as label-limit evidence |
| `weak_business_label` | Usage depends on merchant context not visible in pixels | Human review; avoid stronger model claim |
| `possible_teacher_error` | Supplied metadata appears inconsistent with image | Record, do not silently relabel |
| `product_too_small` | Labelled product has too few pixels | Input/data limitation |
| `product_occluded` | Person or another item hides it | Robustness and review issue |
| `multi_product` | Several items compete for attention | Localisation or data issue |
| `person_shortcut` | Model appears to attend to a person | Ethical/shortcut concern |
| `background_shortcut` | Blank/background style controls prediction | Robustness concern |
| `colour_shortcut` | Colour dominates a socially or semantically weak label | Shortcut concern |
| `article_type_proxy` | Product type maps to dominant gender/usage | Report proxy behaviour |
| `rare_class_no_support` | Training evidence is absent or tiny | Do not claim generalisation |
| `majority_collapse` | Prediction defaults to `Men` or `Casual` | Loss/imbalance issue |
| `rare_class_flooding` | Weighting predicts rare class too often | Precision/calibration issue |
| `transform_damage` | Crop/resize/augmentation removes useful evidence | Change transform only in development cycle |
| `overconfident_error` | Wrong prediction has high calibrated confidence | Review threshold/calibration issue |
| `negative_transfer` | Shared model loses where separate model succeeds | Prefer separate or change sharing experiment |

## 6. Manual review protocol

### 6.1 Fixed selection

Create a deterministic review index from OOF predictions. Include:

- at least five correct and five incorrect examples per class when support allows;
- every `Home` example;
- every `Party` error and a representative sample of its correct predictions;
- representative `Travel`, `Smart Casual`, and `NA` errors;
- highest-confidence errors for each output;
- separate-versus-shared regressions;
- grayscale and unusual-size errors;
- person-visible and multi-product images;
- a random control sample of correct common-class examples.

Do not choose only visually dramatic examples after looking through the whole dataset.

### 6.2 Review fields

Each record should contain:

```text
id
path
product_family_group
fold
run_id
target
true_label
predicted_label
confidence
top2_label
top2_margin
failure_tags
reviewer_note
possible_label_ambiguity
visible_person
multiple_products
product_small_or_occluded
```

### 6.3 Reviewers

Use two reviewers for the final example set where practical. Resolve differences by retaining both
notes or recording a short adjudication. Reviewers must not change ground-truth labels in the shared
dataset.

### 6.4 Decisions supported

- Whether a score gain is visually credible.
- Whether rare-class errors are model or evidence limits.
- Whether the shared model adds a systematic failure.
- Which examples belong in the report.
- Which application outputs need forced review.

## 7. Robustness suite

Common-corruption benchmarks use fixed perturbations so model robustness is not judged by a moving
test. This project adopts that principle at a scale suitable for 60×80 catalogue images.
[ImageNet-C](https://openreview.net/pdf?id=HJz6tiCqYm).

### 7.1 Clean reference

Use the exact OOF validation image and frozen fold model. The clean prediction is the paired reference
for every perturbed copy.

### 7.2 Fixed mild perturbations

| Family | Level 1 | Level 2 | Purpose |
|---|---|---|---|
| JPEG | Quality 75 | Quality 50 | Upload/compression changes |
| Brightness | ×0.85 | ×1.15 | Lighting/exposure change |
| Contrast | ×0.80 | ×1.20 | Low/high contrast catalogue export |
| Blur | Small radius, about 0.5 | Radius about 1.0 | Focus and resampling loss |
| Rotation | −5° | +5° | Small placement change |
| Translation | About 3% | About 5% | Off-centre product |
| Scale/crop-pad | About 3% | About 5% | Framing variation |
| Grayscale | Full grayscale-to-RGB | — | Colour reliance |
| Occlusion | Small corner patch | Small central patch | Missing detail/localisation |

Exact library calls, interpolation, pad colour, and random sign choices must be fixed in the
configuration. If both positive and negative directions are used, evaluate both rather than choosing
the easier one.

### 7.3 Why perturbations remain mild

- Images are already tiny.
- Large blur or crop can destroy the object rather than test realistic stability.
- A robustness test should preserve the intended label.
- Severe changes can be shown separately as stress tests, not mixed with the acceptance gate.

### 7.4 Robustness outputs

For each target, corruption, and level:

- pooled macro-F1;
- absolute macro-F1 drop;
- accuracy and balanced-accuracy drop;
- clean-versus-corrupted prediction consistency;
- per-class recall drop;
- NLL, Brier, and ECE change;
- confidence change on newly wrong predictions;
- worst affected class.

### 7.5 Provisional gates

- Mean mild-corruption macro-F1 drop no greater than 0.05.
- Worst single mild-corruption drop no greater than 0.10.
- No previously working gender class collapses to zero recall.
- A robustness gain must not come from severe clean-performance loss.

These are Task 3 proposals, not universal external thresholds. Approve them before ROB7.1.

### 7.6 No robustness leakage

Do not inspect the fixed suite, add a matching augmentation, and replace the original result without
recording a new development cycle. Keep the original test as evidence. A later augmentation run is a
new experiment and must be compared under the same suite.

## 8. Visual explanation protocol

Grad-CAM uses class gradients to produce a coarse localisation map and can help expose model bias or
failure regions. It does not prove a causal explanation.
[Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).

### Fixed explanation set

Include:

- two correct common-class examples per target;
- two high-confidence errors per target;
- rare-class examples where possible;
- `Home`;
- person-visible errors;
- multi-product images;
- one separate/shared disagreement per target;
- one grayscale and one unusual-size example.

### Review questions

- Is activation on the labelled product?
- Is it on a face, body, or clothing worn by a model instead?
- Is it on the white background?
- Is it on a logo or text?
- Does gender activation differ from usage activation in the shared model?
- Does the model focus on a second product rather than the labelled item?

### Reporting limits

Use explanations to generate and support hypotheses. Do not say “the model reasoned from X” solely
because a heatmap covered X.

## 9. Uncertainty and abstention

### 9.1 Signals

- Calibrated maximum probability.
- Top-1/top-2 probability gap.
- Normalised entropy.
- Seed disagreement where available.
- Large clean-versus-perturbed disagreement.

### 9.2 Development analysis

For each signal:

- plot error rate against confidence;
- calculate risk–coverage;
- inspect per-class coverage;
- review high-confidence errors;
- test whether rare classes are disproportionately rejected.

### 9.3 Operational guidance

The application may show:

```text
Suggested catalogue label: <label>
Confidence: <calibrated value or qualitative band>
Alternative: <second label>
Needs review: yes/no
```

Force review when:

- confidence is below the frozen target threshold;
- the predicted label is `Home`;
- an image fails decode or input checks;
- uncertainty signals disagree strongly;
- the product is outside the intended catalogue-image context.

The official prediction CSV cannot abstain. It must still output one frozen valid label.

### 9.4 Do not turn `Unisex` or `NA` into abstention

Both are supplied labels. Uncertainty is a separate operational state.

## 10. Ethical limits

### 10.1 Catalogue label, not personal identity

Required wording:

> The model suggests the supplied catalogue target-audience and usage labels from a fashion product
> image.

Forbidden wording:

- “detects gender”;
- “identifies whether a person is male or female”;
- “predicts the user’s gender”;
- “decides who can wear the item.”

### 10.2 Why the distinction matters

The image may contain a person, but the training target is catalogue metadata. Presenting the output
as personal inference changes the task, evidence, risk, and affected people.

Research on commercial facial gender classifiers found large intersectional error disparities. This
project is not facial analysis, but that evidence supports a strict non-use boundary.
[Gender Shades](https://proceedings.mlr.press/v81/buolamwini18a.html).

### 10.3 Taxonomy limitations

- The five audience labels are not a complete set of human identities.
- `Boys` and `Girls` combine age and gendered marketing.
- `Unisex` is still a merchant category, not proof of inclusive design.
- Usage labels may be culturally or commercially defined.
- `Ethnic` is a broad supplied label whose meaning may vary and deserves careful wording.
- `NA` has unclear business semantics from the available evidence.

### 10.4 Bias and shortcuts

Potential proxies include:

- article type;
- colour;
- visible model appearance;
- pose;
- background style;
- brand/logo;
- photography source.

NIST recommends finding and managing bias across design, development, evaluation, and use rather
than treating it as one final metric.
[NIST SP 1270](https://www.nist.gov/publications/towards-standard-identifying-and-managing-bias-artificial-intelligence).

### 10.5 Fairness claims that are not supported

The repository does not contain reliable demographic attributes for people shown in images. Do not
claim equal performance by skin tone, ethnicity, gender identity, disability, or another demographic
group.

The class-level gender report measures catalogue-label performance, not demographic fairness.

### 10.6 Harm controls

- Human override for every prediction.
- No customer profiling.
- No product access restriction.
- No automatic pricing or ranking based solely on Task 3 labels.
- Show uncertainty and alternatives.
- Log and review overrides.
- Provide a way to report incorrect or harmful labels.
- Stop or narrow the system if monitoring finds systematic harm.

## 11. Human-review policy

### Reviewer sees

- product image and ID;
- suggested catalogue label;
- calibrated confidence or confidence band;
- second suggestion;
- reason for review, such as low confidence or rare label;
- a simple override control.

### Reviewer does not see

- claims about a photographed person’s identity;
- an unexplained “AI certainty” label;
- a forced acceptance workflow;
- a heatmap presented as proof.

### Review priority

1. `Home` predictions.
2. Low-confidence predictions.
3. High-impact catalogue corrections or bulk operations.
4. Rare usage classes.
5. Images with a visible person or several products.
6. Predictions outside normal image-quality ranges.

### Feedback handling

- Store the original model output and model hash.
- Store the reviewer correction and reason.
- Do not train immediately on one correction.
- Review label definitions and data quality first.
- Retraining must pass the same split, registry, validation, robustness, and holdout governance.

## 12. Required error and ethics artifacts

- Slice-definition file or configuration.
- Full slice table with product/family support.
- Confusion matrices.
- High-confidence-error table.
- Fixed manual-review index.
- Reviewer notes and adjudication.
- Failure-tag summary.
- Robustness configuration and results.
- Clean/corrupted prediction pairs.
- Risk–coverage plots.
- Fixed Grad-CAM index and rendered figures.
- Intended-use and forbidden-use text.
- Human-review policy.
- Known-label-limit statement.
- Monitoring triggers.

Artifact paths and traceability fields are detailed in
[08_reproducibility_and_artifacts.md](08_reproducibility_and_artifacts.md).
