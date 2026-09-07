# Task 3 evaluation framework

[Previous: experiment plan](04_experiment_plan.md) · [Research index](README.md) ·
[Next: error, robustness, and ethics](06_error_robustness_and_ethics.md) ·
[References](references.md)

## 1. Purpose

This document defines the evaluation before Task 3 results exist. It fixes:

- which rows enter each metric;
- the primary metric for each output;
- secondary and class-level measures;
- pooled five-fold aggregation;
- uncertainty and paired comparisons;
- negative-transfer tests;
- calibration and abstention analysis;
- efficiency measurement;
- winner-selection rules;
- acceptance and rejection gates;
- the one-time independent holdout procedure.

Changing this contract after seeing holdout results would invalidate the independent judgement.

## 2. Evaluation principles

1. **Respect the data unit.** Products in one family can be dependent. Keep families together in
   validation and uncertainty analysis.
2. **Evaluate both outputs separately.** A gender gain cannot hide a usage loss.
3. **Give every class visibility.** Macro metrics and per-class reports prevent the majority from
   controlling the story.
4. **Preserve the full taxonomy.** `NA` and `Home` stay in usage outputs and reports.
5. **Show support beside scores.** A score based on one product is not equivalent to a score based on
   thousands.
6. **Use pooled OOF evidence for development.** Each product is predicted once by a model that did
   not train on its family.
7. **Use paired comparisons.** Candidate predictions concern the same products and families.
8. **Separate sample uncertainty from training randomness.** Family bootstrap handles the first;
   finalist seeds show the second.
9. **Measure confidence, robustness, and cost.** Macro-F1 alone cannot support review or deployment.
10. **Use holdout once.** It judges the frozen method; it does not choose another method.
11. **Compare each child with its parent.** Initial lower bounds anchor the story, but the immediate
    accepted parent decides whether one changed factor helped.
12. **Diagnose before continuing.** A child configuration is blocked until the parent's curves,
    OOF, class, failure, cost, core-robustness, and decision artifacts exist.

## 3. Evaluation populations and masks

Let:

- `D` be the 32,773-row development partition;
- `gender_mask[i]` be the gender-valid mask;
- `usage_mask[i]` be the usage-valid mask.

Then:

```text
D_gender = {i in D : gender_mask[i] == 1}, |D_gender| = 32,773
D_usage  = {i in D : usage_mask[i]  == 1}, |D_usage|  = 32,772
```

Rules:

- Product ID `28319` is excluded from usage metrics but included in gender metrics.
- Literal `usage="NA"` is included in usage metrics.
- No invalid or blank target may be silently converted to another class.
- All metrics use the stable development label order from
  [`label_maps.json`](../../../data/processed/label_maps.json).

## 4. Confusion quantities

For each class `c`:

- `TP_c`: true class `c`, predicted class `c`;
- `FP_c`: predicted class `c`, true class different;
- `FN_c`: true class `c`, predicted class different;
- `TN_c`: neither true nor predicted class `c`.

Define:

```text
precision_c = TP_c / (TP_c + FP_c)
recall_c    = TP_c / (TP_c + FN_c)
F1_c        = (2 * TP_c) / (2 * TP_c + FP_c + FN_c)
```

When a denominator is zero, report the metric as undefined in the class table and use the
predeclared aggregate convention `zero_division=0` where a numeric aggregate is required. Always
show support and predicted count so zero is interpretable.

## 5. Primary metrics

### 5.1 Gender primary metric

Let the fixed gender class set be:

```text
gender_classes = {Men, Women, Unisex, Boys, Girls}
```

The primary metric is pooled OOF macro-F1:

```text
gender_macro_F1 = sum(F1_c for c in gender_classes) / 5
```

Decision supported: which eligible method gives the best balanced gender-class performance without
allowing `Men` and `Women` to hide `Boys`, `Girls`, or `Unisex` failure.

### 5.2 Usage primary metric

Let the fixed usage class set be:

```text
usage_classes = {Casual, Sports, Ethnic, Formal, NA,
                 Smart Casual, Travel, Party, Home}
```

The primary metric is pooled OOF macro-F1:

```text
usage_macro_F1 = sum(F1_c for c in usage_classes) / 9
```

Decision supported: which eligible method performs best across the complete development-observed
usage taxonomy.

### 5.3 `Home` influence analysis

`Home` has one development example and no positive training example for the fold-4 model that must
predict it. If its F1 changes from 0 to 1, all-nine macro-F1 changes by:

```text
1 / 9 = 0.1111
```

That possible 11.11-point movement can be a lucky single prediction rather than learned
generalisation.

Therefore calculate the mandatory companion:

```text
usage_macro_F1_without_Home
    = sum(F1_c for c in usage_classes if c != Home) / 8
```

Rules:

- `Home` remains in the model, label map, primary all-nine metric, confusion matrix, and class table.
- The eight-class companion is a stability analysis, not a replacement taxonomy.
- If model ranking changes only because of the single `Home` outcome, declare the development models
  effectively tied on usage and decide from the eight-class companion, paired evidence, robustness,
  cost, and simplicity.
- Do not claim `Home` generalisation regardless of its point estimate.

### 5.4 Why macro-F1 is primary

**Repository facts.** A majority-only predictor has descriptive development accuracy of about 54.17%
for gender and 76.75% for usage, yet approximate macro-F1 values of only 0.1405 and 0.0965.

Macro-F1 gives every class equal contribution. It directly exposes a model that performs well only
on the majority.

It is still insufficient alone because a one-example class can be unstable. That is why support,
class metrics, the `Home` companion, confidence intervals, and seed stability are mandatory.

## 6. Secondary predictive metrics

### 6.1 Accuracy and micro-F1

```text
accuracy = count(prediction_i == truth_i) / N
```

For single-label multiclass classification, micro-F1 equals accuracy.

Decision supported: expected correctness under the natural observed product mix.

Risk: majority classes dominate. Never use it alone.

### 6.2 Balanced accuracy

```text
balanced_accuracy = sum(recall_c for c in classes) / number_of_classes
```

Decision supported: whether the model retrieves each class, regardless of frequency.

Difference from macro-F1: balanced accuracy ignores precision. A model can gain it by predicting a
rare class too often. Always inspect predicted counts and precision.

### 6.3 Weighted-F1

```text
weighted_F1 = sum((support_c / N) * F1_c for c in classes)
```

Decision supported: a middle view between equal-class macro-F1 and row-level accuracy.

Risk: large classes still dominate.

### 6.4 Multiclass Matthews correlation coefficient

Report multiclass MCC as a secondary global summary using the full confusion structure.

Decision supported: whether overall predictions are associated with true labels even under
imbalance.

Do not make it the only metric because class-specific failures remain hidden.

### 6.5 Joint exact match

On rows where both masks are true:

```text
joint_exact_match
    = count(gender_prediction_i == gender_truth_i
            and usage_prediction_i == usage_truth_i)
      / N_rows_with_both_labels
```

Decision supported: the chance that both required Task 3 columns are correct together.

It is secondary because:

- it cannot identify which head failed;
- usage difficulty may dominate;
- it does not replace per-output selection.

### 6.6 Top-2 accuracy

Report only as a diagnostic for human review.

Decision supported: whether showing the second suggestion could help an editor.

It is not an official-output metric because the prediction CSV requires one label.

### 6.7 Why ROC-AUC is not a headline

One-vs-rest ROC-AUC can appear strong under extreme imbalance while precision is poor. Classes with
1, 12, or 22 examples also produce unstable estimates.

Optional class PR-AUC may be reported for classes with enough support, but macro-F1, precision,
recall, calibration, and predicted counts remain more actionable here.

Standard metric definitions can be implemented using the official
[Scikit-learn model-evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html),
with fixed class labels and saved raw confusion counts.

## 7. Mandatory per-class report

For every class, report:

| Field | Why it is needed |
|---|---|
| Product support | Shows raw evidence volume |
| Family support | Shows a more conservative independence count |
| Folds present | Shows validation coverage |
| Minimum training-complement support | Exposes zero-support conditions |
| True positives | Audit input to all class metrics |
| False positives | Shows over-prediction |
| False negatives | Shows missed products |
| Predicted count | Detects class collapse or flooding |
| Precision | Trust when the class is predicted |
| Recall | Coverage of true class examples |
| F1 | Class balance of precision and recall |
| Mean/median confidence | Detects confident rare-class mistakes |
| Main confusion destinations | Supports error analysis |

Class interpretation rules:

- `Home`: descriptive only; no reliable generalisation interval.
- Classes with fewer than about 30 independent families: mark estimates highly uncertain.
- A zero-recall class must be visible even if aggregate performance is good.
- A class with high recall and very low precision may reflect over-weighting or oversampling.

## 8. Confusion matrices

Save both:

1. Raw count matrix.
2. Row-normalised matrix, where each true-class row sums to one when support is nonzero.

The raw matrix preserves operational volume. The row-normalised matrix makes minority patterns
visible.

For usage, inspect especially:

- `Casual` versus every rare class;
- `Formal` versus `Smart Casual`;
- `Casual` versus `Travel` and `Party`;
- `Sports` versus `Casual`;
- `NA` destinations;
- false `Home` predictions.

For gender, inspect:

- `Men`/`Women` versus `Unisex`;
- `Boys` versus `Men`;
- `Girls` versus `Women`;
- whether child categories collapse to adult categories.

## 9. Five-fold OOF aggregation

### 9.1 Pooled estimate

For each fold, train on the other four and predict its validation products. Concatenate all five
validation outputs, then compute the primary metric once on the pooled table.

Reasons:

- every development product is judged once;
- fold sizes do not receive equal weight by accident;
- rare classes are not fragmented into several zero-support fold metrics;
- paired per-product model comparisons become possible;
- the result matches the full development distribution more directly.

### 9.2 Fold report

Also report for each fold:

- validation products and families;
- class supports;
- primary metric;
- accuracy and balanced accuracy;
- NLL and Brier;
- cost if fold cost varies;
- any absent training-complement class.

Summarise fold mean, standard deviation, minimum, and maximum descriptively.

Do not use the standard deviation of five fold scores as a full confidence interval. The training
sets overlap. Research shows there is no universal unbiased variance estimator for k-fold CV and
naive methods can underestimate uncertainty.
[Bengio and Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html).

### 9.3 Development selection optimism

The same five folds are used to compare several candidate configurations. The selected winner’s OOF
score is therefore a development-selection estimate, not independent final performance.

Controls:

- predeclare a bounded matrix;
- report all meaningful runs, not only the winner;
- confirm finalists with frozen settings and three seeds;
- use the untouched holdout once after selection.

## 10. Family-cluster bootstrap uncertainty

### 10.1 Why cluster by family

Rows in one `product_family_group` may be exact, near, or naming-related variants. Row bootstrap would
treat them as independent and can make intervals too narrow.

### 10.2 Procedure

For one saved OOF result:

1. List unique development family groups.
2. Sample that list with replacement until the original number of family groups is reached.
3. Include all product rows belonging to each sampled family, repeated when its family is sampled
   more than once.
4. Compute the metric on valid target rows.
5. Repeat 10,000 times.
6. Report the bootstrap median and 2.5th/97.5th percentiles, or a predeclared BCa interval.

If a supported class is absent from an extremely rare bootstrap replicate, discard and redraw that
replicate for class-dependent inference. `Home` is not suitable for a normal inferential interval.

### 10.3 What the interval means

It estimates sample uncertainty conditional on the already trained fold models and the project’s
family grouping.

It does not include:

- random initialisation;
- data-loader randomness;
- hardware nondeterminism;
- hyperparameter-selection uncertainty;
- future catalogue distribution shift.

## 11. Training-randomness analysis

Run each finalist with three fixed seeds over all five folds.

Report:

- seed-specific pooled primary metric;
- mean across seeds;
- standard deviation;
- range;
- worst seed;
- per-class seed range;
- whether the model ordering changes by seed.

Three seeds do not fully describe the training distribution. They are a practical minimum that
prevents one lucky initialisation from becoming the final method.

PyTorch states that complete reproducibility is not guaranteed across versions, platforms, or
CPU/GPU even with fixed seeds. [PyTorch reproducibility guide](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## 12. Paired model comparisons

For models A and B, calculate the metric on the same OOF rows and the same family-bootstrap sample.

```text
delta_gender = gender_macro_F1_A - gender_macro_F1_B
delta_usage  = usage_macro_F1_A  - usage_macro_F1_B
```

Report:

- point difference;
- paired 95% family-bootstrap interval;
- fold-by-fold difference;
- seed-by-seed difference for finalists;
- class-level changes;
- cost difference.

Avoid unpaired tests because they discard the strong fact that both models predict the same
products.

Avoid a large set of p-values. If a formal test is desired, reserve it for the final predeclared
comparison. A paired permutation test can be a sensitivity analysis, not a replacement for practical
effect size and uncertainty.

## 13. Negative-transfer evaluation

Run this evaluation only when accepted separate parents exist and their measured combined cost
triggers the predeclared shared-system gate. If the gate does not open, record that reason instead
of manufacturing a shared comparison for breadth.

Use a matched backbone to compare:

- gender-only;
- usage-only;
- shared two-head.

The shared model is non-inferior only if both pass:

```text
lower_95_percent_CI(delta_gender) > -0.01
lower_95_percent_CI(delta_usage)  > -0.01
```

The `0.01` margin means one macro-F1 percentage point. It must be approved before the shared child
begins.

Also require:

- no supported-class recall collapse;
- no large calibration regression;
- no robustness regression that changes deployment suitability;
- meaningful combined cost reduction.

If the shared point estimate is better on average but the lower interval crosses the harm margin for
one target, do not claim non-inferiority.

Multitask research supports testing because sharing can reduce inference cost, while also warning
that task objectives can compete. [Standley et al.](https://proceedings.mlr.press/v119/standley20a.html).

## 14. Probability quality

### 14.1 Negative log-likelihood

For predicted probabilities `probability[i, c]`:

```text
NLL = -sum(log(probability[i, true_class_i]) for i in rows) / N
```

NLL strongly penalises confident wrong predictions.

### 14.2 Multiclass Brier score

```text
Brier = sum((probability[i, c] - indicator(true_class_i == c)) ** 2
            for i in rows for c in classes) / N
```

Brier measures full-vector probability error.

### 14.3 Expected calibration error

Use 15 equal-count confidence bins where possible:

```text
ECE = sum((bin_size[b] / N)
          * abs(bin_accuracy[b] - bin_mean_confidence[b])
          for b in bins)
```

ECE depends on binning and can hide class-specific problems. Always include a reliability diagram,
NLL, and Brier.

### 14.4 Per-class calibration

For classes with enough support, show one-vs-rest reliability or confidence distributions.

For tiny classes, report raw confidence examples and avoid a smooth curve that implies more data than
exists.

## 15. Temperature scaling

Modern neural networks can be overconfident. Temperature scaling learns one scalar applied to logits
and often improves probability calibration without changing argmax labels.
[Guo et al.](https://proceedings.mlr.press/v70/guo17a.html).

### Cross-fitted development protocol

1. Save raw OOF logits.
2. Choose one held-out OOF fold for calibration assessment.
3. Fit temperature on the other four folds’ OOF logits.
4. Apply it to the assessment fold.
5. Rotate across all five folds.
6. Pool the cross-fitted calibrated predictions.
7. Compare NLL, Brier, ECE, and reliability with raw logits.

Retain scaling only when:

- cross-fitted NLL or Brier improves;
- ECE does not materially worsen;
- per-class confidence does not become misleading;
- argmax predictions remain unchanged as expected.

After the method is selected, fit the frozen scalar on all development OOF logits. Applying it to the
all-development refit is an approximation because that final model was not one of the fold models.
State this limitation. Do not refit temperature on holdout.

## 16. Uncertainty and abstention

Candidate uncertainty signals:

- maximum class probability;
- top-1 minus top-2 probability margin;
- normalised predictive entropy;
- disagreement across finalist seeds, if an ensemble was predeclared;
- disagreement between separate and shared research models, as an analysis signal only.

### Risk–coverage analysis

For threshold `tau`:

```text
coverage(tau) = count(confidence_i >= tau) / N

selective_error(tau)
    = 1 - count(confidence_i >= tau and prediction_i == truth_i)
          / count(confidence_i >= tau)
```

Report at least:

- 100% coverage;
- 90% coverage;
- 75% coverage.

Also report class coverage. A threshold that removes nearly all rare classes is not an honest safety
improvement.

### Threshold selection

- Choose thresholds from development OOF evidence.
- Choose separately for gender and usage.
- Freeze before holdout.
- Do not choose a threshold only to hit a desired accuracy while hiding low coverage.
- The application may route below-threshold predictions to review.
- The official CSV must still output one valid frozen argmax label because it cannot contain an
  abstention token.

`Home` should always be review-only regardless of confidence.

## 17. Robustness evaluation outputs

The perturbation definitions are in
[06_error_robustness_and_ethics.md](06_error_robustness_and_ethics.md).

For each corruption `r`, compute:

```text
macro_F1_drop_r = clean_macro_F1 - corrupted_macro_F1_r
```

Also compute:

- prediction consistency with clean input;
- accuracy and balanced-accuracy drop;
- per-class recall drop;
- NLL/Brier/ECE change;
- confidence change on wrong predictions.

Use the same products and fold models for clean and corrupted versions. This makes the comparison
paired.

Recommended provisional gate:

- mean mild-corruption macro-F1 drop at most 0.05;
- worst single mild-corruption drop at most 0.10;
- no previously working gender class falls to zero recall.

These are project thresholds, not published universal standards. Freeze them before ROB7.1.

## 18. Efficiency framework

### 18.1 Static measures

- Trainable parameter count.
- Total parameter count.
- Checkpoint bytes on disk.
- MACs or FLOPs at exact input size.
- Number and size of application artifacts.

### 18.2 Training measures

- Seconds per epoch after warmup.
- Total wall-clock train time.
- Number of optimiser steps.
- Peak accelerator memory.
- Energy measure if available; otherwise do not invent one.

### 18.3 Inference measures

- Model load time.
- Batch-1 CPU p50 and p95 latency.
- Batch-1 GPU p50 and p95 latency if GPU deployment is relevant.
- Batch-32 throughput.
- Peak inference RAM/VRAM.
- Cold-start and warmed timing kept separate.

### 18.4 Timing protocol

- Name CPU, GPU, RAM, operating system, framework, thread count, and precision.
- Set models to evaluation mode.
- Disable gradient tracking.
- Use identical preprocessed inputs.
- Warm up before measurement.
- Use at least 100 timed iterations.
- Synchronise accelerator timing.
- Report median and p95, not only the fastest run.

### 18.5 Shared-system comparison

Compare one shared model against the sum of both separate models:

```text
separate checkpoint size = gender model + usage model
separate latency         = gender inference + usage inference
shared checkpoint size   = one backbone + two heads
shared latency           = one inference producing both heads
```

Parameter count or FLOPs alone cannot decide efficiency because hardware kernels differ.

## 19. Output-to-decision map

| Evaluation output | Decision it supports | Misleading conclusion it prevents |
|---|---|---|
| Majority accuracy and macro-F1 | Whether learning beats priors | “76% usage accuracy is strong” |
| Pooled OOF macro-F1 | Balanced development selection | Best-fold cherry-picking |
| Per-fold table | Stability | Hiding one weak fold |
| Per-class report | Rare and minority behaviour | Majority classes hiding collapse |
| `Home` companion metric | Single-row sensitivity | Treating one lucky guess as 11-point progress |
| Confusion matrices | Failure direction | One aggregate score without mechanism |
| Family bootstrap | Sampling uncertainty | Overprecise point estimates |
| Three seeds | Training randomness | Lucky initialisation |
| Paired differences | Direct model comparison | Unpaired noisy comparisons |
| Negative-transfer test | Shared versus separate decision | Hiding one harmed task in an average |
| NLL/Brier/ECE | Confidence quality | Treating softmax as probability truth |
| Risk–coverage | Review threshold | Claiming safer output after rejecting most hard classes |
| Robustness deltas | Input stability | Clean-only success |
| Cost measures | Application suitability | Assuming small FLOPs means fast deployment |
| Holdout gap | Independent generalisation | Treating development selection as final proof |

### Per-candidate diagnostic completeness gate

Before another E1 configuration may be created, the current parent must have:

- per-fold training and validation loss and macro-F1 curves;
- pooled OOF primary and secondary metrics;
- fold and per-class tables with predicted counts;
- raw and row-normalised confusion matrices;
- a fixed failure-example index;
- parameter, checkpoint, training-time, peak-memory, and named-device latency evidence;
- core robustness results for JPEG 75, brightness ×0.85/×1.15, 3% translation, and grayscale;
- a written accept, reject, or stop decision linked to the parent and evidence.

Finalists still require the full calibration, robustness, explanation, cost, and manual-review
suite. The repeated smaller gate prevents error analysis from becoming a post-hoc story.

## 20. Baseline acceptance gate

The primary small CNN is kept as the learnable baseline even if it is weak. Before it can be treated
as a healthy parent for a complexity change, it should satisfy:

1. Split, mask, scratch, and registry integrity passes.
2. Tiny-batch overfit passes and the full baseline diagnostic bundle exists.
3. Its pooled OOF evidence is compared with both the majority lower bound and classical reference.
4. Paired family-bootstrap evidence against majority is reported, including uncertainty.
5. The model predicts more than the majority class.
6. Gender recall is nonzero for all five development classes, or the collapse becomes the next
   written hypothesis.
7. Usage recall is nonzero for every class with meaningful training support, or the collapse becomes
   the next written hypothesis; `Home` remains the stated untrainable exception.

A proposed accuracy guard is no more than a five-percentage-point drop against majority accuracy
unless the team explicitly accepts the tradeoff for a large and credible minority gain. This is a
project rule to freeze, not an external standard.

## 21. Candidate acceptance and rejection gates

### 21.1 Hard integrity rejection

Reject a run from winner selection when:

- its split digest differs;
- family crossing is detected;
- holdout or prediction data fit any parameter;
- label maps or masks are wrong;
- literal `NA` was parsed as missing;
- the eligible model loaded pretrained weights;
- OOF rows are missing or duplicated;
- the required registry or configuration evidence is absent;
- metrics cannot be regenerated from saved predictions.

### 21.2 Predictive rejection

Reject a child and return to its parent when:

- it fails the baseline gate;
- its diagnostic bundle is incomplete;
- it changes more than the one predeclared main factor;
- the hypothesis or trigger evidence was not written before the child ran;
- an adequately supported class collapses to zero recall;
- improvement comes only from the one `Home` product;
- fold or seed instability makes the claimed gain unreliable;
- class weighting floods rare predictions and destroys precision;
- it is dominated by its simpler parent in performance, cost, and stability;
- it does not answer the written parent weakness.

Stop rather than create another child when failure review shows that missing visual evidence,
teacher-label ambiguity, or tiny support is the main limitation.

### 21.3 Shared-model rejection

Reject shared when:

- either lower paired CI is at or below the `-0.01` no-harm margin;
- either task develops a serious class collapse;
- calibration or robustness becomes materially worse;
- cost reduction is too small to justify complexity.

### 21.4 Operational rejection

Do not call a model deployment-ready when:

- robustness gates fail;
- calibrated confidence is not useful for review;
- latency or memory exceeds the named device limit;
- app preprocessing differs from evaluation;
- human-review wording or override is missing;
- the system is presented as person-level gender inference.

It may still be a valid assignment result with honest limitations.

## 22. Winner-selection rules

### 22.1 Per-target eligible winner

For gender and usage separately:

1. Exclude hard integrity failures.
2. Exclude pretrained/ineligible systems.
3. Follow the recorded chain from the primary baseline through accepted children.
4. Exclude rejected children and explain their failed hypotheses.
5. Apply baseline and class-collapse gates to the accepted final parent.
6. Compare that parent with its immediate predecessor using paired family-bootstrap differences.
7. Check three-seed stability.
8. Inspect class, calibration, robustness, failure, and cost evidence.
9. Prefer the simpler predecessor when it is within one macro-F1 percentage point and materially
   cheaper or more stable.

### 22.2 System winner

Possible systems:

- separate gender and usage winners;
- one shared winner that passed both no-harm tests.

Do not form a hidden average of gender and usage macro-F1.

Recommended shared efficiency rule:

- at least 25% reduction in combined checkpoint size or measured end-to-end latency, unless another
  named operational benefit justifies a smaller saving.

Freeze the efficiency rule before final selection.

### 22.3 Ties and practical equivalence

Treat candidates as practically tied when:

- the paired interval includes zero and differences are within the one-point practical margin;
- ranking changes by seed;
- ranking changes only through `Home`;
- the point advantage is smaller than measurement or deployment tradeoffs.

Tie-break order:

1. Fewer serious class failures.
2. Better robustness.
3. Better calibration/risk–coverage.
4. Lower latency and memory.
5. Smaller checkpoint.
6. Simpler training and application path.

## 23. Method-freeze package

Before holdout access, freeze:

- winning development run IDs;
- separate/shared system form;
- architecture and low-resolution stem details;
- random initialisation policy;
- input size, interpolation, padding, and RGB conversion;
- development-fitted normalisation procedure;
- augmentation;
- target masks;
- loss and class-weight formula;
- optimiser, schedule, batch size, precision, and epoch rule;
- random seed or predeclared ensemble membership;
- calibration method and temperature fitting rule;
- review/abstention thresholds;
- primary, secondary, class, calibration, robustness, and cost metrics;
- acceptance thresholds;
- checkpoint and configuration naming;
- official label order;
- application preprocessing path.

The package needs a timestamp and team approval before the protected loader is called.

## 24. Final refit protocol

1. Load development rows only.
2. Verify split, partition, label-map, and mask hashes.
3. Fit final image statistics on development content pixels only.
4. Train the frozen eligible method from random initialisation on all development rows.
5. Use all 32,773 gender labels.
6. Use the 32,772 valid usage labels.
7. Keep all five and nine output logits.
8. Apply the frozen epoch/schedule and seed rule.
9. Save checkpoint, preprocessing, label map, configuration, and environment.
10. Hash and lock every artifact.
11. Append the final-refit run to the registry.

Do not choose a new best epoch using holdout.

If an ensemble is desired, it must have been evaluated as an eligible development candidate and
frozen before this step. Do not add an ensemble after seeing holdout.

## 25. One-time holdout protocol

### 25.1 Unlock

- Use Notebook 06 only.
- Record team approval and timestamp.
- Call the protected loader with `evaluation_unlocked=True`.
- Do not copy holdout targets into normal development artifacts.

### 25.2 Apply

- Load locked checkpoint and preprocessing.
- Do not refit any parameter.
- Do not change thresholds.
- Predict all valid holdout rows.
- Save raw logits, probabilities, labels, masks, and IDs.

### 25.3 Evaluate

Compute the frozen set:

- primary metrics;
- secondary metrics;
- per-class support and metrics;
- confusion matrices;
- NLL, Brier, ECE, and reliability;
- risk–coverage;
- predeclared slices;
- predeclared robustness tests;
- cost on the frozen checkpoint.

### 25.4 Unknown or absent holdout labels

The team must not inspect holdout class coverage before unlock.

After unlock:

- report any class with zero holdout support as not estimable in its class row;
- keep the fixed development label map;
- report an unknown holdout label outside that map as an unsupported target event;
- do not expand the head or retrain;
- report both fixed-map aggregate behaviour and class supports so absence cannot be mistaken for model
  failure or success.

### 25.5 Interpret

Compare holdout with the frozen OOF expectation:

- point difference;
- whether it falls inside the development family-bootstrap interval;
- class changes;
- calibration change;
- robustness change;
- likely distribution shift or development-selection optimism.

A holdout primary drop larger than five percentage points or below the development 95% interval is a
recommended warning trigger. It is not permission to try another model.

### 25.6 Prohibited response

After unlock, do not:

- choose another candidate;
- change the input size;
- add augmentation;
- alter class weights;
- change a metric or class scope;
- move a confidence threshold;
- refit calibration;
- retrain using holdout labels;
- report only a favourable holdout slice.

If the result is poor, report it and mark the system not deployment-ready if necessary.

## 26. Official prediction evaluation boundary

Official prediction images are unlabeled and separate from the independent holdout.

For official predictions:

- use the frozen development-fit checkpoint;
- use the locked preprocessing and label order;
- output one valid gender and usage label for every required ID;
- preserve the exact fixed column order;
- validate row count, ID set/order, duplicates, and missing values;
- do not use prediction images to refit normalisation or choose thresholds.

No accuracy claim can be made from the unlabeled official prediction file.

## 27. Minimum tables and figures for the report

### Main report

- Compact model comparison table.
- Gender and usage primary metrics with uncertainty.
- Pretrained benchmark clearly separated.
- Final system cost.
- One confusion/failure visual selected for decision value.
- Independent holdout result.

### Appendix or evidence artifacts

- Full per-class reports.
- Five-fold table.
- Seed table.
- Paired bootstrap difference table.
- Calibration plots.
- Risk–coverage plots.
- Robustness table.
- Detailed cost table.
- Failure slices and reviewed examples.

All report values should be generated from registered run and prediction artifacts, not copied by
hand.
