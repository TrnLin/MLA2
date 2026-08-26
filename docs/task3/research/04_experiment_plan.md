# Task 3 controlled experiment plan

[Previous: model choice](03_model_choice.md) · [Research index](README.md) ·
[Next: evaluation framework](05_evaluation_framework.md) · [References](references.md)

## 1. Purpose

This plan turns model questions into a bounded set of comparable runs. It is designed to produce:

- enough method breadth for the assignment;
- fair comparisons under one data contract;
- explicit reasons to advance or stop a model family;
- saved OOF evidence for error and statistical analysis;
- a practical route to final selection without an open-ended search.

No result should be entered into the report unless its run, predictions, configuration, and evidence
are traceable through the required registry.

## 2. Common controls

Unless a row explicitly tests one of these variables, keep it fixed:

| Control | Fixed rule |
|---|---|
| Data | Teacher images referenced by [`splits.csv`](../../../data/processed/splits.csv) |
| Development scope | All five saved family-safe folds |
| Validation | Natural fold distribution; no resampling |
| Labels | Stable five-class gender and nine-class usage maps |
| Masks | Target-specific `has_<target>_label` fields |
| Initialisation | Random for every eligible neural model |
| Screening seed | One predeclared seed shared across matched runs |
| Finalist seeds | Three predeclared seeds |
| OOF output | IDs, families, folds, logits, probabilities, labels, masks, run IDs |
| Normalisation | Fitted on each outer training complement |
| Class weights/sampler | Fitted on each outer training complement |
| Holdout | Never accessed during this plan |
| Metrics | Frozen definitions from [05](05_evaluation_framework.md) |
| Cost hardware | Same named device and timing protocol |

One experiment changes one main factor. When two factors must change together for technical reasons,
the run notes must say so.

## 3. Evidence levels

### Level E0: engineering check

- A few batches or one partial fold.
- Used only to catch crashes, bad shapes, masks, or impossible memory use.
- Cannot reject a scientifically plausible model based on score.
- If parameters are updated, record the run as an engineering run.

### Level E1: screening evidence

- All five folds.
- One seed.
- Fixed or clearly recorded training budget.
- Suitable for eliminating clearly weak candidates and choosing finalists.

### Level E2: confirmation evidence

- All five folds.
- Three seeds.
- Frozen architecture and training schedule.
- Suitable for final development selection.

### Level E3: independent evidence

- Frozen all-development refit.
- One holdout opening.
- Performed only under [07](07_final_selection_and_deployment.md).

## 4. Stage 0: integrity and lower bounds

| ID | Hypothesis or question | Method | Budget | Required output | Advance rule |
|---|---|---|---|---|---|
| S0.1 | Class and mask code matches the repository contract | Count classes/masks per training complement and validation fold | No training | Fold support assertions | All counts match saved evidence |
| S0.2 | OOF assembly is correct | Deterministic mock predictions keyed by fold | No training | One-row-per-ID assertions | Exact expected row counts |
| S0.3 | Accuracy is misleading under imbalance | Training-side majority predictor | 5 folds | OOF predictions and metrics | Descriptive baseline stored |
| S0.4 | Metric and seed wiring works | Training-side stratified random predictor | 5 folds, fixed seed | OOF predictions and metrics | Near expected prior/chance behaviour |
| S0.5 | Training loop can learn | Tiny-batch overfit for each target and shared mask path | Small engineering run | Loss and train accuracy curve | Near-memorisation without augmentation |
| S0.6 | No obvious leakage exists | Shuffled training labels | 5 folds only after pipeline works | OOF predictions | Near chance/prior validation behaviour |

Stop the whole experiment programme if S0 integrity fails. Fix the contract before training larger
models.

## 5. Stage 1: classical and capacity baselines

| ID | Main question | A | B or reference | Evidence | Decision |
|---|---|---|---|---|---|
| B1.1 | Do image features beat the majority prior? | HOG+colour multinomial logistic | Majority | E1 | Keep as classical baseline even if it loses |
| B1.2 | Does a margin classifier add value? | Linear SVM | Logistic regression | Optional E1 | Run only if classical breadth is needed and calibration cost is acceptable |
| C1.1 | Do learned features beat HOG? | Small CNN, `[0,1]`, no augmentation, CE | B1.1 | E1 | Advance if learning and primary metrics are credible |
| C1.2 | Does fold-train standardisation help? | Small CNN with fold-train RGB mean/std | C1.1 | E1 | Keep only with stable gain or better optimisation |
| C1.3 | Does light augmentation reduce overfit? | Small CNN plus light augmentation | Best C1.1/C1.2 | E1 | Keep if OOF or train–validation gap improves |

### Stage 1 fixed variables

- Input `(80,60)`.
- One screening seed.
- Unweighted cross-entropy.
- No class sampling.
- Same optimiser family and sample exposure for CNN runs.
- Separate gender and usage models.

### Stage 1 stopping rule

Do not advance a CNN configuration when:

- it cannot pass tiny-batch overfit;
- it fails to improve pooled macro-F1 over the majority predictor;
- it collapses to the majority class;
- its training curve shows an unresolved implementation failure.

A weak but valid small CNN stays in the report as a capacity baseline.

## 6. Stage 2: scratch architecture screen

### Hypothesis

A compact named CNN will improve pooled OOF macro-F1 over the small CNN, but the best performance-cost
tradeoff is unknown.

| ID | Architecture | Input | Initialisation | Loss | Augmentation | Evidence |
|---|---|---|---|---|---|---|
| A2.1 | Small CNN winner | Frozen from Stage 1 | Random | CE | Frozen light/no-light winner | Existing E1 reference |
| A2.2 | MobileNetV3-Small | Same as A2.1 | Random | CE | Same | E1 |
| A2.3 | ResNet-18, low-resolution stem | Same as A2.1 | Random | CE | Same | E1 |

Run separate gender and usage versions. Use the same architecture screen for both targets, but allow
different per-target winners.

### Advance rule

Advance the Pareto set rather than only the highest point estimate. A candidate remains when it is
not clearly dominated on:

- primary metric;
- fold stability;
- per-class behaviour;
- checkpoint size;
- measured latency;
- training cost.

Normally advance at most two architecture families.

### Capacity fallback

Run one of these only if all Stage 2 candidates show underfit:

| ID | Candidate | Trigger |
|---|---|---|
| A2.4 | ResNet-34, low-resolution stem | ResNet-18 train and validation remain close and weak |
| A2.5 | EfficientNet-B0 | A different compact scaling family is justified by compute budget |

Do not run both by default. Choose the one that answers the observed limitation more directly.

## 7. Stage 3: controlled recipe refinement

Recipe refinement uses only the leading eligible architecture family for each target.

### 7.1 Resolution

| ID | Hypothesis | A | B | Fixed controls | Keep when |
|---|---|---|---|---|---|
| R3.1 | Larger feature maps preserve useful detail | `(128,96)` | `(80,60)` | Architecture, loss, augmentation, optimiser, seed, folds | Stable primary gain justifies added cost |

Do not add 224×224 to the minimum matrix. Add it only if both tested sizes show clear resolution
underfit and compute permits.

### 7.2 Augmentation strength

| ID | Hypothesis | A | B | Keep when |
|---|---|---|---|---|
| G3.1 | Light geometry/colour changes improve generalisation | Light augmentation | None | Better OOF or smaller overfit without rare-class harm |
| G3.2 | Strong mixing adds useful regularisation | One mask-safe Mixup or CutMix recipe | Light augmentation | Optional; stable gain across classes and calibration |

Run G3.2 only after unit tests prove correct per-target mask handling.

### 7.3 Imbalance treatment

| ID | Hypothesis | A | B | Keep when |
|---|---|---|---|---|
| I3.1 | Effective-number weights improve minority recall | Capped class-balanced CE | Unweighted CE | Macro-F1/per-class recall improve without severe accuracy or precision loss |
| I3.2 | Exposure rather than loss scale is the main problem | Capped weighted sampler | Best I3.1 result | Optional; better than weights without memorisation/false-positive collapse |
| I3.3 | Easy examples dominate optimisation | Fixed-gamma focal loss | Best CE method | Optional only if I3.1/I3.2 fail diagnostically |

Do not combine weighting and sampling in I3.1. Do not oversample `Home` as if repetition created new
evidence.

### 7.4 Optimiser and regularisation

| ID | Hypothesis | Comparison | Trigger |
|---|---|---|---|
| O3.1 | SGD improves the leading residual model | SGD-momentum versus AdamW | Run only for a ResNet finalist with credible but suboptimal curves |
| W3.1 | Weight decay is limiting fit | Small predeclared log grid | Run only within the leading architecture |
| D3.1 | Head overfit needs dropout | No/low dropout comparison | Run only when train–validation gap remains large |

Keep the grid small. A suggested maximum is:

- two learning rates;
- two weight-decay values;
- at most one extra dropout value.

Do not multiply a large grid across all architecture, resolution, augmentation, and imbalance choices.

## 8. Stage 4: separate versus shared

### M4.1 matched comparison

**Hypothesis.** A shared backbone can reduce the combined system cost without causing practically
important negative transfer.

Build under the best common backbone recipe:

| System | Backbone count | Heads | Purpose |
|---|---:|---:|---|
| Gender-only | 1 | 1 | Separate gender reference |
| Usage-only | 1 | 1 | Separate usage reference |
| Shared | 1 | 2 | Multitask candidate |

Fixed variables:

- architecture;
- input and transform;
- augmentation;
- optimiser and schedule;
- seed;
- fold;
- sample exposure;
- per-head loss type;
- class weighting policy;
- fixed epoch rule.

Use equal normalised task losses first.

Advance shared only when:

- the lower 95% paired family-bootstrap difference is above `-0.01` for gender;
- the same is true for usage;
- no supported class collapses;
- shared system latency or storage falls materially against the two-model sum.

### M4.2 conditional conflict method

Run only if M4.1 is close to the no-harm boundary and shared cost is valuable.

Choose one:

- learned uncertainty weighting; or
- PCGrad.

Do not run a broad task-weight search. Stop sharing work if one output still fails the no-harm rule.

### M4.3 optional partial sharing

Only when full sharing shows useful early common features and harmful later conflict. This is not a
minimum requirement.

## 9. Stage 5: pretrained comparison lane

| ID | Question | Method | Eligibility |
|---|---|---|---|
| P5.1 | How much does external visual pretraining help? | ImageNet-pretrained ResNet-18, full fine-tune | Comparison only |

Use the same folds and target outputs. Use the preprocessing required by the exact weight enum.

Required registry flags:

```text
scratch=false
submission_eligible=false
official_prediction_eligible=false
application_eligible=false
```

Do not let P5.1 change the eligible winner. Its value is explaining the performance gap imposed by
the scratch requirement.

## 10. Stage 6: finalist confirmation

Select a small finalist set before this stage. A reasonable maximum is:

- one or two separate gender candidates;
- one or two separate usage candidates;
- one shared candidate, only if M4 passed.

For each finalist:

- all five folds;
- three fixed seeds;
- frozen input, augmentation, loss, optimiser, schedule, and epoch rule;
- full OOF logits;
- cost measurement;
- error and robustness artifacts.

The confirmation stage must not add new hyperparameters after seeing a bad seed.

## 11. Stage 7: post-training analysis

These are analysis runs, not architecture searches:

| ID | Analysis | Input | Decision |
|---|---|---|---|
| CAL7.1 | Cross-fitted temperature scaling | Finalist OOF logits | Whether calibrated confidence is retained |
| ROB7.1 | Fixed corruption suite | Clean and deterministically perturbed validation images | Whether robustness gates pass |
| ERR7.1 | Fixed slices and manual review | Saved OOF predictions and images | Failure taxonomy and limitations |
| COST7.1 | Latency/memory/storage benchmark | Frozen checkpoints | Practical winner and deployment limit |
| EXPL7.1 | Fixed Grad-CAM review | Predeclared examples | Shortcut and attention diagnosis |

Do not train on the exact robustness suite after seeing ROB7.1 results unless that creates a new,
clearly labelled development cycle. The original robustness result remains reported.

## 12. Suggested run budget

Count one “fold job” as one model trained on four folds and scored on the fifth.

### Minimum credible budget

| Stage | Approximate fold jobs |
|---|---:|
| Classical baseline, both targets | 10 |
| Small CNN reference, both targets | 10 |
| Two named architectures, both targets | 20 |
| Resolution and one recipe refinement, both targets | 20 |
| Matched separate/shared comparison | 15 |
| Pretrained comparison, both targets or two heads | 5–10 |
| Three-seed confirmation for two separate targets | 30 |
| Total | About 110–115 |

This is an estimate, not a requirement. Shared models and reused reference runs reduce duplication.

### Compute-saving rules

- E0 may catch crashes before a five-fold launch.
- Reuse an identical registered reference rather than retraining it.
- Stop adding architecture families after a clear Pareto set exists.
- Run optional imbalance and optimiser methods only when diagnostics justify them.
- Preserve all-five-fold evidence for claims; do not save compute by reporting a lucky fold.
- Prefer analysis of existing runs over an extra weakly motivated training run.

## 13. Stopping rules

### Hard stop

Stop and fix the pipeline when:

- split or family assertions fail;
- label maps or masks differ from the canonical contract;
- a supposedly scratch model loads external weights;
- OOF IDs are duplicated or missing;
- validation data fit preprocessing or weights;
- the registry write fails;
- loss becomes non-finite;
- the model cannot pass the tiny-batch test.

### Candidate stop

Stop investing in a candidate when:

- all-five-fold pooled primary performance fails to beat the required baseline;
- it is dominated in performance, cost, and stability;
- added capacity increases overfit;
- minority gains come with uncontrolled false-positive growth;
- a result depends on one rare row;
- a shared model harms either target beyond the frozen margin;
- the next experiment does not answer a distinct decision.

### Search stop

Stop hyperparameter search when:

- the predeclared grid is exhausted;
- differences are smaller than the practical margin;
- uncertainty intervals overlap enough that complexity, cost, and robustness should decide;
- remaining budget is more valuable for error analysis, three-seed confirmation, or report evidence.

## 14. Run-order recommendation

1. S0.1–S0.5 integrity and tiny-batch checks.
2. S0.3/S0.4 majority and stratified OOF baselines.
3. B1.1 HOG-plus-colour logistic models.
4. C1.1 small CNN without augmentation.
5. C1.2/C1.3 normalisation and light augmentation.
6. A2.2/A2.3 MobileNetV3-Small and low-resolution ResNet-18.
7. R3.1 input-resolution comparison on the leading family.
8. I3.1 class-balanced loss comparison.
9. Any justified optional I3/O3 method.
10. Freeze the best matched separate recipe.
11. M4.1 shared two-head comparison.
12. M4.2 only if M4.1 nearly passes and cost benefit is real.
13. P5.1 pretrained comparison-only benchmark.
14. Select the finalist set.
15. Stage 6 three-seed confirmation.
16. CAL7, ROB7, ERR7, COST7, and EXPL7 analysis.
17. Apply the selection gates from [05](05_evaluation_framework.md).
18. Enter the one-way final process in [07](07_final_selection_and_deployment.md).

## 15. Required record for every experiment decision

For each experiment ID, write:

- question;
- hypothesis;
- reference run ID;
- changed variable;
- fixed variables;
- folds and seeds;
- result table path;
- OOF prediction path;
- cost artifact path;
- conclusion;
- advance/stop decision;
- limitations;
- whether the run is eligible.

Notebook 04 should narrate these decisions and import reusable tables. It should not contain a second
private training implementation.
