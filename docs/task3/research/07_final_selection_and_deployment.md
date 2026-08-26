# Task 3 final selection and deployment plan

[Previous: error, robustness, and ethics](06_error_robustness_and_ethics.md) ·
[Research index](README.md) · [Next: reproducibility and artifacts](08_reproducibility_and_artifacts.md) ·
[References](references.md)

## 1. Purpose

This document defines the one-way path from completed development research to:

- a frozen Task 3 method;
- an all-development scratch refit;
- locked checkpoints and preprocessing;
- one independent holdout evaluation;
- official prediction columns;
- application handoff;
- limited monitored use.

After the holdout unlock, development model selection is over.

## 2. Preconditions for final selection

Do not begin final selection until all of these are true:

- Dummy, classical, and scratch CNN baselines are complete.
- All compared models used the same five-fold scope.
- Every training run is present in the required registry.
- Full OOF logits and probabilities exist for finalists.
- Primary, secondary, class, calibration, robustness, and cost evidence is reproducible.
- Finalists have three-seed confirmation.
- Pretrained systems are marked ineligible.
- Separate/shared negative-transfer evidence is complete.
- Error and ethics review is complete.
- No protected holdout result has been inspected.

The selection gates are defined in
[05_evaluation_framework.md](05_evaluation_framework.md).

## 3. Final system choices

There are only two valid system forms:

### Option A: separate target models

- One frozen gender checkpoint.
- One frozen usage checkpoint.
- Each may use a different eligible architecture or recipe.

Use when:

- shared learning harms either target;
- per-target winners differ materially;
- the application can afford two image passes;
- simpler task-specific diagnosis is more valuable than sharing.

### Option B: one shared backbone with two heads

- One frozen checkpoint.
- Separate gender and usage logits and labels.

Use only when:

- both paired no-harm tests pass;
- no important class collapses;
- robustness and calibration remain acceptable;
- combined latency or storage falls enough.

**Recommendation.** Default to Option A until Option B proves non-inferiority. Sharing is an
experimental result, not an assumption.

## 4. Selection decision record

The decision record should state:

```text
decision_timestamp
decision_owner
development_split_digest
candidate_run_ids
selected_gender_run_id
selected_usage_run_id
selected_system_form
shared_run_id_if_used
primary_metric_results
paired_comparison_intervals
seed_stability
class_failures
calibration_result
robustness_result
cost_result
pretrained_benchmark_result_ineligible
rejected_candidates_and_reasons
accepted_limitations
holdout_not_yet_unlocked=true
```

The narrative should answer:

- Why did the winner win?
- What did the runner-up do better?
- Which classes remain weak?
- Was a performance difference practically meaningful?
- Why is separate or shared the correct system form?
- Which limitation cannot be fixed with this data?

## 5. Method-freeze gate

Freeze every item below before the protected loader is called.

### 5.1 Data contract

- `splits.csv` path and digest.
- Development ID-set digest.
- Label-map path and digest.
- Target mask columns.
- Teacher-image-only scope.
- Quarantine exclusion.

### 5.2 Model contract

- Separate or shared form.
- Architecture name and complete structural parameters.
- Low-resolution stem details.
- Number and order of output classes.
- Random weight initialisation.
- No pretrained checkpoint.
- Head definitions.

### 5.3 Input contract

- Height and width.
- EXIF handling.
- RGB conversion.
- Resize interpolation.
- Aspect-ratio and letterbox rule.
- Padding colour.
- Pixel scaling.
- Normalisation fitting procedure.
- Validation/inference transform.

### 5.4 Training contract

- Augmentation.
- Loss and mask formula.
- Class-weight or sampler formula and cap.
- Shared task weights or conflict method.
- Optimiser and all parameters.
- Scheduler and warmup.
- Batch size and accumulation.
- Precision.
- Epoch or step budget.
- Random seed or frozen ensemble membership.
- Determinism settings.

### 5.5 Post-training contract

- Calibration method.
- Temperature fitting rule.
- Confidence signal.
- Review/abstention thresholds.
- Top-2 display rule.
- `Home` forced-review rule.

### 5.6 Evaluation contract

- Primary and secondary metrics.
- Fixed class order.
- `Home` companion analysis.
- Class reports.
- Bootstrap method and replicate count.
- Robustness suite and gates.
- Cost hardware and timing procedure.
- Holdout warning thresholds.

### 5.7 Output contract

- Checkpoint paths.
- Preprocessing artifact path.
- Configuration path.
- Prediction column order.
- Application model interface.
- Evidence paths.

The freeze record needs a timestamp and team sign-off.

## 6. Final all-development refit

The locked [final-evaluation notebook](../../../notebooks/06_final_evaluation.ipynb) states that the
frozen method is refitted on all development before holdout unlock.

### Procedure

1. Load the canonical split.
2. Assert development, holdout, and quarantine ID digests.
3. Select development rows only.
4. Confirm teacher-image paths only.
5. Fit final RGB statistics on development content pixels only, if the frozen method uses them.
6. Derive final class weights or sampling values from development valid labels only, if used.
7. Initialise the frozen eligible model from random weights.
8. Train with the frozen schedule and seed.
9. Use all 32,773 valid gender labels.
10. Use all 32,772 valid usage labels.
11. Keep all output classes, including `Home` and `NA`.
12. Save the final checkpoint and preprocessing.
13. Write the final-refit run to the registry.
14. Produce no holdout metric yet.

### Epoch rule

The final epoch or step count comes from the frozen development rule. Do not use holdout for early
stopping. If the rule uses the median best development epoch, calculate and freeze that value before
refit.

### Final calibration

If cross-fitted development evidence approved temperature scaling:

- fit the frozen temperature on pooled development OOF logits;
- save it separately from the model checkpoint;
- apply it to final-refit logits at inference;
- state that transferring a calibration value from fold models to the all-development refit is an
  approximation;
- do not fit it on holdout.

## 7. Checkpoint lock

After refit, save and hash:

- model checkpoint(s);
- serialised state dictionary format/version;
- preprocessing configuration;
- fitted RGB statistics;
- label maps;
- calibration parameters;
- confidence thresholds;
- complete run configuration;
- environment/lock data;
- relevant code version identifier;
- final-refit registry row;
- inference smoke-test output.

The lock record should include:

```text
checkpoint_sha256
preprocessing_sha256
label_map_sha256
calibration_sha256
configuration_sha256
environment_sha256
lock_timestamp
locked_by
```

Any checkpoint or configuration change after this point invalidates the planned holdout evaluation.

## 8. Pre-unlock approval

Before opening holdout, the team confirms:

- method-freeze checklist complete;
- final refit complete;
- artifact hashes complete;
- inference smoke test passes;
- no intended code or threshold change remains;
- evaluation notebook contains the frozen metric contract;
- holdout access is necessary and authorised now;
- the team accepts that a poor result will be reported, not tuned away.

Record approval and timestamp in Notebook 06.

## 9. One-time holdout unlock

Use only the protected final loader with explicit:

```text
evaluation_unlocked=True
```

The unlock step should log:

- timestamp;
- approver;
- final checkpoint hash;
- final configuration hash;
- loader call;
- number of joined holdout rows;
- proof that no protected output appeared earlier.

Do not write holdout targets into ordinary development, EDA, or model-selection files.

## 10. Independent holdout evaluation

### Apply without refit

- Load the locked checkpoint.
- Load locked preprocessing and calibration.
- Set evaluation mode.
- Disable gradients.
- Predict holdout images once.
- Save raw logits and probabilities before making summary tables.

### Calculate only frozen outputs

- Gender and usage primary metrics.
- `Home` companion analysis where support permits.
- Secondary metrics.
- Per-class support and metrics.
- Confusion matrices.
- Calibration.
- Risk–coverage.
- Predeclared slices.
- Predeclared robustness.
- Cost.

### Compare with development expectation

Report:

- OOF estimate and interval;
- holdout result;
- absolute difference;
- class-level changes;
- calibration change;
- robustness change;
- plausible reasons for any gap.

### Poor holdout result

If holdout is poor:

- preserve the result;
- investigate only for explanation;
- report distribution shift, selection optimism, class-support change, or operational failure where
  evidence supports it;
- mark the model not deployment-ready if gates fail;
- do not try a second candidate or change the threshold.

## 11. Official prediction generation

Official prediction images are separate and unlabeled.

### Fixed schema

The output must keep exactly:

```text
id,gender,articleType,season,usage
```

Task 3 owns only `gender` and `usage`, but the final merged file must preserve all required columns.

### Generation rules

- Use the locked final-refit checkpoint(s).
- Use the locked inference transform.
- Do not fit statistics on prediction images.
- Preserve official ID order.
- Produce one valid gender and one valid usage value per ID.
- Use the exact label spelling and order from the development maps.
- The operational review threshold does not create blanks in this CSV.

### Validation checks

- Header and column order exact.
- Row count exact.
- ID set and order exact.
- IDs unique.
- No missing required values.
- Every gender value belongs to its five-class map.
- Every usage value belongs to its nine-class map.
- Literal `NA` remains a string value if predicted.
- No index numbers leak into label columns.
- File parses with the expected CSV settings.
- Prediction artifact and checkpoint hashes recorded.

## 12. Application checks

### 12.1 Input checks

- File exists and decodes.
- EXIF orientation is handled.
- Grayscale and alpha inputs convert safely to RGB.
- Unusual dimensions use the frozen aspect-preserving rule.
- Corrupt inputs produce a clear error or review state.
- The application does not fit any statistic from user input.

### 12.2 Model checks

- Correct checkpoint hash loaded.
- Model in evaluation mode.
- Gradients disabled.
- Correct device and precision.
- Output tensor sizes are 5 and 9.
- Probabilities finite and sum to one within tolerance.
- Class-index decoding uses the locked maps.
- Batch and single-image outputs match within numeric tolerance.
- CPU/GPU outputs match within documented tolerance.

### 12.3 User-interface checks

- Say “catalogue target-audience label,” not person gender.
- Display usage as a catalogue label.
- Show confidence only if the frozen calibration passed.
- Show a top alternative when useful.
- Show “needs review” below the frozen threshold.
- Always review `Home`.
- Allow a human override.
- Do not block products or users based on the result.

### 12.4 Performance checks

- Cold and warm load times.
- Batch-1 p50/p95 latency.
- Peak memory.
- Two-model total versus shared latency, if relevant.
- Error handling under repeated requests.
- Deterministic output for repeated identical input under fixed inference mode.

## 13. Deployment limits

This project supports a student demonstration and a catalogue-assistance workflow. It does not
establish suitability for unrestricted production use.

Limits:

- The input distribution is a small teacher catalogue dataset.
- Usage labels are partly subjective and weakly visual.
- `Home` has no useful development generalisation evidence.
- Rare usage estimates are highly uncertain.
- The audience taxonomy is limited and socially sensitive.
- There is no demographic fairness dataset.
- The model may rely on article type, colour, person, or background shortcuts.
- Probability calibration may drift outside the development distribution.
- Official predictions provide no labels for post-release accuracy checking.

The model must not be used for person identity inference, customer profiling, access, pricing, or
another high-impact decision.

## 14. Monitoring plan

### 14.1 Input monitoring

Track:

- decode failure rate;
- image dimensions and aspect ratio;
- colour mode;
- brightness and contrast;
- near-white background proportion;
- blur/compression indicators;
- distance from development feature summaries where available.

Trigger review when a sustained window falls outside a predeclared development reference band.

### 14.2 Output monitoring

Track by model version:

- predicted class distribution;
- `Home`, `NA`, and `Unisex` rates;
- mean confidence and entropy;
- abstention/review rate;
- top-1/top-2 margin;
- error and exception rate.

Unexpected growth in a rare class may indicate drift, calibration failure, or an imbalance method
that does not transfer.

### 14.3 Reviewed-label monitoring

When human review labels are available, track:

- rolling macro-F1;
- per-class precision and recall;
- confusion matrix;
- NLL/Brier/calibration;
- override rate and reason;
- high-confidence-error rate.

Do not report accuracy alone.

### 14.4 Operational monitoring

- p50/p95 latency.
- Peak memory.
- Model load failures.
- Input-processing exceptions.
- Checkpoint/configuration version.
- Batch/single parity alerts.

### 14.5 Ethical monitoring

- Reports of identity-like wording or use.
- Overrides concentrated in one audience label.
- Complaints about stereotyped or exclusionary labelling.
- Model use outside catalogue assistance.
- Staff bypassing review warnings.

### 14.6 Retraining trigger

A retraining proposal may begin after:

- sustained predictive degradation;
- material input or label drift;
- repeated high-confidence errors;
- changed catalogue taxonomy;
- corrected data provenance;
- a new compute/deployment constraint.

Retraining is a new governed cycle. It must use the canonical split rules, registry, full development
evaluation, robustness tests, freeze gate, and a new independent test strategy. Do not silently
update weights online.

## 15. Rollback and retirement

Rollback when:

- wrong checkpoint/configuration is loaded;
- schema validation fails;
- latency or failure rate exceeds the accepted operational limit;
- confidence is shown without valid calibration;
- harmful or out-of-scope use is discovered;
- review controls fail.

Retire or narrow the model when known limits cannot be managed through review, documentation, or a
new evidence cycle.

## 16. Final handoff

Handoff only when the checklist in
[08_reproducibility_and_artifacts.md](08_reproducibility_and_artifacts.md) passes.

The recipient should receive:

- checkpoint(s) and hashes;
- preprocessing and label maps;
- calibration and thresholds;
- configuration and environment;
- final-refit and holdout evidence;
- official prediction artifact;
- application interface and smoke tests;
- intended/forbidden-use statement;
- human-review instructions;
- monitoring and rollback plan;
- known limitations and weak classes.
