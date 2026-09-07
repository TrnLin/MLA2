# Task 3 reproducibility and artifact plan

[Previous: final selection and deployment](07_final_selection_and_deployment.md) ·
[Research index](README.md) · [References](references.md)

## 1. Purpose

Task 3 evidence is reproducible only when another team member can answer:

- Which data and folds were used?
- Which labels and masks were used?
- Which exact model and weights were used?
- Was the model scratch-trained or pretrained?
- Which transform, loss, optimiser, seed, and budget were used?
- Which predictions produced a table or figure?
- Can the reported metrics be regenerated without retraining?
- Which artifact was used for holdout, official predictions, and the application?

This document defines the minimum evidence chain.

## 2. Reproducibility principles

1. Configuration is data. Save it in a machine-readable form.
2. Predictions are primary evidence. Save them before summary metrics.
3. A checkpoint without preprocessing and label maps is incomplete.
4. A metric without run and prediction IDs is not traceable.
5. Every training execution receives a registry row, including failed or debug training runs.
6. Pretrained status and eligibility are explicit fields.
7. Hash immutable artifacts.
8. Keep report tables generated from artifacts, not manually typed values.
9. Record hardware and framework versions because seeds alone do not guarantee exact equality.
10. Never put protected holdout targets into normal development evidence.

PyTorch states that exact reproducibility is not guaranteed across versions, platforms, or CPU/GPU
even with the same seed. [PyTorch reproducibility guide](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## 3. Proposed artifact layout

The exact implementation can follow repository conventions, but a clear logical layout is:

```text
results/
  runs.csv
  task3/
    hypotheses/
    decisions/
    configs/
    predictions/
      oof/
      holdout/
      official/
    checkpoints/
    metrics/
    calibration/
    robustness/
    errors/
    cost/
    logs/
results/figures/task3/
```

Notebook 04 should read these artifacts and narrate the development decision. Notebook 06 should
read the frozen final artifacts and perform the protected final evaluation.

## 4. Data contract artifacts

Every run record should reference or capture:

- canonical split path;
- CV assignment digest;
- development/holdout/quarantine ID-set digests;
- label-map path and hash;
- taxonomy path and hash;
- target-mask column names;
- teacher-image-only scope;
- quarantine rule;
- fold training/validation product counts;
- fold training/validation family counts;
- per-target fold class support;
- image manifest or source hash scope.

Canonical sources include:

- [`splits.csv`](../../../data/processed/splits.csv)
- [`cv_fold_summary.json`](../../../data/processed/cv_fold_summary.json)
- [`label_maps.json`](../../../data/processed/label_maps.json)
- [`taxonomy.json`](../../../data/processed/taxonomy.json)
- [`development_class_summary.csv`](../../../data/processed/development_class_summary.csv)
- [data-preparation evidence](../../../results/evidence/data_preparation/)

## 5. Configuration capture

Use one machine-readable configuration per logical run family, with fold and seed overrides recorded
for each execution.

### 5.1 Identity

```text
run_id
experiment_id
hypothesis_id
decision_id
task=task3
target=gender|usage|gender_usage_shared
parent_experiment_id
parent_run_ids
reference_run_ids
timestamp_start
timestamp_end
status
```

### 5.2 Parent-child question

```text
observed_weakness
trigger_observation_ids
evidence_paths
hypothesis
single_changed_factor
fixed_controls
expected_result
rejection_condition
created_before_child_run
decision=accept|reject|stop
accepted_parent_for_next_cycle
decision_reason
```

### 5.3 Data

```text
split_path
split_digest
partition=development
validation_fold
training_fold_list
training_product_count
validation_product_count
training_family_count
validation_family_count
label_map_digest
teacher_images_only=true
```

### 5.4 Labels and masks

```text
gender_class_order
usage_class_order
gender_mask_column
usage_mask_column
literal_na_is_class=true
home_kept=true
missing_usage_count
zero_training_support_classes
```

### 5.5 Image transform

```text
input_height
input_width
exif_transpose
rgb_conversion
resize_method
interpolation
preserve_aspect_ratio
padding_colour
pixel_range
normalisation_mean
normalisation_std
normalisation_fit_partition
padding_excluded_from_statistics
```

### 5.6 Augmentation

Record every operation, probability, magnitude, and order:

```text
horizontal_flip_probability
rotation_degrees
translation_fraction
scale_range
brightness_range
contrast_range
saturation_range
hue_range
mixup_alpha
cutmix_alpha
random_erasing_area
mask_handling_rule
```

Use explicit null/false values when an operation is disabled.

### 5.7 Model

```text
model_family
architecture_variant
shared_backbone
backbone_parameters
stem_definition
normalisation_layer
activation
dropout
head_definitions
parameter_count
initialisation
weights_argument
pretrained_source
scratch
submission_eligible
official_prediction_eligible
application_eligible
```

### 5.8 Loss and imbalance

```text
loss_name
label_smoothing
focal_gamma
class_weight_method
class_weight_beta
class_weight_cap
resolved_class_weight_vector
sampling_method
resolved_sampling_values
gender_task_weight
usage_task_weight
gradient_conflict_method
mask_normalisation_rule
```

### 5.9 Optimisation

```text
optimizer
learning_rate
weight_decay
momentum_or_betas
scheduler
warmup_steps
batch_size
gradient_accumulation
precision
gradient_clip
epoch_budget
step_budget
checkpoint_rule
```

### 5.10 Randomness

```text
seed
python_seed
numpy_seed
framework_seed
dataloader_generator_seed
worker_seed_rule
deterministic_algorithms
cudnn_benchmark
```

### 5.11 Runtime

```text
python_version
framework_version
torchvision_version
numpy_version
sklearn_version
scikit_image_version
operating_system
cpu
gpu
driver
cuda_or_accelerator_runtime
ram
vram
worker_count
thread_count
```

## 6. Run registry

The project requires every training run to append a row to `results/runs.csv` through the eventual
shared registry.

### Minimum registry fields

| Group | Fields |
|---|---|
| Identity | run ID, experiment ID, hypothesis ID, parent run IDs, timestamp, status |
| Question | observed weakness, changed factor, fixed controls, expected result, rejection condition |
| Scope | task, target, separate/shared, fold, seed |
| Data | split digest, label-map digest, train/validation counts |
| Model | family, variant, scratch/pretrained, eligibility |
| Recipe | input size, augmentation, loss, imbalance, optimiser, schedule, budget |
| Code | code identifier, config path/hash, environment hash |
| Output | checkpoint path/hash, prediction path/hash, log path |
| Metrics | primary metrics, selected secondary metrics, calibration metrics |
| Cost | train time, peak memory, parameter count, checkpoint bytes |
| Failure | exception type/message, last completed stage |

### Registry rules

- One row per actual training execution, not only per model family.
- The registry allocates and appends the execution row before the first optimiser step, then records
  completion or failure atomically.
- Fold and seed are explicit.
- A child run is rejected before training if its hypothesis record is missing, was created after the
  child start time, or names more than one main changed factor.
- A five-fold aggregate is generated from execution rows and OOF predictions; it does not replace
  them.
- Failed training runs remain recorded.
- Debug runs are marked so they cannot enter winner selection.
- Pretrained comparison rows are visually and programmatically separable.
- The final all-development refit is a distinct run.
- The final holdout evaluation record points to the frozen refit run.

## 7. Run ID and naming rules

Use stable, descriptive IDs that do not contain a result claim. One possible pattern is:

```text
t3_<experiment>_<target>_<model>_f<fold>_s<seed>_<short_config_hash>
```

Examples:

```text
t3_baseline_gender_smallcnn_f0_s2753_8c42d91a
t3_h03_usage_loss_f3_s2753_219fe661
```

Do not encode “winner,” “best,” or a metric value in an immutable run ID. Winner status can change as
evidence accumulates.

## 8. Prediction artifacts

Save predictions before calculating aggregate metrics.

### 8.1 OOF prediction schema

```text
id
product_family_group
cv_fold
run_id
model_id
seed
target
target_valid
true_label
predicted_label
logit_<class> for every class
probability_<class> for every class
confidence
top2_label
top2_probability
top2_margin
entropy
```

Shared model outputs may be stored in one wide table or two target tables, but both must point to the
same checkpoint ID.

### 8.2 OOF assertions

- 32,773 valid gender rows per complete fold set.
- 32,772 valid usage rows per complete fold set.
- One target–ID prediction exactly once.
- Fold matches canonical split.
- Family did not enter the model’s training complement.
- True label belongs to the fixed map.
- Predicted label belongs to the fixed map.
- Logits and probabilities finite.
- Probabilities sum to one within tolerance.
- `NA` preserved as a string class.
- Missing usage row absent from usage metrics.

### 8.3 Holdout prediction schema

Use the same core schema plus:

```text
final_refit_run_id
checkpoint_hash
preprocessing_hash
holdout_unlock_timestamp
```

Keep holdout target-bearing prediction artifacts inside the protected final-evaluation evidence
scope.

### 8.4 Official prediction artifact

Capture:

- exact CSV path;
- row count;
- header;
- ID-set/order digest;
- file SHA-256;
- final checkpoint hash;
- generation timestamp;
- schema-validation result.

## 9. Checkpoint package

A deployable checkpoint package includes:

- model state dictionary;
- architecture/configuration needed to reconstruct it;
- target head definitions;
- ordered label maps;
- input preprocessing;
- fitted RGB statistics;
- calibration parameter, if approved;
- confidence thresholds;
- framework/version metadata;
- run ID;
- checkpoint SHA-256;
- intended/forbidden-use note.

Do not rely on serialising a whole runtime model object without recording the reconstruction contract.

## 10. Metric artifacts

For each complete OOF candidate, save:

- aggregate metric JSON/CSV;
- per-class report;
- raw confusion matrices;
- row-normalised confusion matrices;
- per-fold table;
- support table;
- paired-comparison table where relevant;
- family-bootstrap samples or at least reproducible seed/configuration and quantiles;
- three-seed summary for finalists;
- joint exact-match result;
- `Home` influence companion;
- warnings and undefined metrics.

Every table includes:

```text
source_run_ids
source_prediction_hashes
metric_contract_version
generation_timestamp
```

## 11. Calibration artifacts

- Raw OOF logits.
- Cross-fitting fold assignment.
- Temperature per cross-fit training set.
- Cross-fitted calibrated probabilities.
- Raw and calibrated NLL/Brier/ECE.
- Reliability plot data.
- Final pooled OOF temperature.
- Calibration configuration and hash.
- Frozen app threshold and its OOF risk–coverage evidence.
- Note that final-refit temperature transfer is approximate.

## 12. Robustness and error artifacts

### Robustness

- Perturbation configuration.
- Exact severity values.
- Clean and perturbed prediction pairs.
- Corruption-level metric table.
- Per-class drop table.
- Consistency table.
- Calibration-under-corruption table.
- Robustness gate result.

### Errors

- Slice definitions.
- Slice support and metric table.
- High-confidence-error table.
- Fixed manual-review index.
- Reviewer tags and notes.
- Failure taxonomy summary.
- Separate/shared regression list.
- Fixed Grad-CAM index and figures.

## 13. Cost artifacts

Save:

- hardware and runtime configuration;
- warmup count;
- timed iteration count;
- raw timing observations or summary with reproducible command/config;
- CPU/GPU p50 and p95;
- batch throughput;
- load time;
- peak RAM/VRAM;
- parameter count;
- operation estimate method;
- checkpoint bytes;
- combined separate-system and shared-system totals.

Do not copy latency from architecture documentation. Measure the actual frozen model and input size.

## 14. Seed and determinism checklist

- [ ] Python random seed set.
- [ ] NumPy seed or Generator set.
- [ ] Framework CPU seed set.
- [ ] Framework accelerator seeds set.
- [ ] DataLoader generator set.
- [ ] Worker initialisation rule recorded.
- [ ] Sampler seed set.
- [ ] Augmentation RNG source recorded.
- [ ] Deterministic-algorithm setting recorded.
- [ ] cuDNN benchmark/determinism settings recorded where relevant.
- [ ] Same seed means the same fold ordering and initialisation path within one environment.
- [ ] Finalists repeated over three seeds.
- [ ] Non-deterministic warnings stored rather than ignored.

Do not promise bitwise equality across a different framework release or device.

## 15. Environment capture

Before the first real run, pin and record the ML environment. The current
[`pyproject.toml`](../../../pyproject.toml) does not yet include PyTorch, TorchVision, Scikit-learn, or
Scikit-image.

Capture:

- Python version;
- package lock or constraints file;
- installed-package freeze;
- CUDA/accelerator runtime;
- driver;
- OS and kernel;
- CPU and GPU models;
- memory;
- repository code identifier;
- dirty-worktree state at run time;
- configuration hash.

Use the project interpreter convention `./.venv/bin/python` once dependencies are resolved.

## 16. Hashing and immutability

Hash at least:

- split and label maps;
- run configuration;
- environment lock;
- checkpoints;
- preprocessing statistics/configuration;
- calibration parameters;
- OOF predictions;
- holdout predictions;
- official CSV;
- final comparison table.

Use SHA-256 consistently. Store the digest beside the artifact path in the registry or artifact
manifest.

Checkpoint-lock rule:

- a changed byte creates a new hash and a new artifact;
- do not overwrite a checkpoint that has been used for holdout;
- do not reuse the same path for a different seed or configuration;
- retain the exact final file used by the application and official prediction generator.

## 17. Evidence traceability

The expected chain is:

```text
repository decision and split
  → EDA-derived primary baseline configuration
  → registry execution rows
  → checkpoint and OOF predictions
  → curves, metrics, classes, failures, core robustness, and cost
  → written hypothesis with one changed factor
  → child registry rows and evidence
  → accept, reject, or stop decision
  → repeat only from an accepted parent
  → finalist uncertainty, calibration, robustness, explanation, and cost
  → Notebook 04 chain and selected run IDs
  → frozen final-refit run and hashes
  → Notebook 06 holdout evidence
  → official predictions and application package
  → report table/figure
```

Every report number should be traceable backward through that chain.

### Report table metadata

For each table or figure, record:

- source run IDs;
- source prediction paths/hashes;
- generation code or command;
- metric version;
- output path;
- generation timestamp.

## 18. Documentation artifacts

### Model card

Include:

- model and version;
- intended use;
- forbidden use;
- training data scope;
- evaluation data and metrics;
- per-class limitations;
- calibration and review policy;
- robustness and cost;
- ethical risks;
- monitoring and rollback;
- contact/owner.

Model Cards were proposed to document model uses, evaluation, and limitations.
[Model Cards](https://doi.org/10.1145/3287560.3287596).

### Dataset note

Include:

- source and role;
- row/image counts;
- split and family design;
- target semantics and masks;
- class support;
- image properties;
- known shortcuts;
- label and ethical limitations;
- protected holdout boundary.

Datasheets provide a structured way to document dataset motivation, composition, collection, use,
and limits. [Datasheets for Datasets](https://doi.org/10.1145/3458723).

### Decision log

Record:

- primary baseline run IDs;
- parent and child run IDs;
- trigger observations and evidence paths;
- hypotheses and one changed factor;
- accepted, rejected, and stop decisions;
- accepted parent for the next cycle;
- evidence paths;
- practical margin;
- shared no-harm outcome or why the cost gate did not open;
- final winner;
- freeze timestamp;
- holdout unlock timestamp.

## 19. Final handoff checklist

### Data and method

- [ ] Canonical split and digests recorded.
- [ ] Label maps and masks recorded.
- [ ] Teacher-only proof recorded.
- [ ] Selected run IDs documented.
- [ ] Scratch eligibility verified.
- [ ] Separate/shared decision documented.
- [ ] Parent-child hypotheses were created before their child runs.
- [ ] Every child changes one main factor.
- [ ] Accepted, rejected, and stop decisions are traceable.

### Final model

- [ ] All-development refit complete.
- [ ] Registry row complete.
- [ ] Checkpoint hash complete.
- [ ] Preprocessing hash complete.
- [ ] Calibration/threshold hash complete.
- [ ] Reconstruction test passes.
- [ ] Batch/single prediction parity passes.

### Evaluation

- [ ] Five-fold OOF evidence complete.
- [ ] Three-seed finalist evidence complete.
- [ ] Per-class and confusion evidence complete.
- [ ] Paired uncertainty complete.
- [ ] Calibration and risk–coverage complete.
- [ ] Robustness complete.
- [ ] Cost complete.
- [ ] Ethical/manual review complete.

### Holdout

- [ ] Method freeze timestamp precedes unlock.
- [ ] Team approval recorded.
- [ ] Locked hashes recorded at unlock.
- [ ] Holdout predictions saved.
- [ ] Frozen metric set calculated.
- [ ] No post-unlock tuning occurred.
- [ ] OOF–holdout gap explained honestly.

### Official predictions

- [ ] Exact column order.
- [ ] Correct row count and ID order.
- [ ] No duplicate IDs.
- [ ] No missing required values.
- [ ] Every label belongs to the fixed maps.
- [ ] File hash recorded.
- [ ] Generation checkpoint hash recorded.

### Application

- [ ] Correct checkpoint and transform loaded.
- [ ] Input error handling tested.
- [ ] Output decoding tested.
- [ ] Confidence shown only if approved.
- [ ] Review threshold applied.
- [ ] `Home` forced review applied.
- [ ] Human override present.
- [ ] Catalogue-only wording present.
- [ ] Monitoring and rollback owner named.

### Report

- [ ] Comparison table generated from registry/evidence.
- [ ] Final judgement names rejected alternatives.
- [ ] Independent holdout result included.
- [ ] Pretrained comparison clearly ineligible.
- [ ] Limitations and ethical non-use included.
- [ ] All cited figures exist under `results/figures/`.

## 20. Retention and recovery

Retain:

- final and runner-up configs;
- all registry rows;
- all finalist OOF predictions;
- final checkpoints;
- protected holdout evidence;
- official CSV;
- report-generating artifacts.

Large intermediate debug checkpoints may be removed only under an agreed retention rule after their
registry/config/log evidence is safe and they are not referenced by a report or decision.

Never overwrite or delete the exact artifact used for holdout or official predictions without a
verified recoverable copy.
