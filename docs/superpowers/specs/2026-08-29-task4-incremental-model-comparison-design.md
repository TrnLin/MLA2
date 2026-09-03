# Task 4 Incremental Model Comparison Design

- Status: Approved in the 2026-08-29 design interview
- Date: 2026-08-29

## Goal

Freeze a controlled learned-model investigation for Task 4 before training
starts. The study must explain what each change did, compare meaningfully
different approaches, and leave enough evidence for a real deployment
judgement rather than chase one score.

The study uses an adaptive ladder: each incremental candidate changes one main
factor from the highest-scoring completed incremental candidate. A scratch
autoencoder supplies model-family breadth. An ImageNet-pretrained model is
comparison-only.

## Output contract

Every learned candidate encoder in this study maps one preprocessed image to
one finite, L2-normalized float32 embedding with shape `(128,)`. The frozen
non-learned baseline keeps its existing 400-value descriptor.

The search system compares the query embedding with stored gallery embeddings
using exact cosine distance. It returns a ranked Top-K list of distinct product
IDs. The application joins those IDs to the product image and available
metadata. Task 4 does not add a field to `styles_prediction.csv`.

When both teacher and V1 gallery views exist for one product, keep that
product's minimum distance before Top-K so it can occupy only one result slot.

## Safety boundary

- `data/processed/splits.csv` is the only split.
- Candidate comparison uses development fold `1`.
- Ordinary fold-1 runs train on folds `0`, `2`, `3`, and `4`.
- Holdout remains sealed until Notebook 06 after the final method and gallery
  policy are frozen.
- Quarantine is never used.
- Official teacher-test images are prediction-only and never enter training,
  validation, or the search index.
- V1 inherits the teacher product's partition and fold. It is an image variant
  of the same product ID, not independent data.
- Every training attempt registers through `fashion.train.registry`.
- Scratch candidates must initialize every trainable model weight without
  pretrained weights.
- The pretrained benchmark is permanently ineligible for stability selection,
  final submission, and deployment judgement.

## Framework and compute

Use PyTorch and torchvision with exact versions recorded in `pyproject.toml`
and the Python 3.12 constraints file. Use one independent process on each RTX
A6000 rather than distributed training. Each process is pinned to one GPU.

The full investigation has a preferred wall-clock budget of about 48 hours,
which permits at most about 96 GPU-hours across the two GPUs. Before a real
candidate starts, benchmark a synthetic forward/backward step and estimate the
complete matrix. If the approved 100-epoch matrix does not fit, stop and
revisit the budget; do not silently remove candidates, lower epochs, or change
batch rules.

## Frozen input contract

All methods reuse ADR 0021:

- `240×320` RGB;
- EXIF correction;
- transparency composited onto white;
- aspect-preserving LANCZOS resize;
- centred white letterboxing;
- retained content mask;
- lossless uint8 development cache;
- scale to `[0, 1]` at model input.

Use the source-specific RGB mean and standard deviation fitted from the current
round's training folds only. Apply the saved teacher statistics to teacher
views and the saved V1 statistics to V1 views. Standardized padding is exactly
zero. Never fit values on the validation fold or an incoming query.

## Shared encoder contract

The incremental family uses torchvision ResNet encoders:

- no classification head;
- global average pooling;
- a `512 → 512 → 128` projection head;
- BatchNorm and ReLU after the first two projection layers;
- no activation after the final 128-value layer;
- L2 normalization only when writing or searching embeddings.

VICReg receives the unnormalized 128-value projection so its variance and
covariance terms remain meaningful. Ranking receives the normalized projection.

R1 and all scratch descendants pass `weights=None`. B1 alone starts from the
exact pinned torchvision `ResNet18_Weights.IMAGENET1K_V1` state.

## Shared training recipe

Unless a candidate definition says otherwise:

- seed: `2753`;
- batch: 64 product IDs, yielding 64 teacher/V1 view pairs;
- optimizer: AdamW;
- initial learning rate: `3e-4`;
- weight decay: `1e-4`;
- first 5 epochs: linear warm-up;
- remaining epochs: cosine decay to `1e-6`;
- automatic mixed precision: enabled;
- gradient clipping: global norm `5.0`;
- planned epochs: 100;
- checkpoint scoring epochs: 20, 40, 60, 80, and 100;
- checkpoint selection: highest fold-specific equal same-source Protocol A
  query-mean linear nDCG@10;
- deterministic algorithms: required, with deterministic data ordering and
  worker seeds.

The shared recipe is fixed across model comparisons. An out-of-memory result
or unavailable deterministic operation is a failed smoke run, not permission
to alter one candidate. Any shared correction is made before real candidates
start and is applied to the entire matrix.

## R1 — base incremental candidate

R1 is a scratch ResNet-18 trained with VICReg on natural cross-source pairs:

- view A: teacher image;
- view B: V1 image with the same development product ID;
- no synthetic geometry or colour augmentation;
- embedding size: 128.

Use the standard VICReg component weights:

- invariance: `25`;
- variance: `25`;
- covariance: `1`.

This rule gives every training product one label-free positive pair and targets
the baseline's measured source shift. The pair is often visually easy, so R1
is a simple reference rather than an assumed winner.

## R2 — depth candidate

R2 changes only the encoder from scratch ResNet-18 to scratch ResNet-34. Its
pair rule, projection head, objective, optimizer, embedding size, schedule, and
evaluation are identical to R1.

The higher fold-1 development winner score from R1 and R2 becomes the current
champion and parent of R3. Exact score order chooses the parent; supporting
metrics do not replace the frozen winner score.

## R3 — geometry robustness candidate

R3 copies the current champion and changes only the positive-view augmentation
policy. Keep one member of each teacher/V1 pair clean. Apply one seeded geometry
transform to the other member:

- crop while retaining 80–100% of the known content bounds;
- scale retained clothing content to 50–100% of the available canvas;
- sample its valid horizontal and vertical placement uniformly;
- choose normal, 2:1-wide, or 1:2-tall white canvas with probabilities
  `0.50`, `0.25`, and `0.25`;
- run the result through the frozen `240×320` preprocessing contract.

Randomly choose which source receives the geometry transform for each pair.
Do not use colour jitter, grayscale conversion, blur, or compression in this
candidate. The whole predeclared geometry policy is one main experimental
factor and directly addresses the measured white-canvas failure.

The highest winner score among R1–R3 becomes the current champion and parent
of R4.

## R4 — family-aware objective candidate

R4 copies the current champion and adds one family-aware triplet term while
retaining VICReg.

Family positives:

- have a different product ID;
- share `product_family_group`;
- do not share `sha256` or `duplicate_group`.

Family negatives have a different `product_family_group`. Use batch-hard
selection among valid batch members, triplet margin `0.2`, and triplet weight
`1.0`. Anchors without an eligible family positive contribute VICReg only.

Use a family-aware batch arrangement of 16 family pairs plus 32 uniformly
sampled product IDs, totalling the same 64 product IDs and 128 source views as
other incremental runs. This sampler change is part of the single
family-objective factor and must be recorded explicitly.

Only about 41.9% of fold-1 training rows have a family positive. Keeping VICReg
therefore prevents singleton products from becoming unusable. The report must
also state that product-family groups are mostly derived from normalized
product names and are not perfect human similarity labels.

## R5 — scratch breadth candidate

R5 is a convolutional autoencoder trained from scratch. It is a breadth
candidate, not a parent for VICReg-specific changes.

- encoder: scratch ResNet-18;
- bottleneck: 128 values;
- decoder: convolutional upsampling path with no encoder skip connections;
- inputs: teacher and V1 training views treated as separate reconstruction
  examples;
- objective: mean squared reconstruction error over content pixels;
- no synthetic corruption or geometry augmentation;
- retrieval representation: L2-normalized bottleneck.

R5 uses the same preprocessing, optimizer, seed, 100-epoch budget, checkpoint
scoring schedule, and fold-1 evaluation as the incremental candidates. It is
scratch-trained and remains eligible for stability selection.

## B1 — pretrained comparison-only benchmark

B1 matches R1's ResNet-18 architecture, cross-source pairs, VICReg objective,
128-value projection, optimizer, and 100-epoch schedule. The only intended
model factor is ImageNet initialization.

B1 must record:

- `pretrained=true`;
- the exact torchvision weight enum;
- `eligibility=comparison_only`;
- a scratch-compliance failure if any code attempts to mark it eligible.

B1 receives full fold-1 comparison evidence but cannot become the current
champion, a stability finalist, the submitted Task 4 model, or the deployed
model.

## Adaptive execution order

Use the GPUs in independent phases:

1. run R1 and R2 in parallel;
2. score both and select the current champion;
3. run R3 and R5 in parallel;
4. score R3 and update the current champion;
5. run R4 and B1 in parallel;
6. score all five scratch candidates;
7. select the top two scratch configurations;
8. retrain each finalist from scratch with every fold held out in turn.

The stability stage uses five fresh runs per finalist, including a fresh fold-1
run. The original candidate run is not substituted for that fold.

## Frozen evaluation

Every best fold-1 checkpoint receives the complete existing Task 4 evaluation:

- Protocol A at K values 5, 10, and 20;
- Protocol B at K 10;
- teacher→teacher;
- V1→V1;
- teacher→V1;
- V1→teacher;
- exact cosine ranking;
- all exclusions before Top-K;
- deterministic distance then numeric-ID ties;
- product-level minimum-distance variant collapse;
- undefined-query exclusion and visible coverage.

The development winner score is the equal mean of teacher→teacher and V1→V1
Protocol A query-mean linear nDCG@10.

The source robustness ratio is the equal cross-source mean divided by that
same-source mean. A ratio of at least `0.95` is an important target, not an
eligibility gate.

All fold-1 models also report:

- Protocol A class macro and broad/strict precision;
- Protocol B Recall@10, Hit Rate@10, Precision@10, and coverage;
- grayscale, rare-article-type, rare-type/colour, unusual-geometry,
  family-unavailable, and weak-family slices;
- clean, wide-canvas, and tall-canvas scores and Top-10 overlap;
- deterministic qualitative success and failure examples;
- parameter count and checkpoint bytes;
- embedding and index bytes;
- index build time and peak RSS;
- CPU batch-one encoding, search, and end-to-end p50/p95.

Keep the existing teacher-derived grayscale and unusual-geometry slice
definitions in every direction for direct baseline comparability. State that
these labels describe the teacher file even for V1 queries. The two-query
unusual-geometry average is descriptive only.

The learned-model runner must assert that rankings contain every expected query
before metric evaluation. Missing queries are errors, never implicit zeroes.

## Gallery-source study

After the deployment candidate is selected from development evidence, compare
three galleries with its fold-1 checkpoint:

1. teacher embeddings only;
2. V1 embeddings only;
3. both teacher and V1 embeddings, collapsed to one result per product.

Use the equal mean of teacher and V1 query performance against each gallery as
the gallery quality score. If gallery quality is tied within the declared
`1e-5` evaluator tolerance, prefer the smaller and faster gallery. Freeze the
gallery policy before final refit and holdout evaluation.

For an arbitrary user query, apply the saved normalization associated with the
selected single-source gallery. For a two-view gallery, use the V1
normalization because arbitrary uploaded images are operationally closer to
the higher-resolution V1 input; record this as a deployment assumption and
test it in the app robustness work.

## Stability and deployment judgement

Rank R1–R5 by fold-1 development winner score. Retrain the top two from scratch
for all five validation folds and report each unweighted mean and standard
deviation. Stability runs need the primary score and coverage evidence; the
costly full slice, timing, canvas, and example suite remains required only for
the original fold-1 comparison runs.

A deployment candidate must:

- be scratch-trained;
- have CPU batch-one p95 end-to-end time below one second in all four source
  directions;
- keep each searchable index below one GiB.

If the finalist five-fold means differ by more than their pooled fold-to-fold
standard deviation, prefer the higher mean. Define pooled spread as:

`sqrt((sd_a^2 + sd_b^2) / 2)`.

If the mean gap is no larger than that spread, treat the result as a practical
stability tie and compare, in order:

1. source robustness ratio;
2. wide/tall canvas behaviour;
3. CPU p95 end-to-end time;
4. index storage.

This produces the development deployment judgement. It does not open holdout.

## Run registry

Create `fashion.train.registry` before any smoke or real training. The registry
owns one row per attempt in `results/runs.csv`.

Required fields:

- schema version;
- run ID and optional parent run ID;
- UTC start and completion times;
- task, run kind, status, and fold;
- method, architecture, objective, and source policy;
- pretrained flag, exact weight origin, and deployment eligibility;
- seed, embedding dimension, planned epochs, and selected epoch;
- configuration hash, split fingerprint, git commit, and dirty-tree flag;
- parameter count;
- checkpoint path and SHA-256;
- development winner score;
- cross-source score and source robustness ratio;
- Protocol B Recall@10;
- p95 end-to-end seconds and index bytes;
- evidence manifest path;
- error type and short error message.

Run kinds are `smoke`, `candidate`, `benchmark`, `stability`, and `final_refit`.
Statuses are `running`, `completed`, `failed`, and `cancelled`.

Append the initial `running` row under an inter-process file lock. Update only
that row under the same lock using an atomic temporary-file replacement.
Configuration identity fields become immutable after the initial append.
Concurrent GPU processes must never write the CSV directly.

Each attempt gets a new run ID. A failed run remains visible. A retry uses a new
run ID and links the failed attempt in its metadata. Only completed, eligible
run kinds enter report comparison tables.

## Evidence artifacts

Store learned comparison artifacts below `results/evidence/task4/` with run IDs
and configuration hashes:

- experiment matrix and parent choices;
- per-direction summaries;
- per-query metrics;
- family metrics and coverage;
- failure slices;
- canvas robustness;
- timing;
- cost and checkpoint metadata;
- selected examples;
- stability mean and standard deviation;
- gallery-source comparison;
- one machine-readable evidence manifest per run.

Write report figures below `results/figures/task4/`. Notebook 05 loads values
from artifacts rather than hard-coding them. `results/runs.csv` generates the
main model comparison table.

## Python ownership

Add shared training infrastructure below `src/fashion/train/`. Keep Task 4
model, pair, objective, embedding, and evidence adapters below
`src/fashion/task4/`. Keep headless runners below `scripts/task4/` and focused
tests below `tests/train/` and `tests/task4/`.

Generalize existing baseline helpers rather than copy them:

- accept learned method names and folds instead of hard-coding the probe;
- add checkpoint-keyed learned feature caches;
- assert complete query coverage;
- keep the existing ranking, protocol, timing, failure, and evidence contracts.

## Failure handling

- Non-finite loss, gradient, parameter, or embedding fails the run.
- An embedding with the wrong shape or norm fails before ranking.
- A split, fold, source, or scratch-compliance mismatch fails before GPU work.
- An incomplete ranking fails before metrics.
- An out-of-memory or determinism error fails the smoke run.
- A missing checkpoint or evidence artifact prevents completion status.
- No failed candidate is silently omitted from the registry or notebook record.
- No configuration changes after a result is observed without a new run ID and
  an explicit parent/change record.

## Verification

Tests must cover:

- registry schema, locking, atomic updates, retries, and two-process writes;
- split and fold rejection;
- cross-source pair identity;
- family-positive and exclusion rules;
- deterministic family-aware sampling;
- scratch versus pretrained eligibility;
- encoder and autoencoder output shape;
- VICReg and triplet loss finiteness;
- embedding normalization;
- checkpoint configuration and hash validation;
- complete query coverage;
- learned feature-cache invalidation;
- adaptive parent selection;
- top-two scratch-only stability selection;
- pooled-spread deployment rule;
- gallery variant collapse;
- run-to-evidence linkage.

Before training, run focused tests, the full repository suite, lint,
`git diff --check`, and one registered smoke attempt per model family. After
training, validate every checkpoint, registry row, CSV/JSON artifact, notebook,
and figure before claiming the milestone complete.

## Milestone boundary

Milestone 5 ends when this experiment matrix, budget, registry contract,
selection rules, and evidence contract are approved and frozen.

Milestone 5 does not install dependencies, implement training, start a smoke
run, train a candidate, choose a deployed model, or open holdout. Those actions
belong to later implementation and evaluation milestones.
