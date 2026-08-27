# Task 4 Baseline Search Design

- Status: Approved in the 2026-08-27 design interview
- Date: 2026-08-27

## Goal

Freeze and measure one simple, honest non-deep-learning reference system that
later learned retrieval models must beat. Milestone 4 measures this baseline;
it does not tune it to reach a target score.

## Safety boundary

- `data/processed/splits.csv` remains the only split.
- Ordinary evaluation uses validation fold `1` only.
- Holdout, quarantine, and official teacher-test pixels and labels remain
  sealed.
- V1 is an image variant of the same product IDs, not independent data.
- No training run occurs, so Milestone 4 must not create a fake
  `results/runs.csv` row.

## Frozen baseline

Promote the existing `spatial-hsv-edge-4x4-v2` probe unchanged:

- preprocess every query and catalogue image with the frozen `240×320`
  contract from ADR 0021;
- divide content into a `4×4` spatial grid;
- use the existing HSV colour and gradient-orientation histograms;
- exclude letterbox padding through the content mask;
- L2-normalize and equally weight the colour and edge blocks;
- join them into the existing 400-value float32 descriptor;
- rank by exact cosine distance.

Only a test-proven correctness bug may change this descriptor. Grid size, bins,
weights, colour space, and distance are not candidates in Milestone 4.
`96×128` remains historical preprocessing evidence, not a second baseline.

The baseline has zero learned parameters and zero checkpoint bytes. Its stored
descriptor payload is 1,600 bytes per image before container overhead.

## Ranking and evaluation

Reuse ADR 0019 without changing its split, exclusion, product collapse, tie, or
undefined-query rules. Produce Top-20 rankings so Protocol A can report K values
5, 10, and 20. Search-only timing includes the complete real ranking path:
cosine scoring, exclusions, variant collapse, tie handling, sorting, and Top-20
output.

When a product has several image variants in one gallery, keep its smallest
distance before Top-K truncation. This ratifies the fusion rule already used by
the protocol code.

Evaluate all four source directions:

1. teacher → teacher;
2. V1 → V1;
3. teacher → V1;
4. V1 → teacher.

Protocol A query-mean linear nDCG@10 is primary. The headline baseline value is
the equal mean of the two same-source Protocol A values. Cross-source values,
Protocol A class macro and precision, and Protocol B are supporting evidence.
Protocol B undefined queries remain excluded rather than scored as zero, and
coverage is always shown.

Add one deterministic random-ranking sanity floor for Protocol A only. Shuffle
the eligible gallery product IDs once with NumPy seed `2753`, use that fixed
order for every fold-1 query, and apply the normal Top-20 preparation and
metric code. This is not a candidate, does not enter tuning, and does not apply
to Protocol B.

The two baseline hypotheses are:

1. the frozen baseline's same-source mean nDCG@10 is greater than the random
   sanity floor; reject this claim if it is equal or lower;
2. the cross-source mean nDCG@10 is at least 95% of the same-source mean;
   reject the source-robustness claim if it falls below that ratio.

The main notebook table has one row per source direction. It shows the main
K=10 quality values, Protocol B coverage, latency, and index size. Complete
K=5/10/20 evidence and the random sanity floor remain in tracked CSV files.

## Timing and cost policy

Measure on CPU with one fixed thread and batch size one:

- use the first 100 eligible query IDs in ascending order as an untimed warm-up;
- then time every fold-1 query once in ascending ID order;
- keep every sample, including slow outliers;
- report p50 and p95;
- record CPU, logical-core count, operating system, Python and NumPy versions,
  thread count, warm-up count, and timed-query count.

End-to-end query time begins with reading the local image and includes decode,
EXIF handling, preprocessing, descriptor encoding, and the full search path.
Search-only time starts from an already encoded query descriptor.

Measure index construction separately. It includes reading and preprocessing
the gallery, descriptor extraction, and stacking the searchable matrix. Use
cached embeddings for quality scoring, but never use them to shorten
end-to-end query timing.

Report stored descriptor/index bytes separately from peak process memory.
Measure peak memory in a clean child process. State whether p95 end-to-end time
is below one second and whether the index is below 1 GB. These are checks, not
deployment promises.

Timing file schemas and row order are deterministic, but measured timing values
are machine-dependent and are not expected to be byte-identical across runs.

## Failure slices

Derive slice membership from the frozen fold-1 views before inspecting baseline
slice scores:

- **Grayscale:** `mode == "L"`; currently 44 fold-1 queries.
- **Rare article type:** 1–9 broad Protocol A gallery positives; currently 40
  queries across 21 article types. Zero-positive queries remain a separate
  undefined-coverage case.
- **Rare article-type/colour pair:** fewer than 10 strict Protocol A positives,
  including zero; currently 833 queries.
- **Observed unusual geometry:** `aspect_ratio != 0.75`; only two fold-1 queries
  exist, so show examples but do not present their average as reliable.
- **Large background and unusual canvas:** reuse the deterministic clean,
  2:1-wide, and 1:2-tall canvas transforms. Keep this stress result separate
  from the normal headline score.
- **Family unavailable:** zero eligible Protocol B positives; currently 3,937
  queries.
- **Weak but scorable family:** 1–4 eligible Protocol B positives; currently
  1,659 queries.

Every slice result records its score, query count, excluded count, and coverage.
Small slices must remain visibly small.

## Examples and figure

Use V1 → V1 for the illustrative retrieval figure without treating V1 as the
chosen deployed index. Show Top-5 results while metrics remain scored at 10.

Choose IDs deterministically:

- normal success: highest per-query nDCG@10 among clean queries outside the
  declared failure slices;
- failure examples: lowest applicable score inside each declared figure slice;
- large-background example: largest deterministic drop from its clean score;
- ties: ascending numeric product ID.

An ID may represent more than one slice; repeated examples are allowed because
slice overlap is evidence. The figure must include a normal success and visible
grayscale, unusual-geometry or canvas, large-background, and weak-family
failures. Rare-class and rare-pair results remain in the table even when they
do not add a separate readable figure row.

## Evidence artifacts

Write small tracked artifacts below `results/evidence/task4/`:

- `baseline_summary.csv`;
- `baseline_query_metrics.csv`;
- `baseline_failure_slices.csv`;
- `baseline_timing.csv`;
- `baseline_cost.json`;
- `baseline_examples.csv`.

Write `results/figures/task4/baseline_examples.png`. Describe every artifact in
the evidence and figure README files. Notebook values and figures are loaded
from these artifacts rather than hard-coded.

## Python ownership and merge isolation

Task 4 owns the implementation:

- real modules live directly below `src/fashion/task4/`;
- Task 4 runners live below `scripts/task4/`;
- Task-4-only tests live below `tests/task4/`;
- notebooks, results, ADRs, plans, specs, and shared tests keep their current
  repository locations.

Move the existing `external`, `preprocessing`, `cache`, `protocol`, `probe`, and
`preprocessing_experiment` modules into `fashion.task4`. New baseline logic
lives there too. Task 4 scripts and notebooks use `fashion.task4.*` imports.

Keep `fashion.retrieval.*` as thin compatibility modules that only re-export
the corresponding `fashion.task4.*` objects. There must be one real
implementation, not two copies. A Task-4-specific change to shared behaviour
uses a small wrapper or adapter around shared code instead of copying a shared
module.

Moving Task 4 code reduces path overlap but cannot eliminate conflicts in
shared documentation, configuration, or test indexes. Keep such edits small.

## Decision records and documentation

Add a baseline ADR that freezes the descriptor, source summary, fusion,
timing, and failure-slice rules. Clarify superseded wording in ADR 0009 and the
old `96×128` future-candidate wording in ADR 0020 without rewriting their
historical decisions.

Update active documentation and commands to use `fashion.task4` and
`scripts/task4/`. Historical text may retain old paths only when clearly marked
as historical.

## Verification and completion

Tests must cover baseline aggregation, failure-slice boundaries, timing
aggregation, deterministic example selection, and old-import compatibility.
Move only Task-4-only tests; shared documentation and notebook tests remain
shared.

Milestone 4 is complete only when:

1. focused Task 4 tests pass;
2. shared documentation tests pass;
3. the fresh full repository suite passes;
4. lint and `git diff --check` pass;
5. both Task 4 notebooks pass `nbformat.validate`;
6. fresh baseline evidence agrees with notebook values;
7. the known wide/tall failure remains visible;
8. `notebooks/task-4/PROGRESS.md` is updated.

No holdout result, descriptor tuning, learned model, deployed source choice, or
training-registry work belongs to this milestone.
