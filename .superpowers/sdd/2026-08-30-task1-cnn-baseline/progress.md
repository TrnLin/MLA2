# SDD ledger — plan: docs/superpowers/plans/2026-08-30-task1-cnn-baseline.md

Spec: docs/superpowers/specs/2026-08-30-task1-cnn-baseline-design.md
Branch start: c544ca2
Execution location: current checkout on branch task-1-article-type, by user direction.

History note: the approved design and plan are present as `0f7334a` and
`c544ca2` in the current branch history. Earlier local commit IDs were rewritten
before implementation began; no task implementation commit was lost.

## Baseline

- `./.venv/bin/python -m pytest -q`: 63 passed, 1 pre-existing failure.
- Failure: `test_delivered_cache_and_artifact_registry_match_bytes` because local
  `teacher_train_images` has 38,612 files while the delivered cache records 38,613.
- Canonical `splits.csv` has zero rows whose non-blank path is missing on disk.
- Ruling: proceed with Task 1 because the failing inventory check is outside this
  feature and every image path selected by the canonical split exists. Do not
  rewrite the cache or raw data. Cost if wrong: the real-data smoke run may expose
  a second data-integrity problem and final full-suite verification will retain the
  known baseline failure.

## Preflight interface table

| Tasks | Producer / consumer or shared file | Finding |
|---|---|---|
| 1 internal | Shared files, tests, dependencies, path constants | Consistent; `tests/test_config.py` is created rather than modified. |
| 2 internal | Model tests against exact architecture | Consistent. |
| 3 internal | Dataset validation and fold helper tests | Consistent; tests must use the repository split fixture shape. |
| 4 internal | Metric tests against fixed 124-class implementation | Consistent. |
| 5 internal | Smoke/full config, fold trainer, registry, artifacts | Consistent after plan adds `split_path` and `model_factory`. |
| 6 internal | Runner, aggregation, per-class evidence, figures | Consistent; a fold failure raises, so no partial aggregate is returned. |
| 7 internal | Notebook controller and documentation tests | Consistent. |
| 8 internal | Whole suite, safety scan, real-data smoke | One raw-text scan conflict ruled below. |
| 1 → 5 | `RunRegistry`, hashing, seeding, Task 1 result paths | Names and lifecycle match. |
| 1 → 6 | Task 1 figure/evidence paths and atomic CSV writes | Names and ownership match. |
| 1 → 7 | `TASK1_FIGURE_DIR` imported by notebook controller | Name matches. |
| 1 → 8 | `RUNS_CSV`, `RunRegistry`, `verify_artifact` | Names match. |
| 2 ↔ 3 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 2 → 5 | `Task1SmallCNN`, parameter count, model factory | Signatures match. |
| 2 ↔ 4 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 2 ↔ 5 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 2 ↔ 6 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 3 ↔ 4 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 3 → 5 | `Task1TorchDataset`, `get_task1_fold_rows` | Expected sample keys and fold rows match. |
| 3 ↔ 5 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 3 ↔ 6 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 4 → 5 | Metrics and prediction frame | Shapes and fixed class semantics match. |
| 4 → 6 | OOF validation, fold aggregation, per-class metrics | Interfaces match. |
| 4 ↔ 5 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 4 ↔ 6 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 5 → 6 | `Task1TrainConfig`, `Task1FoldResult`, `train_task1_fold` | Result includes preprocessing ID and completed status. |
| 5 → 7 | Training API exported for notebook through experiment runner | Notebook does not copy training logic. |
| 5 → 8 | Physical smoke run artifacts and registry record | Verification fields match. |
| 5 ↔ 6 | Shared `src/fashion/task1/__init__.py` edits | Sequential exports; no interface conflict. |
| 6 → 7 | `Task1ExperimentResult`, runner, figure helpers | Notebook fields and imports match. |
| 6 → 8 | Smoke/full runner and comparison outputs | Commands match. |
| 7 → 8 | Notebook/doc tests become part of full suite | No conflict. |

## Preflight rulings

- Ruling: replace Task 8's raw `rg` expectation with a code-aware check of Python
  files and notebook code cells. The notebook already contains narrative text that
  says not to call `train_test_split`, so raw text search cannot prove a forbidden
  call. Cost if wrong: a deliberately dynamic call could evade the check; task
  tests and final review still inspect the implementation.

## Task 1 review loop

- Ruling: the plan required porting reviewed blobs exactly, but the task reviewer
  found two load-bearing defects in those blobs: unlocked CSV read-modify-write
  and deterministic mode using `warn_only=True`. Permit targeted divergence from
  the blobs: add an inter-process lock around registry mutation and make
  deterministic mode strict. The spec's auditability and reproducibility rules
  bind more strongly than byte-for-byte reuse. Cost if wrong: the Task 1 branch
  will differ from the shared Task 2 implementation and may need conflict
  resolution when branches merge.
- Task 1: fix round 1/5 (2 addressed, 0 open — registry inter-process locking;
  strict deterministic algorithms; commit c343ce8).
- Task 1: complete (commits c544ca2..c343ce8, review clean).
- Task 2: minor (deferred): `fc1` hardcodes the fixed `64 * 4 * 3` adaptive
  output size instead of deriving it from the config; the approved contract fixes
  `(4, 3)`, so this does not affect current behavior.
- Task 2: complete (commits c343ce8..f623ae6, review clean with 1 deferred minor).
- Task 3: complete (commits f623ae6..50b7720, review clean).
- Task 4: complete (commits 50b7720..bdfb274, review clean).

## Task 5 review loop

- Ruling: final-eligible runs must prove that the in-memory split equals the
  canonical redacted frame loaded from `split_path`; hashing the path alone is
  insufficient because training consumes the frame. Cost if wrong: strict frame
  equality may reject a semantically identical caller frame with harmless dtype
  or row-order differences; callers can reload it through the canonical loader.
- Ruling: seal final eligibility around stage `experiment`, 20 epochs, and no
  batch limits, but allow a positive batch-size override because the approved
  design explicitly permits limited-hardware batch changes. Cost if wrong: batch
  size changes optimization dynamics and makes hardware-constrained runs less
  directly controlled, though the exact size remains recorded.
- Ruling: non-default `model_factory` is a test/smoke injection point only.
  Final-eligible runs must use `Task1SmallCNN` by identity before the registry can
  claim `task1_small_cnn_v1` and `scratch=true`. Cost if wrong: legitimate wrapper
  factories for the same architecture cannot be final-eligible without a future
  explicit identity contract.
- Task 5: minor (deferred): the current best-checkpoint test has one epoch and
  does not exercise later-worse epochs, earlier tie-breaking, or state reload.
- Task 5: fix round 1/5 (2 addressed, 1 reviewer request ruled by approved
  design — canonical split/frame equality and model identity addressed; exact
  batch-size lock rejected because the approved design permits a recorded
  limited-hardware batch override; commit 5ffd9a5).
- Task 5: complete (commits bdfb274..5ffd9a5, 1 plan-conflict ruling and 1
  deferred minor).
- Task 6: minor (deferred): aggregation counts five rows per preprocessing but
  does not separately prove fold labels are the unique set `{0,1,2,3,4}`; the
  real runner supplies requested folds sequentially.
- Task 6: minor (deferred): no direct confusion-figure test, though static review
  found the implementation compliant.
- Task 6: complete (commits 5ffd9a5..0a78c64, review clean with 2 deferred minors).
- Task 7: out-of-scope verification note: broad Ruff reports import-order noise in
  unchanged `src/fashion/task1/__init__.py`; Task 8 must correct and verify it.
- Task 7: complete (commits 0a78c64..e0afe64, review clean).

## Task 8 review loop

- Ruling: reject the verification-time model change from adaptive `(4,3)` / 768
  dense inputs to `(5,7)` / 2240. Implement an exact MPS-safe adaptive average
  pooling operation using the same bin boundaries as PyTorch, keep the approved
  model size, compare values against the CPU reference, and rerun the real smoke.
  Cost if wrong: a custom pooling implementation could differ numerically or in
  gradients from PyTorch; focused equivalence and real MPS tests reduce that risk.
- Ruling: waive the pre-existing prepared-data cache test for this feature. The
  raw inventory is one file short, the canonical split references no missing
  paths, and changing raw data or the delivered cache is outside authorized Task
  1 model work. Cost if wrong: the branch will finish with one known failing
  repository test and may require the data owner to restore the missing file
  before submission.
- Task 8: fix round 1/5 (1 addressed, 0 open — approved `(4,3)` / 768 CNN
  restored with exact MPS-safe pooling; corrected smoke and hashes verified;
  commit b00d34c).
- Task 8: complete (commits e0afe64..b00d34c, review clean with 1 explicit
  baseline-test waiver).

## Final whole-branch review

- Important: final eligibility must also seal seed, maximum learning rate,
  weight decay, gradient clipping, and approved preprocessing identity while
  retaining the ruled batch-size and worker-count hardware overrides.
- Important: full mode must compute and persist pooled OOF macro-F1, weighted F1,
  Top-1, and Top-5 per preprocessing candidate.
- Important: notebook must visibly show the frozen pre-run schedule/configuration
  and post-run registry fields including run ID, status, stage, fold, transform,
  and final eligibility.
- Minor fix-wave targets: full-trainer repeatability test, multi-epoch best-state
  selection/tie behavior test, unique fold-label aggregation guard, and direct
  confusion-figure test.
- Deferred after final triage: fixed `fc1` expression is safe while approved
  adaptive size remains `(4,3)`.

## Final fix wave

- Important review fixes implemented directly after the delegated fix agent hit
  its usage limit: final eligibility now seals optimization settings and approved
  preprocessing; full mode computes and persists pooled OOF metrics; the notebook
  displays the schedule, comparison, OOF metrics, and registry rows.
- Cheap test gaps added for sealed settings, unique folds, pooled OOF evidence,
  and confusion-figure output. Repeatability and multi-epoch checkpoint tests
  remain recommended before the expensive ten-run experiment.
- Focused tests: 37 passed. Full suite: 132 passed, 1 pre-existing cache/input
  fingerprint failure, 2 warnings.
- Scoped re-review was attempted with two completed agents, but both were blocked
  by the host usage limit. Manual line-by-line review of `b00d34c..57f55cb`
  found the three important findings addressed and no new regression.
