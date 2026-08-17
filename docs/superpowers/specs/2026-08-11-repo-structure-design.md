# Repo Structure — Design Note

**Date:** 2026-08-11
**Scope:** Directory layout and shared contracts for COSC2753 Assignment 2 (Fashion Intelligence).
**Status:** Layout, staged-parallel execution strategy, and notebook workflow agreed.
Evidence-dependent data policies and the app scope remain open.

---

## Facts that drove these decisions

Established by inspecting the dataset and the spec, not assumed:

| Fact | Source | Consequence |
|---|---|---|
| `styles_prediction.csv` header is `id,gender,articleType,season,usage` | `data/raw/teacher/test/styles_prediction.csv` | Task 3 is **two separate targets**, not a combined class. Four target columns total. |
| Images are 60×80 px | `data/raw/teacher/train/images_train/1163.jpg` | Training is cheap; upscaling beyond ~64–96 px adds no information. |
| 38,617 metadata rows · 38,612 training images · 5,829 test images. 5 metadata rows have no training image (ids `12347`, `39401`, `39403`, `39410`, `39425`); no image lacks a metadata row | `00_eda.ipynb` §2.2 `reconciliation_summary` and `unmatched_ids`; `data/raw/teacher/test/images_test/` directory count | Row count and image count are not interchangeable. Anything image-dependent uses the 38,612 matched rows, so `articleType` has 125 labelled classes but only 124 trainable ones — id `12347` is the dataset's only `Suits` row. |
| `articleType` has 125 classes, from 6,781 down to single-example classes | `styles_train.csv` | Macro-F1 over accuracy; rare-class policy must be decided and defended. |
| `season`: Summer 19,137 · Fall 10,512 · Winter 7,381 · Spring 1,567 · 20 blank | `styles_train.csv` | 4 usable classes, imbalanced, plus blanks to handle. |
| `gender`: Men 20,918 · Women 14,160 · Unisex 2,080 · Boys 814 · Girls 645 | `styles_train.csv` | 5 classes, moderate imbalance. |
| `usage`: Casual 29,641 · Sports 3,940 · Ethnic 2,570 · Formal 2,300 · Smart Casual 55 · Travel 25 · Party 13 · Home 1 · 71 `"NA"` · 1 blank | `styles_train.csv` | Severe tail; `"NA"` string is distinct from blank. |
| Train CSV header ends in two empty columns (`,,`) | `styles_train.csv` | Parser must tolerate trailing empty fields. |
| Compute is not a constraint (Mac + NVIDIA GPU available) | user | Real CNN training and hyperparameter sweeps are affordable. |
| Not currently a git repo; `.venv` is Python 3.13 with no packages | `git status`, `pip list` | Both need initialising before work starts. |

---

## Directory layout

Entries marked GENERATED are produced by code, gitignored, and rebuilt rather than committed.

```
MLA2/
├── README.md                      # env setup, data placement, how to run  (submission requirement)
├── pyproject.toml                 # dependencies + `pip install -e .`
├── .gitignore
├── data/
│   ├── train/
│   │   ├── images_train/          # 38,612 jpgs, filename = id
│   │   └── styles_train.csv
│   ├── test/
│   │   ├── images_test/           # 5,829 jpgs, unlabelled
│   │   └── styles_prediction.csv  # submission format — do not modify
│   └── processed/                 # GENERATED
│       ├── clean.csv              #   labels after NA / missing-image / duplicate handling
│       ├── splits.csv             #   id → train | val | test   ← single source of truth
│       └── embeddings/            #   cached vectors (visual search + classical baselines)
├── src/fashion/
│   ├── config.py                  # paths, seed, image size, target definitions
│   ├── data/
│   │   ├── clean.py               # "NA" vs blank, missing files, duplicates, rare classes
│   │   ├── splits.py              # stratified split, written once
│   │   └── datasets.py            # Dataset + transforms + augmentation
│   ├── features/
│   │   └── classical.py           # colour histogram / HOG / raw pixels for baselines
│   ├── models/
│   │   ├── baselines.py           # majority class, sklearn wrappers
│   │   ├── cnn.py                 # from-scratch CNN (the submittable model)
│   │   └── heads.py               # multi-task heads
│   ├── train/
│   │   ├── loop.py                # the single training function every experiment calls
│   │   └── registry.py            # appends a row to results/runs.csv
│   ├── eval/
│   │   ├── metrics.py             # macro-F1, per-class, confusion matrix
│   │   ├── retrieval.py           # Precision@K, mAP for Task 4
│   │   └── explain.py             # Grad-CAM
│   └── search/
│       ├── embed.py               # image → vector
│       └── index.py               # index + top-K query
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_preprocessing.ipynb     # writes clean.csv + splits.csv
│   ├── 10_task1_articletype.ipynb
│   ├── 20_task2_season.ipynb
│   ├── 30_task3_gender_usage.ipynb
│   ├── 40_task4_visual_search.ipynb
│   └── 90_final_predictions.ipynb # loads saved models → submission CSV
├── models/                        # saved weights  (submission requirement)
├── results/
│   ├── runs.csv                   # one row per experiment  ← report tables come from here
│   └── figures/                   # only plots the report actually cites
├── app/
│   ├── api.py                     # HD requirement: API serving the models
│   └── web/                       # small web UI calling the API
├── scripts/
│   └── predict.py                 # CLI so the marker never has to open a notebook
├── docs/                          # spec PDF, breakdown, design notes
├── rubrics/                       # transcribed marking rubric
└── report/                        # 5-page report source
```

---

## The four load-bearing contracts

Everything else in the layout is negotiable. These four are not.

### 1. `data/processed/splits.csv` is the only split

Written exactly once by `01_preprocessing.ipynb`; read by every notebook, the search index,
and the app. **No other file may call `train_test_split`.**

- Makes cross-model and cross-task comparisons controlled rather than coincidental.
- Prevents evaluation images leaking into the visual-search index — the most likely silent
  failure in Task 4, since the index is built over the same ~38k images.
- Held-out test portion stays untouched during tuning, which is what the rubric means by
  *independent* evaluation.

### 2. `results/runs.csv` is appended by code, never by hand

`train/loop.py` calls `train/registry.py` on every run. One row per experiment:
timestamp, task, model name, hyperparameters, metrics, training time, parameter count.

The Approach criterion is 50 marks and rewards comparison breadth, which means 30+ runs.
Report tables must be a `groupby` over this file, not archaeology through notebook output cells.

### 3. `train/loop.py` is the single training function

Every experiment trains through the same loop with the same early-stopping and logging
behaviour. Otherwise "CNN vs tuned CNN" compares two training regimes, not two models.

### 4. `eval/metrics.py` is the single metrics implementation

All four targets report identical measures. Prevents one task quietly reporting accuracy
while another reports macro-F1.

---

## Notebook-only interface

A teammate who does not want to read `src/` runs `pip install -e .` once and then works
entirely through this surface:

```python
from fashion.data.datasets import make_loaders
from fashion.models.cnn import SmallCNN
from fashion.train.loop import train
from fashion.eval.metrics import report

train_dl, val_dl, classes = make_loaders(target="articleType", img_size=64)
model = SmallCNN(n_classes=len(classes))
history = train(model, train_dl, val_dl, epochs=20, run_name="task1_cnn_baseline")
report(model, val_dl, classes)          # metrics + confusion matrix, auto-saved to results/
```

They inherit the shared split, the shared metrics, and automatic run logging without
knowing those systems exist.

**Accepted cost:** when something breaks inside `train()`, a notebook-only person cannot
debug it alone. Mitigation is keeping these modules small and boring rather than clever.

---

## Execution strategy: sequential foundations, parallel tasks, sequential integration

The project should not run strictly task-by-task, because Tasks 1–4 do not require each
other's final results. It also should not parallelise immediately, because experiments are
not comparable until the shared data and evaluation contracts are fixed.

```text
EDA → cleaning → fixed split → shared training and evaluation code
                              ↓
       ┌──────────────┬──────────────┬──────────────┬─────────────────┐
       Task 1         Task 2         Task 3         Task 4 baseline
       articleType    season         gender/usage   visual retrieval
       └──────────────┴──────────────┴──────────────┴─────────────────┘
                              ↓
                  compare findings and refine
                              ↓
       final models → held-out evaluation → app → predictions → report
```

### Stage 1 — shared foundation, sequential

Complete these before model work branches:

1. `00_eda.ipynb` establishes label distributions, missing-data behaviour, image
   integrity, and evidence for the three open modelling questions.
2. `01_preprocessing.ipynb` writes `clean.csv` and the only `splits.csv`.
3. The shared dataset interface, training loop, metrics module, and run registry are
   established.
4. A common baseline protocol fixes metrics, stopping behaviour, seeds, and what metadata
   every run records.

This stage is a hard dependency. Parallel task results produced before it is complete
would not support controlled comparisons.

### Stage 2 — first experiment round, parallel

Once the foundation is frozen, the task notebooks can progress independently:

- Task 1 investigates `articleType`.
- Task 2 investigates `season`.
- Task 3 investigates `gender` and `usage`.
- Task 4 starts with classical retrieval baselines such as colour histograms and HOG.

Each track reads the same split and uses the same training, evaluation, and registry
interfaces. A task track may not create its own split or silently change a shared metric.

### Stage 3 — evidence-sharing checkpoint

After the first comparable runs, reconvene before tuning broadly. Compare findings about:

- useful image size and augmentation;
- CNN capacity, optimiser, and learning rate;
- imbalance handling;
- common failure cases and label noise.

These are **soft dependencies**, not blockers. A technique that helps Task 1 is a candidate
for Task 2, not a conclusion: each target must validate it independently on the fixed
validation split.

### Stage 4 — targeted improvements, parallel

Each track applies and measures the improvements justified by its own first-round evidence.
Task 4 may now compare its classical baselines with learned embeddings from a promising
from-scratch classifier backbone. This learned-embedding experiment depends on a suitable
backbone; the rest of Task 4 does not.

### Stage 5 — integration and judgement, sequential

The tracks reconverge to:

1. freeze one ultimate-judgement model per required output;
2. run the untouched held-out evaluation once;
3. compare model size, latency, calibration, robustness, and failure modes;
4. integrate the chosen models into the app and prediction script;
5. generate `styles_prediction.csv`;
6. synthesize the cross-task evidence into the report.

The held-out portion must not influence model or hyperparameter selection. Its purpose is
to evaluate a judgement already made from training and validation evidence.

---

## Notebook workflow

### Shared notebook contract

Every notebook must:

- run top-to-bottom from a clean kernel;
- keep reusable logic in `src/fashion/` rather than notebook cells;
- read the single `data/processed/splits.csv` once that artifact exists;
- obtain experiment tables from `results/runs.csv`;
- state a hypothesis for each comparison and change one controlled factor at a time;
- report macro-F1 as the primary classification metric;
- analyse failures and record a decision, not merely display a score.

Task notebooks use only the training and validation partitions while selecting models.
They freeze their chosen configurations without reading held-out labels. Final held-out
evaluation is centralised in `90_final_predictions.ipynb`.

### `00_eda.ipynb` — establish evidence before decisions

**Purpose:** describe the data as it exists and provide evidence for cleaning, splitting,
and modelling choices. It does not write processed data.

The notebook must:

1. audit the CSV schema, trailing empty columns, duplicate IDs, missing labels,
   image/CSV mismatches, corrupt files, image dimensions and colour modes;
2. detect exact duplicates and identify candidate near-duplicate groups that must not
   cross data partitions;
3. measure all four target distributions, imbalance ratios, long tails, label
   co-occurrence, and hierarchy consistency across `masterCategory`, `subCategory`, and
   `articleType`;
4. show stratified image grids for common, rare, ambiguous, and suspicious labels,
   distinguishing likely label noise from model difficulty;
5. establish split requirements without creating a split: duplicate groups stay together,
   every target receives a balance check, and classes too small for fair evaluation are
   identified;
6. gather evidence for the three open decisions: rare-`articleType` handling, whether
   `gender` and `usage` are related enough to benefit from shared learning, and candidate
   relevance definitions for visual search;
7. produce one report-worthy composite dataset figure and a written decision ledger.

**Completion gate:** every cleaning and split decision required by
`01_preprocessing.ipynb` has an evidence reference or is explicitly marked unresolved.

### `01_preprocessing.ipynb` — create the immutable data contract

**Purpose:** convert the evidence-backed EDA decisions into deterministic, validated
artifacts that every later notebook shares.

The notebook must:

1. parse the raw CSV while preserving literal `"NA"`, remove phantom columns, reconcile
   rows with image files, and apply the approved missing-label and rare-class policies
   without discarding labels needed by other tasks;
2. assign exact-duplicate and accepted near-duplicate groups atomically so related images
   cannot cross partitions;
3. write `data/processed/clean.csv` and the only
   `data/processed/splits.csv`, using a fixed seed and documented
   train/validation/held-out ratios;
4. assert split disjointness, complete usable-ID coverage, duplicate-group isolation,
   per-target balance, class coverage, deterministic reproduction, and an unchanged
   official test set;
5. print a compact data card with counts before and after every operation and target
   distributions for each partition.

This is the only notebook permitted to assign rows to partitions. It delegates cleaning
and split logic to `src/fashion/data/clean.py` and `src/fashion/data/splits.py`, where
load-bearing assertions live.

**Outputs:** `data/processed/clean.csv` and `data/processed/splits.csv`.

**Completion gate:** every assertion passes, both artifacts reproduce byte-for-byte from
the same inputs and seed, and no later notebook needs to clean or split data again.

### `10_task1_articletype.ipynb` — investigate the 125-class long tail

**Question:** which from-scratch model gives the most defensible item-type predictions
under severe class imbalance?

The controlled experiment ladder is:

1. majority-class sanity baseline;
2. classical HOG and colour features with a linear classifier;
3. a small from-scratch CNN;
4. a stronger from-scratch CNN with comparable training controls;
5. one evidence-backed imbalance intervention, such as class-balanced or focal loss.

Only justified factors are tuned: image size, augmentation, model capacity, optimiser, and
loss. Every run is registered automatically.

Analysis must include macro-F1, top-k accuracy, per-class recall against class support,
the largest confusion pairs, calibration, parameter count, and inference latency. It must
compare rare-class policies on the same validation examples and state what any reported
metric excludes. Representative error grids and Grad-CAM test whether the model attends to
the product rather than the background. An optional pretrained benchmark is clearly marked
as comparison-only and ineligible for submission.

**Outputs:** registered experiment rows, report-candidate figures, a documented ultimate
judgement, and the frozen from-scratch candidate checkpoint.

**Completion gate:** the chosen configuration is justified using validation evidence,
saved with its class mapping and preprocessing configuration, and no held-out labels have
been read.

### `20_task2_season.ipynb` — measure weak visual signal honestly

**Question:** how much season information is present in the product image, and where does
the target become a merchandising label rather than a visual one?

The controlled experiment ladder is:

1. majority-class sanity baseline;
2. classical colour and texture features with a linear classifier;
3. a small from-scratch CNN;
4. a stronger from-scratch CNN;
5. one targeted improvement justified by the first-round evidence, such as imbalance
   handling or auxiliary learning.

The notebook tests whether augmentation and product category change performance rather
than assuming they transfer from Task 1.

Analysis must include macro-F1, per-season recall, the Summer/Fall/Winter/Spring confusion
patterns, calibration, category-conditional performance, model size, and latency. Error
grids and Grad-CAM distinguish genuine model failures from examples whose season label is
not visually recoverable. The conclusion states an evidence-backed practical ceiling
instead of treating every remaining error as an architecture problem.

**Outputs:** registered experiment rows, report-candidate figures, a documented ultimate
judgement, and the frozen from-scratch candidate checkpoint.

**Completion gate:** the selected configuration is supported by validation and failure
evidence, saved with its class mapping and preprocessing configuration, and no held-out
labels have been read.

### `30_task3_gender_usage.ipynb` — compare separate and shared learning

**Question:** do `gender` and `usage` benefit enough from shared visual features to justify
a two-head model, or do separate models avoid negative transfer?

Both remain separate prediction targets throughout, matching the required submission CSV.
The controlled experiment ladder is:

1. majority baselines for each target;
2. classical-feature baselines for each target;
3. two independent from-scratch CNNs;
4. one from-scratch backbone with separate `gender` and `usage` heads;
5. an imbalance intervention focused on the severe `usage` tail.

The shared and separate variants must use equivalent transforms, training budgets, split
rows, and approximately comparable capacity. Otherwise the experiment would not isolate
the value of multi-task learning.

Analysis reports macro-F1 and per-class recall for each head separately. Joint exact-match
is secondary only. It examines label correlation, negative transfer, loss weighting,
calibration, minority-class failures, parameter count, and latency. The deployment decision
uses those practical measures as well as predictive performance; novelty alone is not a
reason to choose the shared model.

**Outputs:** registered experiment rows, separate-versus-shared comparison figures, a
documented deployment decision, and either one frozen multi-head checkpoint or two frozen
single-target checkpoints.

**Completion gate:** the selected deployment form is supported by a controlled validation
comparison, carries both target mappings and preprocessing configuration, and no held-out
labels have been read.

### `40_task4_visual_search.ipynb` — define similarity before optimising it

**Question:** which representation retrieves visually useful neighbours, under a relevance
definition that can be reproduced and defended?

Before comparing methods, the notebook defines two evaluation views:

1. a reproducible metadata-based relevance proxy;
2. a small human-rated query set.

Their agreement and limitations are reported separately. The retrieval index is restricted
to the training pool; validation images are development queries. The query itself and
accepted duplicate-group matches are excluded where appropriate.

The controlled experiment ladder is:

1. raw-pixel retrieval;
2. colour-histogram retrieval;
3. HOG retrieval;
4. embeddings from a promising from-scratch classifier backbone;
5. a metric-learning improvement if the baseline evidence justifies it.

Analysis reports Precision@K, mAP or nDCG under the predeclared relevance definition,
query latency, index size, and qualitative success/failure grids. It tests whether results
preserve item type, colour, gender, or merely background and composition.

**Outputs:** registered retrieval experiments, versioned cached embeddings under
`data/processed/embeddings/`, report-candidate retrieval grids, and the frozen selected
index and embedding model with split and checkpoint provenance.

**Completion gate:** relevance was defined before model comparison, the selected
configuration is frozen from validation evidence, and no held-out query results have been
examined.

### `90_final_predictions.ipynb` — one controlled final gate

**Purpose:** evaluate already-frozen ultimate judgements exactly once, then produce the
official prediction artifact without any further model selection.

The notebook must:

1. load classification and retrieval artifacts by explicit run and checkpoint identifiers;
2. assert that every artifact includes its target mapping, preprocessing configuration,
   split provenance, and eligibility for submission;
3. evaluate every ultimate judgement once on the untouched held-out partition, including
   macro-F1, calibration, latency, robustness, and task-specific failure summaries;
4. evaluate visual search on held-out queries under the relevance rules fixed in Task 4;
5. compare held-out and validation estimates and explain meaningful gaps without returning
   to tuning;
6. run deterministic inference over all 5,829 official test images using the exact training
   preprocessing and mappings;
7. write a new submission artifact without modifying
   `data/raw/teacher/test/styles_prediction.csv`, preserving ID order and the exact header
   `id,gender,articleType,season,usage`;
8. assert the row count, unique IDs, allowed label vocabularies, absence of blank
   predictions, absence of extra columns, and parity with `scripts/predict.py`;
9. write an artifact manifest linking each output column and the visual-search app to its
   checkpoint, run ID, preprocessing configuration, and final evaluation.

No model is selected, tuned, or retrained after held-out results are visible.

**Outputs:** final held-out evaluation tables and figures, the validated submission CSV,
and the final artifact manifest.

**Completion gate:** all validation assertions pass, every prediction is traceable to a
frozen eligible model, and rerunning the notebook reproduces the same submission file.

---

## Deliberate exclusions

| Excluded | Reason |
|---|---|
| `tests/` directory | Notebooks are the verification for a one-off analysis; unit tests score nothing on this rubric. **Exception:** `clean.py` and `splits.py` warrant a few inline assertions, since a silent bug there poisons every downstream result. |
| Config framework (Hydra/OmegaConf) | `config.py` constants plus function arguments are sufficient. |
| Experiment tracking service (MLflow/W&B) | `runs.csv` covers the need; a service risks the "runs on any standard machine" requirement. |
| DVC / data versioning | Dataset is fixed and supplied. |

---

## Open questions

- Rare-class policy for `articleType` (drop, merge via `subCategory`, or keep and report
  the failure) — needs an evidence-backed decision in `01_preprocessing.ipynb`.
- Whether Task 3 uses two independent models or one shared backbone with two heads.
  The submission format permits either; multi-task is the stronger "beyond class material" claim.
- Definition of relevance for Task 4 retrieval evaluation, since no ground-truth
  similarity labels exist.

---

## Remaining work

- Exact cleaning, rare-class, and retrieval-relevance policies remain evidence-dependent
  decisions for `00_eda.ipynb` and `01_preprocessing.ipynb`; they should not be guessed in
  advance.
- The app's user flow, API boundary, and minimum demonstration scope are not yet designed.
- After the app scope is approved, translate the complete design into a separate code
  implementation plan covering repository setup, shared modules, notebook construction,
  experiment execution, and integration.
