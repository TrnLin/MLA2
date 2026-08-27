# Notebooks

Notebooks tell the investigation story. Reusable code lives in `src/fashion/`.

## Reading order

| # | Notebook | Status | Purpose |
|---|---|---|---|
| 00 | `00_problem_definition.ipynb` | complete | users, task boundaries, risks, and success dimensions |
| 01 | `01_data_preparation.ipynb` | complete and executed | teacher audit, sole split, five folds, and development-only evidence |
| 02 | `02_task1_article_type.ipynb` | planning scaffold | article-type comparisons and judgement |
| 03 | `03_task2_season.ipynb` | planning scaffold | season comparisons and judgement |
| 04 | `04_task3_gender_usage.ipynb` | planning scaffold | separate gender and usage outputs |
| 05a | `task-4/01_v1_eda.ipynb` | complete and executed | V1 provenance, geometry, and paired-image audit |
| 05b | `task-4/05_task4_visual_search.ipynb` | baseline complete; model work open | Top-K search choices and comparisons |
| 06 | `06_final_evaluation.ipynb` | locked scaffold | one holdout evaluation and ultimate judgement |

The Task 4 EDA contains audit code only. The main Task 4 notebook has frozen its
protocol, preprocessing, and untrained baseline from development evidence.
Learned-model and final-winner choices remain open. Other modelling notebook
`TODO(owner)` items still belong to their task owners.

## Shared rules

- Load only `data/processed/splits.csv` through the shared APIs.
- Choose one fixed `cv_fold` or all five folds before experiments.
- Fit learned preprocessing only on the training folds of each round.
- Keep holdout and quarantine targets sealed until Notebook 06.
- Train submitted models from scratch.
- Write every run to `results/runs.csv` through `fashion.train.registry`.
- Tasks 1–3 use `data/raw/teacher` images.
- Task 4 owns query size, image size, optional external images, query/gallery rules,
  relevance, K, the index, and ranking evaluation.

## Required fold block for modelling notebooks

Every teammate working on Notebooks 02–05 **must use this block** to obtain training
and validation folds. It is the same data-access method explained in Notebook 01,
Section 17.

```python
from fashion.data.dataset import get_cv_split, iter_cv_folds, load_splits

splits = load_splits()

# Option 1: use one fold selected before experiments.
SELECTED_FOLD = 0
training, validation = get_cv_split(splits, validation_fold=SELECTED_FOLD)

# Option 2: evaluate the same experiment on all five saved folds.
for fold, training, validation in iter_cv_folds(splits):
    ...
```

Use one option, not both, for a training run. Do not call `train_test_split`, `KFold`,
`StratifiedKFold`, `GroupKFold`, or write another random fold generator. New folds would
break fair model comparison and could place related product images on both sides of a
validation boundary.

## Notebook 01

Notebook 01 is the official shared preparation workflow. Cached mode is the default.
Use full mode only after teacher inputs change:

```bash
FASHION_DATA_PREPARATION_MODE=full ./.venv/bin/python -m jupyter lab
```

It hashes raw bytes before decode, reconciles exact ID sets, controls duplicate and
family leakage, validates five folds, describes development labels and images, and
writes report evidence. Every code result is followed by a short finding.

### How `src/fashion` helps Notebook 01

Notebook 01 shows the audit, graphs, and explanations. The helper files keep the data
rules in one place, so later notebooks cannot quietly use different rules.

```text
+----------------------+       +----------------------+
| Teacher CSV + images |       | config.py            |
| raw inputs           |       | paths/targets/seed   |
+----------+-----------+       +----------+-----------+
           \                              /
            +-------------+--------------+
                          v
               +----------------------+
               | pipeline.py          |
               | full build/cache check|
               +----------+-----------+
                          |
                          v
               +----------------------+
               | audit.py + hashing.py|
               | checks + raw hashes  |
               +----------+-----------+
                          |
                          v
             +--------------------------+
             | metadata.py + manifests.py|
             | repair names + join images|
             +------------+-------------+
                          |
                          v
               +----------------------+
               | perceptual.py        |
               | near-duplicate pairs |
               +----------+-----------+
                          |
                          v
               +----------------------+
               | families.py          |
               | safe product groups  |
               +----------+-----------+
                          |
                          v
               +----------------------+
               | splits.py            |
               | one split + CV folds |
               +----------+-----------+
                          |
               +----------+----------+
               |                     |
               v                     v
      +-------------------+  +-------------------+
      | dataset.py        |  | evidence.py       |
      | safe fold views   |  | analysis tables   |
      +---------+---------+  +---------+---------+
                \                    /
                 +---------+--------+
                           v
                +--------------------+
                | Notebook 01        |
                | graphs + findings  |
                +---------+----------+
                          |
                          v
                 Notebooks 02--06
              reuse the same contracts
```

| Helper file | How it helps Notebook 01 |
|---|---|
| [`config.py`](../src/fashion/config.py) | Gives the notebook the same paths, four target names, random seed, and five-fold setting as the rest of the project. |
| [`audit.py`](../src/fashion/data/audit.py) and [`hashing.py`](../src/fashion/data/hashing.py) | Check raw CSV and image structure, inspect image pixels, and hash raw file bytes before image decoding. |
| [`metadata.py`](../src/fashion/data/metadata.py) and [`manifests.py`](../src/fashion/data/manifests.py) | Repair names split across CSV columns, treat product-name `NA` as missing, keep the valid `usage=NA` teacher label, and join metadata to real image files. |
| [`perceptual.py`](../src/fashion/data/perceptual.py) | Finds possible near-duplicate images with one fixed, label-free rule. The notebook then shows pairs near that rule for human review. |
| [`families.py`](../src/fashion/data/families.py) | Groups equal names, equal file hashes, and accepted near-duplicates into conservative split groups. These blocks protect the split; they are not verified independent products. It also finds duplicate images with conflicting labels for quarantine. |
| [`splits.py`](../src/fashion/data/splits.py) | Builds and checks the only development, holdout, quarantine, and five-fold assignments. It proves that one family does not cross boundaries. |
| [`pipeline.py`](../src/fashion/data/pipeline.py) | Runs the full preparation steps in order. In normal cached mode, it checks that the saved data still matches the teacher inputs and expected artifacts. |
| [`dataset.py`](../src/fashion/data/dataset.py) | Loads the official split safely, hides protected labels, and gives Notebook 01 the same CV fold views that training notebooks will use. |
| [`evidence.py`](../src/fashion/data/evidence.py) | Calculates development-only tables for family size, fold support, near-threshold pairs, and shortcut risk. The graph code stays visible in Notebook 01. |

In full mode, `pipeline.py` calls the preparation helpers and rebuilds the shared data.
In cached mode, it validates that data. Notebook 01 then uses `dataset.py` and
`evidence.py` to explain the result. Later notebooks reuse the same contracts instead
of making another split or another meaning for a label.

Notebook 01 does not open holdout targets. It does not read external images. It does
not select a transform, model, metric, or retrieval protocol.
