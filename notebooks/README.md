# Notebooks

Notebooks tell the investigation story. Reusable code lives in `src/fashion/`.

File names use lowercase words separated by underscores. Shared notebooks use
`NN_description.ipynb`; task notebooks use `NN_taskN_partN_description.ipynb`.
The prefix `NN` runs once from `00` to `11` in reading order, with no gaps or repeats:
shared setup, Task 1, Task 2, Task 3, then Task 4.
`part1`, `part2`, and so on give the reading order within each task, so sorting
by name puts development before final evaluation. Tasks 1–3 share the EDA in
`01_data_preparation.ipynb`;
their first task notebook covers model development and comparisons. Task 4 has
its own image EDA, followed by model development, final evaluation, and the demo.
Every task's final evaluation uses the suffix `final_evaluation.ipynb`.
The separate `task3_training/` experiment names are exempt.

## Reading order

| Order | Notebook | Status | Purpose |
|---|---|---|---|
| 00 | `00_problem_definition.ipynb` | complete | users, task boundaries, risks, and success dimensions |
| 01 | `01_data_preparation.ipynb` | complete and executed | teacher audit, sole split, five folds, and development-only evidence |
| 02 | `02_task1_part1_article_type.ipynb` | model comparisons | article-type comparisons and judgement |
| 03 | `03_task1_part2_final_evaluation.ipynb` | staged execution and replay | final article-type training, evaluation, and judgement |
| 04 | `04_task2_part1_season.ipynb` | complete and executed | season comparisons and judgement replayed from frozen evidence |
| 05 | `05_task2_part2_final_evaluation.ipynb` | complete evaluation replay | Kai's frozen Season bundle, one-shot holdout evidence, teacher-test output, and Assessment 3 evidence |
| 06 | `06_task3_part1_gender_usage.ipynb` | saved-results report | model comparisons, failures, and final choices |
| 07 | `07_task3_part2_final_evaluation.ipynb` | saved Task 3 evaluation | Gender and Usage holdout results, errors and ultimate judgement |
| 08 | `task-4/08_task4_part1_image_eda.ipynb` | complete and executed | V1 provenance, geometry, and paired-image audit |
| 09 | `task-4/09_task4_part2_visual_search.ipynb` | frozen development comparison | Top-K search choices and comparisons |
| 10 | `task-4/10_task4_part3_final_evaluation.ipynb` | complete and executed | replay-only independent holdout evaluation, uncertainty, failures, and ultimate judgement |
| 11 | `task-4/11_task4_part4_search_demo.ipynb` | search demo | development or outside-image visual search |

Each task has its own final-evaluation notebook; there is no shared template.
The Task 2 owner notebook replays the one completed, hash-verified holdout evaluation.
Run All cannot independently unlock raw labels, retrain the model, or create a second
holdout score.
Task 3 has one report and [40 retained training notebooks](task3_training/README.md). Each
`TODO(owner)` belongs to the task owner.

Task 2, part 1 is the completed Season report notebook. It contains one code cell per leaf
subsection and freezes the Season metric, experiment order, leakage controls, and final
decision rule. **Run All does not train a model.** It uses `artifact_replay` to verify and
display the saved manifests, tables, figures, and final bundle. A missing or changed
artifact stops the notebook instead of starting a new run. Git-tracked SHA-256 locks cover
the replay roots and loose figures. Fold-0 preprocessing uses the already-frozen statistics,
so it cannot fit or write a cache. Reusable implementations still belong in `src/fashion/`;
the notebook only checks evidence and explains each output.

The Task 4 EDA contains audit code only. The main Task 4 notebook records its
frozen development choices. The evaluation notebook replays the saved holdout results.

## Shared rules

- Load only `data/processed/splits.csv` through the shared APIs.
- Choose one fixed `cv_fold` or all five folds before experiments.
- Fit learned preprocessing only on the training folds of each round.
- Keep holdout targets sealed until each task's authorised final evaluation; keep quarantine targets sealed.
- Train submitted models from scratch.
- Write every run to `results/runs.csv` through `fashion.train.registry`.
- Tasks 1–3 use `data/raw/teacher` images.
- Task 4 owns query size, image size, optional external images, query/gallery rules,
  relevance, K, the index, and ranking evaluation.

## Required fold block for modelling notebooks

Every teammate working on the model-development notebooks for Tasks 1–4 **must use this block** to obtain training
and validation folds. It is the same data-access method explained in Notebook 01,
Section 5.2.

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
                 Notebooks 02--11
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
