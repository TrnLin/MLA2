# Notebooks

Notebooks tell the investigation story. Reusable code lives in `src/fashion/`.

## Reading order

| # | Notebook | Status | Purpose |
|---|---|---|---|
| 00 | `00_problem_definition.ipynb` | complete | users, task boundaries, risks, and success dimensions |
| 01 | `01_data_preparation.ipynb` | complete and executed | teacher audit, sole split, five folds, and development-only evidence |
| 02 | `02_task1_article_type.ipynb` | planning scaffold | article-type comparisons and judgement |
| 03 | `03_task2_season.ipynb` | planning scaffold | season comparisons and judgement |
| 04 | `04_task3_gender_usage.ipynb` | baseline evidence notebook | model, five-fold analysis, and next hypotheses |
| 04a | `04a_task3_smallcnn_baseline_training.ipynb` | executed baseline runner | reproducible five-fold SmallCNN baselines |
| 04b | `04b_task3_smallcnn_child_experiments.ipynb` | child training runner | brightness-only Gender and loss-only Usage experiments |
| 04c | `04c_task3_smallcnn_e3_experiments.ipynb` | E3 training runner | loss-only Gender and dropout-only Usage experiments |
| 04d | `04d_task3_tinyresnet18_pm_e4_experiments.ipynb` | E4 training runner | parameter-matched TinyResNet architecture experiments |
| 04e | `04e_task3_compactblurcnn_label_smoothing_e5_experiments.ipynb` | E5 training runner | compact Gender architecture and Usage label-smoothing experiments |
| 04f | `04f_task3_gem_focal_e6_experiments.ipynb` | E6 training runner | Gender GeM pooling and Usage focal-loss experiments |
| 04g | `04g_task3_tinyconvnext_tinyhrnet_e7_experiments.ipynb` | E7 training runner | Usage TinyConvNeXt and Gender TinyHRNet architecture experiments |
| 04h | `04h_task3_early_stopping_translation_e8_experiments.ipynb` | E8 training runner | Gender checkpoint selection and Usage translation experiments |
| 04i | `04i_task3_semantic_filter_exception_balance_e9_experiments.ipynb` | E9 training runner | Gender semantic filtering and Usage exception-balance experiments |
| 04j | `04j_task3_audience_aux_e10_experiment.ipynb` | E10 training runner | Gender E6 plus a training-only three-way catalogue-audience head |
| 04k | `04k_task3_clean_slate_eda.ipynb` | clean-slate EDA | teacher-only foreground, nuisance, family, representation, and fold audits |
| 04l | `04l_task3_clean_slate_screen_1.ipynb` | local CPU training runner | two-fold clean-slate screen with separate Gender HOG-SVM and Usage type-posterior models |
| 05 | `05_task4_visual_search.ipynb` | planning scaffold | Top-K search choices and comparisons |
| 06 | `06_final_evaluation.ipynb` | locked scaffold | one holdout evaluation and ultimate judgement |

Notebooks 02, 03, 05, and 06 are planning or locked Markdown scaffolds. Task 3 has one
narrative notebook, separate E1–E10 training runners, one clean-slate EDA notebook, and separate
clean-slate training runners so Run All cannot accidentally retrain an earlier stage. Each
`TODO(owner)` belongs to the task owner.

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
