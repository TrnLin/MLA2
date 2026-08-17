# Fashion Intelligence — COSC2753 Assignment 2

This project studies a fashion-product dataset for:

- `articleType` classification;
- `season` classification;
- `gender` classification;
- `usage` classification; and
- image-based product retrieval.

The repository currently contains the dataset audits and comparison report. Model
preprocessing, training, and evaluation will be added in later notebooks.

## Project decisions

Important choices are stored as one Markdown record per decision in
[`docs/decisions/`](docs/decisions/README.md). Start there before changing data,
preprocessing, or split behaviour.

See [the dataset quality report](docs/dataset-quality-comparison.md) for the supporting
evidence.

## Repository structure

```text
MLA2/
├── data/                              # Ignored by Git
│   ├── raw/                           # Immutable source datasets
│   │   ├── original/                  # Full-resolution catalogue
│   │   │   ├── styles.csv             # 44,446 labelled metadata rows
│   │   │   ├── images.csv             # Image filename-to-URL index
│   │   │   ├── images/                # 44,441 original JPG files
│   │   │   └── styles/                # Per-product JSON records
│   │   └── teacher/
│   │       ├── train/
│   │       │   ├── styles_train.csv   # 38,617 labelled rows
│   │       │   └── images_train/      # 38,612 low-resolution JPG files
│   │       └── test/
│   │           ├── styles_prediction.csv # 5,829 blank prediction rows
│   │           └── images_test/       # 5,829 low-resolution JPG files
│   └── processed/                     # Rebuildable manifests and one shared split
├── notebooks/
│   ├── 00_eda.ipynb                   # Teacher-training dataset audit
│   ├── 00_eda_full_dataset.ipynb      # Original full dataset audit
│   └── 00_dataset_comparison.ipynb    # Original versus teacher comparison
├── src/fashion/
│   ├── config.py                      # Repository paths, targets, and seed
│   ├── eda.py                         # Reusable EDA helpers
│   └── dataset_comparison.py          # Deep comparison and report data
├── scripts/
│   ├── build_dataset_comparison_report.py
│   └── verify_dataset_comparison.py
├── docs/
│   ├── dataset-quality-comparison.md  # Main comparison report
│   ├── decisions/                     # One Markdown record per accepted decision
│   └── assignment-breakdown.html      # Assignment summary
├── results/
│   └── figures/
│       └── dataset-comparison/         # Summary JSON and report figures
├── pyproject.toml
└── README.md
```

## Setup

Python 3.11 or newer is required.

```bash
cd /localhome/local-lintran/MLA2
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
```

If `.venv` already exists, only run the final install command.

Do not use the machine's global Python environment. Its pandas and NumPy packages may be
incompatible.

## Open the notebooks

Start Jupyter:

```bash
cd /localhome/local-lintran/MLA2
./.venv/bin/jupyter lab
```

Suggested reading order:

1. `notebooks/00_eda.ipynb`
2. `notebooks/00_eda_full_dataset.ipynb`
3. `notebooks/00_dataset_comparison.ipynb`

## Run the dataset comparison

The normal command uses the latest machine-readable comparison summary:

```bash
cd /localhome/local-lintran/MLA2
PYTHONPATH=src MPLBACKEND=Agg \
  ./.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=600 \
  notebooks/00_dataset_comparison.ipynb
```

To force a full rescan of the metadata, JSON records, image headers, and approximately
14 GiB of image hashes:

```bash
cd /localhome/local-lintran/MLA2
MLA2_RECOMPUTE_COMPARISON=1 PYTHONPATH=src MPLBACKEND=Agg \
  ./.venv/bin/jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/00_dataset_comparison.ipynb
```

## Rebuild and verify the report

```bash
cd /localhome/local-lintran/MLA2
PYTHONPATH=src \
  ./.venv/bin/python scripts/build_dataset_comparison_report.py

PYTHONPATH=src \
  ./.venv/bin/python scripts/verify_dataset_comparison.py
```

A successful verification ends with:

```text
OK — IDs, labels, image coverage, figures, report claims, and 7 executed notebook cells reconcile
```

## Important data notes

- Five teacher-train metadata IDs have no image in either dataset:
  `12347`, `39401`, `39403`, `39410`, and `39425`.
- `usage` contains the literal label `"NA"`. It is different from a blank value.
- The target classes are heavily imbalanced. Use macro-F1 and per-class support rather
  than accuracy alone.
- Exact duplicate products must remain in the same development split.
- Raw files under `data/` must not be modified or committed.
