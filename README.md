# Fashion Intelligence — COSC2753 Assignment 2

Team repository for four fashion classification targets and a Top-K visual
search system. Shared data preparation, including its focused EDA checks, is reproducible and protects
the official prediction and internal holdout data.

## Start here

1. Read `docs/COSC2753_2026B_Assignment 2.pdf`.
2. Read `rubrics/RUBRIC.md`.
3. Read `AGENTS.md` for project rules.
4. Check `docs/decisions/` before making a choice that affects later work.

## Setup

Python 3.12, 3.13, or 3.14 is required. Python 3.12 is the shared locked baseline.

```bash
cd MLA2-eda
python3.12 -m venv .venv
./.venv/bin/python -m pip install -c requirements/constraints-py312.txt -e ".[dev]"
```

For a normal (non-editable) wheel install, point data and report paths at this
checkout before running commands from another folder:

```bash
export FASHION_PROJECT_ROOT="/absolute/path/to/MLA2-eda"
```

The supplied dataset stays under `data/` and is ignored by Git.

## Run shared data preparation

Place the teacher files in the layout described by `data/raw/README.md`, and place
the high-resolution Fashion Product Images Dataset under `data/fashion-dataset/`.
Then open `notebooks/01_data_preparation.ipynb`, start a fresh kernel, and choose **Run All**.

Normal **Run All** uses a lean prepared pack delivered with the project. The pack keeps
`splits.csv` readable and stores large repeat-heavy tables as deterministic `.csv.gz`
files. It checks every source image name and size, content-hashes a fixed sample, fully
hashes every protected-safe prepared artifact, checks the split and both image variants,
then performs the shared data checks and train-only analysis. Prediction IDs and quarantine never enter
the variant manifest. No separate data-preparation helper command is needed.

The source-image guard is intentionally cheap. It detects missing files, path or size
changes, and changes in the fixed content sample. It cannot detect every same-size edit
to an unsampled raw image. Use full mode after any deliberate raw-image replacement.

If raw files changed, launch Jupyter with `FASHION_DATA_PREPARATION_MODE=full` and use **Run All**.
That explicit forensic mode rebuilds and fully audits the low- and high-resolution
collections, regenerates label-free alignment evidence, and writes both the paired
main-policy normalization and the original-only baseline. It is I/O-bound and
intentionally much slower than normal cached mode.

The saved code-free report is `results/notebooks/01_data_preparation.html`. External tests run
the notebook twice in fresh temporary projects, compare stable hashes, and check
the HTML structure and all 17 figures.

`data/processed/splits.csv` is the only shared split. Its persisted holdout and
quarantine target cells are blank. Only the explicit unlocked final-evaluation loader
may join those targets from the evaluator's local teacher CSV. Do not create another
split in a notebook or model script.

## Reproducibility boundary

`requirements/constraints-py312.txt` freezes the complete shared Python 3.12
environment. Direct packages are also pinned in `pyproject.toml`. Decision `0006`
explains how to update and verify both files.

Notebook provenance records raw-file ID-only digests, the redacted split file digest,
image inventory, audits, both-resolution manifest, notebook code, runtime,
benchmark, and generated outputs. `results/evidence/data_preparation/summary.json` lists stable
report-output hashes without machine-specific checkout paths.
Wall-time benchmark files are marked as hardware-dependent.

## Repository structure

```text
data/             Supplied raw data and rebuildable processed data
docs/             Assignment material and project decisions
notebooks/        Problem definition, shared data preparation, and task notebooks
results/          Report figures and compact preparation or experiment evidence
scripts/           Utility entry points outside notebook workflows
src/fashion/      Reusable Python code
tests/            Automated tests
```

Each folder contains a short guide explaining what belongs there.
