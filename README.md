# Fashion Intelligence — COSC2753 Assignment 2

Team repository for four fashion classification targets and a Top-K visual
search system. The data preparation and EDA stages are reproducible and protect
the official prediction and internal holdout data.

## Start here

1. Read `docs/COSC2753_2026B_Assignment 2.pdf`.
2. Read `rubrics/RUBRIC.md`.
3. Read `AGENTS.md` for project rules.
4. Check `docs/decisions/` before making a choice that affects later work.

## Setup

Python 3.11 or newer is required.

```bash
cd MLA2
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
```

The supplied dataset stays under `data/` and is ignored by Git.

## Prepare data and EDA

Place the teacher files in the layout described by `data/raw/README.md`, then run:

```bash
./.venv/bin/python scripts/prepare_data.py
./.venv/bin/python scripts/generate_eda.py
./.venv/bin/python -m pytest
```

`data/processed/splits.csv` is the only shared split. Do not create another split
in a notebook or model script.

## Reproducibility boundary

The template does not yet define a dependency lock or constraints-file convention.
The checks therefore prove the environment in which they were run, but a later
install may resolve different package versions. Agree and record a locking policy
before shared model training; decision `0006` tracks this collaboration risk.

## Repository structure

```text
data/             Supplied raw data and rebuildable processed data
docs/             Assignment material and project decisions
notebooks/        Short narrative notebooks
results/          Report figures and compact EDA evidence
scripts/           Small command-line entry points
src/fashion/      Reusable Python code
tests/            Automated tests
```

Each folder contains a short guide explaining what belongs there.
