# Fashion Intelligence — COSC2753 Assignment 2

Team project for four image-classification outputs and a Top-K visual search system.
The shared data workflow is teacher-only, repeatable, and keeps the internal holdout sealed.

## Start here

1. Read `docs/COSC2753_2026B_Assignment 2.pdf`.
2. Read `rubrics/RUBRIC.md`.
3. Read `AGENTS.md`.
4. Check `docs/decisions/` before changing a shared rule.

## Setup

Python 3.12 is the shared locked baseline.

```bash
cd MLA2-eda
python3.12 -m venv .venv
./.venv/bin/python -m pip install -c requirements/constraints-py312.txt -e ".[dev]"
```

For a wheel install, point the package at this checkout:

```bash
export FASHION_PROJECT_ROOT="/absolute/path/to/MLA2-eda"
```

## Shared data preparation

Put the supplied teacher data under `data/raw/teacher/` as shown in
`data/raw/README.md`. Then open `notebooks/01_data_preparation.ipynb` in a fresh
kernel and use **Run All**.

Normal Run All validates the delivered cache. If teacher files changed, start Jupyter
with full mode and use Run All:

```bash
FASHION_DATA_PREPARATION_MODE=full ./.venv/bin/python -m jupyter lab
```

Full mode rebuilds in a child process. This keeps protected target values out of the
notebook kernel. It hashes raw teacher image bytes before decoding them, rebuilds all
shared artifacts, and then runs development-only analysis. It does not read optional
external images and does not fit a model or any learned image statistics.

The saved code-free report is `results/notebooks/01_data_preparation.html`.

## One split, five folds

`data/processed/splits.csv` is the only split:

- 32,773 `development` rows, each assigned one `cv_fold` from 0 to 4;
- 5,778 sealed `holdout` rows;
- 61 `quarantine` rows.

Task owners use `fashion.data.dataset.get_cv_split` or `iter_cv_folds`. Any value
learned from data is fitted on that round's training folds only. Notebook 06 may open
the holdout once, after every choice is frozen.

Tasks 1–3 use teacher images. Task 4 decides later whether to use the optional local
collection at `data/raw/external/fashion_product_images_v1/`. Binary external data is
outside Git and is never required by shared preparation.

## Permanent project rules

- Train submitted models from scratch. Pretrained models are comparison benchmarks only.
- Append every training run through `fashion.train.registry` to `results/runs.csv`.
- Do not create another split.
- Prefer broad comparisons and honest failure analysis over one extra training run.

## Repository structure

```text
data/             Raw teacher data and rebuildable processed data
docs/             Assignment material, provenance, and decisions
notebooks/        Problem, preparation, task, and final-evaluation workflows
results/          Report figures, evidence, and saved notebook HTML
src/fashion/      Reusable Python code
tests/            Automated contract and leakage checks
```
