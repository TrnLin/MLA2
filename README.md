# Fashion Intelligence — COSC2753 Assignment 2

Clean starter repository for four fashion classification targets and a Top-K
visual search system.

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

## Repository structure

```text
data/             Supplied raw data and rebuildable processed data
docs/             Assignment material and project decisions
notebooks/        Short narrative notebooks
results/figures/  Figures and tables used as evidence
scripts/           Small command-line entry points
src/fashion/      Reusable Python code
tests/            Automated tests
```

Each folder contains a short guide explaining what belongs there.
