# Scripts

Put command-line utility entry points here.

Scripts should call reusable functions from `src/fashion/` instead of containing
project logic.

Shared data preparation has no script entry point. Open `notebooks/01_data_preparation.ipynb`
and use **Run All**. Normal notebook use validates the delivered prepared-data cache.
Set `FASHION_DATA_PREPARATION_MODE=full` before launch only for the slow forensic rebuild. The
notebook calls reusable contracts in `src/fashion/data/` directly.

Task 4 has two thin runners:

```bash
./.venv/bin/python scripts/task4/run_preprocessing.py
./.venv/bin/python scripts/task4/run_baseline.py
```

The first rebuilds development-only preprocessing evidence. The second rebuilds
the frozen untrained baseline evidence. Reusable work lives in `fashion.task4`;
the scripts only parse arguments and call that package.
