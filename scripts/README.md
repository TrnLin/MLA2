# Scripts

Put command-line utility entry points here.

Scripts should call reusable functions from `src/fashion/` instead of containing
project logic.

Shared data preparation has no script entry point. Open `notebooks/01_data_preparation.ipynb`
and use **Run All**. Normal notebook use validates the delivered prepared-data cache.
Set `FASHION_DATA_PREPARATION_MODE=full` before launch only for the slow forensic rebuild. The
notebook calls reusable contracts in `src/fashion/data/` directly.
