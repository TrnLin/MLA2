# Scripts

Put non-EDA command-line entry points here.

Scripts should call reusable functions from `src/fashion/` instead of containing
project logic.

EDA and EDA preparation have no script entry point. Open `notebooks/00_eda.ipynb`
and use **Run All**. Normal notebook use validates the delivered prepared-data cache.
Set `FASHION_EDA_MODE=full` before launch only for the slow forensic rebuild. The
notebook calls reusable contracts in `src/fashion/data/` directly.
