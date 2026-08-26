# Scripts

Put command-line utility entry points here.

Scripts should call reusable functions from `src/fashion/` instead of containing
project logic.

Shared data preparation has no script entry point. Open `notebooks/01_data_preparation.ipynb`
and use **Run All**. Normal notebook use validates the delivered prepared-data cache.
Set `FASHION_DATA_PREPARATION_MODE=full` before launch only for the slow forensic rebuild. The
notebook calls reusable contracts in `src/fashion/data/` directly.

## Task 2 experiment launcher

Run one immutable Task 2 config from the repository root:

```powershell
& '.\.venv\Scripts\python.exe' scripts/run_task2_experiment.py configs/task2/g2_p1_c2_resnet18.json --mode run_or_load
```

`run_or_load` verifies every declared artifact hash before it reuses a fold. A cache miss
creates a new physical run ID and appends it to `results/runs.csv`. The launcher uses a
Windows-safe `__main__` guard, so multi-worker PyTorch loaders do not start the experiment
again inside child processes.
