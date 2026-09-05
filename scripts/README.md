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

Run the complete matched pretraining benchmark only through its paired launcher:

```powershell
& '.\.venv\Scripts\python.exe' scripts/run_task2_pretraining_benchmark.py --mode run_or_load
```

It validates both frozen declarations before starting P0S followed by P*. The first P*
run may download TorchVision's declared ResNet18 ImageNet weights. Both rows remain
benchmark-only and cannot be selected as the submitted scratch model.

After all ten folds exist, rebuild the comparison in load-only mode:

```powershell
& '.\.venv\Scripts\python.exe' scripts/build_task2_pretraining_evidence.py
```

This command cannot start training. It verifies the cached hashes and writes the matched
P* minus P0S tables plus teacher-style five-fold learning curves.

After the C2 and I2 stability gate is closed, build the declared shortcut and error
analysis from the same frozen OOF files:

```powershell
& '.\.venv\Scripts\python.exe' scripts/build_task2_slice_evidence.py
```

This command never trains or opens holdout labels. It verifies the G5 manifest, all 20
OOF hashes, canonical IDs, folds, targets, and post-prediction slice assignments before
writing G6 tables and figures.

After the slice boundary is closed, run or hash-load the frozen image-stress and
machine-cost probes:

```powershell
& '.\.venv\Scripts\python.exe' scripts/build_task2_robustness_evidence.py --mode run_or_load
```

This command does not train. It reuses the primary-seed C2 and I2 fold checkpoints,
applies only the declared image perturbations to development rows, and records latency
and memory on the current machine. Use `--mode load` to require every exact cache.

After the robustness/cost boundary is closed, build cross-fitted calibration evidence:

```powershell
& '.\.venv\Scripts\python.exe' scripts/build_task2_calibration_evidence.py
```

This command does not train. It verifies the frozen upstream manifests and uses only the
primary-seed development OOF probabilities. Each fold is calibrated by a temperature fit
on the other four folds. The holdout stays sealed, and no app threshold is frozen.

After calibration is closed, build paired product-family bootstrap intervals:

```powershell
& '.\.venv\Scripts\python.exe' scripts/build_task2_bootstrap_evidence.py
```

This command does not train. It verifies all 20 frozen C2/I2 OOF files, joins them to
canonical `product_family_group` values by ID, and runs the declared 10,000 paired draws
for each seed pair. The same family draw is reused across candidates and seeds. The
result describes fitted-pair uncertainty only; it does not open holdout or freeze the
ultimate winner.

After paired uncertainty is closed, build the fixed Grad-CAM and failure review:

```powershell
& '.\.venv\Scripts\python.exe' scripts/build_task2_gradcam_evidence.py
```

This command selects three high-confidence correct and incorrect rows per true class
for both primary-seed finalists. It restores only each row's matching validation-fold
checkpoint, reconciles the raw OOF probabilities, and writes two contact sheets plus
non-causal attention and failure tables. Metadata is review context only. Holdout stays
sealed, and this command does not change or freeze the candidate.

## Task 4 preprocessing and baseline runners

Task 4 has two thin runners:

```bash
./.venv/bin/python scripts/task4/run_preprocessing.py
./.venv/bin/python scripts/task4/run_baseline.py
```

The first rebuilds development-only preprocessing evidence. The second rebuilds
the frozen untrained baseline evidence. Reusable work lives in `fashion.task4`;
the scripts only parse arguments and call that package.
