# Fashion package

Reusable project code lives here. Notebooks import it instead of copying shared logic.

- `config.py` owns shared paths, target names, split sizes, and fixed seeds.
- `data/` owns teacher audits, metadata cleanup, the sole split, fold loaders, image
  loading, data evidence, and the protected final-evaluation boundary.
  - `__init__.py` exposes the small public data API used by notebooks.
  - `audit.py` checks raw CSV files and images without changing them.
  - `dataset.py` loads safe splits, CV folds, samples, and the dataset adapter.
  - `evidence.py` builds development-only tables used to explain data choices.
  - `families.py` groups duplicates and related products for safe splitting.
  - `hashing.py` creates stable file hashes and deterministic CSV files.
  - `images.py` provides image transforms and streaming RGB statistics.
  - `manifests.py` builds image-backed training and prediction manifests.
  - `metadata.py` repairs product names and creates label maps and encodings.
  - `perceptual.py` finds and checks possible near-duplicate images.
  - `pipeline.py` runs and validates the teacher-only preparation workflow.
  - `splits.py` builds and validates the sole split and its five CV folds.
  - `taxonomy.py` describes and validates development target labels.
- `train/` owns registered model execution and evaluation contracts.
  - `augmentation.py` contains the locked brightness-only child transform.
  - `config.py` freezes the Task 3 primary baseline and its parameter count.
  - `data.py` fits fold-only RGB statistics and loads traceable PyTorch samples.
  - `metrics.py` calculates fixed-class OOF, calibration, and per-class metrics.
  - `model.py` defines only the exact Task 3 scratch baseline CNN.
  - `registry.py` appends a durable row before training and preserves failed runs.
  - `task3_baseline.py` checks the Colab runtime and runs the five-fold baseline.
  - `task3_experiments.py` locks and runs the two one-factor SmallCNN children.

Update this if you are adding more scripts.
