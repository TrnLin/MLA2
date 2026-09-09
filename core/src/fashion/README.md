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
- `task4/` owns the real Task 4 implementation: variant audits, evaluation,
  preprocessing, caching, the fixed probe, baseline analysis, evidence, and
  CPU cost measurement. Import reusable code through `fashion.task4`.
  - `external.py` audits and reconciles the V1 image variant.
  - `protocol.py` implements the frozen development-only retrieval metrics.
  - `preprocessing.py` defines the Task 4 image-input contract.
  - `cache.py` builds guarded lossless development-image caches.
  - `probe.py` provides the fixed HSV-and-edge comparison descriptor.
  - `preprocessing_experiment.py` runs the size and source comparison.
  - `baseline.py`, `analysis.py`, `benchmark.py`, and `baseline_evidence.py`
    own the untrained baseline, slices, timing, and tracked evidence.
- `retrieval/` contains compatibility-only exports for older
  `fashion.retrieval` imports. New code must not place logic there.

- `task1/` owns reusable Task 1 article-type classification code.
  - `registry.py` gives Task 1 a private view of the shared run ledger without changing
    the Task 2/3 registry class.
  - `image_contract.py` defines the shared image size, padding colour, and tensor shape.
  - `analysis.py` turns prepared EDA and completed runs into decision and failure evidence.
  - `preprocessing.py` defines deterministic image transforms and fold-fitted normalization.
  - `dataset.py` builds validated Task 1 tensor samples.
  - `models.py` defines the scratch small-CNN architecture.
  - `cnn_engine.py` runs one CNN fold while `training.py` keeps the public training facade.
  - `cnn_experiments.py` runs the controlled CNN comparison and writes full-run evidence.
  - `classical_features.py` defines and caches grayscale HOG features.
  - `classical_models.py` defines the approved classic-model settings.
  - `classical_training.py` runs one classic fold and registers it.
  - `classical_experiments.py` stages classic smoke, tuning, and selected five-fold evidence.
  - `evaluation.py` calculates fixed-124-class metrics and checks out-of-fold predictions.
  - `plotting.py` writes report figures from completed evidence.
  - `classical.py` and `experiments.py` are compatibility facades for earlier imports.
- `train/` owns shared artifacts, reproducibility seeds, and the run registry.

### Task 1 classic-run handoff

Run the controller in [`notebooks/02_task1_part1_article_type.ipynb`](../../notebooks/02_task1_part1_article_type.ipynb).
It defaults both `RUN_MODE` and `CLASSICAL_STAGE` to `"smoke"`. After smoke passes, run
the CNN controller once with `RUN_MODE = "full"`, then run the classic controller once with
`CLASSICAL_STAGE = "tune"`, then with `CLASSICAL_STAGE = "final"`. The final run
requires `results/evidence/task1/classical_selection.json` from tuning and writes
registered five-fold evidence. HOG uses scikit-image; KNN and LinearSVC use
scikit-learn. Smoke checks the path only; tune selects settings and final produces
the report evidence.

Update this if you are adding more scripts.
