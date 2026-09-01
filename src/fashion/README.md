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

- `task1/` owns reusable Task 1 article-type classification code.
  - `models.py` defines the scratch small-CNN architecture.
  - `dataset.py` builds validated Task 1 tensor samples.
  - `evaluation.py` calculates fixed-124-class metrics and checks out-of-fold predictions.
  - `training.py` trains and registers one fold at a time.
  - `classical.py` extracts and caches grayscale HOG features and runs fixed-class KNN
    and Linear SVM fold candidates.
  - `experiments.py` orchestrates both the CNN comparison and staged classic-model
    tuning/five-fold evidence.
- `train/` owns shared artifacts, reproducibility seeds, and the run registry.

### Task 1 classic-run handoff

Run the controller in [`notebooks/02_task1_article_type.ipynb`](../../notebooks/02_task1_article_type.ipynb).
It defaults to `CLASSICAL_STAGE = "smoke"`. After smoke passes, run it once with
`CLASSICAL_STAGE = "tune"`, then with `CLASSICAL_STAGE = "final"`. The final run
requires `results/evidence/task1/classical_selection.json` from tuning and writes
registered five-fold evidence. HOG uses scikit-image; KNN and LinearSVC use
scikit-learn. Smoke checks the path only; tune selects settings and final produces
the report evidence.

Update this if you are adding more scripts.
