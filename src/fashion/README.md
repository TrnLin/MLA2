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
- `retrieval/` owns Task 4 variant audits, evaluation, preprocessing, caching,
  the fixed selection probe, and experiment orchestration.
  - `external.py` audits and reconciles the V1 image variant.
  - `protocol.py` implements the frozen development-only retrieval metrics.
  - `preprocessing.py` defines the Task 4 image-input contract.
  - `cache.py` builds guarded lossless development-image caches.
  - `probe.py` provides the fixed HSV-and-edge comparison descriptor.
  - `preprocessing_experiment.py` runs the size and source comparison.

Update this if you are adding more scripts.
