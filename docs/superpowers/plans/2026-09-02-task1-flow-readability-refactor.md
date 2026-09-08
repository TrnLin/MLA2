# Task 1 Flow and Readability Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Task 1 into small readable modules and make Notebook 02 explain every preprocessing and model choice using visible EDA and problem evidence.

**Architecture:** Keep the existing public Task 1 imports as compatibility facades while moving image rules, CNN mechanics, classical features/models/training, experiment controllers, plots, and analysis into focused modules. Notebook 02 becomes a thin evidence-led story: problem, EDA, hypotheses, controlled runs, results, failures, and decision.

**Tech Stack:** Python 3.12, pandas, NumPy, Pillow, PyTorch, scikit-image, scikit-learn, matplotlib, pytest, Ruff, Jupyter/nbformat

**Spec:** `docs/superpowers/specs/2026-09-02-task1-flow-readability-refactor-design.md`

## Global Constraints

- Read splits only from `data/processed/splits.csv`; never call `train_test_split` elsewhere.
- Keep holdout and quarantine labels sealed until Notebook 06.
- Train submitted models from scratch; never add `pretrained=True`.
- Append every physical training run through `fashion.train.registry`.
- Preserve the fixed 124-class label order, metric formulas, evidence columns, and final prediction format `id,gender,articleType,season,usage`.
- Keep reusable dataset, split, preprocessing, training, and evaluation logic in `src/fashion/`.
- Keep `fashion.task1`, `fashion.task1.classical`, and `fashion.task1.experiments` public imports working.
- Preserve unrelated user changes. Before editing the already-modified Notebook 02, inspect its current diff and edit that current version.
- Never delete `data/processed/splits.csv`, label maps, raw images, or shared data-preparation evidence.
- Use `./.venv/bin/python` for project Python commands.
- Do not start full CNN, classical tuning, or classical final runs during this refactor.

---

### Task 1: Add one shared Task 1 image contract

**Files:**
- Create: `src/fashion/task1/image_contract.py`
- Modify: `src/fashion/task1/preprocessing.py`
- Modify: `src/fashion/task1/classical.py`
- Modify: `src/fashion/task1/dataset.py`
- Modify: `src/fashion/task1/__init__.py`
- Create: `tests/task1/test_image_contract.py`
- Modify: `tests/task1/test_preprocessing.py`
- Modify: `tests/task1/test_classical.py`

**Interfaces:**
- Produces: `TASK1_IMAGE_SIZE: tuple[int, int]`, `TASK1_PAD_COLOR: tuple[int, int, int]`, and `TASK1_TENSOR_SHAPE: tuple[int, int, int]`.
- Consumes: `fashion.data.images.transform_image_with_mask` remains the shared resize/letterbox implementation.
- Compatibility: `Task1PreprocessingConfig.to_dict()` and `Task1HogSpec.to_dict()` keep their existing `image_size` and `pad_color` fields.

- [ ] **Step 1: Write failing shared-contract tests**

Create `tests/task1/test_image_contract.py`:

```python
from fashion.task1.classical import TASK1_HOG_COARSE, TASK1_HOG_FINE
from fashion.task1.image_contract import (
    TASK1_IMAGE_SIZE,
    TASK1_PAD_COLOR,
    TASK1_TENSOR_SHAPE,
)
from fashion.task1.preprocessing import (
    DEFAULT_TASK1_PREPROCESSING,
    TASK1_CONTROL_PREPROCESSING,
)


def test_task1_models_share_one_image_geometry() -> None:
    assert TASK1_IMAGE_SIZE == (80, 60)
    assert TASK1_PAD_COLOR == (255, 255, 255)
    assert TASK1_TENSOR_SHAPE == (3, 80, 60)
    assert DEFAULT_TASK1_PREPROCESSING.image_size == TASK1_IMAGE_SIZE
    assert TASK1_CONTROL_PREPROCESSING.image_size == TASK1_IMAGE_SIZE
    assert DEFAULT_TASK1_PREPROCESSING.pad_color == TASK1_PAD_COLOR
    assert TASK1_CONTROL_PREPROCESSING.pad_color == TASK1_PAD_COLOR
    assert TASK1_HOG_COARSE.image_size == TASK1_IMAGE_SIZE
    assert TASK1_HOG_FINE.image_size == TASK1_IMAGE_SIZE
    assert TASK1_HOG_COARSE.pad_color == TASK1_PAD_COLOR
    assert TASK1_HOG_FINE.pad_color == TASK1_PAD_COLOR
```

Also replace literal `(3, 80, 60)` assertions in `test_preprocessing.py` with
`TASK1_TENSOR_SHAPE` so tests describe the contract rather than repeat it.

- [ ] **Step 2: Run the tests and confirm the missing module failure**

```bash
./.venv/bin/python -m pytest tests/task1/test_image_contract.py tests/task1/test_preprocessing.py -q
```

Expected: collection fails because `fashion.task1.image_contract` does not exist.

- [ ] **Step 3: Add the shared constants**

Create `src/fashion/task1/image_contract.py`:

```python
"""Shared deterministic image geometry for every Task 1 model family."""

TASK1_IMAGE_SIZE: tuple[int, int] = (80, 60)
TASK1_PAD_COLOR: tuple[int, int, int] = (255, 255, 255)
TASK1_TENSOR_SHAPE: tuple[int, int, int] = (3, *TASK1_IMAGE_SIZE)
```

Use these constants as defaults in both `Task1PreprocessingConfig` and `Task1HogSpec`:

```python
image_size: tuple[int, int] = TASK1_IMAGE_SIZE
pad_color: tuple[int, int, int] = TASK1_PAD_COLOR
```

Use `TASK1_TENSOR_SHAPE` in `Task1TorchDataset.__getitem__`:

```python
if array.shape != TASK1_TENSOR_SHAPE or array.dtype != np.float32:
    raise ValueError(
        f"Task 1 transform must return float32 shape {TASK1_TENSOR_SHAPE}"
    )
```

Export the three constants from `fashion.task1`.

- [ ] **Step 4: Run geometry, preprocessing, dataset, and HOG tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_image_contract.py tests/task1/test_preprocessing.py tests/task1/test_dataset.py tests/task1/test_classical.py -q
./.venv/bin/python -m ruff check src/fashion/task1/image_contract.py src/fashion/task1/preprocessing.py src/fashion/task1/dataset.py tests/task1/test_image_contract.py
```

Expected: all selected tests pass and Ruff prints `All checks passed!`.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/fashion/task1/image_contract.py src/fashion/task1/preprocessing.py src/fashion/task1/classical.py src/fashion/task1/dataset.py src/fashion/task1/__init__.py tests/task1/test_image_contract.py tests/task1/test_preprocessing.py tests/task1/test_classical.py
git commit -m "refactor(task1): share image geometry"
```

---

### Task 2: Build reusable EDA-to-decision evidence

**Files:**
- Create: `src/fashion/task1/analysis.py`
- Modify: `src/fashion/task1/__init__.py`
- Create: `tests/task1/test_analysis.py`

**Interfaces:**
- Consumes: prepared `splits` and `data/processed/development_class_summary.csv`.
- Produces: `Task1ProblemProfile`, `build_task1_problem_profile(splits, class_summary)`, `build_task1_decision_evidence(profile)`, `build_task1_weak_class_table(per_class)`, and `build_task1_confusion_pairs(oof_predictions)`.
- Safety: helpers accept prepared data only, perform no model fitting, ignore protected rows, and reject a missing `articleType` summary.

- [ ] **Step 1: Write failing profile and decision-table tests**

Create a compact development frame with columns `id`, `partition`, `articleType`, `width`,
`height`, and `mode`. Create an `articleType` class-summary frame with `rare_warning` and
`untrainable_fold_count`. Assert:

```python
def test_problem_profile_summarises_task1_eda() -> None:
    profile = build_task1_problem_profile(_splits(), _class_summary())
    assert profile.development_products == 4
    assert profile.class_count == 3
    assert profile.minimum_class_products == 1
    assert profile.maximum_class_products == 2
    assert profile.grayscale_images == 1
    assert profile.unusual_geometry_images == 1
    assert profile.classes_with_fold_warnings == 2
    assert profile.classes_untrainable_in_any_fold == 1


def test_decision_evidence_connects_data_problem_and_test() -> None:
    table = build_task1_decision_evidence(
        build_task1_problem_profile(_splits(), _class_summary())
    )
    assert list(table.columns) == ["evidence", "problem", "choice_to_test"]
    assert len(table) == 7
    assert table["choice_to_test"].str.contains("macro-F1", regex=False).any()
    assert table["choice_to_test"].str.contains("HOG", regex=False).any()
    assert table["choice_to_test"].str.contains("scratch CNN", regex=False).any()
```

Add rejection tests for missing geometry columns, duplicate development IDs, a missing
`articleType` summary, and negative `untrainable_fold_count`. Add a separate test that changes
protected-row labels and geometry and proves the returned development profile does not change.

Add result-analysis tests with two named candidates:

```python
def test_failure_tables_keep_candidate_identity_and_example_ids() -> None:
    weak = build_task1_weak_class_table(_per_class(), limit=2)
    pairs = build_task1_confusion_pairs(_oof_predictions(), limit=2)
    assert list(weak.columns) == [
        "candidate_id", "class_index", "class_name", "support",
        "precision", "recall", "f1",
    ]
    assert weak.groupby("candidate_id").size().eq(2).all()
    assert list(pairs.columns) == [
        "candidate_id", "true_label", "predicted_label", "error_count", "example_ids",
    ]
    assert pairs["example_ids"].str.split(",").map(len).le(3).all()
```

- [ ] **Step 2: Run the new tests and confirm missing imports**

```bash
./.venv/bin/python -m pytest tests/task1/test_analysis.py -q
```

Expected: collection fails because the new analysis types and functions do not exist.

- [ ] **Step 3: Implement the immutable profile**

Add:

```python
@dataclass(frozen=True)
class Task1ProblemProfile:
    development_products: int
    class_count: int
    minimum_class_products: int
    maximum_class_products: int
    grayscale_images: int
    unusual_geometry_images: int
    classes_with_fold_warnings: int
    classes_untrainable_in_any_fold: int


def build_task1_problem_profile(
    splits: pd.DataFrame,
    class_summary: pd.DataFrame,
) -> Task1ProblemProfile:
    required = {"id", "partition", "articleType", "width", "height", "mode"}
    if missing := required.difference(splits.columns):
        raise ValueError(f"Task 1 EDA rows are missing columns: {sorted(missing)}")
    development = splits.loc[splits["partition"].eq("development")].copy()
    if development.empty or development["id"].duplicated().any():
        raise ValueError("Task 1 EDA requires unique development products")
    if not set(splits["partition"]).issubset({"development", "holdout", "quarantine"}):
        raise ValueError("Task 1 EDA contains an unknown partition")
    summary = class_summary.loc[class_summary["target"].eq("articleType")].copy()
    if summary.empty:
        raise ValueError("Task 1 EDA requires the articleType class summary")
    untrainable = pd.to_numeric(summary["untrainable_fold_count"], errors="raise")
    if untrainable.lt(0).any():
        raise ValueError("untrainable fold counts must be non-negative")
    support = development["articleType"].astype(str).value_counts()
    unusual = ~(
        pd.to_numeric(development["width"], errors="raise").eq(TASK1_IMAGE_SIZE[1])
        & pd.to_numeric(development["height"], errors="raise").eq(TASK1_IMAGE_SIZE[0])
    )
    return Task1ProblemProfile(
        development_products=len(development),
        class_count=len(support),
        minimum_class_products=int(support.min()),
        maximum_class_products=int(support.max()),
        grayscale_images=int(development["mode"].astype(str).eq("L").sum()),
        unusual_geometry_images=int(unusual.sum()),
        classes_with_fold_warnings=int(
            summary["rare_warning"].fillna("").astype(str).str.strip().ne("").sum()
        ),
        classes_untrainable_in_any_fold=int(untrainable.gt(0).sum()),
    )
```

Reject protected label access by using only development rows for counts. Do not copy holdout or
quarantine labels into returned objects.

- [ ] **Step 4: Implement the seven-row decision table**

Return seven explicit rows covering geometry, grayscale conversion, long-tail metrics,
untrainable classes, HOG, scratch CNN, and augmentation. Use the profile values in the evidence
strings. The first three rows must follow this exact form:

```python
rows = [
    {
        "evidence": (
            f"{profile.unusual_geometry_images} development images differ from the usual "
            "60-by-80 geometry"
        ),
        "problem": "stretching changes shape and centre crops can remove product edges",
        "choice_to_test": "aspect-preserving 60-by-80 white canvas",
    },
    {
        "evidence": f"{profile.grayscale_images} development images are grayscale",
        "problem": "input channel formats are inconsistent",
        "choice_to_test": "deterministic RGB conversion for every model family",
    },
    {
        "evidence": (
            f"{profile.class_count} classes range from "
            f"{profile.minimum_class_products} to {profile.maximum_class_products} products"
        ),
        "problem": "accuracy can hide failure on rare classes",
        "choice_to_test": "fixed-class macro-F1 plus per-class evidence",
    },
]
```

Export the profile and both builders from `fashion.task1`.

- [ ] **Step 5: Implement result-led failure tables**

Add two pure helpers. Validate required columns before aggregation and reject a non-positive
`limit`:

```python
def build_task1_weak_class_table(
    per_class: Mapping[str, pd.DataFrame], *, limit: int = 10
) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit must be positive")
    required = {"class_index", "class_name", "support", "precision", "recall", "f1"}
    rows = []
    for candidate_id, frame in per_class.items():
        if missing := required.difference(frame.columns):
            raise ValueError(f"per-class evidence is missing columns: {sorted(missing)}")
        weakest = frame.sort_values(
            ["f1", "support", "class_index"], kind="stable"
        ).head(limit).copy()
        weakest.insert(0, "candidate_id", candidate_id)
        rows.append(
            weakest.loc[:, [
                "candidate_id", "class_index", "class_name", "support",
                "precision", "recall", "f1",
            ]]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=[
            "candidate_id", "class_index", "class_name", "support",
            "precision", "recall", "f1",
        ]
    )


def build_task1_confusion_pairs(
    oof_predictions: Mapping[str, pd.DataFrame], *, limit: int = 10
) -> pd.DataFrame:
    if limit <= 0:
        raise ValueError("limit must be positive")
    required = {"id", "true_label", "predicted_label"}
    output = []
    for candidate_id, frame in oof_predictions.items():
        if missing := required.difference(frame.columns):
            raise ValueError(f"OOF evidence is missing columns: {sorted(missing)}")
        errors = frame.loc[frame["true_label"].ne(frame["predicted_label"])].copy()
        grouped = (
            errors.groupby(["true_label", "predicted_label"], as_index=False)
            .agg(
                error_count=("id", "size"),
                example_ids=("id", lambda ids: ",".join(
                    str(value) for value in sorted(map(int, ids))[:3]
                )),
            )
            .sort_values(
                ["error_count", "true_label", "predicted_label"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(limit)
        )
        grouped.insert(0, "candidate_id", candidate_id)
        output.append(grouped)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame(
        columns=[
            "candidate_id", "true_label", "predicted_label", "error_count", "example_ids",
        ]
    )
```

Preserve the explicit output column order shown by the tests. Export both helpers from
`fashion.task1`.

- [ ] **Step 6: Run tests and lint**

```bash
./.venv/bin/python -m pytest tests/task1/test_analysis.py tests/task1/test_notebook_baseline.py -q
./.venv/bin/python -m ruff check src/fashion/task1/analysis.py src/fashion/task1/__init__.py tests/task1/test_analysis.py
```

Expected: analysis tests pass. Existing notebook tests remain unchanged and pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/fashion/task1/analysis.py src/fashion/task1/__init__.py tests/task1/test_analysis.py
git commit -m "refactor(task1): expose EDA decision evidence"
```

---

### Task 3: Extract the CNN training engine

**Files:**
- Create: `src/fashion/task1/cnn_engine.py`
- Modify: `src/fashion/task1/training.py`
- Create: `tests/task1/test_cnn_engine.py`
- Modify: `tests/task1/test_training.py`

**Interfaces:**
- Consumes: validated development rows, label mapping, `Task1Normalization`, `Task1PreprocessingConfig`, and `Task1TrainConfig`.
- Produces: internal `Task1CnnEngineResult` and `train_task1_cnn(...)`.
- Compatibility: `Task1TrainConfig`, `Task1FoldResult`, `select_training_device`, and `train_task1_fold` stay in `training.py` with unchanged public signatures.

- [ ] **Step 1: Add failing engine tests**

Create `tests/task1/test_cnn_engine.py` using the small image/split helpers from
`tests/task1/test_training.py`. Test a one-epoch, two-batch CPU run:

```python
result = train_task1_cnn(
    training_rows,
    validation_rows,
    label_to_index,
    class_names,
    normalization=normalization,
    preprocessing=TASK1_CONTROL_PREPROCESSING,
    config=Task1TrainConfig.smoke(),
    root=tmp_path,
    device=torch.device("cpu"),
)
assert result.best_epoch == 1
assert list(result.history.columns) == [
    "epoch", "train_loss", "macro_f1", "weighted_f1",
    "top1_accuracy", "top5_accuracy", "validation_loss",
]
assert len(result.predictions.filter(like="prob_", axis=1).columns) == 124
assert result.metrics["macro_f1"] == result.history.iloc[0]["macro_f1"]
```

Keep the exploding-model test at the registered `train_task1_fold` boundary because registry
failure status belongs to the coordinator.

- [ ] **Step 2: Run the engine test and confirm missing imports**

```bash
./.venv/bin/python -m pytest tests/task1/test_cnn_engine.py -q
```

Expected: collection fails because `cnn_engine.py` does not exist.

- [ ] **Step 3: Define the engine result and entry point**

Create:

```python
@dataclass(frozen=True)
class Task1CnnEngineResult:
    model: nn.Module
    best_epoch: int
    history: pd.DataFrame
    metrics: dict[str, float]
    predictions: pd.DataFrame


def train_task1_cnn(
    training_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    label_to_index: Mapping[str, int],
    class_names: list[str],
    *,
    normalization: Task1Normalization,
    preprocessing: Task1PreprocessingConfig,
    config: Task1TrainConfig,
    root: Path,
    device: torch.device,
    model_factory: Callable[[int], nn.Module] = Task1SmallCNN,
) -> Task1CnnEngineResult:
```

Use `from __future__ import annotations` and import `Task1TrainConfig` only under
`if TYPE_CHECKING:`. `training.py` imports the engine at runtime, so this keeps annotations exact
without creating a circular import.

Move `_limited_batches`, `_evaluate`, `_training_loader`, model/optimizer/scheduler creation,
the epoch loop, best-state selection, and final evaluation from `training.py` into this module.
Keep their calculations unchanged.

- [ ] **Step 4: Reduce `train_task1_fold` to orchestration**

Keep configuration validation, canonical-split checks, run-record creation, normalization,
artifact serialization, hashes, and registry updates in `training.py`. Replace the inlined loop
with:

```python
engine_result = train_task1_cnn(
    training_rows,
    validation_rows,
    label_to_index,
    class_names,
    normalization=normalization,
    preprocessing=preprocessing,
    config=config,
    root=project_root,
    device=selected_device,
    model_factory=model_factory,
)
```

Build checkpoints and `Task1FoldResult` from `engine_result`. Add
`src/fashion/task1/cnn_engine.py` and `src/fashion/task1/image_contract.py` to
`_implementation_hashes()`.

- [ ] **Step 5: Run engine, training, reproducibility, and lint checks**

```bash
./.venv/bin/python -m pytest tests/task1/test_cnn_engine.py tests/task1/test_training.py tests/train/test_reproducibility.py -q
./.venv/bin/python -m ruff check src/fashion/task1/cnn_engine.py src/fashion/task1/training.py tests/task1/test_cnn_engine.py tests/task1/test_training.py
```

Expected: the one-fold artifacts, best checkpoint, predictions, and registry failure tests all
pass without changing schemas.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/fashion/task1/cnn_engine.py src/fashion/task1/training.py tests/task1/test_cnn_engine.py tests/task1/test_training.py
git commit -m "refactor(task1): extract CNN training engine"
```

---

### Task 4: Extract HOG features and cache ownership

**Files:**
- Create: `src/fashion/task1/classical_features.py`
- Modify: `src/fashion/task1/classical.py`
- Create: `tests/task1/test_classical_features.py`
- Modify: `tests/task1/test_classical.py`

**Interfaces:**
- Produces: `Task1HogSpec`, `Task1HogFeatureSet`, `TASK1_HOG_COARSE`, `TASK1_HOG_FINE`, `TASK1_HOG_SPECS`, `extract_task1_hog`, and `load_or_build_task1_hog_features`.
- Consumes: shared `TASK1_IMAGE_SIZE` and `TASK1_PAD_COLOR` plus `transform_image_with_mask`.
- Cache identity includes `classical_features.py`, `image_contract.py`, and `fashion/data/images.py` hashes.

- [ ] **Step 1: Move feature/cache tests to their owner and make imports fail**

Create `test_classical_features.py` by moving these existing behaviours out of
`test_classical.py` without weakening assertions:

```text
HOG fixed IDs and widths
deterministic float32 extraction
invalid geometry rejection
cache hit
stale split/image/config/source/package rejection
invalid cache shape rejection
interrupted atomic write safety
development-only unique inventory validation
```

Change imports to `fashion.task1.classical_features`. Keep one compatibility assertion in
`test_classical.py`:

```python
from fashion.task1.classical import TASK1_HOG_COARSE as facade_hog
from fashion.task1.classical_features import TASK1_HOG_COARSE as owned_hog


def test_classical_facade_reexports_hog_contract() -> None:
    assert facade_hog is owned_hog
```

- [ ] **Step 2: Run feature tests and confirm missing module**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical_features.py -q
```

Expected: collection fails because `classical_features.py` does not exist.

- [ ] **Step 3: Move the exact feature/cache symbol group**

Move these symbols unchanged except for imports and cache source identities:

```text
HOG_CACHE_SCHEMA_VERSION
Task1HogSpec
TASK1_HOG_COARSE
TASK1_HOG_FINE
TASK1_HOG_SPECS
Task1HogFeatureSet
extract_task1_hog
_normalized_development_hog_rows
_hog_inventory
_hog_cache_implementation_identity
load_or_build_task1_hog_features
```

The two HOG specs must still be:

```python
TASK1_HOG_COARSE = Task1HogSpec("task1_gray_hog_ppc16_v1", (16, 16), 288)
TASK1_HOG_FINE = Task1HogSpec("task1_gray_hog_ppc10_v1", (10, 10), 1260)
TASK1_HOG_SPECS = (TASK1_HOG_COARSE, TASK1_HOG_FINE)
```

Re-export the public symbols from `classical.py` during the transition.

- [ ] **Step 4: Run focused and compatibility tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical_features.py tests/task1/test_classical.py -q
./.venv/bin/python -m ruff check src/fashion/task1/classical_features.py src/fashion/task1/classical.py tests/task1/test_classical_features.py
```

Expected: cache and extraction tests pass; the old import path returns the same objects.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/fashion/task1/classical_features.py src/fashion/task1/classical.py tests/task1/test_classical_features.py tests/task1/test_classical.py
git commit -m "refactor(task1): isolate HOG features and cache"
```

---

### Task 5: Extract classical model fitting

**Files:**
- Create: `src/fashion/task1/classical_models.py`
- Modify: `src/fashion/task1/classical.py`
- Create: `tests/task1/test_classical_models.py`
- Modify: `tests/task1/test_classical.py`

**Interfaces:**
- Produces: `Task1KNNConfig`, `Task1LinearSVMConfig`, `Task1ClassicalModelConfig`, both fixed grids/defaults, neighbour-query helpers, KNN scores, Linear SVM scores, and stable fixed-class expansion.
- Consumes: `TASK1_NUM_CLASSES` from evaluation.
- Does not read splits, write files, touch the registry, or choose experiment winners.

- [ ] **Step 1: Move pure estimator tests to a new failing owner test**

Move the existing tests for KNN configs, SVM configs, reused-neighbour voting, fixed 124-column
expansion, stable softmax, and fit/predict functions to `test_classical_models.py`. Import from
`fashion.task1.classical_models`.

Add this boundary test:

```python
def test_classical_model_module_has_no_project_io_dependencies() -> None:
    source = Path(classical_models.__file__).read_text(encoding="utf-8")
    assert "RunRegistry" not in source
    assert "atomic_write" not in source
    assert "load_splits" not in source
```

- [ ] **Step 2: Run and confirm missing module**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical_models.py -q
```

Expected: collection fails because `classical_models.py` does not exist.

- [ ] **Step 3: Move the exact pure-model symbol group**

Move:

```text
Task1KNNConfig
Task1LinearSVMConfig
Task1ClassicalModelConfig
TASK1_KNN_GRID
TASK1_SVM_GRID
TASK1_DEFAULT_KNN
TASK1_DEFAULT_SVM
_expand_observed_scores
_stable_softmax
query_task1_neighbors
knn_probabilities_from_neighbors
fit_predict_task1_knn
fit_predict_task1_linear_svm
```

Keep fixed grids and IDs byte-for-byte equivalent. Re-export the public names from
`classical.py`; private tests now import their owning module.

- [ ] **Step 4: Run model, facade, and lint tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical_models.py tests/task1/test_classical.py -q
./.venv/bin/python -m ruff check src/fashion/task1/classical_models.py src/fashion/task1/classical.py tests/task1/test_classical_models.py
```

Expected: all exact-vote and score-shape assertions pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/fashion/task1/classical_models.py src/fashion/task1/classical.py tests/task1/test_classical_models.py tests/task1/test_classical.py
git commit -m "refactor(task1): isolate classical estimators"
```

---

### Task 6: Extract the registered classical fold runner

**Files:**
- Create: `src/fashion/task1/classical_training.py`
- Replace: `src/fashion/task1/classical.py`
- Create: `tests/task1/test_classical_training.py`
- Modify: `tests/task1/test_classical.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Consumes: feature loading from `classical_features`, estimators from `classical_models`, shared evaluation, canonical split checks, artifacts, and registry.
- Produces: `Task1ClassicalRunConfig`, `Task1ClassicalFoldResult`, and `run_task1_classical_fold(...)`.
- Compatibility: `classical.py` becomes a facade that re-exports all existing non-private classical imports.

- [ ] **Step 1: Move fold-runner tests to the owning module**

Move existing smoke artifact, exact-resume, stale-artifact, canonical-split, failure, interruption,
and final-eligibility tests into `test_classical_training.py`. Patch owned dependencies, for
example:

```python
monkeypatch.setattr(
    "fashion.task1.classical_training.fit_predict_task1_knn",
    explode,
)
```

Keep `test_classical.py` as a small public-facade test that imports every public symbol from the
old path and checks identity against the owning modules.

- [ ] **Step 2: Run and confirm missing module**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical_training.py tests/task1/test_classical.py -q
```

Expected: collection fails for `fashion.task1.classical_training`.

- [ ] **Step 3: Move fold execution and artifact identity**

Move these symbols into `classical_training.py`:

```text
Task1ClassicalRunConfig
Task1ClassicalFoldResult
_classical_model_family
_classical_artifact_path
_resolve_classical_artifact_path
_classical_implementation_sha256
_smoke_hog_features
_aligned_hog_features
_completed_classical_result
run_task1_classical_fold
```

Replace the implementation hash with all behaviour-owning modules:

```python
def _classical_implementation_sha256() -> str:
    paths = (
        ROOT / "src/fashion/task1/image_contract.py",
        ROOT / "src/fashion/task1/classical_features.py",
        ROOT / "src/fashion/task1/classical_models.py",
        Path(__file__),
        ROOT / "src/fashion/task1/evaluation.py",
    )
    return canonical_sha256(
        {path.relative_to(ROOT).as_posix(): compute_sha256(path) for path in paths}
    )
```

- [ ] **Step 4: Replace `classical.py` with a compatibility facade**

The facade contains only a module docstring, imports from the three owners, and an explicit
`__all__`. Include all existing non-private API names, including HOG constants/types, KNN/SVM
constants/types, feature helpers, score helpers, run config/result, and fold runner. Do not
re-export `_stable_softmax` or implementation-hash helpers.

- [ ] **Step 5: Run all classical tests and lint**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical_features.py tests/task1/test_classical_models.py tests/task1/test_classical_training.py tests/task1/test_classical.py -q
./.venv/bin/python -m ruff check src/fashion/task1/classical_features.py src/fashion/task1/classical_models.py src/fashion/task1/classical_training.py src/fashion/task1/classical.py tests/task1/test_classical*.py
```

Expected: all tests pass and `classical.py` is a small import-only facade.

- [ ] **Step 6: Commit Task 6**

```bash
git add src/fashion/task1/classical_training.py src/fashion/task1/classical.py src/fashion/task1/__init__.py tests/task1/test_classical_training.py tests/task1/test_classical.py
git commit -m "refactor(task1): isolate classical fold training"
```

---

### Task 7: Extract CNN experiments and plotting

**Files:**
- Create: `src/fashion/task1/cnn_experiments.py`
- Create: `src/fashion/task1/plotting.py`
- Modify: `src/fashion/task1/experiments.py`
- Create: `tests/task1/test_cnn_experiments.py`
- Create: `tests/task1/test_plotting.py`
- Modify: `tests/task1/test_experiments.py`

**Interfaces:**
- Produces: `Task1ExperimentResult`, `run_task1_experiment(...)`, `write_task1_comparison_figure(...)`, and `write_task1_confusion_figure(...)`.
- Consumes: `train_task1_fold`, shared evaluation, `TASK1_EVIDENCE_DIR`, and `TASK1_FIGURE_DIR`.
- Compatibility: imports from `fashion.task1.experiments` remain valid during the split.

- [ ] **Step 1: Move CNN controller tests to a failing owner test**

Move existing CNN-only experiment tests to `test_cnn_experiments.py`, including smoke/full
schedules, duplicate/missing folds, OOF coverage, evidence atomicity, and aggregation. Change
patches to the owner module:

```python
monkeypatch.setattr(cnn_experiments, "TASK1_EVIDENCE_DIR", tmp_path / "evidence")
```

Create `test_plotting.py` by moving figure-output tests. Assert each writer returns the requested
path and creates a non-empty PNG.

- [ ] **Step 2: Run and confirm missing modules**

```bash
./.venv/bin/python -m pytest tests/task1/test_cnn_experiments.py tests/task1/test_plotting.py -q
```

Expected: collection fails for the two new modules.

- [ ] **Step 3: Move CNN experiment symbols**

Move from `experiments.py`:

```text
Task1ExperimentResult
_fold_metrics_frame
_class_names
_aggregate_comparison
_prediction_probabilities
_write_full_evidence
run_task1_experiment
```

Keep the public runner signature unchanged:

```python
def run_task1_experiment(
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    *,
    mode: Literal["smoke", "full"],
    registry: RunRegistry | None = None,
    root: str | Path = ROOT,
    result_root: str | Path = TASK1_RESULT_DIR,
    fold_runner: Callable[..., Task1FoldResult] = train_task1_fold,
) -> Task1ExperimentResult:
```

Keep evidence output at the existing `TASK1_EVIDENCE_DIR` module constant; tests continue to
redirect it by patching the owning `cnn_experiments` module.

Move `write_task1_comparison_figure` and `write_task1_confusion_figure` unchanged into
`plotting.py`.

- [ ] **Step 4: Re-export during transition and run tests**

Import the moved public names into `experiments.py`. Move `_aggregate_comparison` tests to the
owner module instead of preserving a private facade import.

```bash
./.venv/bin/python -m pytest tests/task1/test_cnn_experiments.py tests/task1/test_plotting.py tests/task1/test_evaluation.py tests/task1/test_experiments.py -q
./.venv/bin/python -m ruff check src/fashion/task1/cnn_experiments.py src/fashion/task1/plotting.py src/fashion/task1/experiments.py tests/task1/test_cnn_experiments.py tests/task1/test_plotting.py
```

Expected: CNN schedules and evidence schemas remain unchanged.

- [ ] **Step 5: Commit Task 7**

```bash
git add src/fashion/task1/cnn_experiments.py src/fashion/task1/plotting.py src/fashion/task1/experiments.py tests/task1/test_cnn_experiments.py tests/task1/test_plotting.py tests/task1/test_experiments.py tests/task1/test_evaluation.py
git commit -m "refactor(task1): isolate CNN experiment flow"
```

---

### Task 8: Extract the classical staged controller and finish the experiment facade

**Files:**
- Create: `src/fashion/task1/classical_experiments.py`
- Replace: `src/fashion/task1/experiments.py`
- Create: `tests/task1/test_classical_experiments.py`
- Modify: `tests/task1/test_experiments.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Consumes: classical features/configs/fold runner, shared evaluation, prepared development IDs, artifacts, and registry.
- Produces: `Task1ClassicalSelection`, `Task1ClassicalExperimentResult`, and `run_task1_classical_experiment(...)`.
- Compatibility: `experiments.py` becomes an explicit facade for both controller families and plots.

- [ ] **Step 1: Move classical controller tests to their owner**

Move smoke, tuning, ranking/tie-break, sealed selection, final five-fold, OOF, missing/duplicate
fold, stale provenance, and aggregate-write-failure tests to `test_classical_experiments.py`.
Change patches from `fashion.task1.experiments` to
`fashion.task1.classical_experiments`.

Keep `test_experiments.py` as a facade test:

```python
def test_experiment_facade_reexports_public_controllers() -> None:
    assert experiments.run_task1_experiment is cnn_experiments.run_task1_experiment
    assert (
        experiments.run_task1_classical_experiment
        is classical_experiments.run_task1_classical_experiment
    )
    assert experiments.write_task1_confusion_figure is plotting.write_task1_confusion_figure
```

- [ ] **Step 2: Run and confirm the missing module**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical_experiments.py tests/task1/test_experiments.py -q
```

Expected: collection fails for `classical_experiments.py`.

- [ ] **Step 3: Move the exact classical controller group**

Move all remaining controller symbols from `experiments.py`, including selection/result
dataclasses, fold-metric conversion, deterministic ranking, selection loading/validation,
final OOF evidence, atomic evidence writers, and `run_task1_classical_experiment`.

Keep its public signature unchanged:

```python
def run_task1_classical_experiment(
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    *,
    stage: Literal["smoke", "tune", "final"],
    registry: RunRegistry | None = None,
    root: str | Path = ROOT,
    result_root: str | Path = TASK1_RESULT_DIR / "classical",
    cache_root: str | Path = TASK1_HOG_CACHE_DIR,
    evidence_root: str | Path = TASK1_EVIDENCE_DIR,
    selection: Task1ClassicalSelection | None = None,
    fold_runner: Callable[..., Task1ClassicalFoldResult] = run_task1_classical_fold,
) -> Task1ClassicalExperimentResult:
```

Update selection implementation identity to hash `classical_experiments.py` and the
`_classical_implementation_sha256()` value from `classical_training.py`.

- [ ] **Step 4: Replace `experiments.py` with a compatibility facade**

The facade re-exports exactly:

```python
__all__ = [
    "Task1ExperimentResult",
    "Task1ClassicalSelection",
    "Task1ClassicalExperimentResult",
    "run_task1_experiment",
    "run_task1_classical_experiment",
    "write_task1_comparison_figure",
    "write_task1_confusion_figure",
]
```

Update `fashion.task1.__init__` to import from owning modules or the facade while preserving its
existing public names.

- [ ] **Step 5: Run all controller, facade, API, and lint tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_cnn_experiments.py tests/task1/test_classical_experiments.py tests/task1/test_plotting.py tests/task1/test_experiments.py tests/task1/test_evaluation.py -q
./.venv/bin/python -m ruff check src/fashion/task1/cnn_experiments.py src/fashion/task1/classical_experiments.py src/fashion/task1/plotting.py src/fashion/task1/experiments.py src/fashion/task1/__init__.py tests/task1/test_*experiments.py
```

Expected: every stage schedule, selection rule, evidence schema, and facade identity passes.

- [ ] **Step 6: Commit Task 8**

```bash
git add src/fashion/task1/classical_experiments.py src/fashion/task1/experiments.py src/fashion/task1/__init__.py tests/task1/test_classical_experiments.py tests/task1/test_experiments.py
git commit -m "refactor(task1): isolate classical experiment flow"
```

---

### Task 9: Rebuild Notebook 02 around EDA and problem evidence

**Files:**
- Modify: `tests/task1/test_notebook_baseline.py`
- Modify: `notebooks/02_task1_article_type.ipynb`
- Modify: `src/fashion/README.md`

**Interfaces:**
- Consumes: the EDA and failure-analysis helpers, both experiment controllers, generated evidence, and the shared public API.
- Produces: a Run-All-safe notebook with ordered sections and both controllers defaulting to smoke.
- Does not contain optimizer loops, estimator fitting, split creation, or hand-written result scores.

- [ ] **Step 1: Inspect and preserve the current notebook diff**

```bash
git diff -- notebooks/02_task1_article_type.ipynb
```

Expected: current user changes are visible, including the classical controller state. Use the
current working file as the edit base; do not restore it from Git.

- [ ] **Step 2: Strengthen the notebook contract test first**

Replace loose section checks with this ordered heading contract:

```python
expected_headings = [
    "## 1. Problem and output",
    "## 2. EDA evidence",
    "## 3. Safety contract",
    "## 4. Evaluation",
    "## 5. Candidate hypotheses",
    "## 6. Controlled preprocessing",
    "## 7. Run plan",
    "## 8. Run controllers",
    "## 9. Results",
    "## 10. Failure analysis",
    "## 11. Decision",
    "## 12. Final handoff",
]
positions = [source.index(heading) for heading in expected_headings]
assert positions == sorted(positions)
```

Require these tokens:

```python
required = (
    "build_task1_problem_profile",
    "build_task1_decision_evidence",
    "build_task1_weak_class_table",
    "build_task1_confusion_pairs",
    'RUN_MODE = "smoke"',
    'CLASSICAL_STAGE = "smoke"',
    "run_task1_experiment",
    "run_task1_classical_experiment",
    "macro-F1",
    "124",
    "HOG",
    "scratch CNN",
    "results/runs.csv",
)
assert all(token in source for token in required)
for forbidden in (
    "train_test_split", "pretrained=True", "optimizer.step()",
    "KNeighborsClassifier(", "LinearSVC(",
):
    assert forbidden not in source
```

Also assert the decision section contains the words `hypothesis`, `evidence`, `passed`,
`failed`, and `not ready` so it cannot become a generic model description.

- [ ] **Step 3: Run the notebook test and confirm heading/default failures**

```bash
./.venv/bin/python -m pytest tests/task1/test_notebook_baseline.py -q
```

Expected: failure because current headings differ and `CLASSICAL_STAGE` is `final`.

- [ ] **Step 4: Reorder the notebook into the twelve approved sections**

Edit the notebook JSON with `apply_patch`. Keep the notebook as the only artifact. The EDA code
cell must load existing prepared evidence and display it:

```python
from dataclasses import asdict

import pandas as pd

from fashion.config import DEVELOPMENT_CLASS_SUMMARY_CSV
from fashion.data.dataset import get_samples, load_label_maps, load_splits
from fashion.task1 import (
    build_task1_decision_evidence,
    build_task1_problem_profile,
)

TARGET = "articleType"
splits = load_splits()
task1_development = get_samples(splits, partition="development", target=TARGET)
article_type_map = load_label_maps()[TARGET]
article_type_classes = tuple(article_type_map["classes"])
class_summary = pd.read_csv(DEVELOPMENT_CLASS_SUMMARY_CSV, keep_default_na=False)
problem_profile = build_task1_problem_profile(splits, class_summary)

display(pd.DataFrame([asdict(problem_profile)]))
display(build_task1_decision_evidence(problem_profile))
```

The markdown beside this table must say that the choices are hypotheses to test, not declared
winners.

- [ ] **Step 5: Make both controller cells safe and clear**

Use exactly:

```python
RUN_MODE = "smoke"  # Change to "full" only for the ten registered CNN runs.
task1_experiment = run_task1_experiment(
    splits,
    article_type_map,
    mode=RUN_MODE,
)
display(task1_experiment.fold_metrics)
if not task1_experiment.comparison.empty:
    display(task1_experiment.comparison)
if not task1_experiment.oof_metrics.empty:
    display(task1_experiment.oof_metrics)
```

and:

```python
CLASSICAL_STAGE = "smoke"  # Then use "tune" and finally "final" in separate runs.
classic_experiment = run_task1_classical_experiment(
    splits,
    article_type_map,
    stage=CLASSICAL_STAGE,
)
display(classic_experiment.fold_metrics)
if not classic_experiment.tuning.empty:
    display(classic_experiment.tuning)
if not classic_experiment.comparison.empty:
    display(classic_experiment.comparison)
if not classic_experiment.oof_metrics.empty:
    display(classic_experiment.oof_metrics)
```

Generate report figures only inside full/final guards. Do not read or display missing final files
in smoke mode.

- [ ] **Step 6: Add conditional failure-evidence cells**

Combine in-memory evidence from completed full/final controllers and call reusable helpers:

```python
from fashion.task1 import (
    build_task1_confusion_pairs,
    build_task1_weak_class_table,
)

per_class_evidence = {
    **{f"cnn:{key}": value for key, value in task1_experiment.per_class.items()},
    **{f"classic:{key}": value for key, value in classic_experiment.per_class.items()},
}
oof_prediction_evidence = {
    **{f"cnn:{key}": value for key, value in task1_experiment.oof_predictions.items()},
    **{f"classic:{key}": value for key, value in classic_experiment.oof_predictions.items()},
}

if per_class_evidence:
    display(build_task1_weak_class_table(per_class_evidence, limit=10))
else:
    print("Weak-class evidence is not ready; complete full/final runs first.")

if oof_prediction_evidence:
    display(build_task1_confusion_pairs(oof_prediction_evidence, limit=10))
else:
    print("Confusion-pair evidence is not ready; complete full/final runs first.")
```

Explain that `example_ids` point back to prepared development rows for representative-image
inspection. Never choose or rank a model from these tables alone.

- [ ] **Step 7: Write evidence-led result, failure, and decision markdown**

Use these exact decision rules in prose:

```text
- The augmentation hypothesis passes only if five-fold mean macro-F1 improves without an
  unacceptable increase in fold standard deviation.
- A classic model earns consideration only from complete selected five-fold evidence, not its
  fold-0 tuning score.
- The final model is not chosen from accuracy alone; use macro-F1, fold stability, per-class
  failures, confusion pairs, and practical limitations.
- If evidence files are absent, the handoff says NOT READY and names the missing run stage.
```

Do not copy the current `0.5265`, `0.5218`, `0.5023`, or `0.4889` values into markdown. The fresh
run tables are the only source of new conclusions.

- [ ] **Step 8: Update the package README module map**

Document each new Task 1 module in one line and identify `classical.py` and `experiments.py` as
compatibility facades. State that Notebook 02 defaults both controller families to smoke.

- [ ] **Step 9: Validate notebook, documentation, and lint**

```bash
./.venv/bin/python -m pytest tests/task1/test_notebook_baseline.py tests/test_documentation.py -q
./.venv/bin/python -m ruff check tests/task1/test_notebook_baseline.py
```

Expected: notebook validates with `nbformat`, headings are ordered, safe defaults are present,
and forbidden implementation tokens are absent.

- [ ] **Step 10: Commit Task 9**

```bash
git add tests/task1/test_notebook_baseline.py notebooks/02_task1_article_type.ipynb src/fashion/README.md
git commit -m "docs(task1): make notebook evidence led"
```

---

### Task 10: Verify, clear old Task 1 outputs, and run fresh smoke checks

**Files:**
- Modify only refactor files when verification exposes a refactor regression.
- Remove generated Task 1 paths only after all code tests pass.
- Atomically rewrite: `results/runs.csv` while preserving every non-Task-1 row.

**Interfaces:**
- Consumes: all Tasks 1-9.
- Produces: verified code, a clean Task 1 result state, one fresh CNN smoke run, one fresh classical smoke stage, and exact commands for the user's later full runs.

- [ ] **Step 1: Run the focused Task 1 suite**

```bash
./.venv/bin/python -m pytest tests/task1 -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 2: Run the complete project suite**

```bash
./.venv/bin/python -m pytest -q
```

Expected: all project tests pass. Fix only failures caused by this refactor.

- [ ] **Step 3: Run Ruff and notebook validation**

```bash
./.venv/bin/python -m ruff check src/fashion/task1 tests/task1
./.venv/bin/python -c "import nbformat; p='notebooks/02_task1_article_type.ipynb'; n=nbformat.read(p, as_version=4); nbformat.validate(n); print('Notebook 02 valid')"
```

Expected: Ruff prints `All checks passed!` and notebook validation prints
`Notebook 02 valid`.

- [ ] **Step 4: Review the exact generated cleanup targets**

Run this PowerShell check before deletion:

```powershell
$projectRoot = (Resolve-Path '.').Path
$task1Targets = @(
    (Join-Path $projectRoot 'results\task1'),
    (Join-Path $projectRoot 'results\evidence\task1'),
    (Join-Path $projectRoot 'results\figures\task1'),
    (Join-Path $projectRoot 'data\processed\task1_hog_cache')
)
foreach ($target in $task1Targets) {
    if (Test-Path -LiteralPath $target) {
        $resolved = (Resolve-Path -LiteralPath $target).Path
        if (-not $resolved.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar)) {
            throw "Refusing out-of-project cleanup target: $resolved"
        }
        Write-Output $resolved
    }
}
```

Expected: every printed path is one of the four exact Task 1 paths above and is inside the
project root.

- [ ] **Step 5: Remove only generated Task 1 directories**

After visually checking Step 4 output, resolve the same four literal targets again in one
PowerShell session and delete only after the guard passes:

```powershell
$projectRoot = (Resolve-Path '.').Path
$task1Targets = @(
    (Join-Path $projectRoot 'results\task1'),
    (Join-Path $projectRoot 'results\evidence\task1'),
    (Join-Path $projectRoot 'results\figures\task1'),
    (Join-Path $projectRoot 'data\processed\task1_hog_cache')
)
foreach ($target in $task1Targets) {
    if (Test-Path -LiteralPath $target) {
        $resolved = (Resolve-Path -LiteralPath $target).Path
        if (-not $resolved.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar)) {
            throw "Refusing out-of-project cleanup target: $resolved"
        }
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
}
```

Expected: only Task 1 model artifacts, evidence, figures, and HOG cache are removed. These files
are generated and will be recreated by the new runs.

- [ ] **Step 6: Atomically remove old Task 1 registry rows**

```bash
./.venv/bin/python -c "from pathlib import Path; import pandas as pd; p=Path('results/runs.csv'); rows=pd.read_csv(p, keep_default_na=False); kept=rows.loc[~rows['task'].eq('task1')]; t=p.with_name('runs.csv.task1-reset.tmp'); kept.to_csv(t, index=False); t.replace(p); print(f'removed {len(rows)-len(kept)} Task 1 rows; kept {len(kept)} other rows')"
```

Then verify:

```bash
./.venv/bin/python -c "import pandas as pd; r=pd.read_csv('results/runs.csv', keep_default_na=False); assert not r['task'].eq('task1').any(); print('Task 1 registry rows cleared')"
```

Expected: the first command reports removed Task 1 rows and the second prints confirmation.

- [ ] **Step 7: Run one real CNN smoke run**

```bash
./.venv/bin/python -c "from fashion.data.dataset import load_label_maps, load_splits; from fashion.task1 import run_task1_experiment; r=run_task1_experiment(load_splits(), load_label_maps()['articleType'], mode='smoke'); print(r.fold_metrics[['run_id','fold','macro_f1']].to_string(index=False))"
```

Expected: one completed fold-0 CNN smoke row prints and its registry row has
`final_eligible=false`.

- [ ] **Step 8: Run one real classical smoke stage**

```bash
./.venv/bin/python -c "from fashion.data.dataset import load_label_maps, load_splits; from fashion.task1 import run_task1_classical_experiment; r=run_task1_classical_experiment(load_splits(), load_label_maps()['articleType'], stage='smoke'); print(r.fold_metrics[['run_id','candidate_id','fold','macro_f1']].to_string(index=False))"
```

Expected: two completed fold-0 rows print, one default KNN and one default Linear SVM; both have
`final_eligible=false`.

- [ ] **Step 9: Verify new smoke artifacts and registry rows**

```bash
./.venv/bin/python -c "from fashion.train.registry import RunRegistry; r=RunRegistry().find(task='task1', stage='smoke', status='completed'); assert len(r)>=3; assert r['final_eligible'].astype(str).str.lower().eq('false').all(); assert r['prediction_path'].astype(str).str.strip().ne('').all(); print(r[['run_id','model_family','fold','primary_metric_value']].to_string(index=False))"
```

Expected: at least three fresh smoke rows print and every assertion passes.

- [ ] **Step 10: Commit any verification-only correction**

If verification required no source correction, do not create an empty commit. If it found a
refactor bug, stage only the corrected refactor files and commit:

```bash
git commit -m "fix(task1): complete refactor verification"
```

Do not commit generated Task 1 model artifacts, cache, evidence, figures, or smoke registry rows
unless the repository already tracks a required generated file and project policy explicitly
requires the update.

- [ ] **Step 11: Hand off the fresh full-run sequence**

Tell the user to run Notebook 02 in this order:

```python
RUN_MODE = "full"
```

Then:

```python
CLASSICAL_STAGE = "tune"
```

After `results/evidence/task1/classical_selection.json` exists:

```python
CLASSICAL_STAGE = "final"
```

Finally return both settings to `smoke` before committing the notebook. State that the final
decision and error-analysis markdown must be updated from the newly generated comparison,
per-class, and confusion evidence rather than copied from the old run.

## Completion Check

Before claiming completion, run:

```bash
git status --short
git log --oneline -12
```

Report only the refactor commits and files. Explicitly state whether the full runs remain for the
user; never claim they completed from smoke evidence.
