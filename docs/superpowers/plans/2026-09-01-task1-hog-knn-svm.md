# Task 1 HOG + KNN/SVM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible grayscale HOG + KNN and HOG + Linear SVM Task 1 candidates with fold-0 tuning, registered five-fold evaluation, and notebook evidence.

**Architecture:** `fashion.task1.classical` owns deterministic HOG features, validated caches, scikit-learn model adapters, and one registered fold evaluation. `fashion.task1.experiments` adds a staged controller that selects one shared HOG setting on fold 0, tunes KNN and Linear SVM, and evaluates both winners through the existing fixed-124-class framework. Notebook 02 remains narrative and calls the reusable controller.

**Tech Stack:** Python 3.12, NumPy, Pandas, Pillow, scikit-image 0.26.0, scikit-learn 1.9.0, pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-09-01-task1-hog-knn-svm-design.md`

## Global Constraints

- Use `./.venv/bin/python` for every Python command.
- Load `data/processed/splits.csv` through `fashion.data.dataset.load_splits()`; never call `train_test_split`.
- Use only labelled development rows and the fixed folds 0, 1, 2, 3, and 4.
- Never use holdout, quarantine, or official prediction images for tuning.
- Preserve all 124 `articleType` score columns even when a training fold lacks classes.
- Register every evaluated configuration/fold through `fashion.train.registry`.
- Mark smoke runs `final_eligible=False`; mark complete canonical-fold runs `final_eligible=True`.
- Set `benchmark_only=False` and `scratch=True`; these models use no pretrained weights.
- Use the shared metrics only: `macro_f1`, `weighted_f1`, `top1_accuracy`, and `top5_accuracy`.
- Rank by the mean of five fold-level `macro_f1` values; pooled OOF scores are supporting evidence.
- Do not add PCA, raw-pixel experiments, RGB/HSV HOG, RBF SVM, probability calibration, augmentation, or `StandardScaler`.
- Do not compare speed or model size in the report. Registry runtime fields remain allowed.
- Preserve unrelated notebook and worktree changes.

## File Structure

- Create `src/fashion/task1/classical.py`: HOG specs, extraction, cache, KNN/SVM configs, fixed-class scores, and one-fold registered execution.
- Create `tests/task1/test_classical.py`: focused tests for all classic feature/model/fold contracts.
- Modify `src/fashion/config.py`: add the ignored HOG cache path.
- Modify `src/fashion/task1/experiments.py`: add staged classic tuning and final-evidence orchestration without changing the CNN runner.
- Modify `tests/task1/test_experiments.py`: add schedule, selection, resume, and five-fold evidence tests.
- Modify `src/fashion/task1/__init__.py`: export the small public classic API used by Notebook 02.
- Modify `tests/task1/test_notebook_baseline.py`: freeze the classic notebook contract.
- Modify `notebooks/02_task1_article_type.ipynb`: add the classic comparison narrative and runner call.
- Modify `src/fashion/README.md`: document the new Task 1 module responsibility.
- Modify `pyproject.toml` and `requirements/constraints-py312.txt`: pin `scikit-image==0.26.0`.

---

### Task 1: Deterministic grayscale HOG extraction

**Files:**
- Create: `src/fashion/task1/classical.py`
- Create: `tests/task1/test_classical.py`
- Modify: `src/fashion/config.py:28-58`
- Modify: `pyproject.toml:10-19`
- Modify: `requirements/constraints-py312.txt`

**Interfaces:**
- Consumes: `fashion.data.images.transform_image_with_mask(image, image_size, pad_color, normalize_range)`.
- Produces: `Task1HogSpec`, `TASK1_HOG_COARSE`, `TASK1_HOG_FINE`, and `extract_task1_hog(path, spec) -> np.ndarray`.

- [ ] **Step 1: Write failing HOG contract tests**

Add this base to `tests/task1/test_classical.py`:

```python
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fashion.task1.classical import (
    TASK1_HOG_COARSE,
    TASK1_HOG_FINE,
    Task1HogSpec,
    extract_task1_hog,
)


def test_hog_specs_have_fixed_ids_and_feature_lengths() -> None:
    assert TASK1_HOG_COARSE.hog_id == "task1_gray_hog_ppc16_v1"
    assert TASK1_HOG_COARSE.expected_features == 288
    assert TASK1_HOG_FINE.hog_id == "task1_gray_hog_ppc10_v1"
    assert TASK1_HOG_FINE.expected_features == 1260


@pytest.mark.parametrize("spec", [TASK1_HOG_COARSE, TASK1_HOG_FINE])
def test_extract_task1_hog_is_float32_deterministic_and_fixed_width(
    tmp_path: Path, spec: Task1HogSpec
) -> None:
    pixels = np.zeros((80, 60, 3), dtype=np.uint8)
    pixels[20:60, 15:45] = (220, 80, 40)
    path = tmp_path / "sample.png"
    Image.fromarray(pixels, mode="RGB").save(path)

    first = extract_task1_hog(path, spec)
    second = extract_task1_hog(path, spec)

    assert first.shape == (spec.expected_features,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_hog_spec_rejects_unapproved_geometry() -> None:
    with pytest.raises(ValueError, match="expected feature count"):
        Task1HogSpec(
            hog_id="bad",
            pixels_per_cell=(8, 8),
            expected_features=1,
        )
```

- [ ] **Step 2: Run the tests and confirm the module is missing**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_classical.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'fashion.task1.classical'`.

- [ ] **Step 3: Pin scikit-image and add the cache path**

Add `"scikit-image==0.26.0",` beside scikit-learn in `pyproject.toml`. Add this exact constraint to `requirements/constraints-py312.txt`:

```text
scikit-image==0.26.0
```

Add this path in `src/fashion/config.py` after `TASK1_EVIDENCE_DIR`:

```python
TASK1_HOG_CACHE_DIR = PROCESSED_DATA_DIR / "task1_hog_cache"
```

- [ ] **Step 4: Add the frozen HOG specification and extractor**

Create `src/fashion/task1/classical.py` with these public values and validation rules:

```python
"""Classic HOG feature and model candidates for Task 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

from fashion.data.images import transform_image_with_mask


@dataclass(frozen=True)
class Task1HogSpec:
    hog_id: str
    pixels_per_cell: tuple[int, int]
    expected_features: int
    orientations: int = 9
    cells_per_block: tuple[int, int] = (2, 2)
    block_norm: str = "L2-Hys"
    transform_sqrt: bool = True
    image_size: tuple[int, int] = (80, 60)
    pad_color: tuple[int, int, int] = (255, 255, 255)

    def __post_init__(self) -> None:
        if not self.hog_id.strip():
            raise ValueError("hog_id must not be blank")
        height, width = self.image_size
        cell_y, cell_x = self.pixels_per_cell
        block_y, block_x = self.cells_per_block
        if min(height, width, cell_y, cell_x, block_y, block_x, self.orientations) <= 0:
            raise ValueError("HOG geometry values must be positive")
        cells_y, cells_x = height // cell_y, width // cell_x
        blocks_y, blocks_x = cells_y - block_y + 1, cells_x - block_x + 1
        calculated = blocks_y * blocks_x * block_y * block_x * self.orientations
        if blocks_y <= 0 or blocks_x <= 0 or calculated != self.expected_features:
            raise ValueError("HOG geometry does not match expected feature count")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


TASK1_HOG_COARSE = Task1HogSpec(
    hog_id="task1_gray_hog_ppc16_v1",
    pixels_per_cell=(16, 16),
    expected_features=288,
)
TASK1_HOG_FINE = Task1HogSpec(
    hog_id="task1_gray_hog_ppc10_v1",
    pixels_per_cell=(10, 10),
    expected_features=1260,
)
TASK1_HOG_SPECS = (TASK1_HOG_COARSE, TASK1_HOG_FINE)


def extract_task1_hog(path: str | Path, spec: Task1HogSpec) -> np.ndarray:
    image_path = Path(path)
    try:
        with Image.open(image_path) as source:
            rgb, _ = transform_image_with_mask(
                source,
                image_size=spec.image_size,
                pad_color=spec.pad_color,
                normalize_range=True,
            )
    except Exception as error:
        raise ValueError(f"cannot extract HOG from {image_path}") from error
    grayscale = rgb2gray(np.asarray(rgb, dtype=np.float32))
    features = hog(
        grayscale,
        orientations=spec.orientations,
        pixels_per_cell=spec.pixels_per_cell,
        cells_per_block=spec.cells_per_block,
        block_norm=spec.block_norm,
        transform_sqrt=spec.transform_sqrt,
        feature_vector=True,
        channel_axis=None,
    ).astype(np.float32, copy=False)
    if features.shape != (spec.expected_features,) or not np.isfinite(features).all():
        raise ValueError(f"invalid HOG feature vector for {image_path}")
    return features
```

- [ ] **Step 5: Run focused tests and lint**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_classical.py -q
./.venv/bin/python -m ruff check src/fashion/task1/classical.py tests/task1/test_classical.py src/fashion/config.py
```

Expected: all tests pass and Ruff prints `All checks passed!`.

- [ ] **Step 6: Commit Task 1**

```bash
git add pyproject.toml requirements/constraints-py312.txt src/fashion/config.py src/fashion/task1/classical.py tests/task1/test_classical.py
git commit -m "feat(task1): add deterministic HOG features"
```

---

### Task 2: Validated atomic HOG cache

**Files:**
- Modify: `src/fashion/task1/classical.py`
- Modify: `tests/task1/test_classical.py`

**Interfaces:**
- Consumes: `Task1HogSpec`, `extract_task1_hog`, `atomic_write_bytes`, and canonical hashing helpers.
- Produces: `Task1HogFeatureSet` and `load_or_build_task1_hog_features(rows, spec, root, cache_root, split_sha256) -> Task1HogFeatureSet`.

- [ ] **Step 1: Add failing cache tests**

Append tests that use a fake extractor so they do not depend on image decoding:

```python
import pandas as pd

from fashion.task1.classical import load_or_build_task1_hog_features


def _cache_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [11, 12],
            "path": ["images/11.jpg", "images/12.jpg"],
            "sha256": ["a" * 64, "b" * 64],
            "partition": ["development", "development"],
            "articleType": ["class-000", "class-001"],
        }
    )


def test_hog_cache_reuses_valid_ordered_float32_features(tmp_path: Path) -> None:
    calls: list[int] = []

    def extractor(path: Path, spec: Task1HogSpec) -> np.ndarray:
        calls.append(int(path.stem))
        return np.full(spec.expected_features, int(path.stem), dtype=np.float32)

    first = load_or_build_task1_hog_features(
        _cache_rows(), TASK1_HOG_COARSE, root=tmp_path,
        cache_root=tmp_path / "cache", split_sha256="c" * 64,
        extractor=extractor,
    )
    second = load_or_build_task1_hog_features(
        _cache_rows(), TASK1_HOG_COARSE, root=tmp_path,
        cache_root=tmp_path / "cache", split_sha256="c" * 64,
        extractor=lambda *_: pytest.fail("valid cache should be reused"),
    )

    assert calls == [11, 12]
    np.testing.assert_array_equal(first.ids, np.array([11, 12]))
    np.testing.assert_array_equal(first.features, second.features)
    assert second.features.dtype == np.float32


def test_hog_cache_rebuilds_when_source_inventory_changes(tmp_path: Path) -> None:
    rows = _cache_rows()
    load_or_build_task1_hog_features(
        rows, TASK1_HOG_COARSE, root=tmp_path,
        cache_root=tmp_path / "cache", split_sha256="c" * 64,
        extractor=lambda *_: np.zeros(288, dtype=np.float32),
    )
    changed = rows.copy()
    changed.loc[0, "sha256"] = "d" * 64
    calls = 0

    def replacement(*_: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.ones(288, dtype=np.float32)

    rebuilt = load_or_build_task1_hog_features(
        changed, TASK1_HOG_COARSE, root=tmp_path,
        cache_root=tmp_path / "cache", split_sha256="c" * 64,
        extractor=replacement,
    )
    assert calls == 2
    assert np.all(rebuilt.features == 1.0)


def test_hog_cache_rejects_non_development_or_duplicate_rows(tmp_path: Path) -> None:
    rows = _cache_rows()
    rows.loc[1, "id"] = 11
    with pytest.raises(ValueError, match="unique development rows"):
        load_or_build_task1_hog_features(
            rows, TASK1_HOG_COARSE, root=tmp_path,
            cache_root=tmp_path / "cache", split_sha256="c" * 64,
        )
```

- [ ] **Step 2: Run only the new cache tests and confirm failure**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical.py -k "hog_cache" -q
```

Expected: import or attribute failures for `load_or_build_task1_hog_features`.

- [ ] **Step 3: Implement one-file NPZ cache identity and validation**

Add these exact interfaces to `classical.py`:

```python
import io
import json
from collections.abc import Callable

import pandas as pd

from fashion.config import ROOT, TASK1_HOG_CACHE_DIR
from fashion.train.artifacts import atomic_write_bytes, canonical_json_bytes, canonical_sha256

HOG_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Task1HogFeatureSet:
    ids: np.ndarray
    features: np.ndarray
    spec: Task1HogSpec
    cache_path: Path


def _hog_inventory(rows: pd.DataFrame) -> list[dict[str, object]]:
    required = {"id", "path", "sha256", "partition"}
    if required.difference(rows.columns):
        raise ValueError("HOG rows are missing required inventory columns")
    if rows.empty or not rows["partition"].eq("development").all():
        raise ValueError("HOG cache requires unique development rows")
    if rows["id"].isna().any() or rows["id"].duplicated().any():
        raise ValueError("HOG cache requires unique development rows")
    ordered = rows.sort_values("id", kind="stable")
    return ordered.loc[:, ["id", "path", "sha256"]].to_dict(orient="records")


def load_or_build_task1_hog_features(
    rows: pd.DataFrame,
    spec: Task1HogSpec,
    *,
    root: str | Path = ROOT,
    cache_root: str | Path = TASK1_HOG_CACHE_DIR,
    split_sha256: str,
    extractor: Callable[[Path, Task1HogSpec], np.ndarray] = extract_task1_hog,
) -> Task1HogFeatureSet:
    inventory = _hog_inventory(rows)
    identity = {
        "schema_version": HOG_CACHE_SCHEMA_VERSION,
        "split_sha256": split_sha256,
        "inventory_sha256": canonical_sha256(inventory),
        "hog": spec.to_dict(),
    }
    cache_path = Path(cache_root) / f"{spec.hog_id}-{canonical_sha256(identity)[:16]}.npz"
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as stored:
                ids = stored["ids"]
                features = stored["features"]
                metadata = json.loads(bytes(stored["metadata"].tolist()).decode("utf-8"))
            expected_ids = np.asarray([int(row["id"]) for row in inventory], dtype=np.int64)
            typed = ids.dtype == np.int64 and features.dtype == np.float32
            if typed and metadata == identity and np.array_equal(ids, expected_ids):
                if features.shape == (len(ids), spec.expected_features) and np.isfinite(features).all():
                    return Task1HogFeatureSet(ids, features, spec, cache_path)
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass

    ordered = rows.sort_values("id", kind="stable")
    project_root = Path(root)
    ids = ordered["id"].to_numpy(dtype=np.int64)
    vectors = [extractor(project_root / str(row.path), spec) for row in ordered.itertuples()]
    features = np.stack(vectors).astype(np.float32, copy=False)
    metadata_bytes = canonical_json_bytes(identity)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        ids=ids,
        features=features,
        metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
    )
    atomic_write_bytes(cache_path, buffer.getvalue())
    return Task1HogFeatureSet(ids, features, spec, cache_path)
```

Keep the validation strict: malformed files rebuild; only a complete matching cache returns.

- [ ] **Step 4: Add interrupted-write coverage**

Monkeypatch `classical.atomic_write_bytes` to raise `RuntimeError("write stopped")`, call the builder, and assert that `cache_root.glob("*.npz")` is empty. This proves the existing atomic writer leaves no final cache file on interruption.

- [ ] **Step 5: Run Task 2 tests and lint**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical.py -k "hog_cache or extract_task1_hog or hog_specs" -q
./.venv/bin/python -m ruff check src/fashion/task1/classical.py tests/task1/test_classical.py
```

Expected: all selected tests pass and Ruff passes.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/fashion/task1/classical.py tests/task1/test_classical.py
git commit -m "feat(task1): cache validated HOG features"
```

---

### Task 3: KNN and Linear SVM fixed-class adapters

**Files:**
- Modify: `src/fashion/task1/classical.py`
- Modify: `tests/task1/test_classical.py`

**Interfaces:**
- Consumes: NumPy feature matrices and integer labels in the global Task 1 label map.
- Produces: `Task1KNNConfig`, `Task1LinearSVMConfig`, `fit_predict_task1_knn(...)`, `fit_predict_task1_linear_svm(...)`, `query_task1_neighbors(...)`, and `knn_probabilities_from_neighbors(...)`.

- [ ] **Step 1: Add failing model configuration and fixed-class tests**

Append:

```python
from sklearn.neighbors import KNeighborsClassifier

from fashion.task1.classical import (
    Task1KNNConfig,
    Task1LinearSVMConfig,
    fit_predict_task1_knn,
    fit_predict_task1_linear_svm,
    knn_probabilities_from_neighbors,
    query_task1_neighbors,
)


def _toy_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.array([[0, 0], [0, 1], [4, 4], [4, 5], [8, 8], [8, 9]], dtype=np.float32)
    y_train = np.array([2, 2, 7, 7, 19, 19], dtype=np.int64)
    x_validation = np.array([[0, 0.2], [4, 4.2], [8, 8.2]], dtype=np.float32)
    return x_train, y_train, x_validation


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_scores_expand_missing_classes_and_sum_to_one(weights: str) -> None:
    x_train, y_train, x_validation = _toy_features()
    config = Task1KNNConfig(n_neighbors=3, weights=weights)
    _, probabilities = fit_predict_task1_knn(x_train, y_train, x_validation, config)
    assert probabilities.shape == (3, 124)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[:, [0, 1, 3, 6, 8, 18, 20]] == 0.0)


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_linear_svm_scores_expand_missing_classes_and_are_finite(
    class_weight: str | None,
) -> None:
    x_train, y_train, x_validation = _toy_features()
    config = Task1LinearSVMConfig(C=1.0, class_weight=class_weight)
    _, probabilities = fit_predict_task1_linear_svm(x_train, y_train, x_validation, config)
    assert probabilities.shape == (3, 124)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities[:, [0, 1, 3, 6, 8, 18, 20]] == 0.0)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("k", [3, 5])
def test_reused_neighbour_votes_match_sklearn(weights: str, k: int) -> None:
    x_train, y_train, x_validation = _toy_features()
    distances, indexes = query_task1_neighbors(x_train, y_train, x_validation, max_k=5)
    reused = knn_probabilities_from_neighbors(
        distances, indexes, y_train, n_neighbors=k, weights=weights
    )
    model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric="euclidean")
    expected_local = model.fit(x_train, y_train).predict_proba(x_validation)
    expected = np.zeros((len(x_validation), 124))
    expected[:, model.classes_.astype(int)] = expected_local
    np.testing.assert_allclose(reused, expected)
```

- [ ] **Step 2: Run the model tests and confirm missing interfaces**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical.py -k "knn or svm or neighbour" -q
```

Expected: import failures for the new model adapters.

- [ ] **Step 3: Add validated model configuration types**

Add:

```python
from typing import Literal, TypeAlias


@dataclass(frozen=True)
class Task1KNNConfig:
    n_neighbors: Literal[3, 5, 11]
    weights: Literal["uniform", "distance"]
    metric: Literal["euclidean"] = "euclidean"

    def __post_init__(self) -> None:
        if self.n_neighbors not in {3, 5, 11}:
            raise ValueError("KNN neighbours must be one of 3, 5, or 11")
        if self.weights not in {"uniform", "distance"} or self.metric != "euclidean":
            raise ValueError("KNN must use approved voting and Euclidean distance")

    @property
    def config_id(self) -> str:
        return f"knn-k{self.n_neighbors}-{self.weights}"


@dataclass(frozen=True)
class Task1LinearSVMConfig:
    C: Literal[0.1, 1.0, 10.0]
    class_weight: Literal["balanced"] | None
    random_state: int = 2753
    dual: Literal["auto"] = "auto"
    tol: float = 1e-4
    max_iter: int = 5000

    def __post_init__(self) -> None:
        if self.C not in {0.1, 1.0, 10.0}:
            raise ValueError("Linear SVM C must be one of 0.1, 1, or 10")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("Linear SVM class_weight must be normal or balanced")
        if self.random_state != 2753 or self.dual != "auto":
            raise ValueError("Linear SVM reproducibility settings are fixed")
        if self.tol != 1e-4 or self.max_iter != 5000:
            raise ValueError("Linear SVM convergence settings are fixed")

    @property
    def config_id(self) -> str:
        weight = "balanced" if self.class_weight else "normal"
        return f"linear-svm-c{self.C:g}-{weight}"


Task1ClassicalModelConfig: TypeAlias = Task1KNNConfig | Task1LinearSVMConfig
```

Expose fixed grids:

```python
TASK1_KNN_GRID = tuple(
    Task1KNNConfig(k, weights)
    for k in (3, 5, 11)
    for weights in ("uniform", "distance")
)
TASK1_SVM_GRID = tuple(
    Task1LinearSVMConfig(C, class_weight)
    for C in (0.1, 1.0, 10.0)
    for class_weight in (None, "balanced")
)
TASK1_DEFAULT_KNN = Task1KNNConfig(5, "distance")
TASK1_DEFAULT_SVM = Task1LinearSVMConfig(1.0, None)
```

- [ ] **Step 4: Implement finite fixed-class score helpers**

Use `TASK1_NUM_CLASSES` from `fashion.task1.evaluation`. Implement `_expand_observed_scores(observed_scores, observed_classes)` so it rejects non-integer/out-of-range/duplicate classes, writes observed columns into a zero-filled `(rows, 124)` matrix, requires finite values, and checks row sums for probability inputs.

For SVM, apply stable softmax before expansion:

```python
def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("SVM decision scores must be a finite matrix")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
```

Construct models exactly as follows:

```python
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import warnings


def fit_predict_task1_knn(x_train, y_train, x_validation, config):
    model = KNeighborsClassifier(
        n_neighbors=config.n_neighbors,
        weights=config.weights,
        metric=config.metric,
        algorithm="brute",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    distances, indexes = query_task1_neighbors(
        x_train, y_train, x_validation,
        max_k=config.n_neighbors,
        batch_size=512,
    )
    probabilities = knn_probabilities_from_neighbors(
        distances, indexes, y_train,
        n_neighbors=config.n_neighbors,
        weights=config.weights,
    )
    return model, probabilities


def fit_predict_task1_linear_svm(x_train, y_train, x_validation, config):
    model = LinearSVC(
        C=config.C,
        class_weight=config.class_weight,
        random_state=config.random_state,
        dual=config.dual,
        tol=config.tol,
        max_iter=config.max_iter,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, y_train)
    decision = np.asarray(model.decision_function(x_validation), dtype=np.float64)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    observed = _stable_softmax(decision)
    return model, _expand_observed_scores(observed, model.classes_)
```

Implement `query_task1_neighbors` in validation batches of 512. Implement `knn_probabilities_from_neighbors` with scikit-learn's zero-distance rule: when any selected distance is zero, distance voting gives equal weight only to zero-distance neighbours; otherwise use `1 / distance`. Accumulate votes with `np.add.at`, then divide every row by its vote sum.

- [ ] **Step 5: Run adapter tests and the existing evaluation tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical.py -k "knn or svm or neighbour" -q
./.venv/bin/python -m pytest tests/task1/test_evaluation.py -q
./.venv/bin/python -m ruff check src/fashion/task1/classical.py tests/task1/test_classical.py
```

Expected: all commands pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/fashion/task1/classical.py tests/task1/test_classical.py
git commit -m "feat(task1): add KNN and linear SVM adapters"
```

---

### Task 4: One registered classic fold run and safe resume

**Files:**
- Modify: `src/fashion/task1/classical.py`
- Modify: `tests/task1/test_classical.py`

**Interfaces:**
- Consumes: `get_task1_fold_rows`, HOG feature sets, model adapters, shared evaluation helpers, and the run registry.
- Produces: `Task1ClassicalRunConfig`, `Task1ClassicalFoldResult`, and `run_task1_classical_fold(...) -> Task1ClassicalFoldResult`.

- [ ] **Step 1: Add a small image-backed split fixture and failing fold-run test**

Reuse the 40-row image fixture shape from `tests/task1/test_training.py`, but keep it local to `test_classical.py`. Add:

```python
from fashion.task1.classical import (
    Task1ClassicalRunConfig,
    run_task1_classical_fold,
)
from fashion.train.registry import RunRegistry


def test_classical_smoke_fold_writes_124_scores_and_registry_row(tmp_path: Path) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")

    result = run_task1_classical_fold(
        splits,
        _label_map(),
        validation_fold=0,
        hog_spec=TASK1_HOG_COARSE,
        model_config=Task1KNNConfig(3, "distance"),
        run_config=Task1ClassicalRunConfig.smoke(),
        registry=registry,
        root=tmp_path,
        result_root=tmp_path / "results",
        cache_root=tmp_path / "cache",
        split_path=split_path,
    )

    predictions = pd.read_csv(result.prediction_path)
    assert result.status == "completed"
    assert len([name for name in predictions if name.startswith("prob_")]) == 124
    assert len(predictions) == 8
    row = registry.read().iloc[0]
    assert row["task"] == "task1"
    assert row["model_family"] == "task1_hog_knn_v1"
    assert row["benchmark_only"] == "false"
    assert row["scratch"] == "true"
    assert row["final_eligible"] == "false"
    assert row["primary_metric_name"] == "macro_f1_124"
```

Add a second call with the same arguments and assert it returns the first `run_id` and leaves one registry row. Add a failure test by monkeypatching `classical.fit_predict_task1_knn` to raise `RuntimeError("classic model exploded")`; assert the registry row becomes `failed`.

- [ ] **Step 2: Run the fold tests and confirm the interfaces are absent**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical.py -k "classical_smoke_fold or classic_model_exploded or resumes" -q
```

Expected: import or attribute failures for the fold-run API.

- [ ] **Step 3: Add run and result dataclasses**

```python
from typing import Literal


@dataclass(frozen=True)
class Task1ClassicalRunConfig:
    stage: Literal["smoke", "experiment"]
    final_eligible: bool
    seed: int = 2753
    validation_batch_size: int = 512

    @classmethod
    def smoke(cls) -> "Task1ClassicalRunConfig":
        return cls(stage="smoke", final_eligible=False)

    @classmethod
    def full(cls) -> "Task1ClassicalRunConfig":
        return cls(stage="experiment", final_eligible=True)


@dataclass(frozen=True)
class Task1ClassicalFoldResult:
    run_id: str
    fold: int
    candidate_id: str
    hog_id: str
    model_family: str
    status: Literal["completed"]
    metrics: dict[str, float]
    model_path: Path
    prediction_path: Path
```

Build candidate IDs with `f"{hog_spec.hog_id}-{model_config.config_id}"`. Model families are exactly `task1_hog_knn_v1` and `task1_hog_linear_svm_v1`.

- [ ] **Step 4: Implement canonical validation, aligned fold matrices, and artifacts**

`run_task1_classical_fold` must:

1. Reject folds outside `range(5)`.
2. For `final_eligible=True`, require `stage="experiment"`, seed 2753, `split_path.resolve() == SPLITS_CSV.resolve()`, and `splits.equals(load_splits(split_path))`.
3. Validate the 124-class label map using the existing `_validate_label_map` contract; move that helper from `training.py` to `evaluation.py` as `validate_task1_label_map()` and update CNN imports/tests in the same commit so there is one implementation.
4. Use `get_task1_fold_rows()` for train and validation rows.
5. In smoke mode only, keep the first 32 training rows and all available validation rows up to 32; do not use the persistent full cache.
6. In full mode, build/load one development-wide HOG cache and align rows by product ID.
7. Fit the chosen model and create fixed scores.
8. Call `classification_metrics()` and `build_prediction_frame()`.
9. Atomically write `model.pkl` and `predictions.csv` below `result_root / run_id`.
10. Fill the registry metric and artifact hash fields before leaving `tracked_run`.

Use `pickle.dumps` with `atomic_write_bytes` for the scikit-learn model bundle:

```python
model_bundle = {
    "model": model,
    "hog": hog_spec.to_dict(),
    "model_config": asdict(model_config),
    "class_names": class_names,
    "validation_fold": validation_fold,
    "seed": run_config.seed,
    "metrics": metrics,
}
atomic_write_bytes(model_path, pickle.dumps(model_bundle, protocol=pickle.HIGHEST_PROTOCOL))
atomic_write_csv(prediction_path, predictions)
```

Use `RunRecord` with `loss_id="not_applicable"`, no epoch fields, and no parameter count. Hash `classical.py`, `dataset.py`, and `evaluation.py` for `implementation_sha256`.

- [ ] **Step 5: Implement exact completed-result reuse**

Before `new_run_id`, calculate the config hash and query:

```python
matches = active_registry.find(
    task="task1",
    experiment_id=experiment_id,
    fold=validation_fold,
    config_sha256=config_sha256,
    status="completed",
)
```

For the newest matching row, resolve project-relative artifact paths, verify both SHA-256 hashes with `verify_artifact`, read predictions, rerun `classification_metrics`, and require it to match registry metrics. Return the reconstructed result only when all checks pass. If verification fails, create a new append-once run; never edit the old row.

- [ ] **Step 6: Run fold, registry, CNN regression, and lint tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_classical.py tests/task1/test_training.py tests/task1/test_evaluation.py -q
./.venv/bin/python -m ruff check src/fashion/task1 tests/task1/test_classical.py tests/task1/test_training.py
```

Expected: all tests and lint pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add src/fashion/task1/classical.py src/fashion/task1/evaluation.py src/fashion/task1/training.py tests/task1/test_classical.py tests/task1/test_training.py
git commit -m "feat(task1): register classic fold runs"
```

---

### Task 5: Fold-0 HOG selection and model tuning controller

**Files:**
- Modify: `src/fashion/task1/experiments.py:29-215`
- Modify: `tests/task1/test_experiments.py`

**Interfaces:**
- Consumes: `run_task1_classical_fold`, the two HOG specs, both default configs, and both six-setting grids.
- Produces: `Task1ClassicalSelection`, `Task1ClassicalExperimentResult`, and `run_task1_classical_experiment(..., stage="smoke" | "tune" | "final")` with `smoke` and `tune` implemented in this task.

- [ ] **Step 1: Add failing smoke and tuning schedule tests**

Use a fake classic fold runner returning deterministic `Task1ClassicalFoldResult` objects and valid 124-column prediction CSV files. Add assertions:

```python
def test_classical_smoke_runs_default_knn_and_svm_on_coarse_hog(tmp_path, monkeypatch):
    result = run_task1_classical_experiment(
        _splits(), _label_map(), stage="smoke",
        fold_runner=_fake_classical_fold_runner,
        evidence_root=tmp_path / "evidence",
    )
    assert [(call.fold, call.hog_id, call.model_id) for call in _classic_calls] == [
        (0, "task1_gray_hog_ppc16_v1", "knn-k5-distance"),
        (0, "task1_gray_hog_ppc16_v1", "linear-svm-c1-normal"),
    ]
    assert result.stage == "smoke"
    assert result.selection is None


def test_classical_tune_selects_shared_hog_then_one_model_per_family(tmp_path, monkeypatch):
    result = run_task1_classical_experiment(
        _splits(), _label_map(), stage="tune",
        fold_runner=_fake_classical_fold_runner,
        evidence_root=tmp_path / "evidence",
    )
    assert len({result.candidate_id for result in result.fold_results}) == 14
    assert result.selection.hog_id == "task1_gray_hog_ppc10_v1"
    assert result.selection.knn_config_id in {
        "knn-k3-uniform", "knn-k3-distance", "knn-k5-uniform",
        "knn-k5-distance", "knn-k11-uniform", "knn-k11-distance",
    }
    assert result.selection.svm_config_id.startswith("linear-svm-")
    assert (tmp_path / "evidence" / "classical_tuning.csv").is_file()
    assert (tmp_path / "evidence" / "classical_selection.json").is_file()
```

Make the fake metrics choose coarse/fine and parameter winners deliberately. Include one exact tie and assert the controller breaks it by weighted-F1, top-1, top-5, then canonical candidate ID.

- [ ] **Step 2: Run the new controller tests and confirm failure**

```bash
./.venv/bin/python -m pytest tests/task1/test_experiments.py -k "classical_smoke or classical_tune" -q
```

Expected: import failures for the new classic experiment types and runner.

- [ ] **Step 3: Add result and selection dataclasses**

Add to `experiments.py` without changing `Task1ExperimentResult` or `run_task1_experiment`:

```python
@dataclass(frozen=True)
class Task1ClassicalSelection:
    hog_id: str
    knn_config_id: str
    svm_config_id: str


@dataclass(frozen=True)
class Task1ClassicalExperimentResult:
    stage: Literal["smoke", "tune", "final"]
    fold_results: tuple[Task1ClassicalFoldResult, ...]
    tuning: pd.DataFrame
    selection: Task1ClassicalSelection | None
    fold_metrics: pd.DataFrame
    comparison: pd.DataFrame
    oof_metrics: pd.DataFrame
    oof_predictions: dict[str, pd.DataFrame]
    per_class: dict[str, pd.DataFrame]
```

Add pure helpers `_classical_metrics_frame`, `_rank_classical_candidates`, `_select_shared_hog`, and `_select_model_config`. Ranking columns are exactly:

```python
ranking = ["macro_f1", "weighted_f1", "top1_accuracy", "top5_accuracy"]
frame.sort_values(ranking + ["candidate_id"], ascending=[False, False, False, False, True])
```

- [ ] **Step 4: Implement `smoke` and `tune` schedules**

The smoke schedule has exactly two coarse-HOG fold-0 runs: default KNN and default SVM.

The tune schedule must:

1. Run the two defaults on both HOG specs: four unique candidates.
2. Average default KNN and SVM macro-F1 by `hog_id` and choose the larger value; exact tie chooses coarse HOG.
3. Run all six KNN and six SVM configs on the selected HOG.
4. Deduplicate the two selected-HOG defaults already completed, leaving 14 unique candidates total.
5. Select one KNN and one SVM using the fixed ranking rule.
6. Write `classical_tuning.csv` atomically.
7. Write `classical_selection.json` atomically with the three exact selected IDs and hashes of the tuning CSV, split, label map, and implementation.

Use dependency injection:

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

For `stage="final"`, raise a clear error in this task; Task 6 replaces it with final execution.

- [ ] **Step 5: Run controller and existing CNN experiment tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_experiments.py -q
./.venv/bin/python -m ruff check src/fashion/task1/experiments.py tests/task1/test_experiments.py
```

Expected: new classic tests and all existing CNN experiment tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add src/fashion/task1/experiments.py tests/task1/test_experiments.py
git commit -m "feat(task1): add classic model tuning controller"
```

---

### Task 6: Selected-model five-fold evidence

**Files:**
- Modify: `src/fashion/task1/experiments.py`
- Modify: `tests/task1/test_experiments.py`

**Interfaces:**
- Consumes: `Task1ClassicalSelection`, selected config ID lookup, existing common metrics, and classic fold results.
- Produces: final-stage `Task1ClassicalExperimentResult` plus the five named classic evidence CSV families.

- [ ] **Step 1: Add failing final-stage tests**

Add a test with a supplied selection and fake fold runner:

```python
def test_classical_final_runs_two_selected_candidates_on_exactly_five_folds(
    tmp_path, monkeypatch
):
    selection = Task1ClassicalSelection(
        hog_id="task1_gray_hog_ppc10_v1",
        knn_config_id="knn-k5-distance",
        svm_config_id="linear-svm-c1-balanced",
    )
    result = run_task1_classical_experiment(
        _splits(), _label_map(), stage="final", selection=selection,
        fold_runner=_fake_classical_fold_runner,
        evidence_root=tmp_path / "evidence",
    )
    assert len(result.fold_results) == 10
    assert {item.fold for item in result.fold_results} == set(range(5))
    assert len(result.comparison) == 2
    assert set(result.oof_predictions) == {
        "task1_gray_hog_ppc10_v1-knn-k5-distance",
        "task1_gray_hog_ppc10_v1-linear-svm-c1-balanced",
    }
    assert all(len(frame) == 124 for frame in result.per_class.values())
    expected = {
        "classical_fold_metrics.csv", "classical_comparison.csv",
        "classical_oof_metrics.csv",
    }
    assert expected.issubset({path.name for path in (tmp_path / "evidence").glob("*.csv")})
```

Add failure cases for a missing selection, unknown config ID, duplicate fold, missing OOF product ID, and a fold runner exception. In every failure case, assert no aggregate classic evidence CSV is written.

- [ ] **Step 2: Run final-stage tests and confirm the deliberate final error**

```bash
./.venv/bin/python -m pytest tests/task1/test_experiments.py -k "classical_final" -q
```

Expected: failure from the temporary `stage="final"` guard added in Task 5.

- [ ] **Step 3: Implement exact final schedule and aggregation**

Resolve IDs only from the frozen constants:

```python
hog_by_id = {spec.hog_id: spec for spec in TASK1_HOG_SPECS}
knn_by_id = {config.config_id: config for config in TASK1_KNN_GRID}
svm_by_id = {config.config_id: config for config in TASK1_SVM_GRID}
```

Run selected KNN on folds 0-4, then selected SVM on folds 0-4. The fold runner's completed-result lookup reuses an exact fold-0 tuning result. Do not skip fold 0 based on filename or candidate ID alone.

Build `classical_fold_metrics.csv` with:

```text
run_id,fold,candidate_id,hog_id,model_family,macro_f1,weighted_f1,top1_accuracy,top5_accuracy
```

Require each candidate to have folds `{0,1,2,3,4}` exactly once. Use `aggregate_fold_metrics` to write `<metric>_mean` and `<metric>_std` columns to `classical_comparison.csv`.

- [ ] **Step 4: Build and validate OOF evidence**

For each selected candidate:

1. Read its five prediction artifacts with `keep_default_na=False`.
2. Concatenate in fold order.
3. Call `validate_oof_predictions` with all eligible labelled development IDs.
4. Extract `prob_000` through `prob_123` in order.
5. Call `classification_metrics` and `per_class_metrics`.
6. Write one row to `classical_oof_metrics.csv`.
7. Write `per_class_classical_<candidate-id>.csv`.

Write all aggregate files only after all ten fold results and both OOF tables validate. Use `atomic_write_csv` for every file.

- [ ] **Step 5: Add selection-file loading and validation**

When `selection=None`, load `classical_selection.json` from `evidence_root`. Require its split and label-map hashes to match current inputs and its tuning CSV hash to verify. Return a clear `ValueError` if it is stale or malformed. A directly supplied `Task1ClassicalSelection` still must resolve to frozen IDs.

- [ ] **Step 6: Run final, CNN regression, and lint tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_experiments.py tests/task1/test_evaluation.py -q
./.venv/bin/python -m ruff check src/fashion/task1/experiments.py tests/task1/test_experiments.py
```

Expected: all tests pass; existing CNN evidence filenames and tests remain unchanged.

- [ ] **Step 7: Commit Task 6**

```bash
git add src/fashion/task1/experiments.py tests/task1/test_experiments.py
git commit -m "feat(task1): add classic five-fold evidence"
```

---

### Task 7: Public API, notebook narrative, and package documentation

**Files:**
- Modify: `src/fashion/task1/__init__.py:1-62`
- Modify: `tests/task1/test_notebook_baseline.py:14-31`
- Modify: `notebooks/02_task1_article_type.ipynb` cells 12-20
- Modify: `src/fashion/README.md:22-28`

**Interfaces:**
- Consumes: approved public classic specs, configs, selection/result types, and runner.
- Produces: a small `fashion.task1` import surface and a Run-All-safe notebook controller.

- [ ] **Step 1: Strengthen the notebook contract test first**

Extend the required notebook tokens with:

```python
required_classical = (
    'CLASSICAL_STAGE = "smoke"',
    "run_task1_classical_experiment",
    "TASK1_HOG_COARSE",
    "TASK1_HOG_FINE",
    "KNeighborsClassifier",
    "LinearSVC",
    "16 x 16",
    "10 x 10",
    "classical_tuning.csv",
    "results/runs.csv",
)
assert all(token in source for token in required_classical)
assert "train_test_split" not in source
assert "pretrained=True" not in source
assert "StandardScaler" not in source
assert "PCA(" not in source
```

Also assert the notebook says the shared Task 1 framework owns the metrics for every candidate family.

- [ ] **Step 2: Run the notebook contract and confirm missing tokens**

```bash
./.venv/bin/python -m pytest tests/task1/test_notebook_baseline.py -q
```

Expected: failure because Notebook 02 does not yet describe or call the classic pipeline.

- [ ] **Step 3: Export the public classic API**

Export only these names from `fashion.task1`:

```python
TASK1_HOG_COARSE
TASK1_HOG_FINE
TASK1_HOG_SPECS
TASK1_KNN_GRID
TASK1_SVM_GRID
Task1ClassicalExperimentResult
Task1ClassicalSelection
Task1HogSpec
Task1KNNConfig
Task1LinearSVMConfig
run_task1_classical_experiment
```

Do not export cache internals, softmax helpers, or private ranking helpers.

- [ ] **Step 4: Update Notebook 02 without embedding model logic**

Edit the notebook JSON with `apply_patch`; do not write a temporary conversion script. Keep the resulting `.ipynb` as the only notebook artifact. Update:

- Section 8: define the CNN as one baseline and HOG + KNN/SVM as two classic candidates. Add a short citation to the original HOG paper as published support for the shape descriptor.
- Section 9: add the three candidate families and explain that KNN/SVM come from scikit-learn while HOG comes from scikit-image.
- Section 10: show the staged matrix: smoke, fold-0 HOG selection, fold-0 KNN/SVM tuning, selected five-fold runs.
- Section 11: add a separate classic controller with `CLASSICAL_STAGE = "smoke"` by default.
- Section 12: predeclare rare-class and confusion-pair analysis using the shared per-class evidence.
- Section 13: remove classic speed/model-size comparison claims.
- Section 14: record the approved choices and rejected PCA/raw-pixel/RBF alternatives.
- Section 15: keep handoff not ready until registered five-fold evidence exists.

The controller cell should contain this shape:

```python
from fashion.task1 import (
    TASK1_HOG_COARSE,
    TASK1_HOG_FINE,
    run_task1_classical_experiment,
)

CLASSICAL_STAGE = "smoke"  # Use "tune", then "final", after smoke passes.
classic_experiment = run_task1_classical_experiment(
    splits,
    load_label_maps()[TARGET],
    stage=CLASSICAL_STAGE,
)
display(classic_experiment.tuning)
display(classic_experiment.comparison)
display(classic_experiment.oof_metrics)

if CLASSICAL_STAGE == "final":
    classical_confusion_figures = {
        candidate_id: write_task1_confusion_figure(
            predictions,
            article_type_classes,
            output=TASK1_FIGURE_DIR / f"classical_oof_confusion_{candidate_id}.png",
        )
        for candidate_id, predictions in classic_experiment.oof_predictions.items()
    }
```

Mention `KNeighborsClassifier` and `LinearSVC` in markdown, not as notebook implementation imports. Keep all training and score logic in `src/fashion/`.

- [ ] **Step 5: Document the module boundary**

Add to `src/fashion/README.md` under `task1/`:

```markdown
  - `classical.py` extracts and caches grayscale HOG features and runs fixed-class KNN
    and Linear SVM fold candidates.
  - `experiments.py` orchestrates both the CNN comparison and staged classic-model
    tuning/five-fold evidence.
```

- [ ] **Step 6: Run notebook, API, and documentation tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_notebook_baseline.py tests/test_documentation.py -q
./.venv/bin/python -m ruff check src/fashion/task1/__init__.py tests/task1/test_notebook_baseline.py
```

Expected: tests pass, Ruff passes, and the notebook validates with `nbformat`.

- [ ] **Step 7: Commit Task 7**

```bash
git add src/fashion/task1/__init__.py tests/task1/test_notebook_baseline.py notebooks/02_task1_article_type.ipynb src/fashion/README.md
git commit -m "docs(task1): add classic model experiment story"
```

---

### Task 8: End-to-end smoke verification and handoff

**Files:**
- Modify only files that fail the verification below, and only when the failure is caused by this feature.

**Interfaces:**
- Consumes: the completed Tasks 1-7.
- Produces: a verified smoke path and a clear command handoff for long tuning/final runs.

- [ ] **Step 1: Run the complete Task 1 test suite**

```bash
./.venv/bin/python -m pytest tests/task1 -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 2: Run the full project test suite**

```bash
./.venv/bin/python -m pytest -q
```

Expected: all project tests pass. Investigate any failure before changing code; preserve unrelated user work.

- [ ] **Step 3: Run lint on changed Python files**

```bash
./.venv/bin/python -m ruff check src/fashion/config.py src/fashion/task1 tests/task1
```

Expected: `All checks passed!`.

- [ ] **Step 4: Execute the real classic smoke stage**

Run:

```bash
./.venv/bin/python - <<'PY'
from fashion.data.dataset import load_label_maps, load_splits
from fashion.task1 import run_task1_classical_experiment

result = run_task1_classical_experiment(
    load_splits(),
    load_label_maps()["articleType"],
    stage="smoke",
)
print(result.stage)
print(result.fold_metrics[["candidate_id", "fold", "macro_f1"]].to_string(index=False))
PY
```

Expected: prints `smoke`, then two completed fold-0 rows, one KNN and one Linear SVM. Confirm both new run IDs have `status=completed`, `stage=smoke`, and `final_eligible=false` in `results/runs.csv`.

- [ ] **Step 5: Check generated artifacts without starting long runs**

```bash
./.venv/bin/python - <<'PY'
from fashion.train.registry import RunRegistry

rows = RunRegistry().find(task="task1", stage="smoke", status="completed")
classic = rows[rows["model_family"].isin(["task1_hog_knn_v1", "task1_hog_linear_svm_v1"])]
assert set(classic["model_family"]) == {"task1_hog_knn_v1", "task1_hog_linear_svm_v1"}
assert classic["prediction_path"].str.strip().ne("").all()
print(classic[["run_id", "model_family", "primary_metric_value"]].tail(2).to_string(index=False))
PY
```

Expected: assertions pass and the two latest classic smoke rows print.

- [ ] **Step 6: Commit any verification-only correction**

If Steps 1-5 required no code correction, do not create an empty commit. If a scoped correction was required, review `git diff` and stage only this feature's fixed paths with this explicit command:

```bash
git add pyproject.toml requirements/constraints-py312.txt src/fashion/config.py src/fashion/task1/classical.py src/fashion/task1/evaluation.py src/fashion/task1/experiments.py src/fashion/task1/training.py src/fashion/task1/__init__.py tests/task1/test_classical.py tests/task1/test_evaluation.py tests/task1/test_experiments.py tests/task1/test_training.py tests/task1/test_notebook_baseline.py notebooks/02_task1_article_type.ipynb src/fashion/README.md
git commit -m "fix(task1): complete classic smoke verification"
```

- [ ] **Step 7: Hand off the long-run commands**

Do not start tuning or five-fold KNN automatically. Give the user these exact notebook choices:

```python
CLASSICAL_STAGE = "tune"
```

After tuning completes and `classical_selection.json` exists:

```python
CLASSICAL_STAGE = "final"
```

State clearly that exact five-fold KNN may run for more than three hours and must be allowed to finish, as approved in the design.

## Completion Check

Before reporting implementation complete, verify:

```bash
git status --short
git log --oneline -8
```

Report only feature commits and feature files. Do not claim the long tuning/final experiments completed unless their registered rows and five-fold evidence files actually exist.
