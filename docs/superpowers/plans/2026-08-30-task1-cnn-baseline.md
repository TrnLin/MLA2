# Task 1 Small-CNN Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a scratch-trained small-CNN baseline for Task 1 with safe five-fold evaluation, smoke and full run modes, hashed artifacts, and complete run registration.

**Architecture:** The Task 1 notebook controls experiments but delegates data loading, model definition, metrics, fold training, and cross-validation aggregation to focused modules in `src/fashion/task1/`. Shared artifact, reproducibility, and registry code is ported from the reviewed Task 2 branch so all tasks write the same `results/runs.csv` schema.

**Tech Stack:** Python 3.12, PyTorch 2.12.1, scikit-learn 1.9.0, pandas 3.0.5, NumPy 2.5.2, Pillow 12.3.0, pytest 9.1.1, matplotlib 3.11.1.

**Spec:** `docs/superpowers/specs/2026-08-30-task1-cnn-baseline-design.md`

## Global Constraints

- Use `data/processed/splits.csv` through `fashion.data.dataset`; never call `train_test_split`.
- Use development folds 0--4 only during model comparison; holdout and quarantine labels stay sealed.
- Train `Task1SmallCNN` from scratch with no pretrained weights.
- Every physical training run, including smoke, failed, and interrupted runs, must pass through `fashion.train.registry` and append to `results/runs.csv`.
- Full comparison means ten runs: five folds for each of `task1_rgb_60x80_no_aug_v1` and `task1_rgb_60x80_mild_aug_v1`.
- Primary metric is fixed-label macro-F1 over all 124 classes with `zero_division=0`.
- Use `./.venv/bin/python` for every Python, pytest, and notebook command.
- Preserve unrelated untracked files and user changes.

## File Structure

- Modify `pyproject.toml`: declare training dependencies already used by the shared runtime.
- Modify `src/fashion/config.py`: define shared registry and Task 1 result paths.
- Modify `.gitignore`: ignore generated Task 1 checkpoints, histories, and predictions.
- Create `src/fashion/train/artifacts.py`: atomic writes, canonical hashes, artifact verification.
- Create `src/fashion/train/reproducibility.py`: deterministic seeds and runtime provenance.
- Create `src/fashion/train/registry.py`: append-once run lifecycle.
- Create `src/fashion/train/__init__.py`: export only training contracts present on this branch.
- Create `src/fashion/task1/models.py`: small CNN and model configuration.
- Create `src/fashion/task1/dataset.py`: validated PyTorch dataset and fold row construction.
- Create `src/fashion/task1/evaluation.py`: fixed-class metrics, prediction evidence, and CV checks.
- Create `src/fashion/task1/training.py`: configuration, one-fold trainer, artifact writing, and run registration.
- Create `src/fashion/task1/experiments.py`: smoke/full orchestration, fold aggregation, and figures.
- Modify `src/fashion/task1/__init__.py`: expose the small public Task 1 training API.
- Modify `src/fashion/README.md`: document the added Task 1 and shared training modules.
- Modify `notebooks/02_task1_article_type.ipynb`: control and explain smoke/full baseline runs.
- Create tests under `tests/train/` and `tests/task1/` for every new contract.

---

### Task 1: Shared Artifact, Reproducibility, and Run Registry Foundation

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/fashion/config.py`
- Modify: `.gitignore`
- Create: `src/fashion/train/__init__.py`
- Create: `src/fashion/train/artifacts.py`
- Create: `src/fashion/train/reproducibility.py`
- Create: `src/fashion/train/registry.py`
- Create: `tests/train/test_artifacts.py`
- Create: `tests/train/test_reproducibility.py`
- Create: `tests/train/test_registry.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: `fashion.data.hashing.compute_sha256`, `fashion.config.ROOT`, `pandas`, `torch`, `psutil`.
- Produces: `atomic_write_bytes`, `atomic_write_csv`, `canonical_sha256`, `verify_artifact`, `seed_everything`, `seed_worker`, `make_torch_generator`, `RunRecord`, `RunRegistry`, `new_run_id`, and `tracked_run`.

- [ ] **Step 1: Add the missing dependency and path tests**

Add tests which assert that `RUNS_CSV == RESULTS_DIR / "runs.csv"`,
`TASK1_RESULT_DIR == RESULTS_DIR / "task1"`, `TASK1_FIGURE_DIR == FIGURE_DIR /
"task1"`, and `TASK1_EVIDENCE_DIR == EVIDENCE_DIR / "task1"`. Port the reviewed
shared tests from these exact Git objects:

```text
a09b1ac:tests/train/test_reproducibility.py  blob f937d8ff4d6a7c1d3d29f333cfe55f9cf552f390
0ac5038:tests/train/test_artifacts.py        blob 5ad78bd8fcb300b760ce01a072c77a838c903b8a
e87030c:tests/train/test_registry.py         blob baf2e51e6a2f27df636daabe018bbc154bedbe40
```

Use `git show <object>` to inspect the reviewed text, then add it with `apply_patch`. Do not cherry-pick Task 2-specific paths or engines.

- [ ] **Step 2: Run the tests and verify the imports fail**

Run:

```bash
./.venv/bin/python -m pytest tests/train tests/test_config.py -q
```

Expected: FAIL because `fashion.train` modules and Task 1 paths do not exist on this branch.

- [ ] **Step 3: Declare exact dependencies and paths**

Add these dependency lines to `pyproject.toml`:

```toml
"psutil==7.2.2",
"scikit-learn==1.9.0",
"torch==2.12.1",
```

Add these constants after the existing result paths in `src/fashion/config.py`:

```python
RUNS_CSV = RESULTS_DIR / "runs.csv"
TASK1_RESULT_DIR = RESULTS_DIR / "task1"
TASK1_FIGURE_DIR = FIGURE_DIR / "task1"
TASK1_EVIDENCE_DIR = EVIDENCE_DIR / "task1"
```

Add `/results/task1/` to `.gitignore`. Report-selected files in
`results/figures/task1/` remain trackable.

- [ ] **Step 4: Port the reviewed shared implementation**

Port these exact reviewed source objects with `apply_patch`:

```text
a09b1ac:src/fashion/train/reproducibility.py  blob 2086809bed690caa45974df71d663131b380b1d4
0ac5038:src/fashion/train/artifacts.py        blob 27cd0837e71ac6e5724e767b9ecd84de6d05eed0
e87030c:src/fashion/train/registry.py         blob ce5687b919bbfe8c4f28e9e90997a27ed95c74ea
```

Create a branch-appropriate `src/fashion/train/__init__.py` which imports only
those three modules:

```python
"""Shared experiment artifacts, reproducibility, and run registration."""

from fashion.train.artifacts import (
    atomic_write_bytes,
    atomic_write_csv,
    canonical_sha256,
    verify_artifact,
)
from fashion.train.registry import RunRecord, RunRegistry, new_run_id, tracked_run
from fashion.train.reproducibility import make_torch_generator, seed_everything, seed_worker

__all__ = [
    "RunRecord",
    "RunRegistry",
    "atomic_write_bytes",
    "atomic_write_csv",
    "canonical_sha256",
    "make_torch_generator",
    "new_run_id",
    "seed_everything",
    "seed_worker",
    "tracked_run",
    "verify_artifact",
]
```

- [ ] **Step 5: Run shared foundation tests**

Run:

```bash
./.venv/bin/python -m pytest tests/train tests/test_config.py -q
./.venv/bin/python -m ruff check src/fashion/train src/fashion/config.py tests/train tests/test_config.py
```

Expected: all tests and lint checks PASS.

- [ ] **Step 6: Commit the shared foundation**

```bash
git add pyproject.toml .gitignore src/fashion/config.py src/fashion/train tests/train tests/test_config.py
git commit -m "feat(train): add shared run registry foundation"
```

---

### Task 2: Scratch Small-CNN Model

**Files:**
- Create: `src/fashion/task1/models.py`
- Create: `tests/task1/test_models.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Consumes: a float `torch.Tensor` shaped `(N, 3, H, W)`.
- Produces: `Task1SmallCNN(num_classes: int = 124)`, `Task1ModelConfig`, and `count_trainable_parameters(model) -> int`; forward output shape is `(N, num_classes)`.

- [ ] **Step 1: Write failing model contract tests**

Create tests with these assertions:

```python
import torch

from fashion.task1.models import Task1SmallCNN, count_trainable_parameters


def test_small_cnn_maps_fashion_images_to_124_logits():
    model = Task1SmallCNN(num_classes=124)
    output = model(torch.zeros(2, 3, 80, 60))
    assert output.shape == (2, 124)
    assert count_trainable_parameters(model) == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def test_small_cnn_uses_distinct_late_convolutions():
    model = Task1SmallCNN(num_classes=124)
    assert model.conv4 is not model.conv5


def test_small_cnn_rejects_wrong_channel_count():
    model = Task1SmallCNN(num_classes=124)
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.zeros(2, 1, 80, 60))
```

- [ ] **Step 2: Run tests and verify the module is missing**

```bash
./.venv/bin/python -m pytest tests/task1/test_models.py -q
```

Expected: FAIL with `ModuleNotFoundError: fashion.task1.models`.

- [ ] **Step 3: Implement the exact model shape**

Implement a frozen `Task1ModelConfig` with `num_classes=124`,
`adaptive_size=(4, 3)`, and `hidden_features=128`. Implement the module with
separate layers:

```python
self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
self.conv5 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
self.pool = nn.MaxPool2d(2, 2)
self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 3))
self.fc1 = nn.Linear(64 * 4 * 3, 128)
self.fc2 = nn.Linear(128, num_classes)
```

The forward order is `conv1`, `conv2`, pool, `conv3`, pool, `conv4`, `conv5`,
pool, adaptive pool, flatten, `fc1`, `fc2`, with ReLU after every convolution
and after `fc1`. Return raw logits; do not add softmax to the model.

- [ ] **Step 4: Export and test the model**

Export the three public names from `fashion.task1.__init__`, then run:

```bash
./.venv/bin/python -m pytest tests/task1/test_models.py -q
./.venv/bin/python -m ruff check src/fashion/task1/models.py tests/task1/test_models.py
```

Expected: PASS.

- [ ] **Step 5: Commit the model**

```bash
git add src/fashion/task1/models.py src/fashion/task1/__init__.py tests/task1/test_models.py
git commit -m "feat(task1): add scratch small cnn"
```

---

### Task 3: Validated PyTorch Dataset and Fold Rows

**Files:**
- Create: `src/fashion/task1/dataset.py`
- Create: `tests/task1/test_dataset.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Consumes: development rows, `label_to_index: Mapping[str, int]`, a callable image transform, and project root.
- Produces: `Task1TorchDataset`, `get_task1_fold_rows(splits, validation_fold)`, and samples shaped `{"image": Tensor[3,80,60], "label": LongTensor[], "id": LongTensor[]}`.

- [ ] **Step 1: Write failing dataset tests with generated images**

Use `PIL.Image.new` and a two-row DataFrame. Assert:

```python
dataset = Task1TorchDataset(
    rows,
    transform=lambda path: np.zeros((3, 80, 60), dtype=np.float32),
    label_to_index={"Shirts": 0, "Shoes": 1},
    root=tmp_path,
)
sample = dataset[0]
assert sample["image"].shape == (3, 80, 60)
assert sample["image"].dtype == torch.float32
assert sample["label"].dtype == torch.long
assert sample["id"].item() == 11
```

Add separate tests that reject holdout and quarantine rows, duplicate IDs, a blank path, an
unknown label, and a transform returning a shape other than `(3, 80, 60)`.
Add a fold test proving `get_task1_fold_rows` returns disjoint training and
validation IDs and filters rows without valid `articleType` labels.

- [ ] **Step 2: Run tests and verify the module is missing**

```bash
./.venv/bin/python -m pytest tests/task1/test_dataset.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement validation and sample conversion**

Implement this public contract:

```python
class Task1TorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        transform: Callable[[Path], np.ndarray],
        label_to_index: Mapping[str, int],
        *,
        root: str | Path = ROOT,
    ) -> None:
        required = {"id", "path", "partition", "articleType"}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"Task 1 rows are missing columns: {sorted(missing)}")
        if rows.empty or not rows["partition"].eq("development").all():
            raise ValueError("Task 1 training rows must be non-empty development rows")
        if rows["id"].isna().any() or rows["id"].duplicated().any():
            raise ValueError("Task 1 row IDs must be present and unique")
        if rows["path"].astype(str).str.strip().eq("").any():
            raise ValueError("Task 1 image paths must not be blank")
        unknown = set(rows["articleType"].astype(str)) - set(label_to_index)
        if unknown:
            raise ValueError(f"unknown articleType labels: {sorted(unknown)}")
        self.rows = rows.reset_index(drop=True).copy()
        self.transform = transform
        self.label_to_index = dict(label_to_index)
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index]
        array = self.transform(self.root / str(row["path"]))
        if array.shape != (3, 80, 60) or array.dtype != np.float32:
            raise ValueError("Task 1 transform must return float32 shape (3, 80, 60)")
        return {
            "image": torch.from_numpy(array),
            "label": torch.tensor(self.label_to_index[str(row["articleType"])], dtype=torch.long),
            "id": torch.tensor(int(row["id"]), dtype=torch.long),
        }


def get_task1_fold_rows(
    splits: pd.DataFrame,
    validation_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training, validation = get_cv_split(splits, validation_fold)
    return (
        get_samples(training, target="articleType"),
        get_samples(validation, target="articleType"),
    )
```

`get_task1_fold_rows` must call `get_cv_split` and then `get_samples` with
`target="articleType"`; it must not reproduce split logic.

- [ ] **Step 4: Run dataset and existing preprocessing tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_dataset.py tests/task1/test_preprocessing.py -q
./.venv/bin/python -m ruff check src/fashion/task1/dataset.py tests/task1/test_dataset.py
```

Expected: PASS.

- [ ] **Step 5: Commit the dataset adapter**

```bash
git add src/fashion/task1/dataset.py src/fashion/task1/__init__.py tests/task1/test_dataset.py
git commit -m "feat(task1): add fold-safe torch dataset"
```

---

### Task 4: Fixed-Class Evaluation and Prediction Evidence

**Files:**
- Create: `src/fashion/task1/evaluation.py`
- Create: `tests/task1/test_evaluation.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Consumes: product IDs, true class indexes, probability matrix, and exactly 124 class names.
- Produces: `classification_metrics`, `per_class_metrics`, `build_prediction_frame`, `validate_oof_predictions`, and `aggregate_fold_metrics`.

- [ ] **Step 1: Write failing metric tests**

Test a three-sample probability matrix embedded in 124 columns. Compute expected
values with explicit labels:

```python
expected_macro = f1_score(
    y_true,
    probabilities.argmax(axis=1),
    labels=np.arange(124),
    average="macro",
    zero_division=0,
)
metrics = classification_metrics(y_true, probabilities, num_classes=124)
assert metrics["macro_f1"] == pytest.approx(expected_macro)
assert metrics["top1_accuracy"] == pytest.approx(2 / 3)
assert metrics["top5_accuracy"] == pytest.approx(1.0)
```

Add tests rejecting NaN probabilities, wrong matrix width, labels outside
`0..123`, duplicate OOF IDs, missing expected OOF IDs, and class-name lists not
of length 124.

- [ ] **Step 2: Run tests and verify they fail**

```bash
./.venv/bin/python -m pytest tests/task1/test_evaluation.py -q
```

Expected: FAIL because the evaluation module does not exist.

- [ ] **Step 3: Implement exact metric contracts**

Implement these contracts. Keep validation in a private helper so every public
function rejects invalid shapes and non-finite values in the same way:

```python
def _validated_arrays(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != (len(labels), num_classes):
        raise ValueError("labels and probability matrix shapes do not align")
    if len(labels) == 0 or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("true labels must be non-empty valid class indexes")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities must be finite")
    return labels, scores


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    num_classes: int = 124,
) -> dict[str, float]:
    y_true, probabilities = _validated_arrays(y_true, probabilities, num_classes)
    y_pred = probabilities.argmax(axis=1)
    top_k = min(5, num_classes)
    top_indexes = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    return {
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=np.arange(num_classes),
            average="macro", zero_division=0,
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, labels=np.arange(num_classes),
            average="weighted", zero_division=0,
        )),
        "top1_accuracy": float(np.mean(y_pred == y_true)),
        "top5_accuracy": float(np.mean((top_indexes == y_true[:, None]).any(axis=1))),
    }


def build_prediction_frame(
    product_ids: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    if len(class_names) != 124:
        raise ValueError("Task 1 prediction evidence requires 124 class names")
    y_true, probabilities = _validated_arrays(y_true, probabilities, 124)
    product_ids = np.asarray(product_ids, dtype=np.int64)
    if len(product_ids) != len(y_true) or len(np.unique(product_ids)) != len(product_ids):
        raise ValueError("prediction product IDs must be unique and align with labels")
    y_pred = probabilities.argmax(axis=1)
    frame = pd.DataFrame({
        "id": product_ids,
        "true_index": y_true,
        "predicted_index": y_pred,
        "true_label": [class_names[index] for index in y_true],
        "predicted_label": [class_names[index] for index in y_pred],
    })
    for index in range(124):
        frame[f"prob_{index:03d}"] = probabilities[:, index]
    return frame


def validate_oof_predictions(
    predictions: pd.DataFrame,
    expected_ids: Collection[int],
) -> None:
    if predictions["id"].duplicated().any():
        raise ValueError("OOF predictions contain duplicate product IDs")
    actual = set(predictions["id"].astype(int))
    expected = {int(product_id) for product_id in expected_ids}
    if actual != expected:
        raise ValueError("OOF predictions do not match expected development IDs")


def aggregate_fold_metrics(fold_metrics: Sequence[Mapping[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(fold_metrics)
    if len(frame) != 5:
        raise ValueError("full Task 1 evidence requires exactly five folds")
    return pd.DataFrame({"mean": frame.mean(), "std": frame.std(ddof=1)})
```

`build_prediction_frame` must write `id`, `true_index`, `predicted_index`,
`true_label`, `predicted_label`, and probability columns named `prob_000` through
`prob_123`. `aggregate_fold_metrics` returns one row per metric with `mean` and
sample `std` using `ddof=1`.

Also implement `per_class_metrics(y_true, probabilities, class_names) ->
pd.DataFrame` with `precision_recall_fscore_support(labels=np.arange(124),
zero_division=0)`. Its 124 rows contain `class_index`, `class_name`, `precision`,
`recall`, `f1`, and `support`.

- [ ] **Step 4: Run and lint evaluation tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_evaluation.py -q
./.venv/bin/python -m ruff check src/fashion/task1/evaluation.py tests/task1/test_evaluation.py
```

Expected: PASS.

- [ ] **Step 5: Commit evaluation contracts**

```bash
git add src/fashion/task1/evaluation.py src/fashion/task1/__init__.py tests/task1/test_evaluation.py
git commit -m "feat(task1): add fixed-class evaluation"
```

---

### Task 5: One-Fold Training, Best Checkpoint, and Registered Artifacts

**Files:**
- Create: `src/fashion/task1/training.py`
- Create: `tests/task1/test_training.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Consumes: canonical split DataFrame, label-map dictionary, one validation fold, one preprocessing config, one train config, and a `RunRegistry`.
- Produces: `Task1TrainConfig`, `Task1FoldResult`, `select_training_device`, and `train_task1_fold`.

- [ ] **Step 1: Write failing configuration and device tests**

Assert the frozen configurations:

```python
smoke = Task1TrainConfig.smoke()
assert (smoke.stage, smoke.epochs, smoke.batch_size) == ("smoke", 1, 16)
assert (smoke.max_train_batches, smoke.max_validation_batches) == (2, 2)
assert smoke.final_eligible is False

full = Task1TrainConfig.full()
assert (full.stage, full.epochs, full.batch_size) == ("experiment", 20, 128)
assert full.max_train_batches is None
assert full.final_eligible is True
```

Test device priority by monkeypatching availability: CUDA, then MPS, then CPU.

- [ ] **Step 2: Write a failing tiny end-to-end fold test**

Generate 40 RGB images and a valid five-fold development DataFrame. Use a
124-class map whose first two labels occur in the images. Write that DataFrame to
`tmp_path / "splits.csv"` before calling:

```python
result = train_task1_fold(
    splits,
    label_map,
    validation_fold=0,
    preprocessing=TASK1_CONTROL_PREPROCESSING,
    config=Task1TrainConfig.smoke(),
    registry=RunRegistry(tmp_path / "runs.csv"),
    root=tmp_path,
    result_root=tmp_path / "results",
    device=torch.device("cpu"),
    split_path=tmp_path / "splits.csv",
)
```

Assert status `completed`, `final_eligible=false`, one registry row, 124
probability columns, one epoch in history, a loadable checkpoint, and matching
SHA-256 hashes. Add a test that injects a model which raises `RuntimeError`
through the public `model_factory` argument and asserts the registry row is
finalized as `failed`.

- [ ] **Step 3: Run tests and verify the trainer is missing**

```bash
./.venv/bin/python -m pytest tests/task1/test_training.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement immutable configuration and result types**

Define:

```python
@dataclass(frozen=True)
class Task1TrainConfig:
    stage: Literal["smoke", "experiment"]
    epochs: int
    batch_size: int
    max_train_batches: int | None
    max_validation_batches: int | None
    final_eligible: bool
    seed: int = RANDOM_SEED
    max_lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    num_workers: int = 0

    @classmethod
    def smoke(cls) -> "Task1TrainConfig":
        return cls(
            stage="smoke", epochs=1, batch_size=16,
            max_train_batches=2, max_validation_batches=2,
            final_eligible=False,
        )

    @classmethod
    def full(cls) -> "Task1TrainConfig":
        return cls(
            stage="experiment", epochs=20, batch_size=128,
            max_train_batches=None, max_validation_batches=None,
            final_eligible=True,
        )


@dataclass(frozen=True)
class Task1FoldResult:
    run_id: str
    fold: int
    preprocessing_id: str
    status: Literal["completed"]
    metrics: dict[str, float]
    checkpoint_path: Path
    history_path: Path
    prediction_path: Path
```

Smoke mode deterministically keeps the lowest 32 training IDs and lowest 32
validation IDs before fitting normalization. This keeps smoke fast and remains
development-only; its registry row is never final-eligible.

- [ ] **Step 5: Implement one-fold data and optimization flow**

Implement `train_task1_fold` with these exact parameters and return type:

```text
train_task1_fold(
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    *,
    validation_fold: int,
    preprocessing: Task1PreprocessingConfig,
    config: Task1TrainConfig,
    registry: RunRegistry | None = None,
    root: str | Path = ROOT,
    result_root: str | Path = TASK1_RESULT_DIR,
    device: torch.device | None = None,
    split_path: str | Path = SPLITS_CSV,
    model_factory: Callable[[int], nn.Module] = Task1SmallCNN,
) -> Task1FoldResult
```

The function must:

1. Validate the map has exactly 124 classes and a matching `label_to_index`.
2. Obtain rows with `get_task1_fold_rows`.
3. Apply the smoke-only 32/32 restriction when requested.
4. Fit `Task1Normalization` on training rows only.
5. Create a fixed validation dataset once.
6. Rebuild the seeded training transform and loader for each epoch so `epoch`
   changes augmentation deterministically.
7. Train with unweighted `F.cross_entropy`, Adam, OneCycleLR, and
   `clip_grad_norm_(model.parameters(), 1.0)`.
8. Stop iteration at configured batch limits without changing the saved split.
9. Evaluate probabilities under `model.eval()` and `torch.no_grad()`.
10. Reject non-finite loss or probabilities.
11. Keep the state with highest fixed-label validation macro-F1; break exact ties
    by the earlier epoch.
12. Set OneCycleLR `steps_per_epoch` to the actual number of training batches
    used after applying a smoke batch limit.
13. Reload the best state and evaluate it once more; final metrics and saved
    predictions must describe that best state, not the last epoch.

- [ ] **Step 6: Register the run and write artifacts atomically**

Build `RunRecord` before model work with:

```python
RunRecord(
    run_id=new_run_id(experiment_id, validation_fold, config.seed),
    task="task1",
    stage=config.stage,
    experiment_id=experiment_id,
    model_family="task1_small_cnn_v1",
    benchmark_only=False,
    final_eligible=config.final_eligible,
    scratch=True,
    fold=validation_fold,
    seed=config.seed,
    transform_id=preprocessing.preprocessing_id,
    loss_id="cross_entropy_unweighted_v1",
    epochs_requested=config.epochs,
    primary_metric_name="macro_f1_124",
    config_sha256=canonical_sha256(run_config),
    split_sha256=compute_sha256(split_path),
    label_map_sha256=canonical_sha256(label_map),
    implementation_sha256=canonical_sha256(source_hashes),
)
```

Run all training inside `tracked_run`. Serialize the checkpoint to `io.BytesIO`
with `torch.save`, then use `atomic_write_bytes`. Use `atomic_write_csv` for
history and predictions. Hash all three files with `compute_sha256`, populate the
tracked record's artifact fields, metrics, best epoch, epochs completed, parameter
count, and CUDA peak VRAM when CUDA is used.

Store paths relative to the project root when possible. The checkpoint payload
must contain `model_state_dict`, `model_config`, `train_config`, `preprocessing`,
`normalization`, `class_names`, `label_map_sha256`, `validation_fold`, `seed`,
`best_epoch`, and `metrics`.

Build `source_hashes` from `compute_sha256` over the checked-in Task 1 model,
dataset, preprocessing, evaluation, and training source files. Sort the path keys
before passing the mapping to `canonical_sha256`.

- [ ] **Step 7: Run training tests and targeted regression tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_training.py tests/task1/test_dataset.py tests/task1/test_evaluation.py tests/train -q
./.venv/bin/python -m ruff check src/fashion/task1/training.py tests/task1/test_training.py
```

Expected: PASS.

- [ ] **Step 8: Commit the fold trainer**

```bash
git add src/fashion/task1/training.py src/fashion/task1/__init__.py tests/task1/test_training.py
git commit -m "feat(task1): train and register cnn folds"
```

---

### Task 6: Smoke/Full Experiment Orchestration and CV Evidence

**Files:**
- Create: `src/fashion/task1/experiments.py`
- Create: `tests/task1/test_experiments.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Consumes: canonical splits, Task 1 label map, mode `"smoke" | "full"`, and optional registry/result roots.
- Produces: `Task1ExperimentResult`, `run_task1_experiment`, and `write_task1_comparison_figure`.

- [ ] **Step 1: Write failing orchestration tests with a fake fold runner**

Inject a fake callable and assert:

```python
smoke = run_task1_experiment(
    splits,
    label_map,
    mode="smoke",
    fold_runner=fake_fold_runner,
)
assert [(call.fold, call.transform_id) for call in calls] == [
    (0, "task1_rgb_60x80_no_aug_v1")
]

calls.clear()
full = run_task1_experiment(
    splits,
    label_map,
    mode="full",
    fold_runner=fake_fold_runner,
)
assert len(calls) == 10
assert {call.fold for call in calls} == set(range(5))
```

Add tests that the runner propagates a fold exception without producing an
aggregate, and that full aggregation rejects duplicate prediction IDs, missing
expected development IDs, and fewer than five metrics for either preprocessing
ID.

- [ ] **Step 2: Run tests and verify the module is missing**

```bash
./.venv/bin/python -m pytest tests/task1/test_experiments.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the orchestration contract**

Define the result type exactly as shown. `fold_metrics` keeps one row per physical
run. `comparison` keeps one summary row per preprocessing candidate. Implement
the runner with the parameter contract listed after it:

```python
@dataclass(frozen=True)
class Task1ExperimentResult:
    mode: Literal["smoke", "full"]
    fold_results: tuple[Task1FoldResult, ...]
    fold_metrics: pd.DataFrame
    comparison: pd.DataFrame
    oof_predictions: dict[str, pd.DataFrame]
    per_class: dict[str, pd.DataFrame]
```

```text
run_task1_experiment(
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    *,
    mode: Literal["smoke", "full"],
    registry: RunRegistry | None = None,
    root: str | Path = ROOT,
    result_root: str | Path = TASK1_RESULT_DIR,
    fold_runner: Callable[..., Task1FoldResult] = train_task1_fold,
) -> Task1ExperimentResult
```

Smoke calls fold 0 with `TASK1_CONTROL_PREPROCESSING` and
`Task1TrainConfig.smoke()`. Full loops preprocessing outermost in the fixed order
control then mild, and folds `0..4` inside each preprocessing. Read each completed
prediction artifact, validate one OOF row for every eligible Task 1 development
ID per preprocessing candidate, and aggregate fold metrics with sample standard
deviation.

- [ ] **Step 4: Add one report-ready comparison figure**

Implement these callable contracts:

```text
write_task1_comparison_figure(
    fold_metrics: pd.DataFrame,
    *,
    output: str | Path = TASK1_FIGURE_DIR / "cnn_preprocessing_macro_f1.png",
) -> Path

write_task1_confusion_figure(
    predictions: pd.DataFrame,
    class_names: Sequence[str],
    *,
    output: str | Path = TASK1_FIGURE_DIR / "cnn_oof_confusion_matrix.png",
) -> Path
```

Plot five fold macro-F1 points for each preprocessing ID plus the mean and sample
standard-deviation error bar. Label axes and preprocessing IDs clearly. Do not
write a confusion matrix until full OOF predictions exist. The confusion helper
uses `confusion_matrix(..., labels=np.arange(124), normalize="true")`, a large
figure, small class labels, and an explicit note that blank-support rows are zero.
Write one figure per preprocessing candidate by adding the preprocessing ID to
the output filename. Store the matching 124-row per-class CSV beside the other
Task 1 evidence.

Full mode also writes `fold_metrics.csv`, `comparison.csv`, and one
`per_class_<preprocessing_id>.csv` under `TASK1_EVIDENCE_DIR` using
`atomic_write_csv`. These files are report evidence and remain trackable.

- [ ] **Step 5: Run orchestration tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_experiments.py tests/task1/test_evaluation.py -q
./.venv/bin/python -m ruff check src/fashion/task1/experiments.py tests/task1/test_experiments.py
```

Expected: PASS.

- [ ] **Step 6: Commit orchestration**

```bash
git add src/fashion/task1/experiments.py src/fashion/task1/__init__.py tests/task1/test_experiments.py
git commit -m "feat(task1): orchestrate cnn baseline experiments"
```

---

### Task 7: Notebook Controller and Project Documentation

**Files:**
- Modify: `notebooks/02_task1_article_type.ipynb`
- Modify: `src/fashion/README.md`
- Modify: `tests/test_notebook_scaffolds.py`
- Create: `tests/task1/test_notebook_baseline.py`

**Interfaces:**
- Consumes: `load_splits`, `load_label_maps`, and `run_task1_experiment`.
- Produces: a narrative notebook with an explicit `RUN_MODE`, registered run table, and full-mode comparison evidence.

- [ ] **Step 1: Write failing notebook structure tests**

Parse the notebook JSON and assert that its joined source contains all of:

```python
required = (
    'RUN_MODE = "smoke"',
    "run_task1_experiment",
    "Task1SmallCNN",
    "macro-F1",
    "results/runs.csv",
    "task1_rgb_60x80_no_aug_v1",
    "task1_rgb_60x80_mild_aug_v1",
)
```

Also assert the joined source does not contain `train_test_split`,
`pretrained=True`, or a copied local training loop such as `optimizer.step()`.

- [ ] **Step 2: Run notebook tests and verify they fail**

```bash
./.venv/bin/python -m pytest tests/task1/test_notebook_baseline.py tests/test_notebook_scaffolds.py -q
```

Expected: FAIL because Section 8 onward still contains unresolved owner markers.

- [ ] **Step 3: Replace the baseline and experiment owner markers**

Update the notebook with these controller cells:

```python
from fashion.config import TASK1_FIGURE_DIR
from fashion.task1 import (
    Task1SmallCNN,
    run_task1_experiment,
    write_task1_comparison_figure,
    write_task1_confusion_figure,
)

RUN_MODE = "smoke"  # Change to "full" only for the ten report runs.

task1_experiment = run_task1_experiment(
    splits,
    load_label_maps()[TARGET],
    mode=RUN_MODE,
)

registered_runs = task1_experiment.fold_results
task1_experiment.comparison

if RUN_MODE == "full":
    comparison_figure = write_task1_comparison_figure(task1_experiment.fold_metrics)
    confusion_figures = {
        preprocessing_id: write_task1_confusion_figure(
            predictions,
            article_type_classes,
            output=(
                TASK1_FIGURE_DIR
                / f"cnn_oof_confusion_matrix_{preprocessing_id}.png"
            ),
        )
        for preprocessing_id, predictions in task1_experiment.oof_predictions.items()
    }
```

Explain that the hypothesis is: mild augmentation should improve mean five-fold
macro-F1 without a large rise in fold standard deviation. The rejection rule is
that the mean does not improve or variability grows enough to erase the gain.
Describe the CNN as a scratch baseline, not the chosen winner.

Update the experiment matrix with one smoke row and ten full physical runs. Leave
actual run IDs and scores generated from the registry; do not invent results.

- [ ] **Step 4: Document module ownership**

Add short entries to `src/fashion/README.md` for:

```text
task1/models.py       scratch small-CNN architecture
task1/dataset.py      validated Task 1 tensor samples
task1/evaluation.py   fixed-124-class metrics and OOF checks
task1/training.py     one-fold registered trainer
task1/experiments.py  smoke/full CV orchestration
train/                shared artifacts, seeds, and run registry
```

- [ ] **Step 5: Run notebook and documentation tests**

```bash
./.venv/bin/python -m pytest tests/task1/test_notebook_baseline.py tests/test_notebook_scaffolds.py tests/test_documentation.py -q
./.venv/bin/python -m ruff check src/fashion tests/task1/test_notebook_baseline.py
```

Expected: PASS.

- [ ] **Step 6: Commit notebook integration**

```bash
git add notebooks/02_task1_article_type.ipynb src/fashion/README.md tests/test_notebook_scaffolds.py tests/task1/test_notebook_baseline.py
git commit -m "docs(task1): wire cnn baseline into notebook"
```

---

### Task 8: Full Verification and Real-Data Smoke Run

**Files:**
- Generated and ignored: `results/runs.csv`
- Generated and ignored: `results/task1/<run_id>/checkpoint.pt`
- Generated and ignored: `results/task1/<run_id>/history.csv`
- Generated and ignored: `results/task1/<run_id>/predictions.csv`

**Interfaces:**
- Consumes: the completed implementation and local prepared data.
- Produces: one completed, non-final real-data smoke record proving the pipeline works before full training.

- [ ] **Step 1: Run the complete automated suite**

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m ruff check src tests
git diff --check
```

Expected: all tests PASS, Ruff reports no errors, and `git diff --check` prints
nothing.

- [ ] **Step 2: Run repository safety scans**

```bash
rg -n "train_test_split|pretrained\s*=\s*True" src notebooks
rg -n "fashion\.train\.registry|tracked_run" src/fashion/task1 src/fashion/train
```

Expected: the first command finds no forbidden implementation. The second shows
the Task 1 trainer entering the shared registry lifecycle.

- [ ] **Step 3: Commit any verification-only corrections, then require a clean tracked tree**

If verification required a code correction, repeat its focused failing test,
apply the smallest fix, rerun the full suite, and commit only that correction.
Then run:

```bash
git status --short --untracked-files=no
```

Expected: no tracked changes. This makes the smoke run's Git provenance clean.

- [ ] **Step 4: Run one real-data smoke experiment**

```bash
./.venv/bin/python - <<'PY'
from fashion.data.dataset import load_label_maps, load_splits
from fashion.task1 import run_task1_experiment

result = run_task1_experiment(
    load_splits(),
    load_label_maps()["articleType"],
    mode="smoke",
)
fold = result.fold_results[0]
print(fold.run_id)
print(fold.metrics)
print(fold.checkpoint_path)
PY
```

Expected: one completed fold-0 smoke result, one checkpoint, one history file,
one prediction file, and no access to protected labels.

- [ ] **Step 5: Verify the real run record and artifacts**

```bash
./.venv/bin/python - <<'PY'
from pathlib import Path

from fashion.config import RUNS_CSV
from fashion.train import RunRegistry, verify_artifact

rows = RunRegistry(RUNS_CSV).find(task="task1", stage="smoke", final_eligible=False)
row = rows.iloc[-1]
assert row["status"] == "completed"
assert row["scratch"] == "true"
assert row["primary_metric_name"] == "macro_f1_124"
for path_column, hash_column in (
    ("checkpoint_path", "checkpoint_sha256"),
    ("prediction_path", "prediction_sha256"),
    ("history_path", "history_sha256"),
):
    verify_artifact(Path(row[path_column]), row[hash_column])
print(row[["run_id", "status", "primary_metric_value"]].to_dict())
PY
```

Expected: all assertions PASS and artifact hashes match.

- [ ] **Step 6: Hand off the full experiment command without running it automatically**

Report this exact command to the user:

```bash
./.venv/bin/python - <<'PY'
from fashion.data.dataset import load_label_maps, load_splits
from fashion.task1 import run_task1_experiment

result = run_task1_experiment(
    load_splits(),
    load_label_maps()["articleType"],
    mode="full",
)
print(result.comparison)
PY
```

Do not start the ten full runs unless the user asks. They are expensive and the
rubric rewards analysis of completed evidence over unnecessary extra runs.
