# Task 1 Gentle Class-Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one gentle class-weighted scratch-CNN candidate, train only its five new folds, merge it safely with the ten completed CNN folds, and show learning-curve evidence in Task 1.

**Architecture:** Represent preprocessing and loss as one explicit CNN candidate so runs that share image transforms cannot be mixed. Compute capped inverse-square-root weights from each fold's training rows, use them only for training loss, and keep validation loss and model selection unchanged. A weighted-only controller produces five new runs and atomically merges them with verified old evidence.

**Tech Stack:** Python 3.12, PyTorch, pandas, NumPy, scikit-learn metrics, Matplotlib, pytest, Jupyter notebooks.

**Spec:** `docs/superpowers/specs/2026-09-05-task1-gentle-class-weighting-design.md`

## Global Constraints

- Train every CNN from scratch; do not load pretrained weights.
- `data/processed/splits.csv` is the only split; never call `train_test_split`.
- Every physical run appends through `fashion.train.registry` to `results/runs.csv`.
- Keep the existing ten completed unweighted CNN folds and train only five new weighted folds.
- Calculate class weights from each fold's development-training rows only.
- Keep validation loss unweighted and select checkpoints by validation macro-F1 over all 124 classes.
- Do not read holdout or quarantine labels during development.
- Write report figures under `results/figures/task1/`.
- Use `./.venv/bin/python` for project commands.

---

### Task 1: Define loss weighting and CNN candidate identities

**Files:**
- Create: `src/fashion/task1/losses.py`
- Create: `src/fashion/task1/candidates.py`
- Create: `tests/task1/test_losses.py`
- Create: `tests/task1/test_candidates.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Produces: `Task1LossConfig`, `Task1LossWeights`, `build_task1_loss_weights(...)`, `TASK1_UNWEIGHTED_LOSS`, and `TASK1_GENTLE_WEIGHTED_LOSS`.
- Produces: `Task1CnnCandidate`, `TASK1_NO_AUG_CANDIDATE`, `TASK1_MILD_AUG_CANDIDATE`, and `TASK1_GENTLE_WEIGHTED_CANDIDATE`.
- Depends on: existing `Task1PreprocessingConfig`, `TASK1_CONTROL_PREPROCESSING`, and `DEFAULT_TASK1_PREPROCESSING`.

- [ ] **Step 1: Write failing loss-weight tests**

Add tests with literal expected values. The production break caught is using validation rows, full inverse weights, a wrong normalizer, no cap, or a non-zero absent-class weight.

```python
from __future__ import annotations

import pandas as pd
import pytest

from fashion.task1.losses import (
    TASK1_GENTLE_WEIGHTED_LOSS,
    TASK1_UNWEIGHTED_LOSS,
    build_task1_loss_weights,
)


def _training_rows() -> pd.DataFrame:
    labels = ["a"] + ["b"] * 4 + ["c"] * 9
    return pd.DataFrame(
        {
            "id": range(1, 15),
            "partition": ["development"] * 14,
            "cv_fold": [1] * 14,
            "articleType": labels,
        }
    )


def test_gentle_weights_use_sqrt_mean_normalisation_and_absent_zero() -> None:
    result = build_task1_loss_weights(
        _training_rows(),
        {"a": 0, "b": 1, "c": 2, "d": 3},
        validation_fold=0,
        config=TASK1_GENTLE_WEIGHTED_LOSS,
    )

    assert result.class_counts == (1, 4, 9, 0)
    assert result.class_weights == pytest.approx((1.6363636, 0.8181818, 0.5454545, 0.0))


def test_unweighted_loss_returns_no_training_tensor() -> None:
    result = build_task1_loss_weights(
        _training_rows(),
        {"a": 0, "b": 1, "c": 2, "d": 3},
        validation_fold=0,
        config=TASK1_UNWEIGHTED_LOSS,
    )
    assert result.tensor is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"partition": "holdout"},
        {"cv_fold": 0},
    ],
)
def test_weights_reject_rows_outside_the_fold_training_set(mutation: dict[str, object]) -> None:
    rows = _training_rows()
    for column, value in mutation.items():
        rows.loc[rows.index[0], column] = value
    with pytest.raises(ValueError, match="development training rows"):
        build_task1_loss_weights(
            rows,
            {"a": 0, "b": 1, "c": 2, "d": 3},
            validation_fold=0,
            config=TASK1_GENTLE_WEIGHTED_LOSS,
        )
```

- [ ] **Step 2: Run the loss tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_losses.py -q
```

Expected: collection fails because `fashion.task1.losses` does not exist.

- [ ] **Step 3: Implement the loss types and weight builder**

Create these public shapes in `losses.py`:

```python
@dataclass(frozen=True)
class Task1LossConfig:
    loss_id: str
    weighting: Literal["none", "sqrt_balanced"]
    minimum_weight: float = 0.25
    maximum_weight: float = 4.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Task1LossWeights:
    tensor: torch.Tensor | None
    class_counts: tuple[int, ...]
    class_weights: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "class_counts": list(self.class_counts),
            "class_weights": list(self.class_weights),
        }


TASK1_UNWEIGHTED_LOSS = Task1LossConfig(
    loss_id="cross_entropy_unweighted_v1", weighting="none"
)
TASK1_GENTLE_WEIGHTED_LOSS = Task1LossConfig(
    loss_id="cross_entropy_sqrt_class_weighted_v1", weighting="sqrt_balanced"
)
```

Implement `build_task1_loss_weights(training_rows, label_to_index, *, validation_fold, config)` with these exact operations:

```python
counts = np.zeros(len(label_to_index), dtype=np.int64)
for label, count in training_rows["articleType"].value_counts().items():
    counts[label_to_index[str(label)]] = int(count)
present = counts > 0
weights = np.zeros(len(counts), dtype=np.float64)
if config.weighting == "sqrt_balanced":
    median = float(np.median(counts[present]))
    raw = np.sqrt(median / counts[present])
    normalised = raw / raw.mean()
    weights[present] = np.clip(
        normalised, config.minimum_weight, config.maximum_weight
    )
    tensor = torch.tensor(weights, dtype=torch.float32)
else:
    tensor = None
```

Validate non-empty rows, development-only partition, no row with `cv_fold == validation_fold`, known labels, finite bounds, `0 < minimum_weight <= maximum_weight`, and a non-empty set of present classes.

- [ ] **Step 4: Write failing candidate-identity tests**

```python
from fashion.task1.candidates import (
    TASK1_GENTLE_WEIGHTED_CANDIDATE,
    TASK1_MILD_AUG_CANDIDATE,
    TASK1_NO_AUG_CANDIDATE,
)


def test_three_cnn_candidates_have_distinct_explicit_identities() -> None:
    candidates = (
        TASK1_NO_AUG_CANDIDATE,
        TASK1_MILD_AUG_CANDIDATE,
        TASK1_GENTLE_WEIGHTED_CANDIDATE,
    )
    assert [candidate.candidate_id for candidate in candidates] == [
        "task1_cnn_no_aug_unweighted_v1",
        "task1_cnn_mild_aug_unweighted_v1",
        "task1_cnn_no_aug_sqrt_weighted_v1",
    ]
    assert TASK1_GENTLE_WEIGHTED_CANDIDATE.preprocessing == TASK1_NO_AUG_CANDIDATE.preprocessing
    assert TASK1_GENTLE_WEIGHTED_CANDIDATE.loss.loss_id != TASK1_NO_AUG_CANDIDATE.loss.loss_id
```

- [ ] **Step 5: Run the candidate test and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_candidates.py -q
```

Expected: collection fails because `fashion.task1.candidates` does not exist.

- [ ] **Step 6: Implement candidate identities and exports**

```python
@dataclass(frozen=True)
class Task1CnnCandidate:
    candidate_id: str
    preprocessing: Task1PreprocessingConfig
    loss: Task1LossConfig


TASK1_NO_AUG_CANDIDATE = Task1CnnCandidate(
    "task1_cnn_no_aug_unweighted_v1",
    TASK1_CONTROL_PREPROCESSING,
    TASK1_UNWEIGHTED_LOSS,
)
TASK1_MILD_AUG_CANDIDATE = Task1CnnCandidate(
    "task1_cnn_mild_aug_unweighted_v1",
    DEFAULT_TASK1_PREPROCESSING,
    TASK1_UNWEIGHTED_LOSS,
)
TASK1_GENTLE_WEIGHTED_CANDIDATE = Task1CnnCandidate(
    "task1_cnn_no_aug_sqrt_weighted_v1",
    TASK1_CONTROL_PREPROCESSING,
    TASK1_GENTLE_WEIGHTED_LOSS,
)
```

Validate non-empty IDs and export the new public types/constants/functions from `fashion.task1`.

- [ ] **Step 7: Run Task 1 tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_losses.py tests/task1/test_candidates.py -q
```

Expected: PASS.

Commit:

```bash
git add src/fashion/task1/losses.py src/fashion/task1/candidates.py src/fashion/task1/__init__.py tests/task1/test_losses.py tests/task1/test_candidates.py
git commit -m "feat(task1): define gentle class-weighted candidate"
```

---

### Task 2: Apply weights only to CNN training loss

**Files:**
- Modify: `src/fashion/task1/cnn_engine.py:125-240`
- Modify: `tests/task1/test_cnn_engine.py`

**Interfaces:**
- Consumes: `Task1LossWeights.tensor` from Task 1.
- Produces: `train_task1_cnn(..., training_class_weights: torch.Tensor | None = None)`.
- Preserves: `_evaluate(...)` always calls unweighted cross-entropy.

- [ ] **Step 1: Write failing engine tests**

Add a direct validation test and a real one-epoch integration test. The integration mutation caught is ignoring `training_class_weights`.

```python
import fashion.task1.cnn_engine as cnn_engine
import torch.nn.functional as functional
from fashion.train.reproducibility import seed_everything


def test_validation_cross_entropy_is_unweighted() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([0, 1])
    expected = functional.cross_entropy(logits, target)
    assert torch.equal(
        cnn_engine.task1_validation_cross_entropy(logits, target), expected
    )


def test_training_class_weights_change_real_one_epoch_updates(tmp_path: Path) -> None:
    splits = _splits_with_images(tmp_path)
    label_map = _label_map()
    training_rows, validation_rows = get_task1_fold_rows(splits, validation_fold=0)
    normalisation = fit_task1_normalization(
        training_rows,
        validation_fold=0,
        root=tmp_path,
        config=TASK1_CONTROL_PREPROCESSING,
    )
    weights = torch.ones(124)
    weights[0] = 4.0

    seed_everything(2753)
    plain = train_task1_cnn(
        training_rows,
        validation_rows,
        label_map["label_to_index"],
        label_map["classes"],
        normalization=normalisation,
        preprocessing=TASK1_CONTROL_PREPROCESSING,
        config=Task1TrainConfig.smoke(),
        root=tmp_path,
        device=torch.device("cpu"),
    )
    seed_everything(2753)
    weighted = train_task1_cnn(
        training_rows,
        validation_rows,
        label_map["label_to_index"],
        label_map["classes"],
        normalization=normalisation,
        preprocessing=TASK1_CONTROL_PREPROCESSING,
        config=Task1TrainConfig.smoke(),
        root=tmp_path,
        device=torch.device("cpu"),
        training_class_weights=weights,
    )

    assert not torch.equal(plain.model.fc2.weight, weighted.model.fc2.weight)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_cnn_engine.py -q
```

Expected: FAIL because the new keyword and validation-loss helper do not exist.

- [ ] **Step 3: Implement weighted training and fixed validation loss**

Add:

```python
def task1_validation_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return functional.cross_entropy(logits, target)
```

Extend `train_task1_cnn` with:

```python
training_class_weights: torch.Tensor | None = None,
```

Before training, require a weighted tensor to have shape `(124,)`, finite values, and no negative values, then move it to `device`. Change only the training call:

```python
loss = functional.cross_entropy(logits, target, weight=device_class_weights)
```

Change `_evaluate` to call `task1_validation_cross_entropy(logits, target)` without weights.

- [ ] **Step 4: Run engine tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_cnn_engine.py -q
```

Expected: PASS.

Commit:

```bash
git add src/fashion/task1/cnn_engine.py tests/task1/test_cnn_engine.py
git commit -m "feat(task1): apply class weights to training loss"
```

---

### Task 3: Register candidate and loss provenance per fold

**Files:**
- Modify: `src/fashion/task1/training.py:34-283`
- Modify: `tests/task1/test_training.py`

**Interfaces:**
- Consumes: `Task1CnnCandidate` and `build_task1_loss_weights(...)`.
- Produces: `train_task1_fold(..., candidate: Task1CnnCandidate, ...)`.
- Produces: `Task1FoldResult` fields `candidate_id`, `preprocessing_id`, and `loss_id`.

- [ ] **Step 1: Write failing registry/checkpoint tests**

Update test calls to pass `candidate=TASK1_NO_AUG_CANDIDATE`. Add:

```python
def test_weighted_fold_records_loss_candidate_and_fold_only_weights(tmp_path: Path) -> None:
    splits = _splits_with_images(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits.to_csv(split_path, index=False)
    registry = RunRegistry(tmp_path / "runs.csv")
    result = train_task1_fold(
        splits,
        _label_map(),
        validation_fold=0,
        candidate=TASK1_GENTLE_WEIGHTED_CANDIDATE,
        config=Task1TrainConfig.smoke(),
        registry=registry,
        root=tmp_path,
        result_root=tmp_path / "results",
        device=torch.device("cpu"),
        split_path=split_path,
    )

    row = registry.read().iloc[0]
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert result.candidate_id == "task1_cnn_no_aug_sqrt_weighted_v1"
    assert result.loss_id == "cross_entropy_sqrt_class_weighted_v1"
    assert row["loss_id"] == result.loss_id
    assert row["experiment_id"] == "task1-cnn-task1_cnn_no_aug_sqrt_weighted_v1"
    assert checkpoint["candidate_id"] == result.candidate_id
    assert checkpoint["loss"]["config"]["loss_id"] == result.loss_id
    assert len(checkpoint["loss"]["class_weights"]) == 124
```

Use the test fixture's saved split path and keep this smoke run non-final eligible, so canonical production split enforcement does not interfere with the isolated test.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_training.py -q
```

Expected: FAIL because `candidate` and the new result fields are missing.

- [ ] **Step 3: Implement candidate-aware fold training**

Replace the `preprocessing` argument with `candidate`, then derive:

```python
preprocessing = candidate.preprocessing
loss_weights = build_task1_loss_weights(
    training_rows,
    label_to_index,
    validation_fold=validation_fold,
    config=candidate.loss,
)
```

Pass `loss_weights.tensor` to the engine. Use:

```python
experiment_id = f"task1-cnn-{candidate.candidate_id}"
```

Add candidate and loss dictionaries to `run_config`; set `RunRecord.loss_id` from the candidate;
store `candidate_id` and
`loss={"config": candidate.loss.to_dict(), **loss_weights.to_dict()}` in the checkpoint; and
return all three identity fields.

Final-eligible validation accepts only the three declared candidate constants, the canonical split, `Task1SmallCNN`, seed 2753, 20 epochs, and the existing fixed optimizer values.

Add `src/fashion/task1/losses.py` and `src/fashion/task1/candidates.py` to `_implementation_hashes()`.

- [ ] **Step 4: Run training tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_training.py tests/task1/test_cnn_engine.py -q
```

Expected: PASS.

Commit:

```bash
git add src/fashion/task1/training.py tests/task1/test_training.py
git commit -m "feat(task1): register CNN candidate loss provenance"
```

---

### Task 4: Make existing CNN evidence candidate-aware

**Files:**
- Modify: `src/fashion/task1/cnn_experiments.py`
- Modify: `tests/task1/test_cnn_experiments.py`

**Interfaces:**
- Consumes: `TASK1_NO_AUG_CANDIDATE` and `TASK1_MILD_AUG_CANDIDATE`.
- Preserves: `run_task1_experiment(..., mode="smoke" | "full")` for the existing two-candidate experiment.
- Produces: evidence keyed by `candidate_id` while retaining preprocessing and loss columns.

- [ ] **Step 1: Change schedule tests first**

Change `_FoldCall` to hold `candidate_id`, `transform_id`, `loss_id`, and `stage`. Change the fake fold runner to accept `candidate` and return all identity fields. Set these literal expectations:

```python
assert [call.candidate_id for call in calls] == [
    "task1_cnn_no_aug_unweighted_v1"
] * 5 + ["task1_cnn_mild_aug_unweighted_v1"] * 5
assert list(full.fold_metrics.columns[:5]) == [
    "run_id", "fold", "candidate_id", "preprocessing_id", "loss_id"
]
assert set(full.oof_predictions) == {
    "task1_cnn_no_aug_unweighted_v1",
    "task1_cnn_mild_aug_unweighted_v1",
}
```

Rename the coverage test expectation from “per preprocessing” to “per candidate.”

- [ ] **Step 2: Run and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_cnn_experiments.py -q
```

Expected: FAIL because the controller still schedules and groups by preprocessing only.

- [ ] **Step 3: Refactor the existing controller minimally**

Schedule `Task1CnnCandidate` objects and pass `candidate=candidate` into the fold runner. Include all identity fields in `_fold_metrics_frame`. Replace `preprocessing_ids` with `candidate_ids` in aggregation, OOF dictionaries, and per-class filenames. Keep preprocessing and loss as descriptive columns, but group and validate by `candidate_id`.

The old full controller still schedules exactly ten runs; it is not used for the new five-fold execution.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_cnn_experiments.py tests/task1/test_training.py -q
```

Expected: PASS.

Commit:

```bash
git add src/fashion/task1/cnn_experiments.py tests/task1/test_cnn_experiments.py
git commit -m "refactor(task1): key CNN evidence by candidate"
```

---

### Task 5: Add the weighted-only runner and safe evidence merge

**Files:**
- Create: `src/fashion/task1/weighted_experiments.py`
- Create: `tests/task1/test_weighted_experiments.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Produces: `run_task1_weighted_experiment(splits, label_map, *, mode, registry=None, root=ROOT, result_root=TASK1_RESULT_DIR, evidence_root=TASK1_EVIDENCE_DIR, fold_runner=train_task1_fold) -> Task1ExperimentResult`.
- Consumes: completed old fold evidence, `RunRegistry`, canonical hashes, three candidate constants, and the candidate-aware evidence builders from Task 4.
- Writes: common combined evidence only after all 15 folds and OOF artifacts validate.

- [ ] **Step 1: Write failing weighted schedule tests**

Use the existing fake prediction approach, but seed an old fold frame with ten rows and registry-compatible artifact records. Assert:

```python
smoke = run_task1_weighted_experiment(
    splits,
    label_map,
    mode="smoke",
    fold_runner=fake_fold_runner,
    result_root=tmp_path / "runs",
    evidence_root=tmp_path / "evidence",
)
assert [(row.fold, row.candidate_id) for row in smoke.fold_results] == [
    (0, "task1_cnn_no_aug_sqrt_weighted_v1")
]
assert smoke.comparison.empty

full = run_task1_weighted_experiment(
    splits,
    label_map,
    mode="full",
    registry=registry,
    fold_runner=fake_fold_runner,
    result_root=tmp_path / "runs",
    evidence_root=tmp_path / "evidence",
)
assert len(full.fold_results) == 15
assert len(full.comparison) == 3
assert set(full.fold_metrics["candidate_id"]) == {
    "task1_cnn_no_aug_unweighted_v1",
    "task1_cnn_mild_aug_unweighted_v1",
    "task1_cnn_no_aug_sqrt_weighted_v1",
}
```

Also add separate tests that corrupt an old registry loss ID, prediction SHA-256, fold number, and OOF ID. Each must raise before any shared evidence file changes. Capture file bytes before the call and compare them after the failure.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_weighted_experiments.py -q
```

Expected: collection fails because `fashion.task1.weighted_experiments` does not exist.

- [ ] **Step 3: Implement registry-backed old-result loading**

Implement these private helpers. `_verified_prediction_path` must be complete before
`_load_verified_old_results` calls it:

```python
from fashion.data.hashing import compute_sha256


def _resolve_artifact_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _verified_prediction_path(row: pd.Series, *, root: Path) -> Path:
    value = str(row["prediction_path"]).strip()
    if not value:
        raise ValueError("completed Task 1 fold is missing a prediction path")
    path = _resolve_artifact_path(value, root=root)
    if not path.is_file():
        raise ValueError(f"Task 1 prediction artifact is missing: {path}")
    if compute_sha256(path) != str(row["prediction_sha256"]):
        raise ValueError(f"Task 1 prediction SHA-256 mismatch: {path}")
    return path


def _load_verified_old_results(
    *, registry: RunRegistry, evidence_root: Path, root: Path, split_sha256: str
) -> tuple[Task1FoldResult, ...]:
    evidence = pd.read_csv(evidence_root / "fold_metrics.csv", keep_default_na=False)
    if "candidate_id" not in evidence:
        evidence["candidate_id"] = evidence["preprocessing_id"].map(
            {
                TASK1_CONTROL_PREPROCESSING.preprocessing_id:
                    TASK1_NO_AUG_CANDIDATE.candidate_id,
                DEFAULT_TASK1_PREPROCESSING.preprocessing_id:
                    TASK1_MILD_AUG_CANDIDATE.candidate_id,
            }
        )
        evidence["loss_id"] = TASK1_UNWEIGHTED_LOSS.loss_id
    expected = {
        TASK1_NO_AUG_CANDIDATE.candidate_id,
        TASK1_MILD_AUG_CANDIDATE.candidate_id,
    }
    if set(evidence["candidate_id"]) != expected:
        raise ValueError("old CNN evidence must contain exactly the two unweighted candidates")

    registry_rows = registry.read().set_index("run_id", drop=False)
    results: list[Task1FoldResult] = []
    metric_columns = [
        "macro_f1", "weighted_f1", "top1_accuracy", "top5_accuracy", "validation_loss"
    ]
    for item in evidence.itertuples(index=False):
        if item.run_id not in registry_rows.index:
            raise ValueError(f"old CNN run is missing from registry: {item.run_id}")
        row = registry_rows.loc[item.run_id]
        if (
            row["status"] != "completed"
            or row["final_eligible"] != "true"
            or row["split_sha256"] != split_sha256
            or row["loss_id"] != TASK1_UNWEIGHTED_LOSS.loss_id
        ):
            raise ValueError(f"old CNN registry identity is invalid: {item.run_id}")
        prediction_path = _verified_prediction_path(row, root=root)
        results.append(
            Task1FoldResult(
                run_id=item.run_id,
                fold=int(item.fold),
                candidate_id=str(item.candidate_id),
                preprocessing_id=str(item.preprocessing_id),
                loss_id=str(item.loss_id),
                status="completed",
                metrics={name: float(getattr(item, name)) for name in metric_columns},
                checkpoint_path=_resolve_artifact_path(row["checkpoint_path"], root=root),
                history_path=_resolve_artifact_path(row["history_path"], root=root),
                prediction_path=prediction_path,
            )
        )
    return tuple(results)
```

Read `evidence_root / "fold_metrics.csv"`, accept exactly the two old candidate IDs and folds 0–4, and match every run ID to one completed, final-eligible registry row. Require canonical split hash, unweighted loss ID, expected transform, non-empty prediction/history/checkpoint paths, and matching SHA-256 values. Recreate `Task1FoldResult` using metrics parsed from the evidence row and paths resolved under `root`.

For backward compatibility with the current evidence, allow rows without `candidate_id` and `loss_id` only when their preprocessing ID maps uniquely to one of the two unweighted candidates. Immediately emit the new explicit columns in the merged output.

- [ ] **Step 4: Implement the weighted runner and atomic merge**

For smoke, run weighted fold 0 only and return no aggregates. For full, schedule weighted folds 0–4. After they succeed, concatenate the ten verified old fold results with the five new results and call the shared candidate-aware evidence builders. Pass `evidence_root` explicitly into the evidence writer; do not rely on a patched module constant.

Write each output to a sibling temporary path, then replace final evidence only after every frame, per-class table, and OOF prediction set has validated. Do not alter old physical artifacts or registry rows.

- [ ] **Step 5: Run weighted-controller tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_weighted_experiments.py tests/task1/test_cnn_experiments.py -q
```

Expected: PASS.

Commit:

```bash
git add src/fashion/task1/weighted_experiments.py src/fashion/task1/__init__.py tests/task1/test_weighted_experiments.py
git commit -m "feat(task1): add incremental weighted experiment runner"
```

---

### Task 6: Add candidate comparison and learning-curve figures

**Files:**
- Modify: `src/fashion/task1/plotting.py`
- Modify: `tests/task1/test_plotting.py`
- Modify: `src/fashion/task1/__init__.py`

**Interfaces:**
- Changes: `write_task1_comparison_figure` groups by `candidate_id`.
- Produces: `write_task1_learning_curve_figure(histories: Mapping[str, Sequence[pd.DataFrame]], *, output: str | Path) -> Path`.
- Consumes: existing history columns `epoch`, `train_loss`, `validation_loss`, and `macro_f1`.

- [ ] **Step 1: Write failing plotting tests**

```python
def test_learning_curve_figure_accepts_existing_history_schema(tmp_path: Path) -> None:
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [2.0, 1.0, 0.5],
            "validation_loss": [2.1, 1.2, 1.4],
            "macro_f1": [0.1, 0.3, 0.25],
        }
    )
    output = write_task1_learning_curve_figure(
        {"task1_cnn_no_aug_unweighted_v1": [history] * 5},
        output=tmp_path / "learning.png",
    )
    assert output.is_file()
    assert output.stat().st_size > 1_000
```

Update the comparison test fixture to use `candidate_id` and assert a candidate comparison file is produced from exactly five rows per candidate.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_plotting.py -q
```

Expected: FAIL because the learning-curve writer is missing and comparison still requires preprocessing IDs.

- [ ] **Step 3: Implement the figures**

The learning figure has two panels:

- Panel 1: per-candidate mean training loss as dashed lines and mean unweighted validation loss as solid lines; add a light ±1 sample-standard-deviation band when five histories are present.
- Panel 2: per-candidate mean validation macro-F1 with the same fold band.

Use epochs on both x-axes, “Loss” on the first y-axis, and “Validation macro-F1 (124 classes)” on the second. Use candidate IDs in the legend and save at 180 DPI. Validate equal epoch grids within a candidate and required columns before creating output directories.

- [ ] **Step 4: Run plotting tests and commit**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_plotting.py -q
```

Expected: PASS.

Commit:

```bash
git add src/fashion/task1/plotting.py src/fashion/task1/__init__.py tests/task1/test_plotting.py
git commit -m "feat(task1): plot CNN learning curves by candidate"
```

---

### Task 7: Reorder the Task 1 notebook and add the weighted run controls

**Files:**
- Modify: `notebooks/02_task1_article_type.ipynb`
- Modify: `tests/task1/test_notebook_baseline.py`

**Interfaces:**
- Consumes: `run_task1_weighted_experiment` and `write_task1_learning_curve_figure`.
- Preserves: default Run All performs smoke work only; five-fold work requires an explicit mode change.
- Produces: the approved eight-part experiment story and current combined evidence displays.

- [ ] **Step 1: Write failing notebook contract tests**

Add assertions based on parsed cells, not raw JSON text:

```python
def test_task1_notebook_orders_diagnosis_before_weighted_experiment() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    headings = [
        line.strip()
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("## ")
    ]
    assert headings.index("## 8. Learning-curve diagnosis") < headings.index(
        "## 9. Gentle class-weighted loss"
    )


def test_task1_notebook_defaults_weighted_controller_to_smoke() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'WEIGHTED_MODE = "smoke"' in source
    assert "run_task1_weighted_experiment(" in source
    assert "train_test_split" not in source
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_notebook_baseline.py -q
```

Expected: FAIL because the weighted section and controller are absent.

- [ ] **Step 3: Update notebook cells**

Keep EDA and safety cells. Reorder the experiment narrative to:

1. Classical baselines.
2. No-augmentation scratch CNN.
3. Learning-curve diagnosis.
4. Mild augmentation.
5. Gentle class weighting.
6. Combined five-fold and OOF comparison.
7. Weak-class/confusion analysis.
8. Development decision and Notebook 06 handoff.

Add this safe weighted controller cell:

```python
from fashion.task1 import run_task1_weighted_experiment

WEIGHTED_MODE = "smoke"  # Use "full" only for the five new weighted folds.
weighted_experiment = run_task1_weighted_experiment(
    splits,
    article_type_map,
    mode=WEIGHTED_MODE,
)
display(weighted_experiment.fold_metrics)
if not weighted_experiment.comparison.empty:
    display(weighted_experiment.comparison)
    display(weighted_experiment.oof_metrics)
```

Add a learning-curve cell that loads each `history_path` from the 15 combined fold rows through matching registry records and calls `write_task1_learning_curve_figure`. Guard it with an existence/completeness check and print a short “complete weighted full first” message when only smoke evidence exists.

Do not hand-write weighted scores. State the known augmentation result exactly: lower mean validation loss and lower fold variance, but current mean macro-F1 changed from 0.5291 to 0.5218 and therefore did not pass the macro-F1 improvement rule.

- [ ] **Step 4: Validate notebook and tests**

Run:

```bash
./.venv/bin/python -m pytest tests/task1/test_notebook_baseline.py -q
./.venv/bin/python -c "import nbformat; p='notebooks/02_task1_article_type.ipynb'; n=nbformat.read(p, as_version=4); nbformat.validate(n); print('Notebook 02 valid')"
```

Expected: tests PASS and `Notebook 02 valid`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/02_task1_article_type.ipynb tests/task1/test_notebook_baseline.py
git commit -m "docs(task1): add weighted experiment story"
```

---

### Task 8: Verify code, run the new experiment, and inspect evidence

**Files:**
- Append: `results/runs.csv` through the registry only
- Create: `results/task1/<weighted-run-id>/...` for one smoke and five full folds
- Update: `results/evidence/task1/fold_metrics.csv`
- Update: `results/evidence/task1/comparison.csv`
- Update: `results/evidence/task1/oof_metrics.csv`
- Create: `results/evidence/task1/per_class_task1_cnn_no_aug_sqrt_weighted_v1.csv`
- Update/Create: `results/figures/task1/cnn_candidate_macro_f1.png`
- Create: `results/figures/task1/cnn_learning_curves.png`

**Interfaces:**
- Consumes: all prior tasks and canonical project data.
- Produces: one non-reportable weighted smoke row, five final-eligible weighted rows, combined 15-fold evidence, and report figures.

- [ ] **Step 1: Run focused and full static verification**

Run:

```bash
./.venv/bin/python -m pytest tests/task1 -q
./.venv/bin/python -m ruff check src/fashion/task1 tests/task1
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 2: Run the weighted smoke stage**

Run:

```bash
./.venv/bin/python -c "from fashion.data.dataset import load_label_maps, load_splits; from fashion.task1 import run_task1_weighted_experiment; r=run_task1_weighted_experiment(load_splits(), load_label_maps()['articleType'], mode='smoke'); print(r.fold_metrics.to_string(index=False))"
```

Expected: one completed fold-0 row with candidate ID `task1_cnn_no_aug_sqrt_weighted_v1`, loss ID `cross_entropy_sqrt_class_weighted_v1`, and `final_eligible=false` in the registry.

- [ ] **Step 3: Check the smoke artifacts before full training**

Run:

```bash
./.venv/bin/python -c "import pandas as pd; r=pd.read_csv('results/runs.csv', keep_default_na=False); x=r[(r.task=='task1') & (r.loss_id=='cross_entropy_sqrt_class_weighted_v1')].tail(1); assert len(x)==1 and x.iloc[0].status=='completed' and str(x.iloc[0].final_eligible).lower()=='false'; print(x[['run_id','status','loss_id','primary_metric_value']].to_string(index=False))"
```

Expected: assertions pass and the row is printed.

- [ ] **Step 4: Run only the five weighted full folds**

Run:

```bash
./.venv/bin/python -c "from fashion.data.dataset import load_label_maps, load_splits; from fashion.task1 import run_task1_weighted_experiment; r=run_task1_weighted_experiment(load_splits(), load_label_maps()['articleType'], mode='full'); print(r.comparison.to_string(index=False)); print(r.oof_metrics.to_string(index=False))"
```

Expected: five new completed final-eligible rows and a three-candidate comparison. This command can run for a long time; do not interrupt it while a fold is training.

- [ ] **Step 5: Verify combined evidence and registry counts**

Run:

```bash
./.venv/bin/python -c "import pandas as pd; f=pd.read_csv('results/evidence/task1/fold_metrics.csv'); c=pd.read_csv('results/evidence/task1/comparison.csv'); r=pd.read_csv('results/runs.csv', keep_default_na=False); assert len(f)==15 and set(f.groupby('candidate_id').size())=={5}; assert len(c)==3; w=r[(r.task=='task1') & (r.stage=='experiment') & (r.loss_id=='cross_entropy_sqrt_class_weighted_v1') & (r.status=='completed')]; assert set(w.fold.astype(int))==set(range(5)); print(c.to_string(index=False))"
```

Expected: assertions pass and the three comparison rows print.

- [ ] **Step 6: Generate and inspect figures**

Run the notebook figure cell or the public plotting functions against the combined evidence. Confirm both PNGs exist, contain readable labels, use all five folds per candidate, and do not claim training macro-F1 exists.

- [ ] **Step 7: Run final regression verification**

Run:

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m ruff check src tests
git diff --check
```

Expected: all tests pass, Ruff reports no errors, and `git diff --check` prints nothing.

- [ ] **Step 8: Commit source and notebook work only**

Do not commit generated checkpoints, predictions, histories, cache files, or registry rows unless the repository policy explicitly tracks them.

```bash
git status --short
git add src/fashion/task1 tests/task1 notebooks/02_task1_article_type.ipynb
git commit -m "feat(task1): compare gentle class-weighted CNN"
```

Review the staged paths before committing so unrelated user changes remain untouched.
