"""Run only the new weighted candidate, then safely merge verified CNN evidence."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fashion.config import ROOT, SPLITS_CSV, TASK1_EVIDENCE_DIR, TASK1_RESULT_DIR
from fashion.data.dataset import get_samples
from fashion.data.hashing import compute_sha256
from fashion.task1.candidates import (
    TASK1_GENTLE_WEIGHTED_CANDIDATE,
    TASK1_MILD_AUG_CANDIDATE,
    TASK1_NO_AUG_CANDIDATE,
)
from fashion.task1.cnn_experiments import (
    Task1ExperimentResult,
    _aggregate_comparison,
    _fold_metrics_frame,
    _prediction_probabilities,
)
from fashion.task1.evaluation import (
    classification_metrics,
    per_class_metrics,
    validate_oof_predictions,
    validate_task1_label_map,
)
from fashion.task1.training import Task1FoldResult, Task1TrainConfig, train_task1_fold
from fashion.train.artifacts import atomic_write_csv, canonical_sha256
from fashion.train.registry import RunRegistry

_OLD_CANDIDATES = (TASK1_NO_AUG_CANDIDATE, TASK1_MILD_AUG_CANDIDATE)
_CANDIDATES = {
    candidate.candidate_id: candidate
    for candidate in (*_OLD_CANDIDATES, TASK1_GENTLE_WEIGHTED_CANDIDATE)
}
_METRIC_COLUMNS = ("macro_f1", "weighted_f1", "top1_accuracy", "top5_accuracy", "validation_loss")


def _resolve_artifact_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _verified_artifact_path(row: pd.Series, kind: str, *, root: Path) -> Path:
    value = str(row[f"{kind}_path"]).strip()
    if not value:
        raise ValueError(f"completed Task 1 fold is missing a {kind} path")
    path = _resolve_artifact_path(value, root=root)
    if not path.is_file():
        raise ValueError(f"Task 1 {kind} artifact is missing: {path}")
    if compute_sha256(path) != str(row[f"{kind}_sha256"]):
        raise ValueError(f"Task 1 {kind} SHA-256 mismatch: {path}")
    return path


def _verified_prediction_path(row: pd.Series, *, root: Path) -> Path:
    return _verified_artifact_path(row, "prediction", root=root)


def _fold_number(value: object) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Task 1 fold must be an integer from 0 to 4") from error
    if not np.isfinite(number) or not number.is_integer() or number not in range(5):
        raise ValueError("Task 1 fold must be an integer from 0 to 4")
    return int(number)


def _verified_result(
    item: pd.Series,
    registry_rows: pd.DataFrame,
    *,
    root: Path,
    split_sha256: str,
) -> Task1FoldResult:
    run_id = str(item["run_id"])
    if run_id not in registry_rows.index:
        raise ValueError(f"Task 1 run is missing from registry: {run_id}")
    candidate = _CANDIDATES.get(str(item["candidate_id"]))
    if candidate is None:
        raise ValueError(f"unknown Task 1 candidate: {item['candidate_id']}")
    fold = _fold_number(item["fold"])
    row = registry_rows.loc[run_id]
    expected = {
        "task": "task1",
        "stage": "experiment",
        "status": "completed",
        "final_eligible": "true",
        "scratch": "true",
        "benchmark_only": "false",
        "model_family": "task1_small_cnn_v1",
        "split_sha256": split_sha256,
        "loss_id": candidate.loss.loss_id,
        "transform_id": candidate.preprocessing.preprocessing_id,
    }
    if (
        any(row[name] != value for name, value in expected.items())
        or _fold_number(row["fold"]) != fold
        or item["preprocessing_id"] != candidate.preprocessing.preprocessing_id
        or item["loss_id"] != candidate.loss.loss_id
    ):
        raise ValueError(f"Task 1 registry identity is invalid: {run_id}")
    metrics = {name: float(item[name]) for name in _METRIC_COLUMNS}
    if any(not np.isfinite(value) or value < 0 for value in metrics.values()) or any(
        metrics[name] > 1 for name in _METRIC_COLUMNS if name != "validation_loss"
    ):
        raise ValueError(f"Task 1 fold metrics are invalid: {run_id}")
    return Task1FoldResult(
        run_id=run_id,
        fold=fold,
        candidate_id=candidate.candidate_id,
        preprocessing_id=candidate.preprocessing.preprocessing_id,
        loss_id=candidate.loss.loss_id,
        status="completed",
        metrics=metrics,
        prediction_path=_verified_prediction_path(row, root=root),
        checkpoint_path=_verified_artifact_path(row, "checkpoint", root=root),
        history_path=_verified_artifact_path(row, "history", root=root),
    )


def _load_verified_old_results(
    *,
    registry: RunRegistry,
    evidence_root: Path,
    root: Path,
    split_sha256: str,
) -> tuple[Task1FoldResult, ...]:
    """Accept legacy identity columns only at this registry-backed merge boundary."""
    evidence = pd.read_csv(evidence_root / "fold_metrics.csv", keep_default_na=False)
    required = {"run_id", "fold", "preprocessing_id", *_METRIC_COLUMNS}
    if missing := required.difference(evidence.columns):
        raise ValueError(f"old Task 1 fold evidence is missing columns: {sorted(missing)}")
    identity_columns = {"candidate_id", "loss_id"}.intersection(evidence.columns)
    if not identity_columns:
        evidence["candidate_id"] = evidence["preprocessing_id"].map(
            {
                candidate.preprocessing.preprocessing_id: candidate.candidate_id
                for candidate in _OLD_CANDIDATES
            }
        )
        evidence["loss_id"] = TASK1_NO_AUG_CANDIDATE.loss.loss_id
    elif len(identity_columns) != 2:
        raise ValueError("old Task 1 evidence must provide both candidate_id and loss_id")
    expected = {candidate.candidate_id for candidate in _OLD_CANDIDATES}
    if set(evidence["candidate_id"]) != expected:
        raise ValueError("old CNN evidence must contain exactly the two unweighted candidates")
    if evidence["run_id"].duplicated().any():
        raise ValueError("old CNN evidence contains duplicate run IDs")
    evidence["fold"] = evidence["fold"].map(_fold_number)
    _aggregate_comparison(evidence, list(expected))
    registry_rows = registry.read().set_index("run_id", drop=False)
    return tuple(
        _verified_result(item, registry_rows, root=root, split_sha256=split_sha256)
        for _, item in evidence.iterrows()
    )


def _validate_label_identity(
    results: Sequence[Task1FoldResult],
    registry: RunRegistry,
    label_map: Mapping[str, object],
) -> None:
    rows = registry.read().set_index("run_id")
    expected = canonical_sha256(label_map)
    if any(rows.loc[result.run_id, "label_map_sha256"] != expected for result in results):
        raise ValueError("Task 1 registry label-map SHA-256 mismatch")


def _build_full_evidence(
    results: tuple[Task1FoldResult, ...],
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    candidate_ids: Sequence[str],
) -> Task1ExperimentResult:
    """Build every table and verify each fold's OOF membership before any writes."""
    if len({result.run_id for result in results}) != len(results):
        raise ValueError("Task 1 evidence contains duplicate run IDs")
    fold_metrics = _fold_metrics_frame(results)
    comparison = _aggregate_comparison(fold_metrics, candidate_ids)
    development = get_samples(splits, partition="development", target="articleType")
    expected_ids = development["id"].tolist()
    label_to_index, class_names = validate_task1_label_map(label_map)
    oof_predictions: dict[str, pd.DataFrame] = {}
    per_class: dict[str, pd.DataFrame] = {}
    oof_rows = []
    for candidate_id in candidate_ids:
        candidate = _CANDIDATES[candidate_id]
        predictions_by_fold = []
        for result in results:
            if result.candidate_id != candidate_id:
                continue
            predictions = pd.read_csv(result.prediction_path)
            fold_ids = development.loc[development["cv_fold"].eq(result.fold), "id"].tolist()
            validate_oof_predictions(predictions, fold_ids)
            predictions_by_fold.append(predictions)
        predictions = pd.concat(predictions_by_fold, ignore_index=True)
        validate_oof_predictions(predictions, expected_ids)
        probabilities = _prediction_probabilities(predictions)
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        ):
            raise ValueError("Task 1 prediction probabilities are invalid")
        labels = pd.to_numeric(predictions["true_index"], errors="raise").to_numpy()
        if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
            raise ValueError("Task 1 prediction true indexes must be finite integers")
        if "articleType" in development:
            expected_labels = (
                predictions["id"]
                .map(development.set_index("id")["articleType"].map(label_to_index))
                .to_numpy()
            )
            if not np.array_equal(labels, expected_labels):
                raise ValueError("Task 1 prediction true indexes do not match development labels")
        labels = labels.astype(np.int64)
        pooled = classification_metrics(labels, probabilities)
        oof_rows.append(
            {
                "candidate_id": candidate_id,
                "preprocessing_id": candidate.preprocessing.preprocessing_id,
                "loss_id": candidate.loss.loss_id,
                "macro_f1_124": pooled["macro_f1"],
                **{
                    name: pooled[name] for name in ("weighted_f1", "top1_accuracy", "top5_accuracy")
                },
            }
        )
        oof_predictions[candidate_id] = predictions
        per_class[candidate_id] = per_class_metrics(labels, probabilities, class_names)
    return Task1ExperimentResult(
        mode="full",
        fold_results=results,
        fold_metrics=fold_metrics,
        comparison=comparison,
        oof_metrics=pd.DataFrame(oof_rows),
        oof_predictions=oof_predictions,
        per_class=per_class,
    )


def _write_full_evidence(result: Task1ExperimentResult, *, evidence_root: Path) -> None:
    """Stage all CSVs beside their destinations before atomically replacing files."""
    frames = {
        "fold_metrics.csv": result.fold_metrics,
        "comparison.csv": result.comparison,
        "oof_metrics.csv": result.oof_metrics,
        **{
            f"per_class_{candidate_id}.csv": frame
            for candidate_id, frame in result.per_class.items()
        },
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for name, frame in frames.items():
            with tempfile.NamedTemporaryFile(
                dir=evidence_root,
                prefix=f".{name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            staged.append((temporary, evidence_root / name))
            atomic_write_csv(temporary, frame)
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def run_task1_weighted_experiment(
    splits: pd.DataFrame,
    label_map: Mapping[str, object],
    *,
    mode: Literal["smoke", "full"],
    registry: RunRegistry | None = None,
    root: str | Path = ROOT,
    result_root: str | Path = TASK1_RESULT_DIR,
    evidence_root: str | Path = TASK1_EVIDENCE_DIR,
    fold_runner: Callable[..., Task1FoldResult] = train_task1_fold,
) -> Task1ExperimentResult:
    """Run weighted smoke/five folds; merge only complete, verified fifteen-fold evidence."""
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    active_registry = registry if registry is not None else RunRegistry()
    project_root, evidence_directory = Path(root), Path(evidence_root)
    old_results: tuple[Task1FoldResult, ...] = ()
    if mode == "full":
        split_sha256 = compute_sha256(SPLITS_CSV)
        old_results = _load_verified_old_results(
            registry=active_registry,
            evidence_root=evidence_directory,
            root=project_root,
            split_sha256=split_sha256,
        )
        _validate_label_identity(old_results, active_registry, label_map)
        _build_full_evidence(
            old_results,
            splits,
            label_map,
            [candidate.candidate_id for candidate in _OLD_CANDIDATES],
        )
    config = Task1TrainConfig.smoke() if mode == "smoke" else Task1TrainConfig.full()
    new_results = tuple(
        fold_runner(
            splits,
            label_map,
            validation_fold=fold,
            candidate=TASK1_GENTLE_WEIGHTED_CANDIDATE,
            config=config,
            registry=active_registry,
            root=root,
            result_root=result_root,
        )
        for fold in (range(1) if mode == "smoke" else range(5))
    )
    if mode == "smoke":
        return Task1ExperimentResult(
            mode=mode,
            fold_results=new_results,
            fold_metrics=_fold_metrics_frame(new_results),
            comparison=pd.DataFrame(),
            oof_metrics=pd.DataFrame(),
            oof_predictions={},
            per_class={},
        )

    # Recheck old files after training: nothing may drift during the five new runs.
    old_results = _load_verified_old_results(
        registry=active_registry,
        evidence_root=evidence_directory,
        root=project_root,
        split_sha256=split_sha256,
    )
    registry_rows = active_registry.read().set_index("run_id", drop=False)
    for fold, result in enumerate(new_results):
        if (
            result.status != "completed"
            or result.fold != fold
            or result.candidate_id != TASK1_GENTLE_WEIGHTED_CANDIDATE.candidate_id
        ):
            raise ValueError(
                "weighted Task 1 results must contain the five scheduled completed folds"
            )
        verified = _verified_result(
            _fold_metrics_frame((result,)).iloc[0],
            registry_rows,
            root=project_root,
            split_sha256=split_sha256,
        )
        for name in ("checkpoint_path", "history_path", "prediction_path"):
            actual = _resolve_artifact_path(str(getattr(result, name)), root=project_root).resolve()
            if actual != getattr(verified, name).resolve():
                raise ValueError(f"Task 1 returned {name} does not match registry: {result.run_id}")
    all_results = (*old_results, *new_results)
    _validate_label_identity(all_results, active_registry, label_map)
    result = _build_full_evidence(all_results, splits, label_map, list(_CANDIDATES))
    _write_full_evidence(result, evidence_root=evidence_directory)
    return result
